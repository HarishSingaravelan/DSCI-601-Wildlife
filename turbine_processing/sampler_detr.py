# turbine_processing/sampler_detr.py
"""
DETR-optimized sampler that reduces overfitting on minority classes
while still maintaining class balance
"""

from __future__ import annotations

import time
import random
from typing import Dict, List, Iterable, Optional

import torch
from torch.utils.data import Sampler


class DETRBalancedSampler(Sampler):
    """
    Modified balanced sampler for DETR that:
    1. Still does 50/50 background/object split
    2. Balances classes BUT limits repetition of minority classes
    3. Increases variety by using sqrt weighting instead of pure balance
    
    This prevents DETR from memorizing the 88 yellow_bellied_sapsucker images
    repeated 42x per epoch.
    
    Args:
        dataset: CocoDetection-like dataset
        epoch_size: number of samples in an epoch
        balance_mode: 'equal' (old behavior) or 'sqrt' (DETR-optimized)
        seed: optional RNG seed
    """

    def __init__(
        self, 
        dataset, 
        epoch_size: Optional[int] = None,
        balance_mode: str = 'sqrt',  # 'equal' or 'sqrt'
        seed: Optional[int] = None
    ):
        self.dataset = dataset
        self.coco = getattr(dataset, "coco", None)
        if self.coco is None:
            raise ValueError("Sampler requires dataset with `.coco` attribute")

        self.dataset_len = len(dataset)
        self.epoch_size = epoch_size if epoch_size is not None else self.dataset_len
        self.balance_mode = balance_mode
        self.seed = seed

        # Build index maps
        self.index_to_imgid = {idx: imgid for idx, imgid in enumerate(dataset.ids)}
        self.background_indices: List[int] = []
        self.class_to_indices: Dict[int, List[int]] = {}

        for idx, imgid in self.index_to_imgid.items():
            ann_ids = self.coco.getAnnIds(imgIds=[imgid])
            if not ann_ids:
                self.background_indices.append(idx)
            else:
                anns = self.coco.loadAnns(ann_ids)
                cats = set(int(a["category_id"]) for a in anns)
                for cat in cats:
                    self.class_to_indices.setdefault(cat, []).append(idx)

        self.classes = sorted(self.class_to_indices.keys())

        # Print class distribution
        print("\n" + "="*60)
        print("DETR Balanced Sampler - Class Distribution")
        print("="*60)
        print(f"Background images: {len(self.background_indices)}")
        for cls in self.classes:
            n = len(self.class_to_indices[cls])
            cat_name = self.coco.loadCats([cls])[0]['name']
            print(f"  Class {cls} ({cat_name:30s}): {n:5d} images")
        print(f"\nBalance mode: {balance_mode}")
        print("="*60 + "\n")

    def __len__(self) -> int:
        return self.epoch_size

    def _rng(self):
        if self.seed is None:
            return random.Random(time.time_ns())
        return random.Random(self.seed + int(time.time_ns() & 0xFFFFF))

    def __iter__(self) -> Iterable[int]:
        rng = self._rng()

        bg_half = self.epoch_size // 2
        animal_half = self.epoch_size - bg_half

        # ── Background sampling (unchanged) ──
        if len(self.background_indices) >= bg_half:
            bg_indices = rng.sample(self.background_indices, k=bg_half)
        else:
            bg_indices = list(self.background_indices) if self.background_indices else []
            needed = bg_half - len(bg_indices)
            if self.background_indices:
                bg_indices += [rng.choice(self.background_indices) for _ in range(needed)]
            else:
                bg_indices = [rng.randrange(self.dataset_len) for _ in range(bg_half)]

        # ── Animal sampling with balance mode ──
        animal_indices: List[int] = []

        if not self.classes:
            animal_indices = [rng.randrange(self.dataset_len) for _ in range(animal_half)]
        else:
            if self.balance_mode == 'equal':
                animal_indices = self._sample_equal(rng, animal_half)
            elif self.balance_mode == 'sqrt':
                animal_indices = self._sample_sqrt(rng, animal_half)
            elif self.balance_mode == 'log':
                animal_indices = self._sample_log(rng, animal_half)
            else:
                raise ValueError(f"Unknown balance_mode: {self.balance_mode}")


        # Merge and shuffle
        final = bg_indices + animal_indices
        rng.shuffle(final)

        # Adjust length
        if len(final) > self.epoch_size:
            final = final[:self.epoch_size]
        elif len(final) < self.epoch_size:
            deficit = self.epoch_size - len(final)
            final += [rng.randrange(self.dataset_len) for _ in range(deficit)]

        return iter(final)

    def _sample_equal(self, rng, n: int) -> List[int]:
        """
        OLD BEHAVIOR: Equal samples per class
        
        Example with 5 classes and n=14,932:
          Each class gets 14,932 / 5 = 2,986 samples
          yellow_bellied_sapsucker (88 images) → repeated 34x
        """
        indices = []
        num_classes = len(self.classes)
        per_class = n // num_classes
        remainder = n % num_classes

        for i, cls in enumerate(self.classes):
            k = per_class + (1 if i < remainder else 0)
            cls_list = self.class_to_indices[cls]

            if k <= len(cls_list):
                indices += rng.sample(cls_list, k)
            else:
                # Oversample with replacement
                indices += list(cls_list)
                need = k - len(cls_list)
                indices += [rng.choice(cls_list) for _ in range(need)]

        return indices


    def _sample_log(self, rng, n: int) -> List[int]:
        """
        NEW: Log-weighted sampling (strongest anti-overfitting)

        Very conservative oversampling for rare classes.
        Best when some classes have <100 images.

        Example:
        trash (4742 images)                  → log(4742) ≈ 8.46
        red_winged_blackbird (264 images)    → log(264)  ≈ 5.58
        yellow_bellied_sapsucker (88 images) → log(88)   ≈ 4.48

        Rare classes still appear, but repetition is tightly controlled.
        """
        import math

        # Compute log weights (add +1 to avoid log(0))
        weights = {}
        total_log = 0.0
        for cls in self.classes:
            n_images = len(self.class_to_indices[cls])
            weight = math.log(n_images + 1)
            weights[cls] = weight
            total_log += weight

        indices = []

        for cls in self.classes:
            cls_list = self.class_to_indices[cls]
            target_samples = int((weights[cls] / total_log) * n)

            if target_samples <= len(cls_list):
                indices += rng.sample(cls_list, target_samples)
            else:
                # Minimal oversampling
                indices += list(cls_list)
                need = target_samples - len(cls_list)
                indices += [rng.choice(cls_list) for _ in range(need)]

        # Fill any remaining slots conservatively
        if len(indices) < n:
            deficit = n - len(indices)
            combined = [idx for lst in self.class_to_indices.values() for idx in lst]
            indices += [rng.choice(combined) for _ in range(deficit)]

        return indices[:n]


    def _sample_sqrt(self, rng, n: int) -> List[int]:
        """
        NEW: Square-root weighted sampling
        
        Gives minority classes more samples than their count, but not equal to majority.
        Reduces overfitting while maintaining some balance.
        
        Example with your data:
          trash (4742 images)                  → sqrt(4742) = 68.9 → 31% of samples
          red_winged_blackbird (264 images)    → sqrt(264)  = 16.2 → 7% of samples
          yellow_bellied_sapsucker (88 images) → sqrt(88)   = 9.4  → 4% of samples
          
        Instead of each class getting 20%, minority classes get proportional sqrt weight.
        yellow_bellied_sapsucker: 88 images → sampled ~600 times (7x repetition, not 34x)
        """
        import math

        # Calculate sqrt weights
        weights = {}
        total_sqrt = 0.0
        for cls in self.classes:
            n_images = len(self.class_to_indices[cls])
            weight = math.sqrt(n_images)
            weights[cls] = weight
            total_sqrt += weight

        # Sample proportionally to sqrt weights
        indices = []
        for cls in self.classes:
            cls_list = self.class_to_indices[cls]
            target_samples = int((weights[cls] / total_sqrt) * n)

            if target_samples <= len(cls_list):
                # Sample without replacement
                indices += rng.sample(cls_list, target_samples)
            else:
                # Oversample with replacement (but much less than 'equal' mode)
                indices += list(cls_list)
                need = target_samples - len(cls_list)
                indices += [rng.choice(cls_list) for _ in range(need)]

        # Fill any remaining slots
        if len(indices) < n:
            deficit = n - len(indices)
            combined = [idx for lst in self.class_to_indices.values() for idx in lst]
            indices += [rng.choice(combined) for _ in range(deficit)]

        return indices[:n]