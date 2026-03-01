# turbine_processing/sampler_adaptive.py
"""
Adaptive sampler that adjusts class distribution based on validation performance.

Strategy:
  - Starts with equal class distribution (all classes get equal samples)
  - After each evaluation, analyzes per-class AP/precision
  - Increases sampling for poorly-performing classes
  - Decreases sampling for well-performing classes
  - Gradually learns optimal distribution for your specific dataset
"""

from __future__ import annotations

import time
import random
import math
from typing import Dict, List, Iterable, Optional

import torch
from torch.utils.data import Sampler


class AdaptiveDETRSampler(Sampler):
    """
    Adaptive sampler that adjusts class distribution based on performance.
    
    How it works:
      1. Initially: all classes get equal samples (equal mode)
      2. After evaluation: update_class_weights(per_class_metrics)
      3. Next epoch: sample based on updated weights
      4. Repeat
    
    Classes with low AP get boosted, classes with high AP get reduced.
    This automatically balances the learning difficulty across classes.
    
    Args:
        dataset: CocoDetection-like dataset
        epoch_size: number of samples per epoch
        initial_mode: 'equal', 'sqrt', or 'log' for first few epochs
        adaptation_rate: how quickly to adapt (0.1 = slow, 0.5 = fast)
        min_weight: minimum weight for any class (prevents starvation)
        max_weight: maximum weight for any class (prevents domination)
        seed: optional RNG seed
    """

    def __init__(
        self,
        dataset,
        epoch_size: Optional[int] = None,
        initial_mode: str = 'equal',
        adaptation_rate: float = 0.3,
        min_weight: float = 0.1,
        max_weight: float = 5.0,
        background_ratio: float = 0.5,       
        dynamic_background: bool = False,    
        min_bg_ratio: float = 0.15,          
        max_bg_ratio: float = 0.5,           
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.coco = getattr(dataset, "coco", None)
        if self.coco is None:
            raise ValueError("Sampler requires dataset with `.coco` attribute")

        self.dataset_len = len(dataset)
        self.epoch_size = epoch_size if epoch_size is not None else self.dataset_len
        self.initial_mode = initial_mode
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # New Background Tracking
        self.bg_ratio = background_ratio
        self.dynamic_bg = dynamic_background
        self.min_bg_ratio = min_bg_ratio
        self.max_bg_ratio = max_bg_ratio
        
        self.seed = seed

        # Build index maps (Keep your existing index map code here...)
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
        self.class_weights = self._initialize_weights()
        self.adaptation_history = [] 

        self._print_distribution()

    def _initialize_weights(self) -> Dict[int, float]:
        """Initialize weights based on initial_mode"""
        weights = {}
        
        if self.initial_mode == 'equal':
            # All classes get weight 1.0
            for cls in self.classes:
                weights[cls] = 1.0
                
        elif self.initial_mode == 'sqrt':
            # Sqrt of sample count
            for cls in self.classes:
                n = len(self.class_to_indices[cls])
                weights[cls] = math.sqrt(n)
                
        elif self.initial_mode == 'log':
            # Log of sample count
            for cls in self.classes:
                n = len(self.class_to_indices[cls])
                weights[cls] = math.log(n + 1)
                
        else:
            raise ValueError(f"Unknown initial_mode: {self.initial_mode}")
        
        # Normalize weights
        total = sum(weights.values())
        weights = {cls: w / total for cls, w in weights.items()}
        
        return weights

    # def print_current_distribution(self):
    #     """Print current expected sampling distribution for next epoch"""
    #     print("\n" + "="*100)
    #     print("Next Epoch - Expected Class Distribution")
    #     print("="*100)
        
    #     animal_half = self.epoch_size // 2
    #     total_weight = sum(self.class_weights.values())
        
    #     print(f"\n{'ID':<4} {'Class Name':<32} {'Weight':<8} {'Images':<8} {'Samples/Epoch':<15} {'Repetition':<12}")
    #     print("-" * 100)
        
    #     # Sort by weight (descending) to show most-sampled classes first
    #     sorted_classes = sorted(self.classes, key=lambda c: self.class_weights[c], reverse=True)
        
    #     for cls in sorted_classes:
    #         n_images = len(self.class_to_indices[cls])
    #         weight = self.class_weights[cls]
    #         expected = int((weight / total_weight) * animal_half)
    #         repetition = expected / n_images if n_images > 0 else 0
            
    #         cat_name = self.coco.loadCats([cls])[0]['name']
            
    #         # Highlight high repetition
    #         if repetition > 10:
    #             marker = "⚠️ "
    #         elif repetition > 5:
    #             marker = "🟡"
    #         else:
    #             marker = "🟢"
            
    #         print(f"{marker} {cls:<2} {cat_name[:30]:<32} {weight:.4f}   {n_images:<8} {expected:<15} {repetition:.1f}x")
        
    #     print("="*100 + "\n")

    def print_current_distribution(self):
        """Print current expected sampling distribution for next epoch"""
        print("\n" + "="*100)
        print("Next Epoch - Expected Class Distribution")
        print("="*100)
        
        # 1. FIX THE MATH: Use the dynamic bg_ratio instead of hardcoded 50%
        bg_count = int(self.epoch_size * self.bg_ratio)
        animal_count = self.epoch_size - bg_count
        total_weight = sum(self.class_weights.values())
        
        print(f"\n{'ID':<4} {'Class Name':<32} {'Weight':<8} {'Images':<8} {'Samples/Epoch':<15} {'Repetition':<12}")
        print("-" * 100)
        
        # 2. ADD THE BACKGROUND ROW AT THE TOP
        n_bg_images = len(self.background_indices)
        bg_repetition = bg_count / n_bg_images if n_bg_images > 0 else 0
        
        if bg_repetition > 10:
            bg_marker = "⚠️ "
        elif bg_repetition > 5:
            bg_marker = "🟡"
        else:
            bg_marker = "🟢"
            
        print(f"{bg_marker} {'0':<2} {'background':<32} {self.bg_ratio:.4f}   {n_bg_images:<8} {bg_count:<15} {bg_repetition:.1f}x")
        print("-" * 100)
        
        # 3. PRINT THE BIRD CLASSES
        # Sort by weight (descending) to show most-sampled classes first
        sorted_classes = sorted(self.classes, key=lambda c: self.class_weights[c], reverse=True)
        
        for cls in sorted_classes:
            n_images = len(self.class_to_indices[cls])
            weight = self.class_weights[cls]
            expected = int((weight / total_weight) * animal_count)
            repetition = expected / n_images if n_images > 0 else 0
            
            cat_name = self.coco.loadCats([cls])[0]['name']
            
            # Highlight high repetition
            if repetition > 10:
                marker = "⚠️ "
            elif repetition > 5:
                marker = "🟡"
            else:
                marker = "🟢"
            
            print(f"{marker} {cls:<2} {cat_name[:30]:<32} {weight:.4f}   {n_images:<8} {expected:<15} {repetition:.1f}x")
        
        print("="*100 + "\n")
    
    def update_class_weights(self, class_metrics: Dict[int, float], bg_accuracy: float = None):
        """
        Update class weights based on per-class performance metrics.
        
        Args:
            class_metrics: Dict mapping class_id -> performance metric (e.g., AP)
                          Higher values = better performance
                          Range: 0.0 to 1.0
        
        Strategy:
            - Classes with low performance get weight boost
            - Classes with high performance get weight reduction
            - Adaptation is gradual (controlled by adaptation_rate)
        
        Example:
            class_metrics = {
                1: 0.85,  # red_tailed_hawk: high AP → reduce weight
                2: 0.20,  # red_winged_blackbird: low AP → increase weight
                3: 0.05,  # ovenbird: very low AP → large increase
                ...
            }
        """
        if not class_metrics:
            return
        
        print("\n" + "="*70)
        print("Adaptive Sampler - Updating Weights")
        print("="*70)

        # --- DYNAMIC BACKGROUND LOGIC ---
        if self.dynamic_bg and bg_accuracy is not None:
            old_bg = self.bg_ratio
            # Target ratio: High accuracy = lower ratio. 
            # (e.g., 90% acc -> 1.0 - 0.90 + 0.1 = 0.20 ratio target)
            target_bg = 1.0 - bg_accuracy + 0.1
            
            blended_bg = (1 - self.adaptation_rate) * old_bg + (self.adaptation_rate * target_bg)
            self.bg_ratio = max(self.min_bg_ratio, min(self.max_bg_ratio, blended_bg))
            
            bg_change = ((self.bg_ratio - old_bg) / old_bg * 100) if old_bg > 0 else 0
            direction = "↑" if bg_change > 0 else "↓"
            print(f"Background Ratio  : {old_bg:.3f} -> {self.bg_ratio:.3f}  {direction}{abs(bg_change):.1f}% (Acc: {bg_accuracy:.2f})")
            print("-" * 70)

        # --- BIRD CLASS LOGIC (Your existing code) ---
        new_weights = {}
        for cls in self.classes:
            if cls not in class_metrics:
                new_weights[cls] = self.class_weights[cls]
                continue
            
            ap = class_metrics[cls]
            target_weight = 1.0 - ap + 0.1
            old_weight = self.class_weights[cls]
            blended_weight = ((1 - self.adaptation_rate) * old_weight + self.adaptation_rate * target_weight)
            
            clamped_weight = max(self.min_weight, min(self.max_weight, blended_weight))
            new_weights[cls] = clamped_weight
        
        # Normalize
        total = sum(new_weights.values())
        new_weights = {cls: w / total for cls, w in new_weights.items()}
        
        # Print changes
        print(f"{'Class':<10} {'Name':<30} {'Old':<8} {'New':<8} {'Change':<8} {'AP':<6}")
        print("-" * 70)
        
        for cls in sorted(self.classes):
            old_w = self.class_weights[cls]
            new_w = new_weights[cls]
            change = ((new_w - old_w) / old_w * 100) if old_w > 0 else 0
            ap = class_metrics.get(cls, 0.0)
            
            if abs(change) > 10:
                cat_name = self.coco.loadCats([cls])[0]['name']
                direction = "↑" if change > 0 else "↓"
                print(f"{cls:<10} {cat_name[:28]:<30} {old_w:.4f}   {new_w:.4f}   {direction}{abs(change):>5.1f}%  {ap:.3f}")
        
        self.class_weights = new_weights
        print("="*70 + "\n")
        
        # Print expected distribution for next epoch
        self.print_current_distribution()

    def _print_distribution(self):
        """Print current class distribution"""
        print("\n" + "="*70)
        print("Adaptive DETR Sampler - Class Distribution")
        print("="*70)
        print(f"Background images: {len(self.background_indices)}")
        print(f"Initial mode: {self.initial_mode}")
        print(f"Adaptation rate: {self.adaptation_rate}")
        print(f"\nClass weights:")
        
        # Sort by weight (descending)
        sorted_classes = sorted(
            self.classes,
            key=lambda c: self.class_weights[c],
            reverse=True
        )
        
        for cls in sorted_classes:
            n = len(self.class_to_indices[cls])
            weight = self.class_weights[cls]
            cat_name = self.coco.loadCats([cls])[0]['name']
            print(f"  Class {cls:2d} ({cat_name:30s}): {n:4d} images, weight={weight:.4f}")
        
        print("="*70 + "\n")

    def __len__(self) -> int:
        return self.epoch_size

    def _rng(self):
        if self.seed is None:
            return random.Random(time.time_ns())
        return random.Random(self.seed + int(time.time_ns() & 0xFFFFF))

    def __iter__(self) -> Iterable[int]:
        rng = self._rng()

        # DYNAMIC SPLIT BASED ON CURRENT RATIO
        bg_count = int(self.epoch_size * self.bg_ratio)
        animal_count = self.epoch_size - bg_count

        # ── Background sampling ──
        if len(self.background_indices) >= bg_count:
            bg_indices = rng.sample(self.background_indices, k=bg_count)
        else:
            bg_indices = list(self.background_indices) if self.background_indices else []
            needed = bg_count - len(bg_indices)
            if self.background_indices:
                bg_indices += [rng.choice(self.background_indices) for _ in range(needed)]
            else:
                bg_indices = [rng.randrange(self.dataset_len) for _ in range(bg_count)]

        # ── Animal sampling with adaptive weights ──
        animal_indices = self._sample_weighted(rng, animal_count)

        # Merge and shuffle
        final = bg_indices + animal_indices
        rng.shuffle(final)

        # Adjust length (safety catch)
        if len(final) > self.epoch_size:
            final = final[:self.epoch_size]
        elif len(final) < self.epoch_size:
            deficit = self.epoch_size - len(final)
            final += [rng.randrange(self.dataset_len) for _ in range(deficit)]

        return iter(final)

    def _sample_weighted(self, rng, n: int) -> List[int]:
        """
        Sample according to current class weights.
        
        Classes with higher weights get more samples.
        """
        if not self.classes:
            return [rng.randrange(self.dataset_len) for _ in range(n)]
        
        indices = []
        total_weight = sum(self.class_weights.values())
        
        for cls in self.classes:
            cls_list = self.class_to_indices[cls]
            weight = self.class_weights[cls]
            
            # Number of samples for this class
            target_samples = int((weight / total_weight) * n)
            
            if target_samples <= len(cls_list):
                # Sample without replacement
                indices += rng.sample(cls_list, target_samples)
            else:
                # Oversample with replacement
                indices += list(cls_list)
                need = target_samples - len(cls_list)
                indices += [rng.choice(cls_list) for _ in range(need)]
        
        # Fill deficit if any
        if len(indices) < n:
            deficit = n - len(indices)
            combined = [idx for lst in self.class_to_indices.values() for idx in lst]
            if combined:
                indices += [rng.choice(combined) for _ in range(deficit)]
            else:
                indices += [rng.randrange(self.dataset_len) for _ in range(deficit)]
        
        return indices[:n]

    def get_current_weights(self) -> Dict[int, float]:
        """Get current class weights (for debugging/visualization)"""
        return self.class_weights.copy()

    def get_adaptation_history(self) -> List[Dict]:
        """Get history of weight adaptations"""
        return self.adaptation_history.copy()

