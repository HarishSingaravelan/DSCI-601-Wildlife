# turbine_processing/sampler.py

from __future__ import annotations

import time
import random
from typing import Dict, List, Iterable, Optional

import torch
from torch.utils.data import Sampler


class DynamicBalancedSampler(Sampler):
    """
    Dynamic per-epoch sampler that builds a balanced index list each time it's iterated.
    - 50% background images (images with 0 annotations) drawn WITHOUT replacement where possible.
    - 50% animal-containing images, with each animal class equally represented.
      Minority classes are oversampled WITH replacement when necessary.
    - Final index list is shuffled and returned.

    Args:
        dataset: a CocoDetection-like dataset (has .ids and .coco)
        epoch_size: number of samples in an epoch (defaults to len(dataset))
        seed: optional RNG seed (if you want reproducible epochs)
    """

    def __init__(self, dataset, epoch_size: Optional[int] = None, seed: Optional[int] = None):
        self.dataset = dataset
        self.coco = getattr(dataset, "coco", None)
        if self.coco is None:
            raise ValueError("DynamicBalancedSampler requires a dataset with `.coco` attribute (torchvision CocoDetection).")

        self.dataset_len = len(dataset)
        self.epoch_size = epoch_size if epoch_size is not None else self.dataset_len
        self.seed = seed

        # Precompute maps: index (0..N-1) -> image_id
        # dataset.ids is ordered list of image ids (COCO)
        self.index_to_imgid = {idx: imgid for idx, imgid in enumerate(dataset.ids)}

        # Build background and per-class image index lists
        self.background_indices: List[int] = []
        self.class_to_indices: Dict[int, List[int]] = {}

        # Populate lists
        for idx, imgid in self.index_to_imgid.items():
            ann_ids = self.coco.getAnnIds(imgIds=[imgid])
            if not ann_ids:
                # No annotations -> background image
                self.background_indices.append(idx)
            else:
                anns = self.coco.loadAnns(ann_ids)
                # Collect set of category_ids present in this image
                cats = set()
                for a in anns:
                    cats.add(int(a["category_id"]))
                for cat in cats:
                    self.class_to_indices.setdefault(cat, []).append(idx)

        # Ensure consistent ordering of classes
        self.classes = sorted(self.class_to_indices.keys())

        if len(self.background_indices) == 0:
            print("[WARNING] No background images found (images with zero annotations). Sampler will sample replacements.")

    def __len__(self) -> int:
        return self.epoch_size

    def _rng(self):
        # Create a per-iteration RNG
        if self.seed is None:
            return random.Random(time.time_ns())
        return random.Random(self.seed + int(time.time_ns() & 0xFFFFF))

    def __iter__(self) -> Iterable[int]:
        rng = self._rng()

        # Half sizes
        bg_half = self.epoch_size // 2
        animal_half = self.epoch_size - bg_half

        # --- Sample background indices (without replacement when possible) ---
        bg_indices = []
        if len(self.background_indices) >= bg_half:
            bg_indices = rng.sample(self.background_indices, k=bg_half)
        else:
            # take all available without replacement, then sample with replacement for remainder
            if len(self.background_indices) > 0:
                bg_indices = list(self.background_indices)
                needed = bg_half - len(bg_indices)
                bg_indices += [rng.choice(self.background_indices) for _ in range(needed)]
            else:
                # No background images: fallback to sampling from entire dataset (with replacement)
                bg_indices = [rng.randrange(self.dataset_len) for _ in range(bg_half)]
                print("[WARNING] No background images available; sampled fallback indices for background portion.")

        # --- Sample animal indices balanced across classes ---
        animal_indices: List[int] = []
        num_classes = max(1, len(self.classes))
        per_class = animal_half // num_classes
        remainder = animal_half % num_classes

        # If there are zero classes (no annotations anywhere), fallback to random sampling
        if len(self.classes) == 0:
            animal_indices = [rng.randrange(self.dataset_len) for _ in range(animal_half)]
        else:
            for i, cls in enumerate(self.classes):
                k = per_class + (1 if i < remainder else 0)
                cls_list = self.class_to_indices.get(cls, [])
                if len(cls_list) == 0:
                    # No images for class, skip
                    continue
                if k <= len(cls_list):
                    # sample without replacement
                    animal_indices += rng.sample(cls_list, k)
                else:
                    # need to oversample with replacement
                    animal_indices += list(cls_list)
                    need = k - len(cls_list)
                    animal_indices += [rng.choice(cls_list) for _ in range(need)]

            # If due to skipping some classes we have fewer animal_indices than needed, fill randomly
            if len(animal_indices) < animal_half:
                deficit = animal_half - len(animal_indices)
                combined = [idx for lst in self.class_to_indices.values() for idx in lst]
                if combined:
                    animal_indices += [rng.choice(combined) for _ in range(deficit)]
                else:
                    animal_indices += [rng.randrange(self.dataset_len) for _ in range(deficit)]

        # Merge and shuffle
        final = bg_indices + animal_indices
        rng.shuffle(final)

        # In case final length differs (edge conditions), adjust
        if len(final) > self.epoch_size:
            final = final[: self.epoch_size]
        elif len(final) < self.epoch_size:
            # pad with random indices
            final += [rng.randrange(self.dataset_len) for _ in range(self.epoch_size - len(final))]

        return iter(final)
