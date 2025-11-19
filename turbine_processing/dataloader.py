from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .sampler import DynamicBalancedSampler


class TurbineDataLoader:
    """
    Wraps PyTorch DataLoader creation for train/val datasets.

    Parameters:
        train_dataset, val_dataset: PyTorch Dataset objects
        batch_size: int
        num_workers: int
        use_balanced_sampler: bool = False  # whether to use dynamic oversampling sampler
    """

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int = 2,
        num_workers: int = 4,
        use_balanced_sampler: bool = False,
    ) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_balanced_sampler = use_balanced_sampler

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for detection models.

        Args:
            batch: list of (image, target) pairs

        Returns:
            images: tuple of images
            targets: tuple of target dicts
        """
        return tuple(zip(*batch))

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        # Enabling pin_memory=True greatly speeds up data transfer to CUDA.
        pin_memory = True 
        
        if self.use_balanced_sampler:
            sampler = DynamicBalancedSampler(
                dataset=self.train_dataset,
                epoch_size=len(self.train_dataset),  # keep epoch size equal to dataset length
            )
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,  # sampler provides ordering
                sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=pin_memory, # Enabled
            )
        else:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=pin_memory, # Enabled
            )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory, # Enabled
        )

        return train_loader, val_loader