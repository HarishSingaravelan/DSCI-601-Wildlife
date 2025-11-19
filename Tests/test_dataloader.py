# tests/test_dataloader.py

import torch
from torch.utils.data import Dataset

from turbine_processing.dataloader import TurbineDataLoader


class DummyDetectionDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        img = torch.rand(3, 64, 64)
        target = {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
            "area": torch.tensor([100.0]),
            "iscrowd": torch.tensor([0]),
            "image_id": torch.tensor([idx]),
        }
        return img, target


def test_turbine_dataloader_batch_structure():
    train_ds = DummyDetectionDataset()
    val_ds = DummyDetectionDataset()

    loader_builder = TurbineDataLoader(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=2,
        num_workers=0,
    )

    train_loader, _ = loader_builder.get_loaders()
    images, targets = next(iter(train_loader))

    assert len(images) == 2
    assert isinstance(images[0], torch.Tensor)
    assert isinstance(targets[0], dict)
    assert "boxes" in targets[0]
    assert "labels" in targets[0]
