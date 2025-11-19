# tests/test_trainer.py

import torch
from torch.utils.data import Dataset, DataLoader

from modeling.model import get_model
from modeling.trainer import Trainer
from processing.dataloader import TurbineDataLoader


class TinyDetectionDataset(Dataset):
    def __len__(self):
        return 2

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


def test_trainer_one_epoch_runs():
    dataset = TinyDetectionDataset()
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=TurbineDataLoader.collate_fn,
    )

    # 5 foreground + 1 background
    model = get_model(num_classes=6)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005,
    )

    device = torch.device("cpu")
    trainer = Trainer(model=model, optimizer=optimizer, device=device)

    avg_loss = trainer.train_one_epoch(data_loader=loader, epoch=0, print_every=10)
    assert avg_loss >= 0.0
