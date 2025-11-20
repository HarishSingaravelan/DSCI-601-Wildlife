# processing/transforms.py

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbumentationsDetectionTransform:
    """
    Adapter to use an Albumentations Compose with our detection dataset.

    Expects:
        - image: PIL.Image or ndarray (H, W, C)
        - target: dict with keys "boxes" (Tensor[N,4]) and "labels" (Tensor[N])

    Returns:
        - image: torch.FloatTensor[C, H, W]
        - target: updated dict with transformed boxes/labels/area tensors
    """

    def __init__(self, augment: A.Compose) -> None:
        self.augment = augment

    def __call__(self, image, target):
       # Convert PIL → numpy
        if not isinstance(image, np.ndarray):
            image_np = np.array(image)
        else:
            image_np = image

        boxes = target["boxes"].tolist()
        labels = target["labels"].tolist()

        transformed = self.augment(
            image=image_np,
            bboxes=boxes,
            labels=labels,
        )

        out_img = transformed["image"]

        # Convert to tensor if needed
        if not isinstance(out_img, torch.Tensor):
            out_img = torch.from_numpy(out_img)

        if out_img.dtype == torch.uint8:
            out_img = out_img.float() / 255.0

        if out_img.ndim == 3 and out_img.shape[-1] == 3:
            out_img = out_img.permute(2, 0, 1)

        # PRINT SHAPE HERE
        #print("[DEBUG] After transform image tensor shape:", out_img.shape)

        # Rebuild boxes
        new_boxes = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
        new_labels = torch.as_tensor(transformed["labels"], dtype=torch.int64)

        if new_boxes.numel() == 0:
            new_boxes = new_boxes.view(0, 4)
            new_labels = new_labels.view(0)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            wh = new_boxes[:, 2:] - new_boxes[:, :2]
            areas = wh[:, 0] * wh[:, 1]

        target["boxes"] = new_boxes
        target["labels"] = new_labels
        target["area"] = areas

        return out_img, target


def get_train_transform() -> AlbumentationsDetectionTransform:
    train_tf = A.Compose(
        [
            A.HorizontalFlip(p=0.5),

            # Safe replacement for ShiftScaleRotate
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.05, 0.05),
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.5,
            ),

            A.RandomBrightnessContrast(p=0.2),

            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                p=0.3,
            ),

            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.3,
        ),
    )

    

    return AlbumentationsDetectionTransform(train_tf)


def get_val_transform() -> AlbumentationsDetectionTransform:
    """
    Validation transform (no heavy augmentations, just tensor conversion).
    """
    val_tf = A.Compose(
        [
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.0,
        ),
    )

    return AlbumentationsDetectionTransform(val_tf)
