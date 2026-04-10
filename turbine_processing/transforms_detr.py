# processing/transforms_detr.py
"""
Lighter transforms for DETR to prevent overfitting
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbumentationsDetectionTransform:
    """Adapter to use an Albumentations Compose with our detection dataset."""

    def __init__(self, augment: A.Compose) -> None:
        self.augment = augment

    def __call__(self, image, target):
        # Convert PIL -> numpy
        if not isinstance(image, np.ndarray):
            image_np = np.array(image)
        else:
            image_np = image

        # Prepare boxes/labels
        boxes = target["boxes"].tolist()
        labels = target["labels"].tolist()

        # Run Albumentations
        transformed = self.augment(
            image=image_np,
            bboxes=boxes,
            labels=labels,
        )

        # Get transformed image
        out_img = transformed["image"]

        # Ensure float32 tensor in [0,1]
        if not isinstance(out_img, torch.Tensor):
            out_img = torch.from_numpy(out_img)

        if out_img.dtype == torch.uint8:
            out_img = out_img.float() / 255.0

        if out_img.ndim == 3 and out_img.shape[-1] == 3:
            out_img = out_img.permute(2, 0, 1)  # HWC → CHW

        # Rebuild boxes back to tensors
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


def get_train_transform_detr_minimal():
    """
    MINIMAL augmentation for DETR - Use this if overfitting badly
    Only horizontal flip + color jitter
    """
    train_tf = A.Compose([
        # Only horizontal flip (least invasive)
        A.HorizontalFlip(p=0.5),
        
        # Minimal color variation
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
            p=0.3
        ),
        
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format="pascal_voc", 
        label_fields=["labels"],
        min_area=1.0, 
        min_visibility=0.3
    ))

    return AlbumentationsDetectionTransform(train_tf)


def get_train_transform_detr_light():
    """
    LIGHT augmentation for DETR - Recommended starting point
    HorizontalFlip + mild color/brightness changes
    """
    train_tf = A.Compose([
        # Geometric - only flip
        A.HorizontalFlip(p=0.5),
        
        # Color variations (mild)
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
            p=0.3
        ),
        
        # Optional: Very mild blur
        A.GaussianBlur(blur_limit=(3, 3), p=0.1),
        
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format="pascal_voc", 
        label_fields=["labels"],
        min_area=1.0, 
        min_visibility=0.3
    ))

    return AlbumentationsDetectionTransform(train_tf)


def get_train_transform_detr_medium():
    """
    MEDIUM augmentation for DETR - Use after model starts converging
    More augmentation but still safe
    """
    train_tf = A.Compose([
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.03,      # Reduced from 0.0625
            scale_limit=0.05,       # Reduced from 0.1
            rotate_limit=10,        # Reduced from 15
            p=0.3                   # Reduced from 0.5
        ),
        
        # Color
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.3
        ),
        
        # Blur/noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            # FIXED: Changed var_limit to var_range to match Albumentations updates
            A.GaussNoise(var_range=(10.0, 30.0), p=1.0),
        ], p=0.2),
        
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format="pascal_voc", 
        label_fields=["labels"],
        min_area=1.0, 
        min_visibility=0.3
    ))

    return AlbumentationsDetectionTransform(train_tf)


def get_val_transform_detr():
    """
    Validation transform for DETR - NO augmentation
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


# ============================================================================
# PROGRESSIVE AUGMENTATION STRATEGY
# ============================================================================

def get_progressive_transform(epoch: int):
    """
    Progressive augmentation: start light, add more as training progresses
    
    Args:
        epoch: Current training epoch
        
    Returns:
        Appropriate transform for current epoch
        
    Usage:
        transform = get_progressive_transform(current_epoch)
    """
    if epoch < 20:
        # Epochs 0-19: Minimal augmentation
        return get_train_transform_detr_minimal()
    elif epoch < 50:
        # Epochs 20-49: Light augmentation
        return get_train_transform_detr_light()
    else:
        # Epochs 50+: Medium augmentation
        return get_train_transform_detr_medium()


# ============================================================================
# USAGE GUIDE
# ============================================================================

"""
HOW TO USE THESE TRANSFORMS:

1. IF OVERFITTING BADLY (val_loss > 2x train_loss):
   Use get_train_transform_detr_minimal()

2. IF MODERATE OVERFITTING (val_loss > 1.5x train_loss):
   Use get_train_transform_detr_light()

3. IF TRAINING IS STABLE (val_loss ~ train_loss):
   Use get_train_transform_detr_medium()

4. FOR PROGRESSIVE TRAINING:
   Use get_progressive_transform(epoch)

EXAMPLES:

# In your training script:
from processing.transforms_detr import (
    get_train_transform_detr_minimal,
    get_train_transform_detr_light,
    get_train_transform_detr_medium,
    get_val_transform_detr
)

# Minimal (if overfitting):
train_transform = get_train_transform_detr_minimal()

# Light (recommended start):
train_transform = get_train_transform_detr_light()

# Medium (after convergence):
train_transform = get_train_transform_detr_medium()

# Validation (always no augmentation):
val_transform = get_val_transform_detr()
"""