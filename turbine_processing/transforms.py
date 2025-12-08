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
    ... (omitting __init__ and __call__ for brevity as they are unchanged)
    """

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

        # --- Ensure float32 tensor in [0,1] ---
        if not isinstance(out_img, torch.Tensor):
            out_img = torch.from_numpy(out_img)

        # If tensor is uint8 → convert to float32
        if out_img.dtype == torch.uint8:
            out_img = out_img.float() / 255.0

        
        if out_img.ndim == 3 and out_img.shape[-1] == 3:
            out_img = out_img.permute(2, 0, 1)  # HWC → CHW

        # --- Rebuild boxes back to tensors ---
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
    """
    Training transform with added Perspective, Coarse Dropout, Shift Scale Rotate, 
    and Random Crop.
    """
    train_tf = A.Compose([
        # --- Geometric/Position Augmentations ---
        A.HorizontalFlip(p=0.5), # Standard flip
        
        # Shift Scale Rotate: good for simulating viewpoint changes
        A.ShiftScaleRotate(
            shift_limit=0.0625,  # Max fractional shift
            scale_limit=0.1,     # Max fractional scale
            rotate_limit=15,     # Max rotation angle
            p=0.5
        ),
        
        # Perspective: adds complex distortion, needs careful use
        A.Perspective(
            scale=(0.05, 0.1), # Controls distortion strength
            p=0.25 # Lower probability to keep bounding boxes valid
        ),
        
        # Random Crop: drastically changes context and object visibility
        A.RandomSizedBBoxSafeCrop(
            width=512, height=512, # Target size
            p=0.25, # Moderate probability
            # The crop area must include all bounding boxes after transformation
            
        ), 

        # --- Color/Pixel Augmentations ---
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(p=0.2),
        
        # --- Structural Augmentations ---
        # Coarse Dropout (Cutout): helps the model learn based on parts, not wholes
        A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32, # Max number and size of holes
            fill_value=0, # Black patches
            p=0.2
        ),

        # --- Final step for PyTorch compatibility ---
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format="pascal_voc", 
        label_fields=["labels"],
        # Drop boxes that fall out of bounds or become too small after geometric transforms
        min_area=1.0, 
        min_visibility=0.1 # Minimum portion of the box visible after crop/transforms
    ))

    return AlbumentationsDetectionTransform(train_tf)


def get_val_transform() -> AlbumentationsDetectionTransform:
    """
    Validation transform - tensor conversions.
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