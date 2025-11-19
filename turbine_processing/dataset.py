# turbine_processing/dataset.py

import os
from typing import Any, Dict, Tuple, List, Optional, Callable

import torch
from torchvision.datasets import CocoDetection
from PIL import Image


class TurbineCocoDataset(CocoDetection):
    """
    COCO-style dataset wrapper for the turbine wildlife project.

    Returns:
        image: PIL Image or Tensor (depending on transforms)
        target: dict with:
            boxes, labels, area, iscrowd, image_id
    """

    def __init__(
        self,
        images_dir: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
    ) -> None:

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not os.path.isfile(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        super().__init__(root=images_dir, annFile=ann_file)
        self._transforms = transforms

    def _safe_load(self, idx: int):
        """
        Try loading image + annotations.
        Catch corrupted images and skip to next index.
        """
        max_attempts = 5
        attempts = 0

        while attempts < max_attempts:
            try:
                img, anns = super().__getitem__(idx)

                # force-load to catch truncated/broken images
                img.load()
                return img, anns, idx

            except Exception as e:
                print(f"[WARNING] Corrupted image at index {idx}, skipping. Error: {e}")

                attempts += 1
                idx = (idx + 1) % len(self)

        # fallback if every attempt failed
        print("[ERROR] Too many corrupted images in a row. Returning blank image.")
        return Image.new("RGB", (224, 224)), [], idx

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        # ---- SAFE IMAGE LOADING ----
        img, anns, valid_idx = self._safe_load(idx)

        # ---- TARGET BUILDING (your original logic) ----
        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for obj in anns:
            bbox = obj["bbox"]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = x_min + bbox[2]
            y_max = y_min + bbox[3]
            boxes.append([x_min, y_min, x_max, y_max])

            labels.append(obj.get("category_id", 1))
            areas.append(float(obj.get("area", bbox[2] * bbox[3])))
            iscrowd.append(int(obj.get("iscrowd", 0)))

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area_tensor = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.int64)

        image_id_tensor = torch.tensor([self.ids[valid_idx]], dtype=torch.int64)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
            "image_id": image_id_tensor,
        }

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target
