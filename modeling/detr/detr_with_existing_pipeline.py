"""
DETR Model with Existing Faster R-CNN Data Pipeline - BULLETPROOF EDITION
"""

import torch
import torch.nn as nn
from transformers import (
    DetrForObjectDetection, 
    DetrImageProcessor,
    DFineForObjectDetection,
    AutoImageProcessor,
    DeformableDetrImageProcessor,
    DeformableDetrForObjectDetection
)
import transformers.models.detr.modeling_detr as modeling_detr
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
import os
import math
import torch.nn.functional as F
from tqdm.auto import tqdm

# ==============================================================================
# INNER-SIoU LOSS PATCH 
# ==============================================================================
def patch_inner_siou_loss():
    def inner_siou_matrix(boxes1, boxes2, ratio=1.25, eps=1e-7):
        x1_1, y1_1, x2_1, y2_1 = boxes1.unsqueeze(1).unbind(dim=2) 
        x1_2, y1_2, x2_2, y2_2 = boxes2.unsqueeze(0).unbind(dim=2) 
        
        w1, h1 = x2_1 - x1_1, y2_1 - y1_1
        w2, h2 = x2_2 - x1_2, y2_2 - y1_2
        cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
        cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
        
        in_w1, in_h1 = w1 * ratio, h1 * ratio
        in_w2, in_h2 = w2 * ratio, h2 * ratio
        
        in_x1_1, in_y1_1 = cx1 - in_w1 / 2, cy1 - in_h1 / 2
        in_x2_1, in_y2_1 = cx1 + in_w1 / 2, cy1 + in_h1 / 2
        in_x1_2, in_y1_2 = cx2 - in_w2 / 2, cy2 - in_h2 / 2
        in_x2_2, in_y2_2 = cx2 + in_w2 / 2, cy2 + in_h2 / 2
        
        inter_x1 = torch.max(in_x1_1, in_x1_2)
        inter_y1 = torch.max(in_y1_1, in_y1_2)
        inter_x2 = torch.min(in_x2_1, in_x2_2)
        inter_y2 = torch.min(in_y2_1, in_y2_2)
        
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        area1, area2 = in_w1 * in_h1, in_w2 * in_h2
        union = area1 + area2 - inter_area
        inner_iou = inter_area / (union + eps)
        
        s_cw = torch.abs(cx2 - cx1)
        s_ch = torch.abs(cy2 - cy1)
        Lambda = 2 * s_cw * s_ch / (s_cw**2 + s_ch**2 + eps)
        gamma = 2 - Lambda
        
        enc_x1 = torch.min(x1_1, x1_2)
        enc_y1 = torch.min(y1_1, y1_2)
        enc_x2 = torch.max(x2_1, x2_2)
        enc_y2 = torch.max(y2_1, y2_2)
        
        cw = torch.clamp(enc_x2 - enc_x1, min=eps)
        ch = torch.clamp(enc_y2 - enc_y1, min=eps)
        
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        Delta = (1 - torch.exp(-gamma * rho_x)) + (1 - torch.exp(-gamma * rho_y))
        
        omega_w = torch.abs(w1 - w2) / torch.clamp(torch.max(w1, w2), min=eps)
        omega_h = torch.abs(h1 - h2) / torch.clamp(torch.max(h1, h2), min=eps)
        Omega = (1 - torch.exp(-omega_w)) ** 4 + (1 - torch.exp(-omega_h)) ** 4
        
        siou_penalty = (Delta + Omega) / 2
        return inner_iou - siou_penalty

    modeling_detr.generalized_box_iou = inner_siou_matrix
    print("Successfully patched Hugging Face DETR to use Inner-SIoU Loss")

# TEMPORARILY DISABLED FOR DEFORMABLE DETR STABILITY
# patch_inner_siou_loss() 
# ==============================================================================


class DETRTransformAdapter:
    def __init__(self, albumentations_transform, processor):
        self.albumentations_transform = albumentations_transform
        self.processor = processor
    
    def __call__(self, image, target):
        # 1. Albumentations
        image, target = self.albumentations_transform(image, target)
        
        # 2. To Numpy HWC
        if isinstance(image, torch.Tensor):
            if image.shape[0] == 3:
                image_np = image.permute(1, 2, 0).numpy()
            else:
                image_np = image.numpy()
        else:
            image_np = np.array(image)

        # 3. BULLETPROOF SANITIZATION & SCALING
        image_np = np.nan_to_num(image_np, nan=0.0, posinf=255.0, neginf=0.0)
        if image_np.dtype != np.uint8:
            if image_np.max() <= 5.0: 
                image_np = np.interp(image_np, (image_np.min(), image_np.max()), (0, 255))
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        
        orig_h, orig_w = image_np.shape[:2]
        
        # 4. BULLETPROOF BOUNDING BOX CLAMPING
        boxes = target["boxes"].cpu().numpy() if isinstance(target["boxes"], torch.Tensor) else target["boxes"]
        labels = target["labels"].cpu().numpy() if isinstance(target["labels"], torch.Tensor) else target["labels"]
        
        valid_annotations = []
        for i in range(len(boxes)):
            if np.any(np.isnan(boxes[i])):
                continue
                
            x_min, y_min, x_max, y_max = float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])
            w, h = x_max - x_min, y_max - y_min
            
            if w > 1.0 and h > 1.0 and x_min >= 0 and y_min >= 0 and x_max <= orig_w + 10 and y_max <= orig_h + 10:
                clamped_x_max = min(x_max, float(orig_w))
                clamped_y_max = min(y_max, float(orig_h))
                clamped_w = clamped_x_max - x_min
                clamped_h = clamped_y_max - y_min
                
                if clamped_w > 0 and clamped_h > 0:
                    valid_annotations.append({
                        "bbox": [x_min, y_min, clamped_w, clamped_h],
                        "category_id": int(labels[i]),
                        "area": float(clamped_w * clamped_h),
                        "iscrowd": int(target["iscrowd"][i]) if "iscrowd" in target else 0,
                    })

        annotations = {
            "image_id": target["image_id"].item() if isinstance(target["image_id"], torch.Tensor) else target["image_id"],
            "annotations": valid_annotations
        }
        
        processed = self.processor(images=image_np, annotations=[annotations], return_tensors="pt")
        
        pixel_values = processed["pixel_values"][0]
        labels = processed["labels"][0]
        labels['orig_size'] = torch.tensor([orig_h, orig_w])
        
        return pixel_values, labels


class DETRWithExistingDataPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['training']['device'])
        
        arch = config['model'].get('architecture', 'standard_detr')
        model_name = config['model'].get('pretrained_model', 'facebook/detr-resnet-50')
        print(f"\n[INFO] Initializing Architecture: {arch.upper()} from {model_name}")

        class_names = config['model']['class_names']
        id2label = {i: name for i, name in enumerate(class_names)}
        label2id = {name: i for i, name in enumerate(class_names)}

        if arch == 'dfine':
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = DFineForObjectDetection.from_pretrained(model_name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
            self.model.to(self.device)
            param_dicts = [
                {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {"params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad], "lr": config['training']['learning_rate'] * 0.1}, 
            ]
        elif arch == 'deformable_detr':
            self.processor = DeformableDetrImageProcessor.from_pretrained(model_name, do_convert_annotations=True)
            self.model = DeformableDetrForObjectDetection.from_pretrained(model_name, num_labels=config['model']['num_object_classes'], ignore_mismatched_sizes=True)
            self.model.to(self.device)
            param_dicts = [
                {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {"params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad], "lr": config['training']['learning_rate'] * 0.1}, 
            ]
        else:
            self.processor = DetrImageProcessor.from_pretrained(model_name, do_convert_annotations=True)
            self.model = DetrForObjectDetection.from_pretrained(model_name, num_labels=config['model']['num_object_classes'], ignore_mismatched_sizes=True)
            self.model.to(self.device)
            param_dicts = [
                {"params": [p for n, p in self.model.named_parameters() if p.requires_grad]}
            ]

        import sys
        from pathlib import Path
        ROOT_DIR = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(ROOT_DIR))

        from turbine_processing.dataset import TurbineCocoDataset
        from turbine_processing.transforms_detr import get_train_transform_detr_medium, get_val_transform_detr

        train_transform = DETRTransformAdapter(get_train_transform_detr_medium(), self.processor)
        val_transform = DETRTransformAdapter(get_val_transform_detr(), self.processor)
        
        self.train_dataset = TurbineCocoDataset(images_dir=config['data']['train_images_dir'], ann_file=config['data']['train_ann_file'], transforms=train_transform)
        self.val_dataset = TurbineCocoDataset(images_dir=config['data']['val_images_dir'], ann_file=config['data']['val_ann_file'], transforms=val_transform)
        self.test_dataset = TurbineCocoDataset(images_dir=config['data']['test_images_dir'], ann_file=config['data']['test_ann_file'], transforms=val_transform)
        
        loader_kwargs = {
            "batch_size": config['training']['batch_size'],
            "num_workers": config['training']['num_workers'],
            "collate_fn": self.collate_fn,
            "pin_memory": True,
            "persistent_workers": True if config['training']['num_workers'] > 0 else False,
            "prefetch_factor": 4 if config['training']['num_workers'] > 0 else None
        }

        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **loader_kwargs)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **loader_kwargs)
        
        self.sampler = None
        self.use_adaptive_sampler = config['data'].get('use_adaptive_sampler', False)
        
        if self.use_adaptive_sampler:
            from turbine_processing.sampler_adaptive import AdaptiveDETRSampler
            self.sampler = AdaptiveDETRSampler(
                dataset=self.train_dataset,
                epoch_size=len(self.train_dataset),
                initial_mode=config['data'].get('initial_mode', 'equal'),
                adaptation_rate=config['data'].get('adaptation_rate', 0.3),
                min_weight=config['data'].get('min_weight', 0.1),
                max_weight=config['data'].get('max_weight', 5.0),
                background_ratio=config['data'].get('background_ratio', 0.5),
                dynamic_background=config['data'].get('dynamic_background', False),
                min_bg_ratio=config['data'].get('min_bg_ratio', 0.15),
                max_bg_ratio=config['data'].get('max_bg_ratio', 0.5),
            )
            self.train_loader = DataLoader(self.train_dataset, shuffle=False, sampler=self.sampler, **loader_kwargs)
            print("\n✓ USING ADAPTIVE SAMPLER")
            
        elif config['data'].get('use_balanced_sampler', False):
            from turbine_processing.sampler_detr import DETRBalancedSampler
            self.sampler = DETRBalancedSampler(dataset=self.train_dataset, epoch_size=len(self.train_dataset), balance_mode=config['data'].get('balance_mode', 'sqrt'))
            self.train_loader = DataLoader(self.train_dataset, shuffle=False, sampler=self.sampler, **loader_kwargs)
            print("\n✓ USING BALANCED SAMPLER")
            
        else:
            self.train_loader = DataLoader(self.train_dataset, shuffle=True, **loader_kwargs)
            print("\n✓ USING REGULAR SHUFFLING (no sampler)")
        
        self.optimizer = torch.optim.AdamW(param_dicts, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['training'].get('scheduler_step_size', 30), gamma=config['training'].get('scheduler_gamma', 0.1))
        
        self.best_map = 0.0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.eval_every_n_epochs = config['evaluation'].get('eval_every_n_epochs', 5)
    
    @staticmethod
    def collate_fn(batch):
        pixel_values_list = [item[0] for item in batch]
        labels_list = [item[1] for item in batch]
        
        max_h = max([pv.shape[1] for pv in pixel_values_list])
        max_w = max([pv.shape[2] for pv in pixel_values_list])
        
        # BULLETPROOF 32x MATH FOR DEFORMABLE DETR
        max_h = math.ceil(max_h / 32) * 32
        max_w = math.ceil(max_w / 32) * 32
        
        padded_pixel_values = []
        pixel_masks = []
        
        for pv in pixel_values_list:
            c, h, w = pv.shape
            pv = torch.nan_to_num(pv, nan=0.0)
            
            pad_h = max_h - h
            pad_w = max_w - w
            padded = F.pad(pv, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
            padded_pixel_values.append(padded)
            
            mask = torch.zeros((max_h, max_w), dtype=torch.long)
            mask[:h, :w] = 1
            pixel_masks.append(mask)
            
        return torch.stack(padded_pixel_values, dim=0), torch.stack(pixel_masks, dim=0), labels_list

    # (Skipping train_one_epoch and validate here because they are completely overridden by train_detr.py anyway)