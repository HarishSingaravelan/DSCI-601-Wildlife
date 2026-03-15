"""
DETR Model with Existing Faster R-CNN Data Pipeline - FIXED for Variable Image Sizes
Handles images of different sizes by padding them to the same dimensions

FULLY INTEGRATED:
- ✅ Adaptive Sampler (adjusts weights based on per-class performance)
- ✅ Balanced Sampler (sqrt/log/equal modes)
- ✅ All config options respected
- ✅ FIXED: Bounding box format conversion (Pascal VOC -> COCO format) for DETR
- ✅ NEW: Inner-SIoU Loss Function Monkey Patch for Small Object Detection (UAV-DETR)
"""

import torch
import torch.nn as nn
from transformers import DetrForObjectDetection, DetrImageProcessor
import transformers.models.detr.modeling_detr as modeling_detr
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
import os
import torch.nn.functional as F
from tqdm.auto import tqdm

# ==============================================================================
# INNER-SIoU LOSS PATCH (From UAV-DETR Paper)
# ==============================================================================
def patch_inner_siou_loss():
    """
    Safely overrides Hugging Face's default GIoU loss with the Inner-SIoU loss
    from the UAV-DETR paper. Uses algebraic simplification for the SIoU angle cost 
    to prevent NaN gradients.
    """
    def inner_siou_matrix(boxes1, boxes2, ratio=1.25, eps=1e-7):
        # Broadcast to create [N, M] matrix comparing all predictions to all targets
        x1_1, y1_1, x2_1, y2_1 = boxes1.unsqueeze(1).unbind(dim=2) 
        x1_2, y1_2, x2_2, y2_2 = boxes2.unsqueeze(0).unbind(dim=2) 
        
        w1, h1 = x2_1 - x1_1, y2_1 - y1_1
        w2, h2 = x2_2 - x1_2, y2_2 - y1_2
        cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
        cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
        
        # ----------------------------------------------------
        # 1. INNER IOU (Scale boxes by 1.25 for sensitivity)
        # ----------------------------------------------------
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
        
        # ----------------------------------------------------
        # 2. SIOU PENALTY (Calculated on original boxes)
        # ----------------------------------------------------
        s_cw = torch.abs(cx2 - cx1)
        s_ch = torch.abs(cy2 - cy1)
        
        # Angle Cost (Algebraic simplification of sin(2*alpha) avoids NaNs)
        Lambda = 2 * s_cw * s_ch / (s_cw**2 + s_ch**2 + eps)
        gamma = 2 - Lambda
        
        # Enclosing box
        enc_x1 = torch.min(x1_1, x1_2)
        enc_y1 = torch.min(y1_1, y1_2)
        enc_x2 = torch.max(x2_1, x2_2)
        enc_y2 = torch.max(y2_1, y2_2)
        
        cw = torch.clamp(enc_x2 - enc_x1, min=eps)
        ch = torch.clamp(enc_y2 - enc_y1, min=eps)
        
        # Distance Cost
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        Delta = (1 - torch.exp(-gamma * rho_x)) + (1 - torch.exp(-gamma * rho_y))
        
        # Shape Cost
        omega_w = torch.abs(w1 - w2) / torch.clamp(torch.max(w1, w2), min=eps)
        omega_h = torch.abs(h1 - h2) / torch.clamp(torch.max(h1, h2), min=eps)
        Omega = (1 - torch.exp(-omega_w)) ** 4 + (1 - torch.exp(-omega_h)) ** 4
        
        siou_penalty = (Delta + Omega) / 2
        
        # Return format aligns with HuggingFace's loss execution (1 - output)
        return inner_iou - siou_penalty

    # Inject the function into the transformers library
    modeling_detr.generalized_box_iou = inner_siou_matrix
    print("✅ Successfully patched Hugging Face DETR to use Inner-SIoU Loss (Ratio=1.25)")

# Trigger the patch immediately so the model utilizes it during initialization
patch_inner_siou_loss()
# ==============================================================================


class DETRTransformAdapter:
    """
    Adapter that wraps your existing Albumentations transforms and converts output to DETR format.
    This sits on top of your get_train_transform() and get_val_transform().
    """
    
    def __init__(self, albumentations_transform, processor):
        """
        Args:
            albumentations_transform: Your AlbumentationsDetectionTransform
            processor: DETR image processor
        """
        self.albumentations_transform = albumentations_transform
        self.processor = processor
    
    def __call__(self, image, target):
        # First apply your existing transforms (Albumentations)
        image, target = self.albumentations_transform(image, target)
        
        # Convert image from [C, H, W] to [H, W, C] for DETR processor
        if isinstance(image, torch.Tensor):
            if image.shape[0] == 3:  # CHW format
                image_np = image.permute(1, 2, 0).numpy()  # Convert to HWC
            else:
                image_np = image.numpy()
        else:
            image_np = np.array(image)
        
        # Convert to 0-255 range for processor
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        # Store original size before processing
        orig_h, orig_w = image_np.shape[:2]
        
        # Prepare DETR-style annotations
        boxes = target["boxes"].cpu().numpy() if isinstance(target["boxes"], torch.Tensor) else target["boxes"]
        labels = target["labels"].cpu().numpy() if isinstance(target["labels"], torch.Tensor) else target["labels"]
        
        annotations = {
            "image_id": target["image_id"].item() if isinstance(target["image_id"], torch.Tensor) else target["image_id"],
            "annotations": [
                {
                    "bbox": [
                        float(boxes[i][0]),                    # xmin
                        float(boxes[i][1]),                    # ymin
                        float(boxes[i][2] - boxes[i][0]),      # width  (xmax - xmin)
                        float(boxes[i][3] - boxes[i][1])       # height (ymax - ymin)
                    ],
                    "category_id": int(labels[i]),
                    "area": float(target["area"][i]) if "area" in target else float((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])),
                    "iscrowd": int(target["iscrowd"][i]) if "iscrowd" in target else 0,
                }
                for i in range(len(boxes))
            ]
        }
        
        # Process with DETR processor (handles normalization, resizing, etc.)
        processed = self.processor(
            images=image_np,
            annotations=[annotations],
            return_tensors="pt"
        )
        
        # Extract and return
        pixel_values = processed["pixel_values"][0]  # Remove batch dimension
        labels = processed["labels"][0]
        
        # Add original size to labels for later use
        labels['orig_size'] = torch.tensor([orig_h, orig_w])
        
        return pixel_values, labels


class DETRWithExistingDataPipeline:
    """
    DETR trainer that uses your existing TurbineCocoDataset and pytorch DataLoader
    
    FULLY SUPPORTS:
    - Adaptive Sampler (learns optimal class weights from validation performance)
    - Balanced Sampler (sqrt/log/equal modes)
    - Regular random shuffling (no sampler)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['training']['device'])
        
        # Initialize DETR processor
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", do_convert_annotations=True)
        
        # Initialize DETR model
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=config['model']['num_object_classes'],
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)

        import sys
        from pathlib import Path

        ROOT_DIR = Path(__file__).resolve().parents[2]  # project root
        sys.path.insert(0, str(ROOT_DIR))

        
        # Import your existing classes
        from turbine_processing.dataset import TurbineCocoDataset
        from turbine_processing.dataloader import TurbineDataLoader
        from turbine_processing.transforms import get_train_transform, get_val_transform
        
        from turbine_processing.transforms_detr import (
            get_train_transform_detr_medium,  # augmentation settings
            get_val_transform_detr
        )

        train_transform_albumentations = get_train_transform_detr_medium()
        val_transform_albumentations = get_val_transform_detr()
        
        train_transform = DETRTransformAdapter(train_transform_albumentations, self.processor)
        val_transform = DETRTransformAdapter(val_transform_albumentations, self.processor)
        
        # Create datasets using your existing TurbineCocoDataset
        self.train_dataset = TurbineCocoDataset(
            images_dir=config['data']['train_images_dir'],
            ann_file=config['data']['train_ann_file'],
            transforms=train_transform
        )
        
        self.val_dataset = TurbineCocoDataset(
            images_dir=config['data']['val_images_dir'],
            ann_file=config['data']['val_ann_file'],
            transforms=val_transform
        )
        
        self.test_dataset = TurbineCocoDataset(
            images_dir=config['data']['test_images_dir'],
            ann_file=config['data']['test_ann_file'],
            transforms=val_transform
        )
        
        # Create initial val and test loaders (these never use samplers)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True if config['training']['num_workers'] > 0 else False,
            prefetch_factor=24 if config['training']['num_workers'] > 0 else None
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True if config['training']['num_workers'] > 0 else False,
            prefetch_factor=24 if config['training']['num_workers'] > 0 else None
        )
        
        # ═══════════════════════════════════════════════════════════════
        # SAMPLER INITIALIZATION - FULLY INTEGRATED
        # ═══════════════════════════════════════════════════════════════
        
        self.sampler = None
        self.use_adaptive_sampler = config['data'].get('use_adaptive_sampler', False)
        
        if self.use_adaptive_sampler:
            # ─────────────────────────────────────────────────────────
            # ADAPTIVE SAMPLER: Learns optimal class weights over time
            # ─────────────────────────────────────────────────────────
            from turbine_processing.sampler_adaptive import AdaptiveDETRSampler
            
            self.sampler = AdaptiveDETRSampler(
                dataset=self.train_dataset,
                epoch_size=len(self.train_dataset),
                initial_mode=config['data'].get('initial_mode', 'equal'),
                adaptation_rate=config['data'].get('adaptation_rate', 0.3),
                min_weight=config['data'].get('min_weight', 0.1),
                max_weight=config['data'].get('max_weight', 5.0),
            )
            
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,  # Sampler controls order
                sampler=self.sampler,
                num_workers=config['training']['num_workers'],
                collate_fn=self.collate_fn,
                pin_memory=True,
                persistent_workers=True if config['training']['num_workers'] > 0 else False,
                prefetch_factor=24 if config['training']['num_workers'] > 0 else None
            )
            
            print("\n" + "="*70)
            print("✓ USING ADAPTIVE SAMPLER")
            print("="*70)
            print(f"  Initial mode     : {config['data'].get('initial_mode', 'equal')}")
            print(f"  Adaptation rate  : {config['data'].get('adaptation_rate', 0.3)}")
            print(f"  Min weight       : {config['data'].get('min_weight', 0.1)}")
            print(f"  Max weight       : {config['data'].get('max_weight', 5.0)}")
            print(f"  Epoch size       : {len(self.train_dataset)}")
            print("="*70 + "\n")
            
        elif config['data'].get('use_balanced_sampler', False):
            # ─────────────────────────────────────────────────────────
            # BALANCED SAMPLER: Fixed balancing strategy (sqrt/log/equal)
            # ─────────────────────────────────────────────────────────
            from turbine_processing.sampler_detr import DETRBalancedSampler
            
            balance_mode = config['data'].get('balance_mode', 'sqrt')
            
            self.sampler = DETRBalancedSampler(
                dataset=self.train_dataset,
                epoch_size=len(self.train_dataset),
                balance_mode=balance_mode,
            )
            
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,  # Sampler controls order
                sampler=self.sampler,
                num_workers=config['training']['num_workers'],
                collate_fn=self.collate_fn,
                pin_memory=True,
                persistent_workers=True if config['training']['num_workers'] > 0 else False,
                prefetch_factor=24 if config['training']['num_workers'] > 0 else None
            )
            
            print("\n" + "="*70)
            print("✓ USING BALANCED SAMPLER")
            print("="*70)
            print(f"  Balance mode     : {balance_mode}")
            print(f"  Epoch size       : {len(self.train_dataset)}")
            print("="*70 + "\n")
            
        else:
            # ─────────────────────────────────────────────────────────
            # NO SAMPLER: Regular random shuffling
            # ─────────────────────────────────────────────────────────
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True,  # Regular shuffling
                num_workers=config['training']['num_workers'],
                collate_fn=self.collate_fn,
                pin_memory=True,
                persistent_workers=True if config['training']['num_workers'] > 0 else False,
                prefetch_factor=24 if config['training']['num_workers'] > 0 else None
            )
            
            print("\n" + "="*70)
            print("✓ USING REGULAR SHUFFLING (no sampler)")
            print("="*70)
            print(f"  Epoch size       : {len(self.train_dataset)}")
            print("="*70 + "\n")
        
        # ═══════════════════════════════════════════════════════════════
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler (important for DETR)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['training'].get('scheduler_step_size', 30),
            gamma=config['training'].get('scheduler_gamma', 0.1)
        )
        
        # Tracking
        self.best_map = 0.0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.eval_every_n_epochs = config['evaluation'].get('eval_every_n_epochs', 5)
    
    @staticmethod
    def collate_fn(batch):
        """
        FIXED: Custom collate function for DETR that handles variable image sizes.
        
        Images from Albumentations can have different sizes, so we need to pad them
        to the same dimensions before batching.
        """
        # Separate pixel_values and labels
        pixel_values_list = []
        labels_list = []
        
        for pixel_values, labels in batch:
            pixel_values_list.append(pixel_values)
            labels_list.append(labels)
        
        # Find the maximum dimensions in this batch
        max_h = max([pv.shape[1] for pv in pixel_values_list])
        max_w = max([pv.shape[2] for pv in pixel_values_list])
        
        # Pad all images to the maximum size
        padded_pixel_values = []
        for pv in pixel_values_list:
            c, h, w = pv.shape
            
            # Calculate padding (pad_left, pad_right, pad_top, pad_bottom)
            pad_h = max_h - h
            pad_w = max_w - w
            
            # Pad with zeros (you can also use mean values if preferred)
            # F.pad expects (left, right, top, bottom) for last 2 dimensions
            padded = F.pad(pv, (0, pad_w, 0, pad_h), mode='constant', value=0)
            padded_pixel_values.append(padded)
        
        # Now all images have the same size, stack them
        pixel_values_batch = torch.stack(padded_pixel_values, dim=0)
        
        # Labels remain as a list of dicts
        return pixel_values_batch, labels_list
    
    def train_one_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for pixel_values, targets in pbar:
            # pixel_values is now a properly batched tensor [B, C, H, W]
            pixel_values = pixel_values.to(self.device)
            
            # Move targets to device
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = self.model(pixel_values=pixel_values, labels=targets)
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (important for DETR stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        for pixel_values, targets in pbar:
            pixel_values = pixel_values.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in t.items()} for t in targets]
            
            outputs = self.model(pixel_values=pixel_values, labels=targets)
            loss = outputs.loss
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def evaluate_with_metrics(self, epoch: int):
        """
        Run detailed evaluation and return per-class metrics.
        This is used by adaptive sampler to adjust class weights.
        
        Returns:
            Dict[int, float]: Per-class AP scores
        """
        print(f"\n{'='*70}")
        print(f"Running detailed evaluation at epoch {epoch+1}...")
        print(f"{'='*70}\n")
        
        try:
            # Import your evaluation function
            from turbine_processing.detr_evaluation_v3 import evaluate_detr
            
            # Run evaluation
            metrics, pr_data, confusion_matrix = evaluate_detr(
                model=self.model,
                data_loader=self.val_loader,
                device=self.device,
                processor=self.processor,
                dataset=self.val_dataset,
                confidence_threshold=self.config['evaluation']['confidence_threshold'],
                iou_threshold=self.config['evaluation']['iou_threshold'],
            )
            
            # Extract per-class AP
            class_ap = {}
            if pr_data:
                for class_id, data in pr_data.items():
                    if class_id > 0:  # Skip background
                        class_ap[class_id] = data.get('ap', 0.0)
            
            print(f"\n✓ Evaluation complete")
            print(f"  mAP: {metrics.get('mAP', 0.0):.4f}")
            print(f"  Per-class APs collected: {len(class_ap)} classes\n")
            
            return class_ap
            
        except Exception as e:
            print(f"⚠ Evaluation failed: {e}")
            print("  Continuing training without metrics update\n")
            return {}
    
    def update_sampler_weights(self, class_metrics: Dict[int, float]):
        """
        Update adaptive sampler weights based on validation metrics.
        Also recreates the DataLoader with updated sampler.
        """
        if not self.use_adaptive_sampler or self.sampler is None:
            return
        
        if not class_metrics:
            print("⚠ No metrics provided, skipping weight update\n")
            return
        
        # Update sampler weights
        self.sampler.update_class_weights(class_metrics)
        
        # Recreate train_loader with updated sampler
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            sampler=self.sampler,
            num_workers=self.config['training']['num_workers'],
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True if self.config['training']['num_workers'] > 0 else False,
            prefetch_factor=24 if self.config['training']['num_workers'] > 0 else None
        )
        
        print("✓ Adaptive sampler weights updated and DataLoader recreated\n")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("DETR TRAINING WITH EXISTING DATA PIPELINE")
        print("="*70)
        print(f"  Training samples   : {len(self.train_dataset)}")
        print(f"  Validation samples : {len(self.val_dataset)}")
        print(f"  Test samples       : {len(self.test_dataset)}")
        print(f"  Number of classes  : {self.config['model']['num_object_classes']}")
        print(f"  Batch size         : {self.config['training']['batch_size']}")
        print(f"  Learning rate      : {self.config['training']['learning_rate']}")
        print(f"  Epochs             : {self.config['training']['num_epochs']}")
        print(f"  Eval every         : {self.eval_every_n_epochs} epoch(s)")
        
        if self.use_adaptive_sampler:
            print(f"  Adaptive sampling  : True")
        elif self.sampler is not None:
            print(f"  Balanced sampling  : True")
        else:
            print(f"  Regular shuffling  : True")
        
        print("="*70 + "\n")
        
        epochs_without_improvement = 0
        patience = self.config['training'].get('early_stopping_patience', 10)
        
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1} / {self.config['training']['num_epochs']}")
            print(f"{'='*70}")
            
            # Train
            train_loss = self.train_one_epoch(epoch)
            print(f"  Train loss : {train_loss:.4f}")
            
            # Validate
            should_eval = (epoch + 1) % self.eval_every_n_epochs == 0
            
            if should_eval:
                val_loss = self.validate()
                print(f"  Val loss   : {val_loss:.4f}")
                
                # Track best 
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    print(f"  ⚠ No improvement for {epochs_without_improvement}/{patience} evaluations")
                
                # Run detailed evaluation for adaptive sampler
                if self.use_adaptive_sampler:
                    print(f"\n  Running detailed evaluation at epoch {epoch+1}…\n")
                    class_metrics = self.evaluate_with_metrics(epoch)
                    self.update_sampler_weights(class_metrics)
                
            else:
                print(f"  (Skipping validation — next eval at epoch {((epoch // self.eval_every_n_epochs) + 1) * self.eval_every_n_epochs})")
            
            # Step scheduler
            self.scheduler.step()
            
            # Save checkpoint
            if self.config['training'].get('save_checkpoint_every', 10) > 0:
                if (epoch + 1) % self.config['training']['save_checkpoint_every'] == 0:
                    checkpoint_dir = self.config['training'].get('checkpoint_dir', 'checkpoints/')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss if should_eval else None,
                        'best_val_loss': self.best_val_loss,
                    }, checkpoint_path)
                    print(f"  Checkpoint saved → {checkpoint_path}")
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                print(f"  No improvement for {patience} consecutive evaluations")
                break
        
        # Final evaluation on test set
        print(f"\n{'='*70}")
        print("Final evaluation on test set…")
        print(f"{'='*70}\n")
        
        try:
            from turbine_processing.detr_evaluation_v3 import evaluate_detr
            
            final_metrics, _, _ = evaluate_detr(
                model=self.model,
                data_loader=self.test_loader,
                device=self.device,
                processor=self.processor,
                dataset=self.test_dataset,
                confidence_threshold=self.config['evaluation']['confidence_threshold'],
                iou_threshold=self.config['evaluation']['iou_threshold'],
            )
            
            final_map = final_metrics.get('mAP', 0.0)
            print(f"\n{'='*70}")
            print("Training Summary")
            print(f"{'='*70}")
            print(f"  Best validation mAP : {self.best_map:.4f}")
            print(f"  Best val loss       : {self.best_val_loss:.4f}")
            print(f"  Final test mAP      : {final_map:.4f}")
            print(f"  Epochs completed    : {epoch + 1}")
            
        except Exception as e:
            print(f"⚠ Final evaluation failed: {e}")
            print(f"\n{'='*70}")
            print("Training Summary")
            print(f"{'='*70}")
            print(f"  Best val loss       : {self.best_val_loss:.4f}")
            print(f"  Epochs completed    : {epoch + 1}")
        
        # Save final model
        output_path = self.config['training']['output_model_path']
        torch.save(self.model.state_dict(), output_path)
        print(f"  Model saved         : {output_path}")
        
        # Print sampler history if adaptive
        if self.use_adaptive_sampler and self.sampler:
            history = self.sampler.get_adaptation_history()
            if history:
                print(f"\n  Adaptive sampler made {len(history)} weight updates")
        
        print(f"{'='*70}\n")
        print("✅ Done!\n")
        
        return self.model


def main():
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = DETRWithExistingDataPipeline(config)
    
    # Train
    model = trainer.train()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()