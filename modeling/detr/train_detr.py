"""
DETR & Faster R-CNN Training with Ablation Controls
Supports Custom Adaptive Sampling, Inner-SIoU, and Multi-Architecture Per-Class Logging
"""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import yaml
import sys
import os
import traceback
import numpy as np
import scipy.optimize as scipy_opt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datetime import datetime  
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modeling.detr.detr_with_existing_pipeline import DETRWithExistingDataPipeline
from modeling.detr.detr_evaluation import DETREvaluator
import torch.nn.functional as F

# ==============================================================================
# FASTER R-CNN STATIC PATCH (Tuple-Safe)
# ==============================================================================
from torchvision.models.detection import roi_heads
orig_fastrcnn_loss = roi_heads.fastrcnn_loss

GLOBAL_FASTRCNN_LOGS = {}

def patched_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    global GLOBAL_FASTRCNN_LOGS
    losses = orig_fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    
    with torch.no_grad():
        cat_labels = torch.cat(labels, dim=0)
        unreduced_loss = F.cross_entropy(class_logits, cat_labels, reduction="none")
        
        GLOBAL_FASTRCNN_LOGS.clear()
        for class_id in cat_labels.unique():
            c_id = class_id.item()
            GLOBAL_FASTRCNN_LOGS[f"log_only_class_{c_id}"] = unreduced_loss[cat_labels == c_id].mean()
            
    return losses

roi_heads.fastrcnn_loss = patched_fastrcnn_loss


# ==============================================================================
# DYNAMIC INNER-SIOU INTERCEPTION LOGIC (GLOBAL OVERRIDE)
# ==============================================================================
def apply_inner_siou_patch(arch):
    print(f"\n🌟 ABLATION ENABLED: Injecting Inner-SIoU into {arch.upper()}!")

    if arch == "faster_rcnn":
        print("   ↳ Inner-SIoU safely bypassed for Faster R-CNN.")
        return

    try:
        import sys
        import numpy as np
        import scipy.optimize as scipy_opt
        import transformers.loss.loss_for_object_detection as hf_loss_module

        matcher_module = sys.modules['transformers.loss.loss_for_object_detection']

        # ── Standard converter: safe, clamped to [0,1] — for matcher assignment ──
        def standard_center_to_corners_format(x):
            x_safe = torch.nan_to_num(x, nan=0.001, posinf=1.0, neginf=0.0)
            x_c, y_c, w, h = x_safe.unbind(-1)
            w = w.abs().clamp(min=1e-4)
            h = h.abs().clamp(min=1e-4)
            x1 = (x_c - 0.5 * w).clamp(0.0, 1.0)
            y1 = (y_c - 0.5 * h).clamp(0.0, 1.0)
            x2 = (x_c + 0.5 * w).clamp(0.0, 1.0)
            y2 = (y_c + 0.5 * h).clamp(0.0, 1.0)
            x1_f = torch.min(x1, x2)
            x2_f = torch.max(x1, x2) + 1e-4
            y1_f = torch.min(y1, y2)
            y2_f = torch.max(y1, y2) + 1e-4
            return torch.stack([x1_f, y1_f, x2_f, y2_f], dim=-1)

        # ── Inner-SIoU converter: 1.25x expansion — for GIoU loss only ──
        def inner_siou_center_to_corners_format(x):
            x_safe = torch.nan_to_num(x, nan=0.001, posinf=1.0, neginf=0.0)
            ratio = 1.25
            x_c, y_c, w, h = x_safe.unbind(-1)
            w = w.abs().clamp(min=1e-4)
            h = h.abs().clamp(min=1e-4)
            inner_w = w * ratio
            inner_h = h * ratio
            x1 = x_c - 0.5 * inner_w
            y1 = y_c - 0.5 * inner_h
            x2 = x_c + 0.5 * inner_w
            y2 = y_c + 0.5 * inner_h
            x1_f = torch.min(x1, x2)
            x2_f = torch.max(x1, x2) + 1e-4
            y1_f = torch.min(y1, y2)
            y2_f = torch.max(y1, y2) + 1e-4
            return torch.stack([x1_f, y1_f, x2_f, y2_f], dim=-1)

        # ── Box sanitizer: clamps and fixes any OOB/NaN boxes ──
        def sanitize_boxes(boxes):
            boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1.0, neginf=0.0)
            boxes = boxes.clamp(0.0, 1.0)
            x1, y1, x2, y2 = boxes.unbind(-1)
            x1_f = torch.min(x1, x2)
            x2_f = (torch.max(x1, x2) + 1e-4).clamp(max=1.0)
            y1_f = torch.min(y1, y2)
            y2_f = (torch.max(y1, y2) + 1e-4).clamp(max=1.0)
            return torch.stack([x1_f, y1_f, x2_f, y2_f], dim=-1)

        # ── Safe LSA: replaces NaN/Inf in cost matrix before scipy sees it ──
        orig_lsa = scipy_opt.linear_sum_assignment

        def safe_linear_sum_assignment(cost_matrix):
            cost_matrix = np.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=-1e6)
            return orig_lsa(cost_matrix)

        # ── Patch 1: HungarianMatcher — standard converter + safe LSA ──
        from transformers.loss.loss_for_object_detection import HungarianMatcher
        orig_matcher_forward = HungarianMatcher.forward

        def patched_matcher_forward(self, outputs, targets):
            orig_conv = matcher_module.center_to_corners_format
            orig_hf_lsa = hf_loss_module.linear_sum_assignment

            matcher_module.center_to_corners_format = standard_center_to_corners_format
            hf_loss_module.linear_sum_assignment = safe_linear_sum_assignment

            try:
                result = orig_matcher_forward(self, outputs, targets)
            finally:
                matcher_module.center_to_corners_format = orig_conv
                hf_loss_module.linear_sum_assignment = orig_hf_lsa

            return result

        HungarianMatcher.forward = patched_matcher_forward
        print("   ↳ HungarianMatcher: standard converter + NaN-safe cost matrix")

        # ── Patch 2: generalized_box_iou — Inner-SIoU 1.25x + sanitize before orig_giou ──
        orig_giou = hf_loss_module.generalized_box_iou

        def patched_generalized_box_iou(boxes1, boxes2):
            orig_conv = matcher_module.center_to_corners_format
            matcher_module.center_to_corners_format = inner_siou_center_to_corners_format
            try:
                result = orig_giou(sanitize_boxes(boxes1), sanitize_boxes(boxes2))
            finally:
                matcher_module.center_to_corners_format = orig_conv
            return result

        hf_loss_module.generalized_box_iou = patched_generalized_box_iou
        matcher_module.generalized_box_iou = patched_generalized_box_iou
        print("   ↳ generalized_box_iou: Inner-SIoU 1.25x + NaN/OOB sanitization")
        print("   ↳ Patch complete!")

    except Exception as e:
        print(f"   ⚠ CRITICAL ERROR injecting Inner-SIoU: {e}")
        traceback.print_exc()


# =====================================================================
# HELPER: Bounding Box Converter for Faster R-CNN
# =====================================================================
def convert_cxcywh_to_xyxy(boxes_cxcywh, img_h, img_w):
    if boxes_cxcywh.numel() == 0:
        return torch.empty((0, 4), device=boxes_cxcywh.device), torch.empty((0,), dtype=torch.bool, device=boxes_cxcywh.device)
        
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x1 = (cx - 0.5 * w) * img_w
    y1 = (cy - 0.5 * h) * img_h
    x2 = (cx + 0.5 * w) * img_w
    y2 = (cy + 0.5 * h) * img_h
    
    x1 = x1.clamp(min=0, max=img_w)
    y1 = y1.clamp(min=0, max=img_h)
    x2 = x2.clamp(min=0, max=img_w)
    y2 = y2.clamp(min=0, max=img_h)
    
    keep = (x2 > x1) & (y2 > y1)
    return torch.stack([x1, y1, x2, y2], dim=-1), keep


# ==============================================================================
# MAIN TRAINER CLASS
# ==============================================================================
class DETRTrainerWithAblations(DETRWithExistingDataPipeline):
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config  
        self.arch = config['model'].get('architecture', 'standard_detr').lower()
        self.num_classes = config['model']['num_object_classes']
        self.ablations = config.get('ablations', {})
        
        # Apply Inner-SIoU Globally upon initialization
        if self.ablations.get('use_inner_siou', False):
            apply_inner_siou_patch(self.arch)
        else:
            print(f"\nABLATION CONTROL: Standard Bounding Box Loss (Inner-SIoU Disabled)")

        if self.arch == "faster_rcnn":
            print(f"\n Architecture flag '{self.arch}' detected. Overriding DETR...")
            print(" Initializing Faster R-CNN (ResNet50-FPN)...")
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            self.model.to(self.device)
            
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=config['training']['learning_rate'], 
                weight_decay=config['training']['weight_decay']
            )
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=config['training'].get('scheduler_step_size', 30), 
                gamma=config['training'].get('scheduler_gamma', 0.1)
            )

        if self.ablations.get('use_adaptive_sampler', False):
            print("ABLATION ENABLED: Adaptive Data Sampler Active!")
            self._setup_adaptive_sampler(config)
        else:
            print("ABLATION CONTROL: Standard Random Dataloader (Sampler Disabled)")

        self.eval_every_n_epochs = config['training'].get('eval_every_n_epochs', 5)
        self.save_best_model = config['training'].get('save_best_model', True)
        self.save_ckpt_every = config['training'].get('save_checkpoint_every', 5)
        self.patience = config['training'].get('patience', 10)

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0    
        self.best_map = 0.0                    

        self.start_epoch = 0
        resume_path = config['training'].get('resume_checkpoint', None)
        
        if resume_path and os.path.exists(resume_path):
            print(f"\n🔄 Resuming cluster training from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                self.start_epoch = checkpoint.get('epoch', 0)
                self.best_map = checkpoint.get('best_map', 0.0)
                self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

                # ==============================================================
                # 1. THE LR OVERRIDE: Force the loaded optimizer to use the config LR
                # ==============================================================
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = config['training']['learning_rate']

                # ==============================================================
                # 2. THE SAMPLER MEMORY: Restore exact class distributions
                # ==============================================================
                if 'sampler_weights' in checkpoint and checkpoint['sampler_weights'] is not None:
                    if hasattr(self, 'adaptive_sampler'):
                        # Force the sampler's internal dictionary to use the saved weights
                        self.adaptive_sampler.class_weights = checkpoint['sampler_weights']
                        print("  ↳ Successfully restored Adaptive Sampler distribution!")

                print(f"  ↳ Successfully restored model, optimizer, and scheduler to epoch {self.start_epoch}!")
            else:
                self.model.load_state_dict(checkpoint)
                print("  ↳ Loaded weights only.")

        self.evaluator = DETREvaluator(
            model=self.model, data_loader=self.val_loader, processor=self.processor, device=self.device, config=config,
        )
        self.test_evaluator = DETREvaluator(
            model=self.model, data_loader=self.test_loader, processor=self.processor, device=self.device, config=config,
        )

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        sampler_str = "AdaptSampler" if self.ablations.get('use_adaptive_sampler') else "NoSampler"
        siou_str = "InnerSIoU" if self.ablations.get('use_inner_siou') else "BaseLoss"
        
        log_dir = f"runs/{self.arch}_{sampler_str}_{siou_str}_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"📊 TensorBoard logging to: {log_dir}")

        

    def _setup_adaptive_sampler(self, config):
        from turbine_processing.sampler_adaptive import AdaptiveDETRSampler
        from torch.utils.data import DataLoader

        self.adaptive_sampler = AdaptiveDETRSampler(
            dataset=self.train_dataset,
            epoch_size=len(self.train_dataset),
            initial_mode=config.get('data', {}).get('initial_mode', 'equal'),
            adaptation_rate=config.get('data', {}).get('adaptation_rate', 0.3),
            min_weight=config.get('data', {}).get('min_weight', 0.1),
            max_weight=config.get('data', {}).get('max_weight', 5.0),
            background_ratio=config.get('data', {}).get('background_ratio', 0.5),
            dynamic_background=config.get('data', {}).get('dynamic_background', True),
            min_bg_ratio=config.get('data', {}).get('min_bg_ratio', 0.15),
            max_bg_ratio=config.get('data', {}).get('max_bg_ratio', 0.50),
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            sampler=self.adaptive_sampler,
            num_workers=config['training']['num_workers'],
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=config['training']['num_workers'] > 0,
            prefetch_factor=8 if config['training']['num_workers'] > 0 else None,
            
        )

    def _update_sampler_weights(self, metrics):
        if not hasattr(self, 'adaptive_sampler') or 'pr_data' not in metrics:
            return

        if not hasattr(self.adaptive_sampler, 'update_class_weights'):
            return

        class_ap = {class_id: data['ap'] for class_id, data in metrics['pr_data'].items() if class_id > 0}
        self.adaptive_sampler.update_class_weights(class_ap, metrics.get('bg_accuracy', None))

        for class_id, weight in self.adaptive_sampler.get_current_weights().items():
            class_name = self.config['model']['class_names'][class_id] if class_id < len(self.config['model']['class_names']) else f"class_{class_id}"
            self.writer.add_scalar(f'Sampler_Weights/{class_name}', weight, self.current_epoch)
            
        if hasattr(self.adaptive_sampler, 'bg_ratio'):
            self.writer.add_scalar('Sampler_Weights/Background_Ratio', self.adaptive_sampler.bg_ratio, self.current_epoch)

    def _extract_rcnn_targets(self, targets, img_h, img_w):
        rcnn_targets = []
        for t in targets:
            hf_boxes = t.get('boxes', torch.empty((0, 4), device=self.device))
            
            if 'class_labels' in t:
                hf_labels = t['class_labels']
            else:
                hf_labels = t.get('labels', torch.empty((0,), dtype=torch.int64, device=self.device))

            if hf_boxes.numel() == 0:
                rcnn_targets.append({
                    "boxes":  torch.empty((0, 4), dtype=torch.float32, device=self.device),
                    "labels": torch.empty((0,),   dtype=torch.int64,   device=self.device),
                })
                continue

            if 'resized_size' in t:
                target_h, target_w = t['resized_size'][0].item(), t['resized_size'][1].item()
            else:
                target_h, target_w = img_h, img_w

            abs_boxes, valid_mask = convert_cxcywh_to_xyxy(hf_boxes, target_h, target_w)
            rcnn_targets.append({
                "boxes":  abs_boxes[valid_mask].to(torch.float32),
                "labels": hf_labels[valid_mask].to(torch.int64),
            })
        return rcnn_targets

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        valid_batches = 0
        last_batch_loss = 0.0
        crash_count = 0
        MAX_EARLY_CRASHES = 3

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1:3d}/{self.config['training']['num_epochs']:3d} Train",
            leave=True, dynamic_ncols=True, unit='batch', mininterval=30.0,
            maxinterval=120.0
        )

        for batch_idx, (pixel_values, pixel_mask, targets) in enumerate(pbar):
            pixel_values = pixel_values.to(self.device)
            pixel_mask   = pixel_mask.to(self.device)
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]

            try:
                if self.arch == "faster_rcnn":
                    img_h, img_w = pixel_values.shape[2], pixel_values.shape[3]
                    rcnn_targets = self._extract_rcnn_targets(targets, img_h, img_w)
                    image_list = list(img for img in pixel_values)
                    loss_dict  = self.model(image_list, rcnn_targets)
                    
                    actual_losses = {k: v for k, v in loss_dict.items() if not k.startswith("log_only")}
                    
                    global GLOBAL_FASTRCNN_LOGS
                    log_metrics = GLOBAL_FASTRCNN_LOGS.copy()
                    
                    loss = sum(l for l in actual_losses.values())
                else:
                    outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=targets)
                    loss = outputs.loss
                    log_metrics = {}

                if torch.isnan(loss) or torch.isinf(loss):
                    tqdm.write(f"⚠ Warning: NaN/Inf loss at batch {batch_idx}. Skipping update.")
                    self.optimizer.zero_grad()
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()

                global_step = (epoch * len(self.train_loader)) + batch_idx
                for k, v in log_metrics.items():
                    class_id = int(k.split("_")[-1])
                    if class_id < len(self.config['model']['class_names']):
                        class_name = self.config['model']['class_names'][class_id]
                    else:
                        class_name = f"Class_{class_id}"
                    self.writer.add_scalar(f"Per_Class_Loss/{class_name}", v.item(), global_step)

            except Exception as e:
                crash_count += 1
                tqdm.write(f"\n⚠ Forward pass crashed at batch {batch_idx} (crash {crash_count}):")
                tqdm.write(traceback.format_exc())
                self.optimizer.zero_grad()

                # Early abort: if too many crashes happen in the first 20 batches,
                # the patch is broken — fail fast instead of wasting hours
                if batch_idx < 20 and crash_count >= MAX_EARLY_CRASHES:
                    raise RuntimeError(
                        f"\n🚨 ABORTING EPOCH {epoch+1}: {crash_count} crashes within the first "
                        f"{batch_idx+1} batches. Inner-SIoU patch is not intercepting correctly. "
                        f"Fix the patch before resuming training."
                    )
                continue

            last_batch_loss = loss.item()
            total_loss  += last_batch_loss
            valid_batches += 1
            avg_loss = total_loss / valid_batches

            pbar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                'batch_loss': f'{last_batch_loss:.4f}'
            }, refresh=False)

        avg_loss = total_loss / max(valid_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss

    def train(self):
        print(f"\n{'='*70}")
        print(f"COMMENCING RESEARCH ABLATION RUN")
        print(f"{'='*70}")
        print(f"  Architecture       : {self.arch.upper()}")
        print(f"  Adaptive Sampler   : {self.ablations.get('use_adaptive_sampler', False)}")
        print(f"  Inner-SIoU Loss    : {self.ablations.get('use_inner_siou', False)}")
        print(f"  Training samples   : {len(self.train_loader.dataset)}")
        print(f"  Learning rate      : {self.config['training']['learning_rate']}")

        for epoch in range(self.start_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch
            print(f"\n{'='*60}")

            train_loss = self.train_one_epoch(epoch)
            self.writer.add_scalar('Loss/Train_Epoch_Average', train_loss, epoch + 1)
            self.scheduler.step()

            if (epoch + 1) % self.config['training']['eval_every_n_epochs'] == 0:
                print(f"\n  Running detailed evaluation at epoch {epoch+1}…")
                metrics, _, _ = self.evaluator.evaluate(epoch=epoch+1)
                
                self.writer.add_scalar('Validation_Metrics/mAP_0.5', metrics.get('mAP', 0.0), epoch + 1)
                self.writer.add_scalar('Validation_Metrics/mAP_0.5_0.95', metrics.get('mAP_50_95', 0.0), epoch + 1)
                self.writer.add_scalar('Validation_Metrics/Recall', metrics.get('Recall', 0.0), epoch + 1)
                self.writer.add_scalar('Validation_Metrics/F1_Score', metrics.get('F1', 0.0), epoch + 1)
                
                if metrics.get('mAP_Small', -1.0) >= 0:
                    self.writer.add_scalar('Validation_Scale/mAP_Small', metrics.get('mAP_Small', 0.0), epoch + 1)
                if metrics.get('mAP_Medium', -1.0) >= 0:
                    self.writer.add_scalar('Validation_Scale/mAP_Medium', metrics.get('mAP_Medium', 0.0), epoch + 1)
                if metrics.get('mAP_Large', -1.0) >= 0:
                    self.writer.add_scalar('Validation_Scale/mAP_Large', metrics.get('mAP_Large', 0.0), epoch + 1)

                if 'pr_data' in metrics:
                    for class_id, data in metrics['pr_data'].items():
                        if class_id > 0:
                            c_name = self.config['model']['class_names'][class_id] if class_id < len(self.config['model']['class_names']) else f"Class_{class_id}"
                            c_name = c_name.replace(" ", "_")
                            self.writer.add_scalar(f'Validation_Per_Class_AP/{c_name}', data.get('ap', 0.0), epoch + 1)

                current_map = metrics.get('mAP', 0.0)
                print(f"\n{'='*60}")
                
                if current_map > getattr(self, 'best_map', 0.0):
                    self.best_map = current_map
                    self.epochs_without_improvement = 0
                    if self.config['training']['save_best_model']:
                        best_model_path = self.config['training']['output_model_path'].replace('.pth', '_best.pth')
                        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                        torch.save(self.model.state_dict(), best_model_path)
                        print(f"  🎉 New best mAP={current_map:.4f} → {os.path.basename(best_model_path)}")
                else:
                    self.epochs_without_improvement += 1
                    print(f"  ⚠ No improvement for {self.epochs_without_improvement}/{self.config['training']['patience']} evaluations")
                    
                self._update_sampler_weights(metrics)
                
                if hasattr(self, 'adaptive_sampler') and hasattr(self.adaptive_sampler, 'print_current_distribution'):
                    print("\n📊 UPDATED SAMPLER DISTRIBUTION BASED ON NEW AP:")
                    self.adaptive_sampler.print_current_distribution()

                print(f"{'='*60}")

                if self.epochs_without_improvement >= self.config['training']['patience']:
                    print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs.")
                    break
            
            if (epoch + 1) % self.config['training'].get('save_checkpoint_every', 10) == 0:
                ckpt_dir = self.config['training'].get('checkpoint_dir', 'checkpoints/')
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth")
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_map': getattr(self, 'best_map', 0.0)
                }, ckpt_path)
                print(f"  💾 Saved cluster checkpoint: {ckpt_path}")

        print("\n============================================================")
        print("Final evaluation on test set…")
        print("============================================================")
        try:
            best_model_path = self.config['training']['output_model_path'].replace('.pth', '_best.pth')
            if os.path.exists(best_model_path):
                print(f"  Loading best checkpoint from: {best_model_path}")
                self.model.load_state_dict(torch.load(best_model_path))
            else:
                print(f"  ⚠ Best checkpoint not found at {best_model_path}.")
                
            self.model.eval()
            self.evaluator.data_loader = self.test_loader
            metrics, _, _ = self.evaluator.evaluate(epoch="final_test")
            
            print("\n" + "="*60)
            print("🏆 FINAL TEST SET SUMMARY (FOR PAPER)")
            print("="*60)
            print(f"  mAP@0.5      : {metrics.get('mAP', 0.0):.4f}")
            print(f"  mAP@0.5:0.95 : {metrics.get('mAP_50_95', 0.0):.4f}")
            print(f"  Recall       : {metrics.get('Recall', 0.0):.4f}")
            print("============================================================\n")
        except Exception as e:
            print(f"⚠ Could not complete final test evaluation. Error: {e}")

        self.writer.close()
        return self.model


def main():
    config_path = 'config/config.yaml'
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    trainer = DETRTrainerWithAblations(config)
    trainer.train()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()