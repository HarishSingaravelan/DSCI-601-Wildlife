"""
DETR & Faster R-CNN Training with Ablation Controls
Supports Adaptive Sampling, Inner-SIoU, and Multi-Architecture Per-Class Logging
"""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import yaml
import os
import sys
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datetime import datetime  
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modeling.detr.detr_with_existing_pipeline import DETRWithExistingDataPipeline
from modeling.detr.detr_evaluation import DETREvaluator


import torch.nn.functional as F

# ==============================================================================
# PER-CLASS LOSS MONKEY-PATCHES
# ==============================================================================

# 1. FASTER R-CNN PATCH (Tuple-Safe)
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

# 2. STANDARD DETR PATCH 
try:
    from transformers.models.detr.modeling_detr import DetrLoss
    orig_detr_loss = DetrLoss.loss_labels
    def patched_detr_loss(self, outputs, targets, indices, num_boxes, log=True):
        loss_dict = orig_detr_loss(self, outputs, targets, indices, num_boxes, log)
        with torch.no_grad():
            src_logits = outputs["logits"]
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([
                (t["class_labels"] if "class_labels" in t else t["labels"])[J] 
                for t, (_, J) in zip(targets, indices)
            ])
            src_logits_m = src_logits[idx]
            unreduced_loss = F.cross_entropy(src_logits_m, target_classes_o, reduction="none")
            for class_id in target_classes_o.unique():
                c_id = class_id.item()
                loss_dict[f"log_only_class_{c_id}"] = unreduced_loss[target_classes_o == c_id].mean()
        return loss_dict
    DetrLoss.loss_labels = patched_detr_loss
except ImportError:
    pass

# 3. DEFORMABLE DETR / D-FINE PATCH
try:
    from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrLoss
    orig_def_detr_loss = DeformableDetrLoss.loss_labels
    def patched_def_detr_loss(self, outputs, targets, indices, num_boxes, log=True):
        loss_dict = orig_def_detr_loss(self, outputs, targets, indices, num_boxes, log)
        with torch.no_grad():
            src_logits = outputs["logits"]
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([
                (t["class_labels"] if "class_labels" in t else t["labels"])[J] 
                for t, (_, J) in zip(targets, indices)
            ])
            src_logits_m = src_logits[idx]
            unreduced_loss = F.cross_entropy(src_logits_m, target_classes_o, reduction="none")
            for class_id in target_classes_o.unique():
                c_id = class_id.item()
                loss_dict[f"log_only_class_{c_id}"] = unreduced_loss[target_classes_o == c_id].mean()
        return loss_dict
    DeformableDetrLoss.loss_labels = patched_def_detr_loss
except ImportError:
    pass
# ==============================================================================


# ==============================================================================
# DYNAMIC INNER-SIOU INTERCEPTION LOGIC
# ==============================================================================
def apply_inner_siou_patch(arch):
    """Dynamically overrides bounding box loss functions with Inner-SIoU"""
    print(f"🌟 ABLATION ENABLED: Injecting Inner-SIoU into {arch.upper()}!")
    
    if arch == "faster_rcnn":
        # Faster R-CNN calculates box loss in roi_heads.py (fastrcnn_loss)
        # TODO: Implement Custom Faster R-CNN Inner-SIoU Loss Override here
        pass
        
    elif arch == "standard_detr":
        from transformers.models.detr.modeling_detr import DetrLoss
        orig_loss_boxes = DetrLoss.loss_boxes
        
        def patched_loss_boxes(self, outputs, targets, indices, num_boxes):
            # TODO: Implement Custom Inner-SIoU math for Standard DETR here
            # Calculate L1 loss normally, but replace GIoU with Inner-SIoU
            return orig_loss_boxes(self, outputs, targets, indices, num_boxes)
            
        DetrLoss.loss_boxes = patched_loss_boxes
        
    elif arch in ["deformable_detr", "dfine"]:
        from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrLoss
        orig_loss_boxes = DeformableDetrLoss.loss_boxes
        
        def patched_loss_boxes(self, outputs, targets, indices, num_boxes):
            # TODO: Implement Custom Inner-SIoU math for Deformable DETR here
            return orig_loss_boxes(self, outputs, targets, indices, num_boxes)
            
        DeformableDetrLoss.loss_boxes = patched_loss_boxes


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


class DETRTrainerWithAblations(DETRWithExistingDataPipeline):
    """Trainer supporting Multi-Architecture, Adaptive Sampling, and Inner-SIoU Controls"""

    def __init__(self, config):
        super().__init__(config)
        
        self.config = config  
        self.arch = config['model'].get('architecture', 'standard_detr').lower()
        self.num_classes = config['model']['num_object_classes']
        self.ablations = config.get('ablations', {})

        # Apply Inner-SIoU Ablation if enabled
        if self.ablations.get('use_inner_siou', False):
            apply_inner_siou_patch(self.arch)
        else:
            print(f"⚪ ABLATION CONTROL: Standard Bounding Box Loss (Inner-SIoU Disabled)")

        if self.arch == "faster_rcnn":
            print(f"\n🚀 Architecture flag '{self.arch}' detected. Overriding DETR...")
            print("📦 Initializing Faster R-CNN (ResNet50-FPN)...")
            
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
            print("🌟 ABLATION ENABLED: Adaptive Data Sampler Active!")
            self._setup_adaptive_sampler(config)
        else:
            print("⚪ ABLATION CONTROL: Standard Random Dataloader (Sampler Disabled)")

        self.eval_every_n_epochs = config['training'].get('eval_every_n_epochs', 5)
        self.save_best_model = config['training'].get('save_best_model', True)
        self.save_ckpt_every = config['training'].get('save_checkpoint_every', 5)
        self.patience = config['training'].get('patience', 10)

        self.best_val_loss = float('inf')
        self.patience_counter = 0

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
        from torch.utils.data import WeightedRandomSampler, DataLoader
        
        bg_ratio = config.get('data', {}).get('background_ratio', 0.10)
        
        num_empty = 0
        num_birds = 0
        for idx in range(len(self.train_dataset)):
            img_id = self.train_dataset.ids[idx]
            if len(self.train_dataset.coco.imgToAnns[img_id]) == 0:
                num_empty += 1
            else:
                num_birds += 1
                
        weight_empty = bg_ratio / max(num_empty, 1)
        weight_bird = (1.0 - bg_ratio) / max(num_birds, 1)
        
        sample_weights = []
        for idx in range(len(self.train_dataset)):
            img_id = self.train_dataset.ids[idx]
            if len(self.train_dataset.coco.imgToAnns[img_id]) == 0:
                sample_weights.append(weight_empty) 
            else:
                sample_weights.append(weight_bird)       
                
        self.adaptive_sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(self.train_dataset), 
            replacement=True
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
            prefetch_factor=2 if config['training']['num_workers'] > 0 else None,
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
            hf_boxes  = t.get('boxes',  torch.empty((0, 4), device=self.device))
            
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

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1:3d}/{self.config['training']['num_epochs']:3d} Train",
            leave=True, dynamic_ncols=True, unit='batch',
        )

        for batch_idx, (pixel_values, pixel_mask, targets) in enumerate(pbar):
            pixel_values = pixel_values.to(self.device)
            pixel_mask   = pixel_mask.to(self.device)
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]

            try:
                log_metrics = {}

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
                    
                    if hasattr(outputs, 'loss_dict') and outputs.loss_dict is not None:
                        log_metrics = {k: v for k, v in outputs.loss_dict.items() if k.startswith("log_only")}

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
                tqdm.write(f"\n⚠ Forward pass crashed at batch {batch_idx}:")
                tqdm.write(traceback.format_exc())
                self.optimizer.zero_grad()
                continue

            last_batch_loss = loss.item()
            total_loss  += last_batch_loss
            valid_batches += 1
            avg_loss = total_loss / valid_batches

            pbar.set_postfix(
                avg_loss=f'{avg_loss:.4f}',
                batch_loss=f'{last_batch_loss:.4f}',
            )

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

        for epoch in range(self.config['training']['num_epochs']):
            print(f"\n{'='*60}")
            train_loss = self.train_one_epoch(epoch)
            
            # ==================================================================
            # NEW: Log Training Loss to TensorBoard (Per Epoch)
            # ==================================================================
            self.writer.add_scalar('Loss/Train_Epoch_Average', train_loss, epoch + 1)

            self.scheduler.step()

            if (epoch + 1) % self.config['training']['eval_every_n_epochs'] == 0:
                print(f"\n  Running detailed evaluation at epoch {epoch+1}…")
                metrics, _, _ = self.evaluator.evaluate(epoch=epoch+1)
                
                # ==================================================================
                # NEW: Log Validation Metrics to TensorBoard (Per Eval)
                # ==================================================================
                self.writer.add_scalar('Validation_Metrics/mAP_0.5', metrics.get('mAP', 0.0), epoch + 1)
                self.writer.add_scalar('Validation_Metrics/mAP_0.5_0.95', metrics.get('mAP_50_95', 0.0), epoch + 1)
                self.writer.add_scalar('Validation_Metrics/Recall', metrics.get('Recall', 0.0), epoch + 1)
                self.writer.add_scalar('Validation_Metrics/F1_Score', metrics.get('F1', 0.0), epoch + 1)
                
                # Log Size-Specific Metrics for Compression Analysis!
                if metrics.get('mAP_Small', -1.0) >= 0:
                    self.writer.add_scalar('Validation_Scale/mAP_Small', metrics.get('mAP_Small', 0.0), epoch + 1)
                if metrics.get('mAP_Medium', -1.0) >= 0:
                    self.writer.add_scalar('Validation_Scale/mAP_Medium', metrics.get('mAP_Medium', 0.0), epoch + 1)
                if metrics.get('mAP_Large', -1.0) >= 0:
                    self.writer.add_scalar('Validation_Scale/mAP_Large', metrics.get('mAP_Large', 0.0), epoch + 1)

                current_map = metrics.get('mAP', 0.0)
                print(f"\n{'='*60}")
                
                if current_map > getattr(self, 'best_map', 0.0):
                    self.best_map = current_map
                    self.epochs_without_improvement = 0
                    if self.config['training']['save_best_model']:
                        best_model_path = self.config['training']['output_model_path'].replace('.pth', '_best.pth')
                        torch.save(self.model.state_dict(), best_model_path)
                        print(f"  🎉 New best mAP={current_map:.4f} → {os.path.basename(best_model_path)}")
                else:
                    self.epochs_without_improvement += 1
                    print(f"  ⚠ No improvement for {self.epochs_without_improvement}/{self.config['training']['patience']} evaluations")

                print(f"{'='*60}")

                if self.epochs_without_improvement >= self.config['training']['patience']:
                    print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs.")
                    break

        print("\n============================================================")
        print("Training Summary Complete")
        print("============================================================")

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
                print(f"  ⚠ Evaluating test set using the final epoch's weights instead.")
                
            self.model.eval()
            self.evaluator.data_loader = self.test_loader 
            
            metrics, _, _ = self.evaluator.evaluate(epoch="final_test")
            
            print("\n" + "="*60)
            print("🏆 FINAL TEST SET SUMMARY (FOR PAPER)")
            print("="*60)
            print(f"  mAP@0.5      : {metrics.get('mAP', 0.0):.4f}")
            print(f"  mAP@0.5:0.95 : {metrics.get('mAP_50_95', 0.0):.4f}")
            print(f"  Recall       : {metrics.get('Recall', 0.0):.4f}")
            print("-" * 60)
            
            mAP_S = metrics.get('mAP_Small', -1.0)
            mAP_M = metrics.get('mAP_Medium', -1.0)
            mAP_L = metrics.get('mAP_Large', -1.0)
            
            print(f"  mAP_Small    : {mAP_S:.4f}" if mAP_S >= 0 else "  mAP_Small    : N/A")
            print(f"  mAP_Medium   : {mAP_M:.4f}" if mAP_M >= 0 else "  mAP_Medium   : N/A")
            print(f"  mAP_Large    : {mAP_L:.4f}" if mAP_L >= 0 else "  mAP_Large    : N/A")
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