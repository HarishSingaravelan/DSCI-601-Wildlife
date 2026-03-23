"""
DETR Evaluation Script - FULLY FIXED & BACKGROUND AWARE
Fixes:
  1. Lower confidence threshold handling
  2. GT boxes normalization
  3. Batch handling
  4. NEW: Background Image Accuracy & True Negative visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
from typing import Dict, List
from inference.visualization_utils import create_readable_visualizations
import os


class DETREvaluator:

    def __init__(self, model, data_loader, processor, device, config, bg_threshold=0.05):
        self.model       = model
        self.data_loader = data_loader
        self.processor   = processor
        self.device      = device
        self.config      = config

        self.iou_threshold  = config.get('evaluation', {}).get('iou_threshold', 0.5)
        self.conf_threshold = config.get('evaluation', {}).get('confidence_threshold', 0.1)
        
        # New threshold specifically for checking "silence" on empty images
        self.bg_threshold = bg_threshold 
        
        self.class_names    = config.get('model', {}).get(
            'class_names',
            [f"class_{i}" for i in range(config['model']['num_object_classes'])]
        )

    @staticmethod
    def box_iou(boxes1, boxes2):
        """IoU between two sets of boxes [N,4] and [M,4] in pixel coords"""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return torch.zeros((len(boxes1), len(boxes2)))

        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)

        lt    = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb    = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh    = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2 - inter
        return inter / (union + 1e-6)

    @staticmethod
    def normalize_to_pixels(boxes_norm, orig_h, orig_w):
        """Convert DETR normalized boxes [cx, cy, w, h] to pixel [x1, y1, x2, y2]."""
        if boxes_norm.numel() == 0:
            return boxes_norm

        cx = boxes_norm[:, 0] * orig_w
        cy = boxes_norm[:, 1] * orig_h
        w  = boxes_norm[:, 2] * orig_w
        h  = boxes_norm[:, 3] * orig_h

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        return torch.stack([x1, y1, x2, y2], dim=1)

    @torch.no_grad()
    def collect_predictions_and_bg_stats(self):
        """
        Collects predictions, GT, AND tracks pure background image performance.
        Returns:
            predictions, ground_truths, bg_stats (dict)
        """
        self.model.eval()

        all_predictions   = []
        all_ground_truths = []
        
        # Background Stats Tracking
        total_empty_images = 0
        correctly_silent_images = 0
        hallucinations_on_empty = 0

        for batch_idx, (pixel_values, targets) in enumerate(self.data_loader):
            pixel_values = pixel_values.to(self.device)
            outputs      = self.model(pixel_values=pixel_values)

            # --- BACKGROUND CHECK (Raw Logits Analysis) ---
            # Check probabilities before post-processing filters them out
            probs = outputs.logits.softmax(-1)[..., :-1] # Exclude 'no-object' class
            max_probs = probs.max(-1).values
            
            # Use orig_size for post-processing
            orig_sizes = torch.stack([t['orig_size'] for t in targets]).to(self.device)

            results = self.processor.post_process_object_detection(
                outputs, 
                target_sizes=orig_sizes, 
                threshold=self.conf_threshold
            )

            for i, (target, result) in enumerate(zip(targets, results)):
                # --- BG Logic ---
                gt_boxes_norm = target['boxes']
                is_empty_image = (len(gt_boxes_norm) == 0)
                
                if is_empty_image:
                    total_empty_images += 1
                    # Did the model predict anything above the strict BG threshold?
                    num_detections = (max_probs[i] > self.bg_threshold).sum().item()
                    
                    if num_detections == 0:
                        correctly_silent_images += 1
                    else:
                        hallucinations_on_empty += num_detections

                # --- Standard Logic ---
                all_predictions.append({
                    'boxes' : result['boxes'].cpu(),
                    'scores': result['scores'].cpu(),
                    'labels': result['labels'].cpu(),
                })

                gt_labels = target['class_labels'].cpu()
                orig_h, orig_w = target['orig_size'].tolist()
                gt_boxes_pixels = self.normalize_to_pixels(gt_boxes_norm.cpu(), orig_h, orig_w)

                all_ground_truths.append({
                    'boxes' : gt_boxes_pixels,
                    'labels': gt_labels,
                })

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(self.data_loader)}")

        bg_stats = {
            "total_empty": total_empty_images,
            "correct_empty": correctly_silent_images,
            "hallucinations": hallucinations_on_empty
        }
        
        return all_predictions, all_ground_truths, bg_stats

    def match_predictions_to_ground_truth(self, predictions, ground_truths):
        """Match predictions to GT using IoU"""
        matches = []

        for pred_dict, gt_dict in zip(predictions, ground_truths):
            pred_boxes  = pred_dict['boxes']
            pred_scores = pred_dict['scores']
            pred_labels = pred_dict['labels']
            gt_boxes    = gt_dict['boxes']
            gt_labels   = gt_dict['labels']

            # Images with no GT annotations (Background images)
            if len(gt_boxes) == 0:
                for j in range(len(pred_boxes)):
                    matches.append({
                        'pred_label': pred_labels[j].item(),
                        'gt_label'  : None, # Represents Background
                        'score'     : pred_scores[j].item(),
                        'matched'   : False,
                    })
                continue

            # Images with no predictions (False Negatives)
            if len(pred_boxes) == 0:
                for j in range(len(gt_boxes)):
                    matches.append({
                        'pred_label': None,
                        'gt_label'  : gt_labels[j].item(),
                        'score'     : 0.0,
                        'matched'   : False,
                    })
                continue

            iou_matrix = self.box_iou(pred_boxes, gt_boxes)
            matched_gt = set()

            for idx in torch.argsort(pred_scores, descending=True):
                pred_label = pred_labels[idx].item()
                pred_score = pred_scores[idx].item()

                ious = iou_matrix[idx]
                best_iou, best_gt_idx = ious.max(0)
                best_gt_idx = best_gt_idx.item()

                if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt:
                    matched_gt.add(best_gt_idx)
                    matches.append({
                        'pred_label': pred_label,
                        'gt_label'  : gt_labels[best_gt_idx].item(),
                        'score'     : pred_score,
                        'matched'   : True,
                    })
                else:
                    matches.append({
                        'pred_label': pred_label,
                        'gt_label'  : None, # False Positive
                        'score'     : pred_score,
                        'matched'   : False,
                    })

            # Unmatched GTs = False Negatives
            for j in range(len(gt_boxes)):
                if j not in matched_gt:
                    matches.append({
                        'pred_label': None,
                        'gt_label'  : gt_labels[j].item(),
                        'score'     : 0.0,
                        'matched'   : False,
                    })

        return matches

    def compute_confusion_matrix(self, matches, bg_stats):
        """
        Compute CM and INJECT True Negatives (Background-Background)
        """
        num_classes = self.config['model']['num_object_classes']
        cm          = np.zeros((num_classes, num_classes), dtype=np.int32)
        count       = 0

        for m in matches:
            if m['matched']:
                # True Positive
                pred, gt = m['pred_label'], m['gt_label']
                cm[gt, pred] += 1
                count += 1
            elif m['gt_label'] is None:
                # False Positive (Background classified as Object)
                pred = m['pred_label']
                cm[0, pred] += 1 # Row 0 is Actual Background
                count += 1
            elif m['pred_label'] is None:
                # False Negative (Object classified as Background/Missed)
                gt = m['gt_label']
                cm[gt, 0] += 1 # Col 0 is Predicted Background
                count += 1

        # --- KEY INTEGRATION: Inject True Negatives ---
        # "Correctly Silent" images are effectively True Negatives for the Background class
        true_negatives = bg_stats['correct_empty']
        cm[0, 0] += true_negatives
        count += true_negatives
        
        print(f"  Matched pairs in confusion matrix: {count}")
        print(f"  (Includes {true_negatives} correctly identified background images)")
        
        return cm

    def compute_pr_curve_data(self, matches):
        num_classes = self.config['model']['num_object_classes']
        pr_data     = {}

        for class_id in range(num_classes):
            y_true   = []
            y_scores = []

            for m in matches:
                # Check for standard True Positive / False Negative
                if m.get('gt_label') == class_id:
                    if m.get('pred_label') == class_id and m['matched']:
                        y_true.append(1);  y_scores.append(m['score'])
                    else:
                        y_true.append(1);  y_scores.append(0.0)
                
                # Check for False Positive
                elif m.get('pred_label') == class_id:
                    y_true.append(0);  y_scores.append(m['score'])

            n_pos = int(sum(y_true)) if y_true else 0
            made_predictions = sum(y_scores) > 0.0

            if n_pos > 0 and made_predictions:
                y_true_arr   = np.array(y_true)
                y_scores_arr = np.array(y_scores)
                precision, recall, _ = precision_recall_curve(y_true_arr, y_scores_arr)
                ap = average_precision_score(y_true_arr, y_scores_arr)
            else:
                # If no ground truths OR no predictions were made, AP is 0.0
                precision = recall = np.array([])
                ap = 0.0
            # ---------------------------------------------------------

            pr_data[class_id] = {
                'precision'  : precision,
                'recall'     : recall,
                'ap'         : ap,
                'class_name' : self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}",
                'num_samples': n_pos,
            }

        return pr_data

    def evaluate(self, epoch=None):
        print(f"\n{'='*60}")
        print(f"Evaluation{f' — Epoch {epoch}' if epoch else ''}")
        print(f"  Confidence threshold : {self.conf_threshold}")
        print(f"  BG Check threshold   : {self.bg_threshold}")
        print(f"{'='*60}")

        print("Collecting predictions…")
        predictions, ground_truths, bg_stats = self.collect_predictions_and_bg_stats()

        # Summary stats
        total_preds = sum(len(p['boxes']) for p in predictions)
        total_gt    = sum(len(g['boxes']) for g in ground_truths)
        imgs_with_gt = sum(1 for g in ground_truths if len(g['boxes']) > 0)
        
        print(f"\n  Total predictions  : {total_preds}")
        print(f"  Total GT objects   : {total_gt}")
        print(f"  Images with GT     : {imgs_with_gt} / {len(ground_truths)}")
        
        # --- BG REPORT ---
        bg_accuracy = 0.0
        if bg_stats['total_empty'] > 0:
            bg_accuracy = (bg_stats['correct_empty'] / bg_stats['total_empty']) * 100  
            print(f"\n  [Background Performance]")
            print(f"  Total Empty Images : {bg_stats['total_empty']}")
            print(f"  Correctly Silent   : {bg_stats['correct_empty']}")
            print(f"  Background Acc     : {bg_accuracy:.2f}%")
        # -----------------

        print("\nMatching predictions to ground truth…")
        matches = self.match_predictions_to_ground_truth(predictions, ground_truths)

        # Confusion matrix (Now includes BG Stats!)
        if self.config.get('evaluation', {}).get('compute_confusion_matrix', True):
            cm = self.compute_confusion_matrix(matches, bg_stats)
        else:
            cm = None

        # PR curves
        if self.config.get('evaluation', {}).get('compute_pr_curve', True):
            pr_data = self.compute_pr_curve_data(matches)
        else:
            pr_data = {}
        
        # Create all visualizations using the new readable format
        if self.config.get('evaluation', {}).get('save_plots', True) and (cm is not None or pr_data):
            plots_dir = self.config.get('evaluation', {}).get('plots_dir', 'evaluation_plots/')
            epoch_suffix = f"_epoch_{epoch}" if epoch else "_final"
            epoch_dir = os.path.join(plots_dir, f"evaluation{epoch_suffix}")
            
            create_readable_visualizations(
                metrics={'mAP': 0.0},  # Will be updated below
                pr_data=pr_data,
                confusion_matrix=cm if cm is not None else np.zeros((len(self.class_names), len(self.class_names))),
                class_names=self.class_names,
                output_dir=epoch_dir
            )
        
        # Calculate mAP
        if pr_data:
            # mAP: exclude background (class 0)
            aps = [d['ap'] for cid, d in pr_data.items()
                   if cid > 0 and d['num_samples'] > 0]
            mAP = float(np.mean(aps)) if aps else 0.0

            print(f"\n  mAP@{self.iou_threshold} (excl. background): {mAP:.4f}")
            
            metrics = {
                'mAP'             : mAP,
                'pr_data'         : pr_data,
                'confusion_matrix': cm,
                'bg_accuracy'     : bg_accuracy,
            }
        else:
            metrics = {'mAP': 0.0, 'pr_data': {}, 'confusion_matrix': None}

        print(f"{'='*60}\n")
        return metrics, pr_data, cm