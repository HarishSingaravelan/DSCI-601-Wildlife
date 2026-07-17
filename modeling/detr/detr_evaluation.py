"""
DETR Evaluation Script
Fixes/Upgrades:
  1. Integrated torchmetrics MeanAveragePrecision (mAP@0.5, mAP@0.5:0.95, Recall, F1)
  2. Maintained custom Background Image Accuracy & True Negative visualization
  3. Maintained Confusion Matrix generation
  4. Fixed Faster R-CNN label indexing, box scaling, and Chatty Model (confidence thresholding) bugs
  5. ADDED Scale-Specific Metrics (mAP_Small, mAP_Medium, mAP_Large) for compression analysis
  6. FIXED Background Accuracy logic to strictly obey evaluation confidence thresholds
"""

import torch
import numpy as np
import os
from sklearn.metrics import precision_recall_curve, average_precision_score
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from inference.visualization_utils import create_readable_visualizations


class DETREvaluator:

    def __init__(self, model, data_loader, processor, device, config, bg_threshold=0.05):
        self.model       = model
        self.data_loader = data_loader
        self.processor   = processor
        self.device      = device
        self.config      = config

        self.iou_threshold  = config.get('evaluation', {}).get('iou_threshold', 0.5)
        self.conf_threshold = config.get('evaluation', {}).get('confidence_threshold', 0.1)

        self.class_names = config.get('model', {}).get(
            'class_names',
            [f"class_{i}" for i in range(config['model']['num_object_classes'])]
        )

        self.arch = config.get('model', {}).get('architecture', 'standard_detr').lower()

        # TorchMetrics COCO Evaluator
        self.coco_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)

    # ------------------------------------------------------------------
    # STATIC HELPERS
    # ------------------------------------------------------------------

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
        """Convert DETR normalized [cx, cy, w, h] → pixel [x1, y1, x2, y2]."""
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

    # ------------------------------------------------------------------
    # TARGET KEY HELPER
    # ------------------------------------------------------------------

    def _get_gt_labels(self, target: dict) -> torch.Tensor:
        """Safely extract ground-truth class labels from a target dict."""
        if 'class_labels' in target:
            return target['class_labels'].cpu()
        if 'labels' in target:
            return target['labels'].cpu()
        return torch.empty((0,), dtype=torch.int64)

    def _get_orig_size(self, target: dict, pixel_values: torch.Tensor, idx: int):
        """Return (orig_h, orig_w) for a single image."""
        if 'orig_size' in target:
            return target['orig_size'].tolist()
        return pixel_values.shape[2], pixel_values.shape[3]

    # ------------------------------------------------------------------
    # MAIN COLLECTION LOOP
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_predictions_and_bg_stats(self):
        """
        Collect predictions + GT, update TorchMetrics, track background stats.
        Supports Faster R-CNN and all DETR variants through a single loop.
        """
        self.model.eval()

        all_predictions   = []
        all_ground_truths = []

        total_empty_images       = 0
        correctly_silent_images  = 0
        hallucinations_on_empty  = 0

        for batch_idx, (pixel_values, pixel_mask, targets) in enumerate(self.data_loader):
            pixel_values = pixel_values.to(self.device)
            pixel_mask   = pixel_mask.to(self.device)

            # ==============================================================
            # FASTER R-CNN BRANCH
            # ==============================================================
            if self.arch == 'faster_rcnn':
                image_list = list(img for img in pixel_values)
                raw_outputs = self.model(image_list)

                batch_preds = []
                batch_gts   = []

                for i, (target, result) in enumerate(zip(targets, raw_outputs)):
                    # ---- Ground Truth ----------------------------------------
                    gt_labels = self._get_gt_labels(target)

                    orig_h, orig_w = self._get_orig_size(target, pixel_values, i)
                    gt_boxes_norm  = target.get('boxes', torch.empty((0, 4)))
                    gt_boxes_pixels = self.normalize_to_pixels(gt_boxes_norm.cpu(), orig_h, orig_w)

                    # ---- Predictions -----------------------------------------
                    keep_mask = result['scores'] >= self.conf_threshold
                    
                    pred_boxes = result['boxes'][keep_mask].cpu().clone()
                    pred_scores = result['scores'][keep_mask].cpu()
                    pred_labels = result['labels'][keep_mask].cpu()

                    is_empty_image = (len(gt_boxes_pixels) == 0)
                    if is_empty_image:
                        total_empty_images += 1
                        above_thresh = len(pred_scores)
                        if above_thresh == 0:
                            correctly_silent_images += 1
                        else:
                            hallucinations_on_empty += above_thresh

                    if 'resized_size' in target:
                        resized_h, resized_w = target['resized_size'].tolist()
                    else:
                        resized_h, resized_w = pixel_values.shape[2], pixel_values.shape[3]

                    if len(pred_boxes) > 0:
                        scale_x = orig_w / resized_w
                        scale_y = orig_h / resized_h
                        pred_boxes[:, 0] *= scale_x
                        pred_boxes[:, 2] *= scale_x
                        pred_boxes[:, 1] *= scale_y
                        pred_boxes[:, 3] *= scale_y

                    pred_dict = {
                        'boxes' : pred_boxes,
                        'scores': pred_scores,
                        'labels': pred_labels,
                    }
                    gt_dict = {
                        'boxes' : gt_boxes_pixels,
                        'labels': gt_labels,
                    }

                    batch_preds.append(pred_dict)
                    batch_gts.append(gt_dict)
                    all_predictions.append(pred_dict)
                    all_ground_truths.append(gt_dict)

                self.coco_metric.update(batch_preds, batch_gts)

                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(self.data_loader)}")

                continue  # skip DETR logic below

            # ==============================================================
            # DETR / DEFORMABLE-DETR / D-FINE BRANCH
            # ==============================================================
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            orig_sizes = torch.stack([t['orig_size'] for t in targets]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=orig_sizes, threshold=self.conf_threshold
            )

            batch_preds = []
            batch_gts   = []

            for i, (target, result) in enumerate(zip(targets, results)):
                gt_boxes_norm = target.get('boxes', torch.empty((0, 4)))
                is_empty_image = (len(gt_boxes_norm) == 0)

                if is_empty_image:
                    total_empty_images += 1
                    # STRICT FIX: The post_process function already dropped boxes below conf_threshold.
                    # We just count how many boxes survived the filter.
                    num_detections = len(result['scores'])
                    if num_detections == 0:
                        correctly_silent_images += 1
                    else:
                        hallucinations_on_empty += num_detections

                pred_dict = {
                    'boxes' : result['boxes'].cpu(),
                    'scores': result['scores'].cpu(),
                    'labels': result['labels'].cpu(),
                }

                orig_h, orig_w  = target['orig_size'].tolist()
                gt_boxes_pixels = self.normalize_to_pixels(gt_boxes_norm.cpu(), orig_h, orig_w)
                gt_labels       = self._get_gt_labels(target)

                gt_dict = {
                    'boxes' : gt_boxes_pixels,
                    'labels': gt_labels,
                }

                batch_preds.append(pred_dict)
                batch_gts.append(gt_dict)
                all_predictions.append(pred_dict)
                all_ground_truths.append(gt_dict)

            self.coco_metric.update(batch_preds, batch_gts)

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(self.data_loader)}")

        bg_stats = {
            'total_empty'  : total_empty_images,
            'correct_empty': correctly_silent_images,
            'hallucinations': hallucinations_on_empty,
        }

        return all_predictions, all_ground_truths, bg_stats

    # ------------------------------------------------------------------
    # MATCHING & METRICS
    # ------------------------------------------------------------------

    def match_predictions_to_ground_truth(self, predictions, ground_truths):
        """Match predictions to GT using IoU (used for Confusion Matrix)"""
        matches = []
        for pred_dict, gt_dict in zip(predictions, ground_truths):
            pred_boxes  = pred_dict['boxes']
            pred_scores = pred_dict['scores']
            pred_labels = pred_dict['labels']
            gt_boxes    = gt_dict['boxes']
            gt_labels   = gt_dict['labels']

            if len(gt_boxes) == 0:
                for j in range(len(pred_boxes)):
                    matches.append({'pred_label': pred_labels[j].item(), 'gt_label': None, 'score': pred_scores[j].item(), 'matched': False})
                continue

            if len(pred_boxes) == 0:
                for j in range(len(gt_boxes)):
                    matches.append({'pred_label': None, 'gt_label': gt_labels[j].item(), 'score': 0.0, 'matched': False})
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
                    matches.append({'pred_label': pred_label, 'gt_label': gt_labels[best_gt_idx].item(), 'score': pred_score, 'matched': True})
                else:
                    matches.append({'pred_label': pred_label, 'gt_label': None, 'score': pred_score, 'matched': False})

            for j in range(len(gt_boxes)):
                if j not in matched_gt:
                    matches.append({'pred_label': None, 'gt_label': gt_labels[j].item(), 'score': 0.0, 'matched': False})

        return matches

    def compute_confusion_matrix(self, matches, bg_stats):
        """Compute CM and inject True Negatives"""
        num_classes = self.config['model']['num_object_classes']
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)

        for m in matches:
            if m['matched']:
                cm[m['gt_label'], m['pred_label']] += 1
            elif m['gt_label'] is None:
                cm[0, m['pred_label']] += 1
            elif m['pred_label'] is None:
                cm[m['gt_label'], 0] += 1

        cm[0, 0] += bg_stats['correct_empty']
        return cm

    def compute_pr_curve_data(self, matches):
        """Generate PR curve data for plotting (Fixed Plotting Artifacts)"""
        num_classes = self.config['model']['num_object_classes']
        pr_data = {}

        for class_id in range(num_classes):
            y_true, y_scores = [], []
            total_gt = 0  # 1. Track total real birds for accurate recall math

            for m in matches:
                if m.get('gt_label') == class_id:
                    total_gt += 1  # 2. Count the ground truth
                    if m.get('pred_label') == class_id and m['matched']:
                        y_true.append(1);  y_scores.append(m['score'])
                elif m.get('pred_label') == class_id:
                    y_true.append(0);  y_scores.append(m['score'])

            n_pos = int(sum(y_true)) if y_true else 0
            made_predictions = sum(y_scores) > 0.0

            if n_pos > 0 and made_predictions:
                precision, recall, _ = precision_recall_curve(np.array(y_true), np.array(y_scores))
                # 4. Mathematically scale recall to reflect missed birds due to the 10% cutoff
                if total_gt > 0:
                    recall = recall * (n_pos / total_gt) 
                    ap = average_precision_score(np.array(y_true), np.array(y_scores)) * (n_pos / total_gt)
                else:
                    ap = 0.0
            else:
                precision = recall = np.array([])
                ap = 0.0

            pr_data[class_id] = {
                'precision'  : precision,
                'recall'     : recall,
                'ap'         : ap,
                'class_name' : self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}",
                'num_samples': total_gt,
            }
        return pr_data

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------

    def evaluate(self, epoch=None):
        print(f"\n{'='*60}")
        print(f"Evaluation{f' — Epoch {epoch}' if epoch else ''}")
        print(f"{'='*60}")

        print("Collecting predictions…")
        predictions, ground_truths, bg_stats = self.collect_predictions_and_bg_stats()

        # --- COCO Metrics ---
        print("\n⏳ Computing COCO Metrics via torchmetrics...")
        coco_results = self.coco_metric.compute()

        map_50_95 = coco_results['map'].item()
        map_50    = coco_results['map_50'].item()
        recall    = coco_results['mar_100'].item()
        f1_score  = 2 * ((map_50 * recall) / (map_50 + recall)) if (map_50 + recall) > 0 else 0.0
        
        # EXTRACTING SIZE-SPECIFIC METRICS
        # Returns -1 if no objects of that size exist in the evaluation set
        map_small  = coco_results.get('map_small', torch.tensor(-1.0)).item()
        map_medium = coco_results.get('map_medium', torch.tensor(-1.0)).item()
        map_large  = coco_results.get('map_large', torch.tensor(-1.0)).item()

        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        print(f"mAP@0.5      (Headline Precision) : {map_50:.4f}  ({map_50 * 100:.1f}%)")
        print(f"mAP@0.5:0.95 (Strict Precision)   : {map_50_95:.4f}  ({map_50_95 * 100:.1f}%)")
        print(f"Recall       (Found Animals)      : {recall:.4f}  ({recall * 100:.1f}%)")
        print(f"F1-Score     (Combined Balance)   : {f1_score:.4f}  ({f1_score * 100:.1f}%)")
        print("-" * 50)
        print("SCALE-SPECIFIC METRICS (FOR COMPRESSION ANALYSIS)")
        print(f"mAP_Small    (Area < 32^2 px)     : {map_small:.4f}  ({map_small * 100:.1f}%)" if map_small >= 0 else "mAP_Small    (Area < 32^2 px)     : N/A (No small objects)")
        print(f"mAP_Medium   (32^2 < Area < 96^2) : {map_medium:.4f}  ({map_medium * 100:.1f}%)" if map_medium >= 0 else "mAP_Medium   (32^2 < Area < 96^2) : N/A (No medium objects)")
        print(f"mAP_Large    (Area > 96^2 px)     : {map_large:.4f}  ({map_large * 100:.1f}%)" if map_large >= 0 else "mAP_Large    (Area > 96^2 px)     : N/A (No large objects)")
        print("="*50)

        self.coco_metric.reset()

        # --- Background Report ---
        bg_accuracy = 0.0
        if bg_stats['total_empty'] > 0:
            bg_accuracy = (bg_stats['correct_empty'] / bg_stats['total_empty']) * 100
            print(f"\n  [Background Performance]")
            print(f"  Total Empty Images : {bg_stats['total_empty']}")
            print(f"  Correctly Silent   : {bg_stats['correct_empty']}")
            print(f"  Background Acc     : {bg_accuracy:.2f}%")

        # --- Confusion Matrix & PR Curves ---
        matches = self.match_predictions_to_ground_truth(predictions, ground_truths)

        cm      = self.compute_confusion_matrix(matches, bg_stats) if self.config.get('evaluation', {}).get('compute_confusion_matrix', True) else None
        pr_data = self.compute_pr_curve_data(matches)              if self.config.get('evaluation', {}).get('compute_pr_curve', True)        else {}

        if self.config.get('evaluation', {}).get('save_plots', True) and (cm is not None or pr_data):
            plots_dir = self.config.get('evaluation', {}).get('plots_dir', 'evaluation_plots/')
            epoch_dir = os.path.join(plots_dir, f"evaluation_{epoch}" if epoch else "evaluation_final")
            create_readable_visualizations(
                metrics={'mAP': map_50},
                pr_data=pr_data,
                confusion_matrix=cm if cm is not None else np.zeros((len(self.class_names), len(self.class_names))),
                class_names=self.class_names,
                output_dir=epoch_dir,
            )

        metrics = {
            'mAP'             : map_50,
            'mAP_50_95'       : map_50_95,
            'mAP_Small'       : map_small,
            'mAP_Medium'      : map_medium,
            'mAP_Large'       : map_large,
            'Recall'          : recall,
            'F1'              : f1_score,
            'pr_data'         : pr_data,
            'confusion_matrix': cm,
            'bg_accuracy'     : bg_accuracy,
        }

        print(f"{'='*60}\n")
        return metrics, pr_data, cm