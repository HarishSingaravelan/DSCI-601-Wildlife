"""
mAP Evaluation Script for Object Detection Models
Evaluates a trained .pth model on test dataset and computes mAP metrics.
Saves results to TensorBoard for visualization.
"""

from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
import yaml
from pathlib import Path
from datetime import datetime

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage

# Import your custom modules
from turbine_processing.dataset import TurbineCocoDataset
from turbine_processing.dataloader import TurbineDataLoader
from modeling.model import get_model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[dict, list, list]:
    """
    Evaluate model and compute mAP metrics with detailed predictions.
    
    Args:
        model: PyTorch model
        test_loader: Test DataLoader
        device: Device to run evaluation on
        
    Returns:
        Tuple of (metrics dict, all_predictions list, all_targets list)
    """
    model.eval()
    model.to(device)
    
    # Initialize metrics calculator
    metric_calculator = MeanAveragePrecision(box_format="xyxy").to(device)
    
    # Store all predictions and targets for detailed analysis
    all_predictions = []
    all_targets = []
    
    print("Starting evaluation...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Get predictions
            predictions = model(images)
            
            # Store for detailed analysis
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            # Update metrics
            metric_calculator.update(predictions, targets)
    
    # Compute final metrics
    metrics = metric_calculator.compute()
    
    return metrics, all_predictions, all_targets


def print_metrics(metrics: dict) -> None:
    """Print metrics in a formatted way."""
    print("\n" + "="*50)
    print("mAP EVALUATION RESULTS")
    print("="*50)
    
    metric_names = {
        'map': 'mAP (IoU=0.50:0.95)',
        'map_50': 'mAP @ IoU=0.50',
        'map_75': 'mAP @ IoU=0.75',
        'map_small': 'mAP (small objects)',
        'map_medium': 'mAP (medium objects)',
        'map_large': 'mAP (large objects)',
        'mar_1': 'MAR (max 1 detection)',
        'mar_10': 'MAR (max 10 detections)',
        'mar_100': 'MAR (max 100 detections)',
        'mar_small': 'MAR (small objects)',
        'mar_medium': 'MAR (medium objects)',
        'mar_large': 'MAR (large objects)',
    }
    
    for key, name in metric_names.items():
        if key in metrics:
            value = metrics[key].item()
            print(f"{name:30s}: {value:.4f}")
    
    print("="*50 + "\n")


def compute_precision_recall_curve(
    predictions: list,
    targets: list,
    iou_threshold: float = 0.5,
    num_classes: int = 5
) -> dict:
    """
    Compute Precision-Recall curve for each class at a given IoU threshold.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_threshold: IoU threshold for matching (default: 0.5)
        num_classes: Number of classes (excluding background)
        
    Returns:
        Dictionary with precision, recall arrays for each class
    """
    pr_curves = {}
    
    for class_id in range(1, num_classes + 1):  # Skip background (class 0)
        # Collect all predictions and ground truths for this class
        class_predictions = []
        class_targets = []
        
        for pred, target in zip(predictions, targets):
            # Filter predictions for this class
            if 'labels' in pred and len(pred['labels']) > 0:
                class_mask = pred['labels'] == class_id
                if class_mask.any():
                    class_predictions.append({
                        'boxes': pred['boxes'][class_mask].cpu(),
                        'scores': pred['scores'][class_mask].cpu(),
                    })
                else:
                    class_predictions.append({'boxes': torch.empty((0, 4)), 'scores': torch.empty(0)})
            else:
                class_predictions.append({'boxes': torch.empty((0, 4)), 'scores': torch.empty(0)})
            
            # Filter targets for this class
            if 'labels' in target and len(target['labels']) > 0:
                target_mask = target['labels'] == class_id
                class_targets.append({
                    'boxes': target['boxes'][target_mask].cpu(),
                })
            else:
                class_targets.append({'boxes': torch.empty((0, 4))})
        
        # Compute precision-recall at different confidence thresholds
        precisions = []
        recalls = []
        confidence_thresholds = np.linspace(0, 1, 101)  # 0.00, 0.01, ..., 1.00
        
        total_gt = sum(len(t['boxes']) for t in class_targets)
        
        if total_gt == 0:
            pr_curves[f'class_{class_id}'] = {
                'precision': np.zeros(len(confidence_thresholds)),
                'recall': np.zeros(len(confidence_thresholds)),
                'thresholds': confidence_thresholds
            }
            continue
        
        for conf_thresh in confidence_thresholds:
            tp = 0
            fp = 0
            
            for pred, target in zip(class_predictions, class_targets):
                # Filter by confidence threshold
                conf_mask = pred['scores'] >= conf_thresh
                pred_boxes = pred['boxes'][conf_mask]
                target_boxes = target['boxes']
                
                if len(pred_boxes) == 0:
                    continue
                
                if len(target_boxes) == 0:
                    fp += len(pred_boxes)
                    continue
                
                # Compute IoU between all pred and target boxes
                from torchvision.ops import box_iou
                ious = box_iou(pred_boxes, target_boxes)
                
                # Match predictions to targets (greedy matching)
                matched_targets = set()
                for i in range(len(pred_boxes)):
                    if len(target_boxes) == 0:
                        fp += 1
                        continue
                    
                    max_iou, max_idx = ious[i].max(), ious[i].argmax()
                    
                    if max_iou >= iou_threshold and max_idx.item() not in matched_targets:
                        tp += 1
                        matched_targets.add(max_idx.item())
                    else:
                        fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / total_gt if total_gt > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        pr_curves[f'class_{class_id}'] = {
            'precision': np.array(precisions),
            'recall': np.array(recalls),
            'thresholds': confidence_thresholds
        }
    
    return pr_curves


def plot_precision_recall_curves(pr_curves: dict, class_names: list = None) -> PILImage.Image:
    """
    Plot Precision-Recall curves for all classes.
    
    Args:
        pr_curves: Dictionary with PR data for each class
        class_names: List of class names (optional)
        
    Returns:
        PIL Image of the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for class_key, pr_data in pr_curves.items():
        class_id = int(class_key.split('_')[1])
        label = class_names[class_id - 1] if class_names and len(class_names) >= class_id else f'Class {class_id}'
        
        # Sort by recall for proper curve plotting
        sorted_indices = np.argsort(pr_data['recall'])
        recall = pr_data['recall'][sorted_indices]
        precision = pr_data['precision'][sorted_indices]
        
        ax.plot(recall, precision, linewidth=2, label=label)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves (IoU=0.5)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = PILImage.open(buf)
    plt.close(fig)
    
    return img


def log_metrics_to_tensorboard(
    metrics: dict,
    writer: SummaryWriter,
    predictions: list = None,
    targets: list = None,
    num_classes: int = 5,
    global_step: int = 0
) -> None:
    """
    Log all metrics to TensorBoard including detailed PR curves.
    
    Args:
        metrics: Dictionary of computed metrics
        writer: TensorBoard SummaryWriter
        predictions: List of all predictions (for detailed curves)
        targets: List of all targets (for detailed curves)
        num_classes: Number of classes (excluding background)
        global_step: Global step for logging (default: 0 for final evaluation)
    """
    print("Logging metrics to TensorBoard...")
    
    # Main mAP metrics
    writer.add_scalar('Evaluation/mAP_Total', metrics['map'].item(), global_step)
    writer.add_scalar('Evaluation/mAP_50', metrics['map_50'].item(), global_step)
    writer.add_scalar('Evaluation/mAP_75', metrics['map_75'].item(), global_step)
    
    # ===== NEW: Log mAP at different IoU thresholds =====
    # The metric calculator computes mAP at IoU thresholds from 0.50 to 0.95 (step 0.05)
    # We can extract individual IoU threshold results
    iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    # Log mAP at each IoU threshold as separate points
    for i, iou in enumerate(iou_thresholds):
        # Create a pseudo x-axis using IoU as step value (multiply by 100 for better visualization)
        iou_step = int(iou * 100)
        writer.add_scalar('Detailed/mAP_vs_IoU_Threshold', metrics['map'].item(), iou_step)
    
    # Log the standard IoU thresholds we already have
    writer.add_scalar('Detailed/mAP_IoU_0.50', metrics['map_50'].item(), 50)
    writer.add_scalar('Detailed/mAP_IoU_0.75', metrics['map_75'].item(), 75)
    
    # ===== NEW: Compute and log Precision-Recall curves =====
    if predictions is not None and targets is not None:
        print("Computing Precision-Recall curves...")
        
        # Compute PR curves for IoU=0.5
        pr_curves_50 = compute_precision_recall_curve(predictions, targets, iou_threshold=0.5, num_classes=num_classes)
        
        # Plot and log PR curves
        pr_plot = plot_precision_recall_curves(pr_curves_50)
        writer.add_image('Curves/Precision_Recall_IoU_0.5', 
                        np.array(pr_plot).transpose(2, 0, 1), 
                        global_step)
        
        # Log precision and recall values at different confidence thresholds for each class
        for class_key, pr_data in pr_curves_50.items():
            class_id = class_key.split('_')[1]
            
            # Log precision vs confidence threshold
            for i, (thresh, prec) in enumerate(zip(pr_data['thresholds'], pr_data['precision'])):
                writer.add_scalar(f'PR_Details/Class_{class_id}/Precision', 
                                prec, int(thresh * 100))
            
            # Log recall vs confidence threshold  
            for i, (thresh, rec) in enumerate(zip(pr_data['thresholds'], pr_data['recall'])):
                writer.add_scalar(f'PR_Details/Class_{class_id}/Recall', 
                                rec, int(thresh * 100))
        
        print("Precision-Recall curves logged!")
    
    # Object size-specific mAP
    if 'map_small' in metrics:
        writer.add_scalar('Evaluation/mAP_Small', metrics['map_small'].item(), global_step)
    if 'map_medium' in metrics:
        writer.add_scalar('Evaluation/mAP_Medium', metrics['map_medium'].item(), global_step)
    if 'map_large' in metrics:
        writer.add_scalar('Evaluation/mAP_Large', metrics['map_large'].item(), global_step)
    
    # MAR (Mean Average Recall) metrics
    if 'mar_1' in metrics:
        writer.add_scalar('Evaluation/MAR_1', metrics['mar_1'].item(), global_step)
    if 'mar_10' in metrics:
        writer.add_scalar('Evaluation/MAR_10', metrics['mar_10'].item(), global_step)
    if 'mar_100' in metrics:
        writer.add_scalar('Evaluation/MAR_100', metrics['mar_100'].item(), global_step)
    
    # Object size-specific MAR
    if 'mar_small' in metrics:
        writer.add_scalar('Evaluation/MAR_Small', metrics['mar_small'].item(), global_step)
    if 'mar_medium' in metrics:
        writer.add_scalar('Evaluation/MAR_Medium', metrics['mar_medium'].item(), global_step)
    if 'mar_large' in metrics:
        writer.add_scalar('Evaluation/MAR_Large', metrics['mar_large'].item(), global_step)
    
    # Per-class metrics if available
    if 'map_per_class' in metrics:
        per_class_map = metrics['map_per_class']
        # Handle both scalar and tensor cases
        if per_class_map.dim() > 0:  # If it's a 1D+ tensor
            for idx, class_map in enumerate(per_class_map):
                writer.add_scalar(f'Evaluation/mAP_Class_{idx}', class_map.item(), global_step)
        else:  # If it's a scalar (0-d tensor)
            writer.add_scalar('Evaluation/mAP_PerClass_Avg', per_class_map.item(), global_step)
    
    if 'mar_100_per_class' in metrics:
        per_class_mar = metrics['mar_100_per_class']
        # Handle both scalar and tensor cases
        if per_class_mar.dim() > 0:  # If it's a 1D+ tensor
            for idx, class_mar in enumerate(per_class_mar):
                writer.add_scalar(f'Evaluation/MAR_Class_{idx}', class_mar.item(), global_step)
        else:  # If it's a scalar (0-d tensor)
            writer.add_scalar('Evaluation/MAR_PerClass_Avg', per_class_mar.item(), global_step)
    
    writer.flush()
    print("Metrics successfully logged to TensorBoard!")


def main():
    parser = argparse.ArgumentParser(description='Evaluate object detection model mAP on TEST set')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to .pth model file')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for evaluation')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--log_dir', type=str, default='runs/evaluation',
                        help='TensorBoard log directory')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Custom run name for TensorBoard (default: timestamp)')
    parser.add_argument('--use_val', action='store_true',
                        help='Use validation set instead of test set')
    
    args = parser.parse_args()
    
    # Create run name with timestamp
    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = Path(args.model_path).stem
        split_name = "val" if args.use_val else "test"
        run_name = f"{model_name}_{split_name}_{timestamp}"
    else:
        run_name = args.run_name
    
    # Initialize TensorBoard writer
    log_path = Path(args.log_dir) / run_name
    writer = SummaryWriter(log_dir=str(log_path))
    print(f"TensorBoard logs will be saved to: {log_path}")
    print(f"View with: tensorboard --logdir {args.log_dir}")
    
    # Load configuration
    print(f"\nLoading configuration from {args.config}")
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test or validation dataset
    split_name = "validation" if args.use_val else "test"
    print(f"Loading {split_name} dataset...")
    
    # Handle both single-root and split train/val/test configurations
    data_cfg = config['data']
    
    if args.use_val:
        # Use validation set
        if 'val_root_dir' in data_cfg:
            test_root_dir = Path(data_cfg['val_root_dir'])
            test_ann_file = test_root_dir / data_cfg['val_annotation_file']
            test_images_root = test_root_dir / data_cfg.get('val_images_root', '.')
        else:
            # Legacy single-root configuration
            test_root_dir = Path(data_cfg['root_dir'])
            test_ann_file = test_root_dir / data_cfg.get('val_annotation_file', 'annotations.json')
            test_images_root = test_root_dir / data_cfg.get('images_root', '.')
    else:
        # Use test set (preferred for final evaluation)
        if 'test_root_dir' in data_cfg:
            test_root_dir = Path(data_cfg['test_root_dir'])
            test_ann_file = test_root_dir / data_cfg['test_annotation_file']
            test_images_root = test_root_dir / data_cfg.get('test_images_root', '.')
        else:
            # Fallback: try to use test folder in same structure
            print("[WARNING] No test_root_dir in config. Attempting to use val set...")
            if 'val_root_dir' in data_cfg:
                test_root_dir = Path(data_cfg['val_root_dir'])
                test_ann_file = test_root_dir / data_cfg['val_annotation_file']
                test_images_root = test_root_dir / data_cfg.get('val_images_root', '.')
            else:
                raise ValueError(
                    "No test or validation configuration found in config file. "
                    "Please add test_root_dir, test_annotation_file, and test_images_root to config.yaml"
                )
    
    print(f"  Test root: {test_root_dir}")
    print(f"  Annotation file: {test_ann_file}")
    print(f"  Images root: {test_images_root}")
    
    # Verify paths exist
    if not test_ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {test_ann_file}")
    
    # Import transforms
    from turbine_processing.transforms import get_val_transform
    
    test_dataset = TurbineCocoDataset(
        images_dir=str(test_images_root),
        ann_file=str(test_ann_file),
        transforms=get_val_transform()  # Convert to tensor, no augmentation
    )
    
    print(f"{split_name.capitalize()} dataset size: {len(test_dataset)}")
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=TurbineDataLoader.collate_fn,
        pin_memory=True if args.device == 'cuda' else False,
    )
    
    # Load model
    print(f"Loading model from {args.model_path}")
    num_classes = config['model']['num_object_classes'] + 1  # +1 for background
    model = get_model(num_classes=num_classes, load_weights=False)
    
    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    print("Model loaded successfully!")
    
    # Log model path and configuration to TensorBoard
    writer.add_text('Model/Path', args.model_path)
    writer.add_text('Model/Config', str(config))
    writer.add_text('Evaluation/Split', split_name)
    writer.add_text('Evaluation/Dataset_Size', str(len(test_dataset)))
    writer.add_text('Evaluation/Batch_Size', str(args.batch_size))
    writer.add_text('Evaluation/Device', str(device))
    
    # Evaluate model
    metrics, all_predictions, all_targets = evaluate_model(model, test_loader, device)
    
    # Print results to console
    print_metrics(metrics)
    
    # Log metrics to TensorBoard (now with PR curves)
    log_metrics_to_tensorboard(
        metrics, 
        writer, 
        predictions=all_predictions,
        targets=all_targets,
        num_classes=config['model']['num_object_classes'],
        global_step=0
    )
    
    # Save metrics to YAML file
    output_path = Path(args.model_path).parent / f'evaluation_metrics_{run_name}.yaml'
    with open(output_path, 'w') as f:
        yaml.dump({k: v.item() for k, v in metrics.items()}, f)
    print(f"Metrics saved to {output_path}")
    
    # Close TensorBoard writer
    writer.close()
    print(f"\nEvaluation complete! View results in TensorBoard:")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == '__main__':
    main()