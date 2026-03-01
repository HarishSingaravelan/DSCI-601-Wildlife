"""
Deep Model Diagnostic - Check if model is actually learning detection
Run this on your latest checkpoint to see what's happening internally
"""

import torch
import yaml
import sys
import numpy as np
from pathlib import Path

def deep_diagnostic(checkpoint_path, config_path='config/config.yaml'):
    print("="*70)
    print("DEEP MODEL DIAGNOSTIC")
    print("="*70)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['training']['device'])
    
    # Load model
    from transformers import DetrForObjectDetection, DetrImageProcessor
    
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=config['model']['num_object_classes'],
        ignore_mismatched_sizes=True
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # Load a batch of data
    sys.path.insert(0, 'modeling/detr')
    from detr_with_existing_pipeline import DETRWithExistingDataPipeline
    
    trainer = DETRWithExistingDataPipeline(config)
    
    # Get one batch
    pixel_values, targets = next(iter(trainer.val_loader))
    pixel_values = pixel_values.to(device)
    
    print(f"\nBatch info:")
    print(f"  Batch size: {len(targets)}")
    print(f"  Pixel values shape: {pixel_values.shape}")
    
    # Find an image with GT
    gt_indices = [i for i, t in enumerate(targets) if len(t['boxes']) > 0]
    if gt_indices:
        img_idx = gt_indices[0]
        print(f"  Using image {img_idx} (has {len(targets[img_idx]['boxes'])} GT boxes)")
    else:
        img_idx = 0
        print(f"  No GT in batch - using image 0")
    
    # Run model
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
    
    print("\n" + "="*70)
    print("MODEL OUTPUTS ANALYSIS")
    print("="*70)
    
    logits = outputs.logits[img_idx]      # [100, num_classes+1]
    boxes  = outputs.pred_boxes[img_idx]  # [100, 4]
    
    print(f"\nLogits shape: {logits.shape}")
    print(f"Boxes shape:  {boxes.shape}")
    
    # Convert logits to probabilities
    probs = logits.softmax(-1)  # [100, num_classes+1]
    
    # Check what the model is predicting
    max_probs, pred_classes = probs.max(-1)
    
    print(f"\n--- Query Predictions (all 100 queries) ---")
    print(f"Max probability per query:")
    print(f"  Mean:   {max_probs.mean():.4f}")
    print(f"  Median: {max_probs.median():.4f}")
    print(f"  Max:    {max_probs.max():.4f}")
    print(f"  Min:    {max_probs.min():.4f}")
    
    # Check class distribution
    num_classes = config['model']['num_object_classes']
    class_names = config['model']['class_names']
    
    print(f"\n--- Predicted Class Distribution (across 100 queries) ---")
    for c in range(num_classes):
        count = (pred_classes == c).sum().item()
        print(f"  Class {c} ({class_names[c] if c < len(class_names) else 'unknown'}): {count} queries")
    
    # Check "no object" class (last index)
    no_obj_class = num_classes
    no_obj_count = (pred_classes == no_obj_class).sum().item()
    print(f"  No-object class (index {no_obj_class}): {no_obj_count} queries")
    
    # Analyze confidence scores
    print(f"\n--- Top 10 Most Confident Predictions ---")
    top_indices = torch.argsort(max_probs, descending=True)[:10]
    for rank, idx in enumerate(top_indices):
        prob = max_probs[idx].item()
        cls  = pred_classes[idx].item()
        box  = boxes[idx].cpu().numpy()
        
        cls_name = class_names[cls] if cls < len(class_names) else f"no-object"
        print(f"  #{rank+1}: query={idx:3d}  prob={prob:.4f}  class={cls:2d} ({cls_name:20s})  box={box}")
    
    # Check if ANY query predicts object class with > 0.1 confidence
    print(f"\n--- Detection Analysis ---")
    object_queries = pred_classes < num_classes  # Exclude no-object
    object_confidences = max_probs[object_queries]
    
    if len(object_confidences) > 0:
        print(f"  Queries predicting an object class: {len(object_confidences)}")
        print(f"  Max confidence among object predictions: {object_confidences.max():.4f}")
        print(f"  Mean confidence among object predictions: {object_confidences.mean():.4f}")
        
        # Count by threshold
        for thresh in [0.01, 0.05, 0.1, 0.2, 0.5]:
            count = (object_confidences > thresh).sum().item()
            print(f"  Predictions > {thresh}: {count}")
    else:
        print(f"  ⚠️  ALL 100 queries predict 'no-object' class!")
        print(f"      Model has collapsed to predicting nothing")
    
    # Check loss components
    print(f"\n--- Loss Analysis ---")
    print(f"  Total loss from checkpoint: {ckpt.get('train_loss', 'N/A')}")
    
    if gt_indices:
        # Compute loss on this image
        single_target = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in targets[img_idx].items()}]
        single_img = pixel_values[img_idx:img_idx+1]
        
        with torch.no_grad():
            loss_output = model(pixel_values=single_img, labels=single_target)
        
        print(f"  Loss on this image: {loss_output.loss.item():.4f}")
        if hasattr(loss_output, 'loss_dict'):
            print(f"  Loss components: {loss_output.loss_dict}")
    
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    if no_obj_count == 100:
        print("❌ CRITICAL: Model predicts 'no-object' for ALL 100 queries")
        print("   This means the model has NOT learned object detection")
        print("\n   Possible causes:")
        print("   1. Learning rate too low (model can't escape 'no-object' local minimum)")
        print("   2. Loss weights need adjustment")
        print("   3. Data preprocessing issue (boxes not in right format)")
        print("   4. Model needs many more epochs")
        print("\n   Recommended fixes:")
        print("   - INCREASE learning rate to 0.0005 or 0.001")
        print("   - Check if GT boxes are being loaded correctly")
        print("   - Train for 50-100 more epochs")
    elif object_confidences.max() < 0.1:
        print("⚠️  Model predicts objects but with very low confidence")
        print(f"   Max object confidence: {object_confidences.max():.4f}")
        print("\n   Recommended:")
        print("   - Lower eval threshold to 0.01")
        print("   - Continue training - confidence will increase")
    else:
        print("✓ Model is making reasonable predictions")
        print(f"  Object predictions with >0.1 conf: {(object_confidences > 0.1).sum().item()}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        # Find latest checkpoint
        ckpt_dir = Path("checkpoints")
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
            if ckpts:
                checkpoint_path = str(ckpts[-1])
                print(f"Using latest checkpoint: {checkpoint_path}")
            else:
                print("No checkpoints found in checkpoints/")
                sys.exit(1)
        else:
            print("Usage: python deep_diagnostic.py <checkpoint_path>")
            sys.exit(1)
    else:
        checkpoint_path = sys.argv[1]
    
    deep_diagnostic(checkpoint_path)