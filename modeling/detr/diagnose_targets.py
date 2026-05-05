"""
Quick diagnostic - run this to see exactly what's in your targets
during evaluation so we can fix the key mismatch
"""

import torch
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.detr.detr_with_existing_pipeline import DETRWithExistingDataPipeline


def diagnose_targets(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Loading model and data pipeline...")
    trainer = DETRWithExistingDataPipeline(config)

    # Grab ONE batch from val loader
    pixel_values, targets = next(iter(trainer.val_loader))

    print("\n" + "="*60)
    print("BATCH INSPECTION")
    print("="*60)
    print(f"pixel_values shape : {pixel_values.shape}")
    print(f"num targets in batch: {len(targets)}")

    print("\n--- Target[0] keys and values ---")
    t = targets[0]
    for key, val in t.items():
        if isinstance(val, torch.Tensor):
            print(f"  key='{key}'  shape={val.shape}  dtype={val.dtype}  values={val[:4]}")
        else:
            print(f"  key='{key}'  value={val}")

    print("\n--- Target[1] keys (just keys) ---")
    for key in targets[1].keys():
        print(f"  '{key}'")

    # Now run the model on this batch and inspect outputs
    print("\n" + "="*60)
    print("MODEL OUTPUT INSPECTION")
    print("="*60)

    from transformers import DetrImageProcessor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    device = torch.device(config['training']['device'])

    trainer.model.eval()
    with torch.no_grad():
        pv = pixel_values.to(device)
        outputs = trainer.model(pixel_values=pv)

    print(f"logits shape    : {outputs.logits.shape}")    # [B, 100, num_classes+1]
    print(f"pred_boxes shape: {outputs.pred_boxes.shape}") # [B, 100, 4]

    # Try post-processing with different target_sizes
    print("\n--- Trying post_process_object_detection ---")

    # Option 1: use orig_size from target if it exists
    if 'orig_size' in targets[0]:
        print("  'orig_size' key FOUND in targets")
        orig_sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
        print(f"  orig_sizes: {orig_sizes[:4]}")
    else:
        print("  'orig_size' key NOT FOUND — using pixel_values spatial dims")
        h = pixel_values.shape[2]
        w = pixel_values.shape[3]
        orig_sizes = torch.tensor([[h, w]] * len(targets)).to(device)
        print(f"  Using fallback size: {h}x{w}")

    results_05 = processor.post_process_object_detection(
        outputs, target_sizes=orig_sizes, threshold=0.5
    )
    results_01 = processor.post_process_object_detection(
        outputs, target_sizes=orig_sizes, threshold=0.1
    )
    results_001 = processor.post_process_object_detection(
        outputs, target_sizes=orig_sizes, threshold=0.01
    )

    print(f"\n  Detections with threshold=0.50 : {sum(len(r['boxes']) for r in results_05)}")
    print(f"  Detections with threshold=0.10 : {sum(len(r['boxes']) for r in results_01)}")
    print(f"  Detections with threshold=0.01 : {sum(len(r['boxes']) for r in results_001)}")

    if len(results_001[0]['boxes']) > 0:
        print(f"\n  Sample prediction (threshold=0.01):")
        print(f"    boxes  : {results_001[0]['boxes'][:3]}")
        print(f"    scores : {results_001[0]['scores'][:3]}")
        print(f"    labels : {results_001[0]['labels'][:3]}")

    # Check ground truth boxes
    print("\n" + "="*60)
    print("GROUND TRUTH INSPECTION")
    print("="*60)

    boxes_key   = None
    labels_key  = None

    # Try to find the right keys
    for k in targets[0].keys():
        if 'box' in k.lower():
            boxes_key = k
        if 'label' in k.lower() or 'class' in k.lower():
            labels_key = k

    print(f"  Detected boxes key  : '{boxes_key}'")
    print(f"  Detected labels key : '{labels_key}'")

    if boxes_key:
        gt_boxes = targets[0][boxes_key]
        print(f"  GT boxes shape: {gt_boxes.shape}")
        if gt_boxes.numel() > 0:
            print(f"  GT boxes sample: {gt_boxes[:3]}")
            print(f"  GT boxes range: min={gt_boxes.min():.3f} max={gt_boxes.max():.3f}")
            print(f"  NOTE: if max <= 1.0, boxes are NORMALIZED (DETR format)")
            print(f"        if max >> 1.0, boxes are in PIXEL coordinates")
        else:
            print("  ⚠ GT boxes are EMPTY for this image!")

    if labels_key:
        print(f"  GT labels: {targets[0][labels_key]}")

    # Count how many images in the batch have GT boxes
    print("\n  GT box counts per image in batch:")
    for i, t in enumerate(targets):
        bk = boxes_key or 'boxes'
        boxes = t.get(bk, torch.zeros(0, 4))
        print(f"    Image {i}: {len(boxes)} GT boxes")

    print("\n" + "="*60)
    print("SUMMARY & FIX RECOMMENDATION")
    print("="*60)

    if boxes_key != 'boxes':
        print(f"⚠  GT boxes key is '{boxes_key}', not 'boxes'")
        print(f"   Fix: change 'boxes' to '{boxes_key}' in collect_predictions()")
    if labels_key not in ('class_labels', 'labels'):
        print(f"⚠  GT labels key is '{labels_key}'")
        print(f"   Fix: change key lookup in collect_predictions()")

    total_dets_01 = sum(len(r['boxes']) for r in results_01)
    if total_dets_01 == 0:
        print("⚠  Model produces 0 detections even at threshold=0.10")
        print("   → Model may need more training epochs")
        print("   → OR orig_size is wrong (boxes scaled incorrectly)")
    else:
        print(f"✓  Model produces {total_dets_01} detections at threshold=0.10")
        print(f"   → Evaluation threshold may be too high")
        print(f"   → Try lowering confidence_threshold in config to 0.1 or 0.2")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config/config.yaml'
    diagnose_targets(config_path)