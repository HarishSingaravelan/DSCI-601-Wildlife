import torch
import yaml
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# ==========================================
# PATH FIX: Add project root to sys.path
# ==========================================
ROOT_DIR = Path(__file__).resolve().parents[1]  
sys.path.insert(0, str(ROOT_DIR))

# Import all possible architectures
from transformers import (
    DetrForObjectDetection, DetrImageProcessor,
    DFineForObjectDetection, AutoImageProcessor,
    DeformableDetrForObjectDetection, DeformableDetrImageProcessor
)

from modeling.detr.detr_evaluation import DETREvaluator
from modeling.detr.detr_with_existing_pipeline import DETRTransformAdapter, DETRWithExistingDataPipeline
from turbine_processing.dataset import TurbineCocoDataset
from turbine_processing.transforms_detr import get_val_transform_detr

# ==========================================
# SETTINGS: TWEAK THESE FOR YOUR EXPERIMENTS
# ==========================================
CHECKPOINT_PATH = "checkpoints/dfine/detr_adaptive_inner_siou/epoch_100.pth" # <-- Update to your D-FINE checkpoint
CUSTOM_THRESHOLD = 0.1  # <--- Change this to test 0.40, 0.50, etc.
CONFIG_PATH = "config/config.yaml"
# ==========================================

def run_manual_eval():
    print(f"\n--- Starting Manual Evaluation ---")
    print(f"Target Threshold: {CUSTOM_THRESHOLD}")
    
    # 1. Load config
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Cannot find config at {CONFIG_PATH}")
        
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")

    # 2. DYNAMICALLY Load Processor and Model based on config
    arch = config['model'].get('architecture', 'standard_detr').lower()
    model_name = config['model'].get('pretrained_model', 'facebook/detr-resnet-50')
    num_classes = config['model']['num_object_classes']
    
    print(f"Initializing Architecture: {arch.upper()} from {model_name}")

    if arch == 'dfine':
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = DFineForObjectDetection.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    elif arch == 'deformable_detr':
        processor = DeformableDetrImageProcessor.from_pretrained(model_name)
        model = DeformableDetrForObjectDetection.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    else:
        processor = DetrImageProcessor.from_pretrained(model_name)
        model = DetrForObjectDetection.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    
    # 3. Load specific weights
    print(f"Loading weights from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    # 4. Setup Validation Dataset and Loader
    print("Preparing validation dataset...")
    val_transform = DETRTransformAdapter(get_val_transform_detr(), processor)
    val_dataset = TurbineCocoDataset(
        images_dir=config['data']['val_images_dir'],
        ann_file=config['data']['val_ann_file'],
        transforms=val_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training'].get('batch_size', 8),
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        collate_fn=DETRWithExistingDataPipeline.collate_fn
    )

    # 5. Run Evaluation
    print("\nGenerating metrics, confusion matrices, and PR curves...")
    
    # Temporarily overwrite the config threshold with our manual setting
    config['evaluation']['confidence_threshold'] = CUSTOM_THRESHOLD
    
    evaluator = DETREvaluator(
        model=model,
        data_loader=val_loader,
        processor=processor,
        device=device,
        config=config
    )

    with torch.no_grad():
        # This string determines the final folder name!
        folder_name = f"manual_test_thresh_{CUSTOM_THRESHOLD}"
        metrics, pr_data, confusion_matrix = evaluator.evaluate(epoch=folder_name)

    # 6. Print Summary
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Tested Checkpoint : {os.path.basename(CHECKPOINT_PATH)}")
    print(f"Confidence Thresh : {CUSTOM_THRESHOLD}")
    print(f"Overall mAP@0.5   : {metrics.get('mAP', 0.0):.4f}")
    print(f"Background Acc    : {metrics.get('bg_accuracy', 0.0):.2f}%")
    print("="*50)
    
    output_path = os.path.join(config['evaluation']['plots_dir'], f"evaluation_{folder_name}")
    print(f"Outputs saved to: {output_path}")

if __name__ == "__main__":
    run_manual_eval()