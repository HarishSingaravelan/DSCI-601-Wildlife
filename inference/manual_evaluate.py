import torch
import yaml
import os
import sys
from pathlib import Path
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import DataLoader

# ==========================================
# PATH FIX: Add project root to sys.path
# ==========================================
# This ensures Python can find 'modeling' and 'turbine_processing'
ROOT_DIR = Path(__file__).resolve().parents[1]  
sys.path.insert(0, str(ROOT_DIR))

# Now we can safely import your custom modules
from modeling.detr.detr_evaluation import DETREvaluator
from modeling.detr.detr_with_existing_pipeline import DETRTransformAdapter, DETRWithExistingDataPipeline
from turbine_processing.dataset import TurbineCocoDataset
from turbine_processing.transforms_detr import get_val_transform_detr

# ==========================================
# SETTINGS: TWEAK THESE FOR YOUR EXPERIMENTS
# ==========================================
CHECKPOINT_PATH = "checkpoints/full_run_log_sampler_confidence_0.5_bg/checkpoint_epoch_290.pth" 
CUSTOM_THRESHOLD = 0.3  # <--- Change this to test 0.20, 0.50, etc. without retraining!
CONFIG_PATH = "config/config.yaml"
# ==========================================

def run_manual_eval():
    print(f"\n--- Starting Manual Evaluation ---")
    print(f"Target Threshold: {CUSTOM_THRESHOLD}")
    
    # 1. Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")

    # 2. Load Processor and Model
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", 
        do_convert_annotations=True
    )
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=config['model']['num_object_classes'],
        ignore_mismatched_sizes=True
    )
    
    # 3. Load specific weights
    print(f"Loading weights from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    # Handle both raw state_dicts and full checkpoint dictionaries
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
        batch_size=config['training']['batch_size'],
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
        config=config,
        bg_threshold = 0.3
    )

    with torch.no_grad():
        metrics, pr_data, confusion_matrix = evaluator.evaluate(
            epoch=f"manual_test_thresh_{CUSTOM_THRESHOLD}"
        )

    # 6. Print Summary
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Tested Checkpoint : {os.path.basename(CHECKPOINT_PATH)}")
    print(f"Confidence Thresh : {CUSTOM_THRESHOLD}")
    print(f"Overall mAP       : {metrics.get('mAP', 0.0):.4f}")
    print("="*50)
    print("Check your evaluation output folder for the newly generated confusion matrices and plots!")

if __name__ == "__main__":
    run_manual_eval()