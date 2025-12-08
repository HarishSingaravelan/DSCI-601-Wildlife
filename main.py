import os
import sys
import datetime

import torch
import yaml

# --- Ensure project root is on sys.path ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# ------------------------------------------


# --- Imports from your own code ---
from turbine_processing.dataset import TurbineCocoDataset
from turbine_processing.dataloader import TurbineDataLoader
from turbine_processing.transforms import get_train_transform, get_val_transform
from modeling.model import get_model
from modeling.trainer import Trainer
# ----------------------------------


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Loads configuration parameters from a YAML file."""
    print(f"[DEBUG] Loading config from: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_datasets(cfg: dict):
    """
    Build train and validation datasets using the TurbineCocoDataset class.
    
    This version supports both single-root and split train/val configurations.

    Args:
        cfg: The configuration dictionary loaded from config.yaml.

    Returns:
        A tuple containing (train_dataset, val_dataset).
    """
    data_cfg = cfg["data"]
    
    # Check if using split configuration (separate train/val paths)
    if "train_root_dir" in data_cfg and "val_root_dir" in data_cfg:
        print("[INFO] Using split train/val configuration")
        
        # Training dataset paths
        train_root = data_cfg["train_root_dir"]
        train_images_root = os.path.join(train_root, data_cfg.get("train_images_root", "."))
        train_ann_file = os.path.join(train_root, data_cfg["train_annotation_file"])
        
        # Validation dataset paths
        val_root = data_cfg["val_root_dir"]
        val_images_root = os.path.join(val_root, data_cfg.get("val_images_root", "."))
        val_ann_file = os.path.join(val_root, data_cfg["val_annotation_file"])
        
        print(f"[DEBUG] Train images root: {train_images_root}")
        print(f"[DEBUG] Train annotation file: {train_ann_file}")
        print(f"[DEBUG] Val images root: {val_images_root}")
        print(f"[DEBUG] Val annotation file: {val_ann_file}")
        
        # Initialize training dataset with augmentations
        train_dataset = TurbineCocoDataset(
            images_dir=train_images_root,
            ann_file=train_ann_file,
            transforms=get_train_transform(),
        )
        
        # Initialize validation dataset with simple transforms
        val_dataset = TurbineCocoDataset(
            images_dir=val_images_root,
            ann_file=val_ann_file,
            transforms=get_val_transform(),
        )
    
    else:
        # Legacy single-root configuration
        print("[INFO] Using single-root configuration (legacy)")
        root_dir = data_cfg["root_dir"]
        images_root = os.path.join(root_dir, data_cfg["images_root"])
        ann_file = os.path.join(root_dir, data_cfg["annotation_file"])
        
        print(f"[DEBUG] Images root: {images_root}")
        print(f"[DEBUG] COCO annotation file: {ann_file}")
        
        # Initialize training dataset with augmentations
        train_dataset = TurbineCocoDataset(
            images_dir=images_root,
            ann_file=ann_file,
            transforms=get_train_transform(),
        )
        
        # Initialize validation dataset with simple transforms
        val_dataset = TurbineCocoDataset(
            images_dir=images_root,
            ann_file=ann_file,
            transforms=get_val_transform(),
        )
    
    return train_dataset, val_dataset


def main():
    print(">>> main() starting up...")

    # Load configuration
    cfg = load_config()

    # Device setup (prioritizes CUDA if available)
    device_str = cfg["training"]["device"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # --- TensorBoard Setup ---
    # Create a unique directory for TensorBoard logs using a timestamp
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", f"faster_rcnn_exp_{current_time}")
    print(f"[INFO] TensorBoard logs will be saved to: {log_dir}")

    # Datasets & DataLoaders
    train_dataset, val_dataset = build_datasets(cfg)
    print(f"[INFO] Train dataset size: {len(train_dataset)}")
    print(f"[INFO] Val dataset size:   {len(val_dataset)}")

    # Initialize the DataLoader builder, enabling the balanced sampler
    dataloader_builder = TurbineDataLoader(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        use_balanced_sampler=True,
    )

    train_loader, val_loader = dataloader_builder.get_loaders()
    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Val batches:   {len(val_loader)}")

    # Model Initialization
    num_object_classes = cfg["model"]["num_object_classes"]
    num_classes = num_object_classes + 1  # add background class
    print(
        f"[INFO] Foreground classes: {num_object_classes}, "
        f"total num_classes (with background): {num_classes}"
    )

    # Get the Faster R-CNN model
    model = get_model(num_classes=num_classes)

    # Optimizer setup - using Adam
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params,
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    print(f"[INFO] Using Adam optimizer with lr={cfg['training']['learning_rate']}")

    # Initialize Trainer with the log directory
    trainer = Trainer(model=model, optimizer=optimizer, device=device, log_dir=log_dir)

    # Training Loop (with validation inside)
    num_epochs = cfg["training"]["num_epochs"]

    for epoch in range(num_epochs):
        epoch_idx = epoch + 1
        print(f"\n[INFO] Starting epoch {epoch_idx}/{num_epochs}")

        # 1. Train and calculate loss, logging to TensorBoard
        trainer.train_one_epoch(
            train_loader=train_loader,
            val_loader=val_loader,
            epoch=epoch_idx,  # Pass 1-based index for cleaner TensorBoard logging
        )
        
        # 2. Calculate and log validation metrics (mAP, etc.)
        trainer.validate_metrics(
            val_loader=val_loader,
            epoch=epoch_idx,
        )

    # Save final model state dictionary
    output_path = cfg["training"]["output_model_path"]
    torch.save(model.state_dict(), output_path)
    print(f"[INFO] Model saved to: {output_path}")
    
    # Close the TensorBoard writer to ensure all data is flushed to disk
    trainer.close()
    print("[INFO] TensorBoard writer closed.")
    print("[INFO] Training complete!")


if __name__ == "__main__":
    print("[DEBUG] __name__ == '__main__', calling main()")
    main()