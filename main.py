import os
import sys
import datetime # for log Directory

import torch
import yaml

# --- Ensures project root is on sys.path ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# ------------------------------------------


# --- Imports from own code ---
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

    Args:
        cfg: The configuration dictionary loaded from config.yaml.

    Returns:
        A tuple containing (train_dataset, val_dataset).
    """
    data_cfg = cfg["data"]
    root_dir = data_cfg["root_dir"]

    # Construct full paths to images and annotation file
    images_root = os.path.join(root_dir, data_cfg["images_root"])
    ann_file = os.path.join(root_dir, data_cfg["annotation_file"])

    print(f"[DEBUG] Train/Val images root: {images_root}")
    print(f"[DEBUG] COCO annotation file: {ann_file}")

    # Initialize training dataset with augmentations
    train_dataset = TurbineCocoDataset(
        images_dir=images_root,
        ann_file=ann_file,
        transforms=get_train_transform(),
    )

    # Initialize validation dataset 
    val_dataset = TurbineCocoDataset(
        images_dir=images_root,
        ann_file=ann_file,
        transforms=get_val_transform(),
    )

    return train_dataset, val_dataset


def main():
    print(">>> main() starting up..")

    cfg = load_config()

    # Device setup 
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

    # Optimizer setup (SGD is standard for object detection fine-tuning)
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(
    #     params,
    #     lr=cfg["training"]["learning_rate"],
    #     momentum=cfg["training"]["momentum"],
    #     weight_decay=cfg["training"]["weight_decay"],
    # )

    # initializing adam
    optimizer = torch.optim.Adam(
        params,
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Initialize Trainer with the log directory
    trainer = Trainer(model=model, optimizer=optimizer, device=device, log_dir=log_dir)

    # Training Loop 
    num_epochs = cfg["training"]["num_epochs"]

    for epoch in range(num_epochs):
        epoch_idx = epoch + 1
        print(f"\n[INFO] Starting epoch {epoch_idx}/{num_epochs}")

        # 1. Train and calculate loss, logging to TensorBoard
        trainer.train_one_epoch(
            train_loader=train_loader,
            val_loader=val_loader,
            epoch=epoch_idx, 
        )
        
        # 2. Calculate and log validation metrics 
        trainer.validate_metrics(
             val_loader=val_loader,
             epoch=epoch_idx,
        )

    # Save final model state dictionary
    output_path = cfg["training"]["output_model_path"]
    torch.save(model.state_dict(), output_path)
    print(f"[INFO] Model saved to: {output_path}")
    
    trainer.close()
    print("[INFO] TensorBoard writer closed.")


if __name__ == "__main__":
    print("[DEBUG] __name__ == '__main__', calling main()")
    main()