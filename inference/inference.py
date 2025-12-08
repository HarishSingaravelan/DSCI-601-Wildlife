import os
import sys
import torch
import numpy as np
from PIL import Image

# -------------------------------------------------------------------
# 1. SETUP PATHS
#    This ensures we can import from the root directory
# -------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# -------------------------------------------------------------------
# 2. PROJECT IMPORTS
# -------------------------------------------------------------------
from modeling.model import get_model
from modeling.trainer import Trainer
from turbine_processing.transforms import get_val_transform

# -------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------------------------

def load_config():
    """
    Mock configuration values necessary for model and device setup.
    Ideally, you would load this from your config.yaml file.
    """
    return {
        "model": {"num_object_classes": 5}, 
        "training": {
            "device": "cuda",
            # Ensure this filename matches exactly what main.py saved
            "output_model_path": "fasterrcnn_turbine_adam_transforms.pth" 
        }
    }

def preprocess_image(image_path: str, device: torch.device):
    """
    Loads an image, applies validation transforms, and prepares it for inference.
    """
    print(f"[INFO] Loading image from: {image_path}")
    
    # 1. Load Image
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"[ERROR] Image not found at {image_path}. Creating a dummy random image.")
        dummy_tensor = torch.rand(3, 800, 800)
        return dummy_tensor.to(device)

    # 2. Create Dummy Target
    dummy_target = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }

    # 3. Apply Transforms
    val_transform = get_val_transform()
    tensor_img, _ = val_transform(img, dummy_target)

    # 4. Final Formatting
    # Ensure float32 and move to GPU/CPU
    return tensor_img.float().to(device)


def run_inference():
    print(">>> Inference script starting up...")

    # --- A. Configuration ---
    cfg = load_config()
    device_str = cfg["training"]["device"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --- B. Model Initialization ---
    num_object_classes = cfg["model"]["num_object_classes"]
    num_classes = num_object_classes + 1
    model = get_model(num_classes=num_classes)

    # --- C. Load Weights ---
    weights_path = cfg["training"]["output_model_path"]
    
    # Check if the weight file exists in the root directory
    full_weights_path = os.path.join(parent_dir, weights_path)
    
    # Fallback: check if it exists relative to current script
    if not os.path.exists(full_weights_path):
        full_weights_path = weights_path

    if not os.path.exists(full_weights_path):
        print(f"[ERROR] Model weights not found at: {full_weights_path}")
        print("Please ensure you have run main.py and the .pth file exists.")
        return

    print(f"[INFO] Loading model weights from: {full_weights_path}")
    model.load_state_dict(torch.load(full_weights_path, map_location=device))

    # --- D. Trainer/Predictor Setup ---
    trainer = Trainer(model=model, optimizer=None, device=device, log_dir="runs/inference_only")

    # --- E. Prepare Input ---
    inference_image_path = os.path.join(parent_dir, "Sample_images", "DJI_0674.JPG")
    
    input_tensor = preprocess_image(inference_image_path, device)
    
    
    input_batch = [input_tensor]

    # --- F. Run Inference ---
    print("[INFO] Running model inference...")
    predictions = trainer.inference(input_batch)

    # --- G. Process Results ---
    if predictions and len(predictions) > 0:
        print("\n--- Model Predictions (First Image) ---")
        pred = predictions[0]
        
        # Move to CPU and convert to numpy for easy printing
        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        
        # Filter by confidence
        confidence_threshold = 0.5
        keep = scores > confidence_threshold

        if np.any(keep):
            print(f"Found {np.sum(keep)} detections above {confidence_threshold}:")
            
            # Map IDs to names based on your dataset
            class_map = {
                1: "red_winged_blackbird",
                2: "mammal_carcass", 
                3: "hawk_species", 
                4: "gull_species", 
                5: "trash"
            } 
            
            # Zip and iterate through the kept predictions
            for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
                label_name = class_map.get(label, f"Class {label}")
                # Convert box to integer coordinates
                box_int = box.astype(int)
                print(f"  {label_name:<20} | Score: {score:.4f} | Box: {box_int}")
        else:
            print(f"No objects detected with confidence > {confidence_threshold}.")
    else:
        print("Model returned empty prediction results.")

if __name__ == "__main__":
    run_inference()