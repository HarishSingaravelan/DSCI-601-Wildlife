import torch
import yaml
import time
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

# ==========================================
# 1. SETTINGS (CHANGE THESE)
# ==========================================
IMAGE_PATH = "Sample_images/DJI_0677.JPG"                               
MODEL_WEIGHTS = "checkpoints/full_run_log_sampler_confidence_0.2_bg/checkpoint_epoch_300.pth" 
CONFIG_PATH = "config/config.yaml"                                      
CONFIDENCE_THRESHOLD = 0.50                                            
# ==========================================

def load_config_and_classes(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    class_names = config['model']['class_names']
    num_classes = config['model']['num_object_classes']
    return class_names, num_classes

def load_ground_truth(json_path, img_width, img_height):
    """Reads the JSON file and converts normalized coordinates to absolute pixels"""
    gt_data = []
    if not os.path.exists(json_path):
        print(f"      --> No Ground Truth JSON found at {json_path}")
        return gt_data
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    for carcass in data.get("carcasses", []):
        locs = carcass.get("location", [])
        if len(locs) >= 2:
            # Extract normalized coordinates
            xs = [float(loc["x"]) for loc in locs]
            ys = [float(loc["y"]) for loc in locs]
            
            # Convert to absolute pixels
            xmin, xmax = min(xs) * img_width, max(xs) * img_width
            ymin, ymax = min(ys) * img_height, max(ys) * img_height
            
            cls_name = carcass.get("class", "unknown")
            gt_data.append({"class": cls_name, "box": [xmin, ymin, xmax, ymax]})
            
    return gt_data

def run_inference():
    total_start = time.time()
    
    print("[1/6] Loading configuration...")
    class_names, num_classes = load_config_and_classes(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      --> Using device: {device.type.upper()}")

    print("[2/6] Loading base model and processor...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    print("[3/6] Injecting custom trained weights...")
    checkpoint = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"[4/6] Opening and preprocessing image: {IMAGE_PATH}")
    step_start = time.time()
    image = Image.open(IMAGE_PATH).convert("RGB")
    img_width, img_height = image.size
    inputs = processor(images=image, return_tensors="pt").to(device)
    print(f"      --> Preprocessing finished in {time.time() - step_start:.2f} seconds.")

    print(f"[5/6] Running model forward pass...")
    step_start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"      --> Forward pass finished in {time.time() - step_start:.2f} seconds.")

    print("[6/6] Post-processing and drawing boxes...")
    target_sizes = torch.tensor([[img_height, img_width]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD
    )[0]

    # Filter out background (Class 0)
    keep = results["labels"] > 0
    boxes = results["boxes"][keep]
    scores = results["scores"][keep]
    labels = results["labels"][keep]

    # Set up matplotlib figure
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    ax.axis("off")

    # --- DRAW GROUND TRUTH (BLUE) ---
    json_path = os.path.splitext(IMAGE_PATH)[0] + ".json"
    gt_boxes = load_ground_truth(json_path, img_width, img_height)
    
    for gt in gt_boxes:
        xmin, ymin, xmax, ymax = gt["box"]
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                 linewidth=3, edgecolor='#00BFFF', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 8, f"GT: {gt['class']}", color='white', fontsize=12, weight='bold',
                bbox=dict(facecolor='#00BFFF', alpha=0.9, edgecolor='none', pad=3))

    # --- DRAW PREDICTIONS (GREEN) ---
    if len(boxes) == 0:
        print("      --> Result: No animals predicted by model.")
    else:
        print(f"      --> Result: Model predicted {len(boxes)} animal(s).")
        for score, label, box in zip(scores, labels, boxes):
            box_coords = [round(i, 2) for i in box.tolist()]
            xmin, ymin, xmax, ymax = box_coords
            class_name = class_names[label.item()]
            
            print(f"          - PRED: {class_name:<20} | Conf: {score.item():.2f} | BBox: [Xmin: {xmin}, Ymin: {ymin}, Xmax: {xmax}, Ymax: {ymax}]")
            
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=3, edgecolor='#00FF00', facecolor='none')
            ax.add_patch(rect)
            
            label_text = f"PRED: {class_name} ({score.item():.2f})"
            ax.text(xmin, ymax + 20, label_text, color='black', fontsize=12, weight='bold',
                    bbox=dict(facecolor='#00FF00', alpha=0.9, edgecolor='none', pad=3))

    output_filename = "inference_result.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"\n DONE! Total time: {time.time() - total_start:.2f} seconds.")
    print(f"Saved visualization to: {output_filename}")

if __name__ == "__main__":
    run_inference()