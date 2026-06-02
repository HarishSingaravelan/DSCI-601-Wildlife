import os
import json
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION - BULLETPROOF ABSOLUTE PATHS
# ==============================================================================
BASE_DIR = "/shared/rc/turbine" 

INPUT_IMG_DIR = BASE_DIR
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, "resized_dataset_statified")

SPLIT_FOLDER_NAME = "turbine_split_stratified"

MAX_EDGE = 1333  
NUM_WORKERS = 30 
# ==============================================================================

def process_single_image(img_info):
    """Worker function to read, resize, and save a single image."""
    img_id = img_info['id']
    file_name = img_info['file_name']
    
    in_path = os.path.join(INPUT_IMG_DIR, file_name)
    out_path = os.path.join(OUTPUT_IMG_DIR, file_name)
    
    img = cv2.imread(in_path)
    if img is None:
        # Return a special flag so we know EXACTLY which path failed
        return {'error': in_path}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    old_h, old_w = img.shape[:2]
    scale = MAX_EDGE / max(old_h, old_w)
    
    if scale >= 1.0:
        scale = 1.0
        new_w, new_h = old_w, old_h
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        new_w = int(old_w * scale)
        new_h = int(old_h * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {
        'id': img_id,
        'file_name': file_name,
        'width': new_w,
        'height': new_h,
        'scale': scale
    }

def process_split(split_name):
    """Processes an entire dataset split (train, val, or test)."""
    
    # DYNAMIC PATHS: Now uses your SPLIT_FOLDER_NAME variable!
    input_json = os.path.join(BASE_DIR, f"{SPLIT_FOLDER_NAME}/{split_name}/annotations.json")
    output_json = os.path.join(OUTPUT_IMG_DIR, f"{SPLIT_FOLDER_NAME}/{split_name}/annotations.json")

    if not os.path.exists(input_json):
        print(f"⏭️  Skipping {split_name} (JSON not found: {input_json})")
        return

    print(f"\n========================================")
    print(f"Processing Split: {split_name.upper()}")
    print(f"========================================")
    
    with open(input_json, 'r') as f:
        coco_data = json.load(f)
        
    print(f"Resizing {len(coco_data['images'])} images...")
    
    scale_dict = {} 
    new_images = []
    missing_count = 0
    first_error_path = None
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_single_image, coco_data['images']), total=len(coco_data['images'])))
        
    for res in results:
        if 'error' in res:
            missing_count += 1
            if first_error_path is None:
                first_error_path = res['error']
        else:
            scale_dict[res['id']] = res['scale']
            new_images.append({
                'id': res['id'],
                'file_name': res['file_name'],
                'width': res['width'],
                'height': res['height']
            })
            
    if missing_count > 0:
        print(f"ERROR: Could not find {missing_count} images!")
        print(f"Example missing path: {first_error_path}")
        print("Stopping script. Please check your BASE_DIR path!")
        return
            
    print("Updating bounding box coordinates...")
    new_annotations = []
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in scale_dict:
            continue 
            
        scale = scale_dict[img_id]
        x, y, w, h = ann['bbox']
        
        new_ann = ann.copy()
        new_ann['bbox'] = [x * scale, y * scale, w * scale, h * scale]
        
        if 'area' in new_ann:
            new_ann['area'] = new_ann['area'] * (scale ** 2)
            
        new_annotations.append(new_ann)
        
    coco_data['images'] = new_images
    coco_data['annotations'] = new_annotations
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco_data, f)
        
    print(f"✅ {split_name.upper()} successfully resized and saved!")

def main():
    if not os.path.exists(BASE_DIR):
        print(f"CRITICAL ERROR: The directory '{BASE_DIR}' does not exist!")
        print("Please check the absolute path to your shared folder.")
        return

    splits = ["train", "val", "test"]
    for split in splits:
        process_split(split)
        
    print("\n🎉 ALL DATASETS COMPRESSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()