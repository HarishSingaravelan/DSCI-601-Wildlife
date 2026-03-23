import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm  # Added tqdm

# --- Configuration ---
ROOT_DIR = "../../../../../../shared/rc/turbine/Test_2021_SY"
OUTPUT_JSON = "../../../../../../shared/rc/turbine/coco_annotations_all_classes.json"

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tiff')
ANNOTATION_EXTENSIONS = ('.json', '.xml')

COCO_INFO = {
    "year": 2026,
    "version": "2.0",
    "description": "Dynamic UAS Wildlife Detection Dataset",
    "contributor": "",
    "date_created": ""
}

# --- Global State for Dynamic Categories ---
category_map = {}
category_id_counter = 1

def get_or_create_category_id(raw_name: str) -> int:
    global category_id_counter
    if not raw_name:
        raw_name = "unknown"
    
    name = raw_name.lower().strip()
    if name not in category_map:
        category_map[name] = category_id_counter
        category_id_counter += 1
    return category_map[name]

def get_image_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return None

def parse_custom_json(json_path, image_id, image_width, image_height, ann_id):
    annotations = []
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except:
        return annotations, ann_id

    for carcass in data.get('carcasses', []):
        raw_cname = carcass.get('class')
        cid = get_or_create_category_id(raw_cname)
        
        loc = carcass.get('location', [])
        if len(loc) < 2: continue

        try:
            nx1, ny1 = float(loc[0]['x']), float(loc[0]['y'])
            nx2, ny2 = float(loc[1]['x']), float(loc[1]['y'])
            
            xmin = int(min(nx1, nx2) * image_width)
            ymin = int(min(ny1, ny2) * image_height)
            xmax = int(max(nx1, nx2) * image_width)
            ymax = int(max(ny1, ny2) * image_height)
            
            w, h = max(0, xmax - xmin), max(0, ymax - ymin)
            if w == 0 or h == 0: continue

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cid,
                "iscrowd": 0,
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "segmentation": [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]],
            })
            ann_id += 1
        except (ValueError, KeyError, IndexError):
            continue
    return annotations, ann_id

def parse_pascal_voc_xml(xml_path, image_id, ann_id):
    annotations = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except:
        return annotations, ann_id

    for obj in root.findall('object'):
        raw_cname = obj.find('name').text
        cid = get_or_create_category_id(raw_cname)
        
        b = obj.find('bndbox')
        try:
            xmin = int(float(b.find('xmin').text))
            ymin = int(float(b.find('ymin').text))
            xmax = int(float(b.find('xmax').text))
            ymax = int(float(b.find('ymax').text))
            
            w, h = max(0, xmax - xmin), max(0, ymax - ymin)
            if w <= 0 or h <= 0: continue

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cid,
                "iscrowd": 0,
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "segmentation": [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]],
            })
            ann_id += 1
        except (ValueError, AttributeError):
            continue
    return annotations, ann_id

def generate_coco_json(root_dir: str, output_path: str):
    coco = {
        "info": COCO_INFO,
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": [],
        "annotations": [],
        "categories": []
    }

    print(f"Scanning: {root_dir}")
    image_paths = []
    # Using tqdm for the initial walk if the directory is massive
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(r, f))

    img_id_counter = 1
    ann_id_counter = 1
    root_basename = os.path.basename(os.path.normpath(root_dir))

    # Main processing loop with tqdm
    for img_path in tqdm(image_paths, desc="Processing Images", unit="img"):
        dims = get_image_dimensions(img_path)
        if not dims: continue
        
        w, h = dims
        rel_path = os.path.relpath(img_path, root_dir)
        file_name = os.path.join(root_basename, rel_path).replace(os.path.sep, "/")

        coco["images"].append({
            "id": img_id_counter,
            "width": w,
            "height": h,
            "file_name": file_name,
            "license": 1
        })

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        folder = os.path.dirname(img_path)

        for ext in ANNOTATION_EXTENSIONS:
            ann_path = os.path.join(folder, base_name + ext)
            if os.path.exists(ann_path):
                if ext == ".json":
                    new_anns, ann_id_counter = parse_custom_json(ann_path, img_id_counter, w, h, ann_id_counter)
                elif ext == ".xml":
                    new_anns, ann_id_counter = parse_pascal_voc_xml(ann_path, img_id_counter, ann_id_counter)
                coco["annotations"].extend(new_anns)

        img_id_counter += 1

    for name, cid in sorted(category_map.items(), key=lambda item: item[1]):
        coco["categories"].append({
            "id": cid,
            "name": name,
            "supercategory": "wildlife"
        })

    print("\n--- Summary ---")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Categories discovered ({len(coco['categories'])}): {list(category_map.keys())}")

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    generate_coco_json(ROOT_DIR, OUTPUT_JSON)