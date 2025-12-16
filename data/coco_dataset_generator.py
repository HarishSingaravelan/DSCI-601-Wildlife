"""
.. module:: coco_dataset_generator
   :synopsis: Generates COCO-style JSON annotations from custom JSON and Pascal VOC XML formats.
.. moduleauthor:: AI Assistant

This script iterates through a directory tree, finds image files, and parses corresponding 
custom JSON or Pascal VOC XML annotation files to build a single COCO-compliant JSON file. 
It handles robust category mapping and provides a detailed summary of processed and skipped files.
"""
import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from typing import Optional, List, Tuple, Dict

# --- Configuration (Global Constants) ---
ROOT_DIR = "Sample_images"
OUTPUT_JSON = "coco_annotations.json"

ANNOTATION_EXTENSIONS = ['.json', '.xml']
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tiff')

CATEGORY_MAPPING = {
    "red_winged_blackbird": 1,
    "eastern_red_bat": 2,
    "yellow_bellied_sapsucker": 3,
    "ring_necked_pheasant": 4,
    "trash": 5,
}

COCO_INFO = {
    "year": 2024,
    "version": "1.0",
    "description": "UAS Wildlife Detection Dataset",
    "contributor": "AI Assistant",
    "date_created": "2025-09-30"
}

COCO_LICENSES = [
    {"id": 1, "name": "Unknown / Custom License", "url": ""}
]

COCO_CATEGORIES = [
    {"id": cid, "name": cname, "supercategory": "wildlife"}
    for cname, cid in CATEGORY_MAPPING.items()
]

TRASH_CATEGORY_ID = CATEGORY_MAPPING.get("trash", 5) 

# --- Helper Maps ---
ROBUST_CATEGORY_MAP = {name.lower().strip(): cid for name, cid in CATEGORY_MAPPING.items()}
ROBUST_CATEGORY_NAMES = {name.lower().strip(): name for name in CATEGORY_MAPPING.keys()}

# Global counters for reporting
global_reassigned_count = 0
global_skipped_count = 0


# --- Helper Functions ---

def get_image_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    """
    Reads image dimensions safely using the PIL library.

    :param image_path: Path to the image file.
    :type image_path: str
    :returns: A tuple containing (width, height) in pixels, or None if the file is invalid.
    :rtype: Optional[Tuple[int, int]]
    """
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return None

def get_category_id_robust(raw_cname: str) -> int:
    """
    Looks up category ID robustly using a case-insensitive, stripped key.
    If the category name is not found, the annotation is assigned to the 
    'trash' category (TRASH_CATEGORY_ID) and the global reassignment counter is updated.

    :param raw_cname: The raw category name extracted from the annotation file.
    :type raw_cname: str
    :returns: The integer category ID (e.g., 1, 2, 5).
    :rtype: int
    """
    global global_reassigned_count
    
    if not raw_cname:
        return TRASH_CATEGORY_ID 

    normalized_cname = raw_cname.lower().strip()
    
    if normalized_cname in ROBUST_CATEGORY_MAP:
        return ROBUST_CATEGORY_MAP[normalized_cname]
    else:
        # Fallback logic: Assign to trash
        global_reassigned_count += 1
        return TRASH_CATEGORY_ID


def parse_custom_json(
    json_path: str, 
    image_id: int, 
    image_width: int, 
    image_height: int, 
    ann_id: int
) -> Tuple[List[Dict], int]:
    """
    Parses custom JSON annotation format (assuming normalized coordinates [0, 1]).

    :param json_path: Path to the custom JSON annotation file.
    :type json_path: str
    :param image_id: The COCO ID of the image currently being processed.
    :type image_id: int
    :param image_width: Image width in pixels, used for coordinate conversion.
    :type image_width: int
    :param image_height: Image height in pixels, used for coordinate conversion.
    :type image_height: int
    :param ann_id: The starting annotation ID counter.
    :type ann_id: int
    :returns: A tuple containing the list of new COCO annotations and the updated annotation ID.
    :rtype: Tuple[List[Dict], int]
    """
    annotations = []
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return annotations, ann_id

    for carcass in data.get('carcasses', []):
        raw_cname = carcass.get('class')
        cid = get_category_id_robust(raw_cname)
        
        # Check for required location data
        if not carcass.get('location') or len(carcass['location']) < 2:
            continue

        try:
            # Extract normalized coordinates (nx1, ny1, nx2, ny2)
            nx1, ny1 = float(carcass['location'][0]['x']), float(carcass['location'][0]['y'])
            nx2, ny2 = float(carcass['location'][1]['x']), float(carcass['location'][1]['y'])
        except (ValueError, KeyError, IndexError):
            continue

        # Convert normalized coordinates to absolute pixel values
        xmin = int(min(nx1, nx2) * image_width)
        ymin = int(min(ny1, ny2) * image_height)
        xmax = int(max(nx1, nx2) * image_width)
        ymax = int(max(ny1, ny2) * image_height)
        
        w = max(0, xmax - xmin)
        h = max(0, ymax - ymin)

        if w <= 0 or h <= 0:
            continue

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

    return annotations, ann_id

def parse_pascal_voc_xml(
    xml_path: str, 
    image_id: int, 
    ann_id: int
) -> Tuple[List[Dict], int]:
    """
    Parses Pascal VOC XML annotation format (assuming absolute pixel coordinates).

    :param xml_path: Path to the Pascal VOC XML file.
    :type xml_path: str
    :param image_id: The COCO ID of the image currently being processed.
    :type image_id: int
    :param ann_id: The starting annotation ID counter.
    :type ann_id: int
    :returns: A tuple containing the list of new COCO annotations and the updated annotation ID.
    :rtype: Tuple[List[Dict], int]
    """
    annotations = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return annotations, ann_id

    for obj in root.findall('object'):
        raw_cname = obj.find('name').text
        cid = get_category_id_robust(raw_cname)
        
        b = obj.find('bndbox')
        
        try:
            # Pascal VOC uses absolute pixel coordinates (x,y max/min)
            xmin = int(float(b.find('xmin').text))
            ymin = int(float(b.find('ymin').text))
            xmax = int(float(b.find('xmax').text))
            ymax = int(float(b.find('ymax').text))
        except (ValueError, AttributeError):
            continue

        w = max(0, xmax - xmin)
        h = max(0, ymax - ymin)

        if w <= 0 or h <= 0:
            continue

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

    return annotations, ann_id


def generate_coco_json(root_dir: str, output_path: str):
    """
    Main function to scan the directory tree, parse annotations, and save the COCO JSON file.

    :param root_dir: The root directory containing the images and their annotations.
    :type root_dir: str
    :param output_path: The path where the final COCO JSON file will be saved.
    :type output_path: str
    """
    global global_skipped_count
    global global_reassigned_count
    
    coco = {
        "info": COCO_INFO,
        "licenses": COCO_LICENSES,
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": []
    }

    print(f"Scanning directory tree: {root_dir}")
    all_files = []

    # FIRST PASS: Collect all image file paths (fastest way, already recursive)
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                all_files.append(os.path.join(r, f))

    total_found_files = len(all_files)
    skipped_images = []
    
    print(f"Found {total_found_files} potential image files. Starting processing...")

    image_id = 1
    ann_id = 1
    root_name = os.path.basename(os.path.normpath(root_dir))

    # MAIN IMAGE LOOP
    for img_path in all_files:
        rel_path = os.path.relpath(img_path, root_dir)
        file_name = os.path.join(root_name, rel_path).replace(os.path.sep, "/")

        dims = get_image_dimensions(img_path)
        
        if not dims:
            skipped_images.append(rel_path)
            global_skipped_count += 1
            continue # Skip image if dimensions cannot be read

        w, h = dims

        coco["images"].append({
            "id": image_id,
            "width": w,
            "height": h,
            "file_name": file_name,
            "license": 1
        })

        base, _ = os.path.splitext(os.path.basename(img_path))
        folder = os.path.dirname(img_path)

        # Process available annotation files for the current image
        for ext in ANNOTATION_EXTENSIONS:
            ann_path = os.path.join(folder, base + ext)
            if os.path.exists(ann_path):
                new_anns = []
                if ext == ".json":
                    new_anns, ann_id = parse_custom_json(ann_path, image_id, w, h, ann_id) 
                elif ext == ".xml":
                    new_anns, ann_id = parse_pascal_voc_xml(ann_path, image_id, ann_id) 

                coco["annotations"].extend(new_anns)
                
        image_id += 1

    # --- Final Summary ---
    print("\n--- Summary ---")
    print(f"Total images successfully processed: {len(coco['images'])}")
    print(f"Total annotations collected: {len(coco['annotations'])}")
    print(f"Total annotations re-assigned to 'trash' (ID {TRASH_CATEGORY_ID}): {global_reassigned_count}")
    
    if skipped_images:
        print(f"\nCRITICAL WARNING: Skipped {len(skipped_images)} images due to reading/dimension errors or corruption.")
        print(f"Examples: {skipped_images[:5]}")
    
    # Save the final COCO JSON file
    try:
        with open(output_path, "w") as f:
            json.dump(coco, f, indent=4)
        print(f"COCO JSON successfully saved to: {output_path}")
    except Exception as e:
        print(f"FATAL ERROR: Could not write COCO JSON to {output_path}. Error: {e}")


if __name__ == "__main__":
    generate_coco_json(ROOT_DIR, OUTPUT_JSON)