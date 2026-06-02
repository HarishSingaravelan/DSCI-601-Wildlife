"""
Stratified Dataset Split Script for Object Detection
Feature: "Tail-Class Aggregation" - Groups rare classes into a common bucket 
so bounding box data is preserved and splitting is mathematically possible.
"""

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import random
from typing import Dict, List, Tuple

def load_coco_annotations(ann_file: Path) -> Dict:
    with open(ann_file, 'r') as f:
        return json.load(f)

def aggregate_rare_classes(coco_data: Dict, min_images: int = 3, target_bucket_name: str = "unknown") -> Dict:
    """Finds classes with < min_images and maps them to a target bucket."""
    print(f"\n[1/6] Aggregating rare classes (< {min_images} images) into '{target_bucket_name}' bucket...")
    
    # 1. Count images per category
    img_counts_per_class = defaultdict(set)
    for ann in coco_data['annotations']:
        img_counts_per_class[ann['category_id']].add(ann['image_id'])
        
    cat_id_to_name = {c['id']: c['name'] for c in coco_data['categories']}
    
    # 2. Identify which classes need to be bucketed
    rare_cat_ids = set()
    rare_details = []
    for cat_id, img_set in img_counts_per_class.items():
        if len(img_set) < min_images and cat_id_to_name.get(cat_id) != target_bucket_name:
            rare_cat_ids.add(cat_id)
            rare_details.append((cat_id_to_name.get(cat_id, "Unnamed"), len(img_set)))
            
    if not rare_cat_ids:
        print("  ✓ All classes meet the minimum image threshold. No bucketing needed.")
        return coco_data

    # 3. Find or Create the Target Bucket ID
    target_cat_id = None
    for c in coco_data['categories']:
        if c['name'] == target_bucket_name:
            target_cat_id = c['id']
            break
            
    if target_cat_id is None:
        # Create the bucket if it doesn't exist in the JSON yet
        target_cat_id = max([c['id'] for c in coco_data['categories']] + [0]) + 1
        coco_data['categories'].append({"id": target_cat_id, "name": target_bucket_name, "supercategory": "none"})
        print(f"  Created new category '{target_bucket_name}' with ID {target_cat_id}")

    # 4. Map the annotations
    print(f"  ⚠ Re-mapping {len(rare_cat_ids)} rare classes into '{target_bucket_name}':")
    for name, count in rare_details:
        print(f"    - {name} ({count} images) -> {target_bucket_name}")
        
    for ann in coco_data['annotations']:
        if ann['category_id'] in rare_cat_ids:
            ann['category_id'] = target_cat_id

    # 5. Clean up the categories list (remove the rare ones we just emptied)
    new_categories = [c for c in coco_data['categories'] if c['id'] not in rare_cat_ids]
    
    print(f"  Surviving Unique Classes: {len(new_categories)} / {len(coco_data['categories'])}")
    
    return {
        'images': coco_data['images'],
        'annotations': coco_data['annotations'],
        'categories': new_categories,
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', [])
    }


def stratified_split(coco_data: Dict, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """Splits dataset using Rarest-First bucket stratification."""
    random.seed(seed)
    
    img_counts_per_class = defaultdict(set)
    for ann in coco_data['annotations']:
        img_counts_per_class[ann['category_id']].add(ann['image_id'])
        
    global_freq = {cat_id: len(imgs) for cat_id, imgs in img_counts_per_class.items()}
    
    buckets = defaultdict(list)
    
    for img in coco_data['images']:
        img_id = img['id']
        img_anns = [a for a in coco_data['annotations'] if a['image_id'] == img_id]
        
        if not img_anns:
            buckets['background'].append(img_id)
        else:
            rarest_cat = min(img_anns, key=lambda a: global_freq[a['category_id']])['category_id']
            buckets[rarest_cat].append(img_id)
            
    train_ids, val_ids, test_ids = [], [], []
    sorted_cats = sorted(global_freq.keys(), key=lambda k: global_freq[k])
    
    for cat_id in sorted_cats:
        bucket_imgs = buckets[cat_id]
        random.shuffle(bucket_imgs)
        
        n = len(bucket_imgs)
        if n == 0:
            continue
        elif n == 1:
            train_ids.append(bucket_imgs[0])
        elif n == 2:
            train_ids.append(bucket_imgs[0])
            val_ids.append(bucket_imgs[1])
        else:
            train_ids.append(bucket_imgs[0])
            val_ids.append(bucket_imgs[1])
            test_ids.append(bucket_imgs[2])
            
            rest = bucket_imgs[3:]
            if rest:
                n_rest = len(rest)
                n_train = int(n_rest * train_ratio)
                n_val = int(n_rest * val_ratio)
                
                train_ids.extend(rest[:n_train])
                val_ids.extend(rest[n_train:n_train + n_val])
                test_ids.extend(rest[n_train + n_val:])
                
    bg_imgs = buckets['background']
    random.shuffle(bg_imgs)
    n_bg = len(bg_imgs)
    n_train_bg = int(n_bg * train_ratio)
    n_val_bg = int(n_bg * val_ratio)
    
    train_ids.extend(bg_imgs[:n_train_bg])
    val_ids.extend(bg_imgs[n_train_bg:n_train_bg + n_val_bg])
    test_ids.extend(bg_imgs[n_train_bg + n_val_bg:])
    
    return train_ids, val_ids, test_ids


def create_split_annotations(coco_data: Dict, split_image_ids: List[int]) -> Dict:
    split_image_ids_set = set(split_image_ids)
    return {
        'images': [img for img in coco_data['images'] if img['id'] in split_image_ids_set],
        'annotations': [ann for ann in coco_data['annotations'] if ann['image_id'] in split_image_ids_set],
        'categories': coco_data['categories'],
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', [])
    }


def copy_images(source_root: Path, dest_root: Path, image_filenames: List[str]):
    dest_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img_file in image_filenames:
        src_path = source_root / img_file
        dest_path = dest_root / img_file
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            copied += 1
    print(f"  Copied: {copied} images")


def verify_distribution(train_coco, val_coco, test_coco):
    print("\n[6/6] Verifying Split Quality (Class Coverage Report)...")
    train_cats = set(ann['category_id'] for ann in train_coco['annotations'])
    val_cats = set(ann['category_id'] for ann in val_coco['annotations'])
    test_cats = set(ann['category_id'] for ann in test_coco['annotations'])
    
    all_cats = set(c['id'] for c in train_coco['categories'])
    
    missing_in_val = all_cats - val_cats
    missing_in_test = all_cats - test_cats
    
    if not missing_in_val and not missing_in_test:
        print(" SUCCESS: Every single class is present in Train, Val, and Test!")
    else:
        print(f"  ⚠ Minor issues detected (Likely extremely rare co-occurrences):")
        if missing_in_val: print(f"    - Missing in Val: {len(missing_in_val)} classes")
        if missing_in_test: print(f"    - Missing in Test: {len(missing_in_test)} classes")


def main():
    parser = argparse.ArgumentParser(description='Stratified COCO Dataset Splitter with Aggregation')
    parser.add_argument('--source_root', type=str, required=True, help='Root directory containing images')
    parser.add_argument('--annotation_file', type=str, required=True, help='COCO annotation JSON file')
    parser.add_argument('--output_root', type=str, required=True, help='Output directory for split dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--min_images', type=int, default=3, help='Classes with fewer images get bucketed')
    parser.add_argument('--bucket_name', type=str, default="unknown", help='Name of the bucket for rare classes')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--copy_images', action='store_true')
    
    args = parser.parse_args()
    source_root, annotation_file, output_root = Path(args.source_root), Path(args.annotation_file), Path(args.output_root)
    
    print("="*60)
    print("STRATIFIED DATASET SPLITTING (With Tail-Class Aggregation)")
    print("="*60)
    
    # 1. Load & Aggregate
    coco_data = load_coco_annotations(annotation_file)
    coco_data = aggregate_rare_classes(coco_data, args.min_images, args.bucket_name)
    
    # 2. Stratified Split
    print("\n[2/6] Executing Rarest-First Stratification...")
    train_ids, val_ids, test_ids = stratified_split(coco_data, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    
    print(f"  Train: {len(train_ids)} images | Val: {len(val_ids)} images | Test: {len(test_ids)} images")
    
    # 3. Create JSONs
    print("\n[3/6] Mapping annotations to splits...")
    train_coco = create_split_annotations(coco_data, train_ids)
    val_coco = create_split_annotations(coco_data, val_ids)
    test_coco = create_split_annotations(coco_data, test_ids)
    
    # 4. Save JSONs
    print("\n[4/6] Saving split annotation files...")
    output_root.mkdir(parents=True, exist_ok=True)
    for name, data in zip(['train', 'val', 'test'], [train_coco, val_coco, test_coco]):
        path = output_root / name / 'annotations.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"  Saved: {path}")
        
    # 5. Copy Images
    if args.copy_images:
        print("\n[5/6] Copying images...")
        id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
        copy_images(source_root, output_root / 'train' / 'images', [id_to_file[i] for i in train_ids])
        copy_images(source_root, output_root / 'val' / 'images', [id_to_file[i] for i in val_ids])
        copy_images(source_root, output_root / 'test' / 'images', [id_to_file[i] for i in test_ids])
    else:
        print("\n[5/6] Skipping image copy (--copy_images not flagged)")
        
    # 6. Verify
    verify_distribution(train_coco, val_coco, test_coco)
    print("\n============================================================")
    print("SPLIT COMPLETE!")
    print("============================================================")

if __name__ == '__main__':
    main()