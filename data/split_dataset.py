"""
Dataset Split Script for Object Detection
Splits images and COCO annotations into train/val/test sets
Ensures no overlap between splits
"""

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import random
from typing import Dict, List, Tuple


def load_coco_annotations(ann_file: Path) -> Dict:
    """Load COCO format annotations."""
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    return coco_data


def split_images_by_id(
    image_ids: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split image IDs into train/val/test sets.
    
    Args:
        image_ids: List of all image IDs
        train_ratio: Proportion for training (default 0.7 = 70%)
        val_ratio: Proportion for validation (default 0.15 = 15%)
        test_ratio: Proportion for test (default 0.15 = 15%)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    shuffled_ids = image_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Calculate split indices
    n_total = len(shuffled_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split
    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]
    
    return train_ids, val_ids, test_ids


def create_split_annotations(
    coco_data: Dict,
    split_image_ids: List[int]
) -> Dict:
    """
    Create COCO annotations for a specific split.
    
    Args:
        coco_data: Full COCO annotation dictionary
        split_image_ids: Image IDs for this split
    
    Returns:
        New COCO annotation dictionary for the split
    """
    split_image_ids_set = set(split_image_ids)
    
    # Filter images
    split_images = [
        img for img in coco_data['images']
        if img['id'] in split_image_ids_set
    ]
    
    # Filter annotations
    split_annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] in split_image_ids_set
    ]
    
    # Create new COCO dict
    split_coco = {
        'images': split_images,
        'annotations': split_annotations,
        'categories': coco_data['categories'],
    }
    
    # Copy other fields if they exist
    if 'info' in coco_data:
        split_coco['info'] = coco_data['info']
    if 'licenses' in coco_data:
        split_coco['licenses'] = coco_data['licenses']
    
    return split_coco


def copy_images(
    source_root: Path,
    dest_root: Path,
    image_filenames: List[str],
    verbose: bool = True
):
    """
    Copy images from source to destination.
    Handles nested folder structure.
    
    Args:
        source_root: Root directory containing source images
        dest_root: Destination directory
        image_filenames: List of image filenames (may include subfolders)
        verbose: Print progress
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    missing = 0
    
    for img_file in image_filenames:
        src_path = source_root / img_file
        
        # Create destination with same subfolder structure
        dest_path = dest_root / img_file
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            copied += 1
        else:
            if verbose:
                print(f"Warning: Image not found: {src_path}")
            missing += 1
    
    if verbose:
        print(f"  Copied: {copied} images")
        if missing > 0:
            print(f"  Missing: {missing} images")


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train/val/test with COCO annotations'
    )
    parser.add_argument('--source_root', type=str, required=True,
                        help='Root directory containing images')
    parser.add_argument('--annotation_file', type=str, required=True,
                        help='COCO annotation JSON file')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Output directory for split dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--copy_images', action='store_true',
                        help='Copy images to split directories (default: False)')
    
    args = parser.parse_args()
    
    source_root = Path(args.source_root)
    annotation_file = Path(args.annotation_file)
    output_root = Path(args.output_root)
    
    # Verify paths
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    print("="*60)
    print("DATASET SPLITTING")
    print("="*60)
    print(f"Source root: {source_root}")
    print(f"Annotation file: {annotation_file}")
    print(f"Output root: {output_root}")
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    # Load COCO annotations
    print("\n[1/5] Loading COCO annotations...")
    coco_data = load_coco_annotations(annotation_file)
    
    n_images = len(coco_data['images'])
    n_annotations = len(coco_data['annotations'])
    n_categories = len(coco_data['categories'])
    
    print(f"  Total images: {n_images}")
    print(f"  Total annotations: {n_annotations}")
    print(f"  Total categories: {n_categories}")
    
    # Get all image IDs
    image_ids = [img['id'] for img in coco_data['images']]
    
    # Split image IDs
    print("\n[2/5] Splitting image IDs...")
    train_ids, val_ids, test_ids = split_images_by_id(
        image_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print(f"  Train: {len(train_ids)} images")
    print(f"  Val: {len(val_ids)} images")
    print(f"  Test: {len(test_ids)} images")
    
    # Verify no overlap
    assert len(set(train_ids) & set(val_ids)) == 0, "Train/Val overlap detected!"
    assert len(set(train_ids) & set(test_ids)) == 0, "Train/Test overlap detected!"
    assert len(set(val_ids) & set(test_ids)) == 0, "Val/Test overlap detected!"
    print("  ✓ No overlap between splits")
    
    # Create split annotations
    print("\n[3/5] Creating split annotations...")
    
    train_coco = create_split_annotations(coco_data, train_ids)
    val_coco = create_split_annotations(coco_data, val_ids)
    test_coco = create_split_annotations(coco_data, test_ids)
    
    print(f"  Train annotations: {len(train_coco['annotations'])}")
    print(f"  Val annotations: {len(val_coco['annotations'])}")
    print(f"  Test annotations: {len(test_coco['annotations'])}")
    
    # Save split annotations
    print("\n[4/5] Saving split annotation files...")
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    train_ann_path = output_root / 'train' / 'annotations.json'
    val_ann_path = output_root / 'val' / 'annotations.json'
    test_ann_path = output_root / 'test' / 'annotations.json'
    
    train_ann_path.parent.mkdir(parents=True, exist_ok=True)
    val_ann_path.parent.mkdir(parents=True, exist_ok=True)
    test_ann_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(train_ann_path, 'w') as f:
        json.dump(train_coco, f, indent=2)
    print(f"  Saved: {train_ann_path}")
    
    with open(val_ann_path, 'w') as f:
        json.dump(val_coco, f, indent=2)
    print(f"  Saved: {val_ann_path}")
    
    with open(test_ann_path, 'w') as f:
        json.dump(test_coco, f, indent=2)
    print(f"  Saved: {test_ann_path}")
    
    # Copy images if requested
    if args.copy_images:
        print("\n[5/5] Copying images to split directories...")
        
        # Get image filenames for each split
        id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        train_filenames = [id_to_filename[img_id] for img_id in train_ids]
        val_filenames = [id_to_filename[img_id] for img_id in val_ids]
        test_filenames = [id_to_filename[img_id] for img_id in test_ids]
        
        print("  Copying train images...")
        copy_images(source_root, output_root / 'train' / 'images', train_filenames)
        
        print("  Copying val images...")
        copy_images(source_root, output_root / 'val' / 'images', val_filenames)
        
        print("  Copying test images...")
        copy_images(source_root, output_root / 'test' / 'images', test_filenames)
    else:
        print("\n[5/5] Skipping image copy (use --copy_images to enable)")
        print("  Note: Images will remain in original location")
    
    print("\n" + "="*60)
    print("SPLIT COMPLETE!")
    print("="*60)
    print(f"Output structure:")
    print(f"{output_root}/")
    print(f"├── train/")
    print(f"│   ├── annotations.json ({len(train_coco['annotations'])} annotations)")
    if args.copy_images:
        print(f"│   └── images/ ({len(train_filenames)} images)")
    print(f"├── val/")
    print(f"│   ├── annotations.json ({len(val_coco['annotations'])} annotations)")
    if args.copy_images:
        print(f"│   └── images/ ({len(val_filenames)} images)")
    print(f"└── test/")
    print(f"    ├── annotations.json ({len(test_coco['annotations'])} annotations)")
    if args.copy_images:
        print(f"    └── images/ ({len(test_filenames)} images)")
    print("="*60)


if __name__ == '__main__':
    main()