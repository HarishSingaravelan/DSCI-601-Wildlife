"""
Create a balanced sample dataset WITH background images.

This script maintains the same 50% background / 50% object distribution
as your full training pipeline, ensuring the sample is representative.

Usage:
    python create_sample_dataset_with_bg.py \
        --input-ann-file path/to/annotations.json \
        --input-images-dir path/to/images \
        --output-dir sample_100 \
        --samples-per-class 100
"""

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm


def load_coco_annotations(ann_file):
    """Load COCO format annotations"""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    return data


def get_images_by_class(coco_data):
    """
    Group images by the classes they contain.
    Returns:
        class_to_images: {class_id: [list of image_ids]}
        background_images: [list of image_ids with NO annotations]
    """
    # Get all image IDs
    all_image_ids = {img['id'] for img in coco_data['images']}
    
    # Get images with annotations
    images_with_annotations = set()
    class_to_images = defaultdict(set)
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        images_with_annotations.add(image_id)
        class_to_images[category_id].add(image_id)
    
    # Background images = all images - images with annotations
    background_images = list(all_image_ids - images_with_annotations)
    
    # Convert sets to lists
    class_to_images = {k: list(v) for k, v in class_to_images.items()}
    
    return class_to_images, background_images


def create_balanced_sample_with_bg(
    input_ann_file,
    input_images_dir,
    output_dir,
    samples_per_class=100,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    bg_ratio=0.5,  # 50% background, 50% object images
    seed=42
):
    """
    Create a balanced sample dataset WITH background images.
    
    Maintains the same distribution as your full training:
    - 50% background images (no annotations)
    - 50% object images (balanced across classes)
    """
    random.seed(seed)
    
    print(f"\n{'='*70}")
    print("Creating Balanced Sample Dataset WITH Background Images")
    print(f"{'='*70}")
    print(f"Input annotations: {input_ann_file}")
    print(f"Input images:      {input_images_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Background ratio:  {bg_ratio:.0%}")
    print(f"Split:             {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")
    print(f"{'='*70}\n")
    
    # Load full dataset
    print("Loading annotations...")
    coco_data = load_coco_annotations(input_ann_file)
    
    # Group images by class and get background images
    print("Grouping images by class and identifying background images...")
    class_to_images, background_images = get_images_by_class(coco_data)
    
    # Print statistics
    print(f"\nFull dataset statistics:")
    print(f"  Total images:             {len(coco_data['images'])}")
    print(f"  Images with annotations:  {sum(len(imgs) for imgs in class_to_images.values())} (unique: {len(set().union(*[set(v) for v in class_to_images.values()]))})")
    print(f"  Background images:        {len(background_images)} ({len(background_images)/len(coco_data['images'])*100:.1f}%)")
    
    # Print class distribution
    print("\nClass distribution in full dataset:")
    print(f"{'Class ID':<10} {'Name':<35} {'Images':<10}")
    print("-" * 55)
    
    categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    for class_id in sorted(class_to_images.keys()):
        class_name = categories_dict.get(class_id, f"class_{class_id}")
        n_images = len(class_to_images[class_id])
        print(f"{class_id:<10} {class_name[:33]:<35} {n_images:<10}")
    
    # Sample images for each class
    print(f"\nSampling {samples_per_class} images per class...")
    sampled_object_images = {}
    
    for class_id, image_ids in class_to_images.items():
        n_available = len(image_ids)
        n_sample = min(samples_per_class, n_available)
        
        sampled = random.sample(image_ids, n_sample)
        sampled_object_images[class_id] = sampled
        
        if n_sample < samples_per_class:
            print(f"  ⚠️  Class {class_id} ({categories_dict.get(class_id, 'unknown')}): "
                  f"only {n_available} images available (wanted {samples_per_class})")
    
    # Get unique object image IDs
    all_sampled_object_ids = set()
    for image_ids in sampled_object_images.values():
        all_sampled_object_ids.update(image_ids)
    
    all_sampled_object_ids = list(all_sampled_object_ids)
    
    print(f"\nTotal unique object images sampled: {len(all_sampled_object_ids)}")
    
    # Calculate how many background images we need to maintain bg_ratio
    # If we want 50% background, and we have N object images, we need N background images
    n_bg_needed = int(len(all_sampled_object_ids) * (bg_ratio / (1 - bg_ratio)))
    n_bg_available = len(background_images)
    n_bg_sample = min(n_bg_needed, n_bg_available)
    
    print(f"\nBackground image sampling:")
    print(f"  Object images sampled:  {len(all_sampled_object_ids)}")
    print(f"  Background needed:      {n_bg_needed} (for {bg_ratio:.0%} ratio)")
    print(f"  Background available:   {n_bg_available}")
    print(f"  Background sampled:     {n_bg_sample}")
    
    sampled_bg_ids = random.sample(background_images, n_bg_sample)
    
    # Combine object and background images
    all_sampled_ids = all_sampled_object_ids + sampled_bg_ids
    random.shuffle(all_sampled_ids)
    
    actual_bg_ratio = len(sampled_bg_ids) / len(all_sampled_ids)
    print(f"\nFinal sample composition:")
    print(f"  Total images:      {len(all_sampled_ids)}")
    print(f"  Object images:     {len(all_sampled_object_ids)} ({len(all_sampled_object_ids)/len(all_sampled_ids)*100:.1f}%)")
    print(f"  Background images: {len(sampled_bg_ids)} ({actual_bg_ratio*100:.1f}%)")
    
    # Split into train/val/test
    n_total = len(all_sampled_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_ids = set(all_sampled_ids[:n_train])
    val_ids = set(all_sampled_ids[n_train:n_train + n_val])
    test_ids = set(all_sampled_ids[n_train + n_val:])
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_ids)} images")
    print(f"  Val:   {len(val_ids)} images")
    print(f"  Test:  {len(test_ids)} images")
    
    # Create output directories
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Create new COCO annotations for each split
    def create_split_annotations(split_name, split_image_ids):
        """Create COCO annotation file for a split"""
        
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
        
        # Count background images in this split
        split_bg_count = len([img_id for img_id in split_image_ids if img_id in sampled_bg_ids])
        split_obj_count = len(split_image_ids) - split_bg_count
        
        # Create new COCO dict
        split_coco = {
            'images': split_images,
            'annotations': split_annotations,
            'categories': coco_data['categories']
        }
        
        # Save annotation file
        ann_output_path = output_path / split_name / 'annotations.json'
        with open(ann_output_path, 'w') as f:
            json.dump(split_coco, f, indent=2)
        
        print(f"\n{split_name.upper()} split:")
        print(f"  Total images:         {len(split_images)}")
        print(f"    Object images:      {split_obj_count} ({split_obj_count/len(split_images)*100:.1f}%)")
        print(f"    Background images:  {split_bg_count} ({split_bg_count/len(split_images)*100:.1f}%)")
        print(f"  Total annotations:    {len(split_annotations)}")
        print(f"  Saved to:             {ann_output_path}")
        
        return split_images
    
    # Create annotations for each split
    train_images = create_split_annotations('train', train_ids)
    val_images = create_split_annotations('val', val_ids)
    test_images = create_split_annotations('test', test_ids)
    
    # Copy images
    print(f"\nCopying images...")
    input_images_path = Path(input_images_dir)
    
    def copy_images(split_name, image_list):
        """Copy images for a split"""
        split_dir = output_path / split_name
        
        for img_info in tqdm(image_list, desc=f"Copying {split_name}", leave=False):
            src = input_images_path / img_info['file_name']
            dst = split_dir / img_info['file_name']
            
            # Create parent directories if they don't exist
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            if src.exists():
                shutil.copy2(src, dst)
            else:
                print(f"  ⚠️  Image not found: {src}")
    
    copy_images('train', train_images)
    copy_images('val', val_images)
    copy_images('test', test_images)
    
    # Print class distribution in each split
    print(f"\n{'='*70}")
    print("Class Distribution in Sample Dataset")
    print(f"{'='*70}")
    
    def print_split_distribution(split_name, split_image_ids):
        """Print class distribution for a split"""
        split_image_ids_set = set(split_image_ids)
        
        # Count annotations per class
        class_counts = defaultdict(int)
        for ann in coco_data['annotations']:
            if ann['image_id'] in split_image_ids_set:
                class_counts[ann['category_id']] += 1
        
        # Count background images
        bg_count = len([img_id for img_id in split_image_ids if img_id in sampled_bg_ids])
        
        print(f"\n{split_name.upper()}:")
        print(f"{'Type':<12} {'Class ID':<10} {'Name':<30} {'Count':<10}")
        print("-" * 62)
        
        # Background
        print(f"{'Background':<12} {'-':<10} {'(no annotations)':<30} {bg_count:<10}")
        
        # Object classes
        for class_id in sorted(class_counts.keys()):
            class_name = categories_dict.get(class_id, f"class_{class_id}")
            count = class_counts[class_id]
            print(f"{'Object':<12} {class_id:<10} {class_name[:28]:<30} {count:<10}")
        
        total_ann = sum(class_counts.values())
        total_img = len(split_image_ids)
        print(f"{'-'*62}")
        print(f"{'TOTALS':<12} {'':<10} {'Images: '+str(total_img):<30} {'Ann: '+str(total_ann):<10}")
    
    print_split_distribution('train', train_ids)
    print_split_distribution('val', val_ids)
    print_split_distribution('test', test_ids)
    
    # Create a summary file
    summary_path = output_path / 'dataset_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Balanced Sample Dataset with Background Images\n")
        f.write("="*70 + "\n\n")
        f.write(f"Source: {input_ann_file}\n")
        f.write(f"Samples per class (target): {samples_per_class}\n")
        f.write(f"Background ratio: {bg_ratio:.0%}\n\n")
        f.write(f"Total images: {len(all_sampled_ids)}\n")
        f.write(f"  Object images:     {len(all_sampled_object_ids)} ({len(all_sampled_object_ids)/len(all_sampled_ids)*100:.1f}%)\n")
        f.write(f"  Background images: {len(sampled_bg_ids)} ({actual_bg_ratio*100:.1f}%)\n\n")
        f.write(f"Train: {len(train_ids)} images\n")
        f.write(f"Val:   {len(val_ids)} images\n")
        f.write(f"Test:  {len(test_ids)} images\n")
        f.write(f"\nCreated: {output_path.absolute()}\n")
    
    print(f"\n{'='*70}")
    print("✅ Sample dataset with background images created successfully!")
    print(f"{'='*70}")
    print(f"Output location: {output_path.absolute()}")
    print(f"Summary saved:   {summary_path}")
    print(f"\nTo use this dataset, update your config.yaml:")
    print(f"  train_images_dir: '{input_images_dir}'")
    print(f"  train_ann_file:   '{output_path / 'train' / 'annotations.json'}'")
    print(f"  val_images_dir:   '{input_images_dir}'")
    print(f"  val_ann_file:     '{output_path / 'val' / 'annotations.json'}'")
    print(f"  test_images_dir:  '{input_images_dir}'")
    print(f"  test_ann_file:    '{output_path / 'test' / 'annotations.json'}'")
    print(f"\n💡 Note: Images stay in the original directory, only annotations are split!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create a balanced sample dataset WITH background images"
    )
    
    parser.add_argument(
        '--input-ann-file',
        type=str,
        required=True,
        help='Path to full annotations.json file'
    )
    
    parser.add_argument(
        '--input-images-dir',
        type=str,
        required=True,
        help='Directory containing all images'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for sample dataset'
    )
    
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=100,
        help='Number of images to sample per class (default: 100)'
    )
    
    parser.add_argument(
        '--bg-ratio',
        type=float,
        default=0.5,
        help='Ratio of background images (default: 0.5 = 50%%)'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Fraction of data for training (default: 0.7)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Fraction of data for validation (default: 0.15)'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Fraction of data for testing (default: 0.15)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"❌ Error: train_ratio + val_ratio + test_ratio must equal 1.0")
        print(f"   Got: {args.train_ratio} + {args.val_ratio} + {args.test_ratio} = {total_ratio}")
        return
    
    create_balanced_sample_with_bg(
        input_ann_file=args.input_ann_file,
        input_images_dir=args.input_images_dir,
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        bg_ratio=args.bg_ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()