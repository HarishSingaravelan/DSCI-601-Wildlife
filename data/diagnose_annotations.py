"""
Diagnostic script to analyze your COCO annotations
Run this BEFORE training to catch category ID issues
"""

import json
import os
from collections import defaultdict


def analyze_annotations(ann_file: str):
    print(f"\n{'='*60}")
    print(f"Analyzing: {ann_file}")
    print(f"{'='*60}")

    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    num_images = len(data.get('images', []))
    print(f"\n IMAGES:")
    print(f"   Total images: {num_images}")

    # ── Categories ────────────────────────────────────────────────
    print("\n CATEGORIES:")
    categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    for cat_id, cat_name in sorted(categories.items()):
        print(f"   id={cat_id}  name={cat_name}")

    cat_ids = sorted(categories.keys())
    print(f"\n   Min category_id : {min(cat_ids)}")
    print(f"   Max category_id : {max(cat_ids)}")
    print(f"   Num categories  : {len(cat_ids)}")

    # ── Annotations ───────────────────────────────────────────────
    annotations = data.get('annotations', [])
    print(f"\n ANNOTATIONS:")
    print(f"   Total annotations: {len(annotations)}")

    # Count per category
    cat_counts = defaultdict(int)
    for ann in annotations:
        cat_counts[ann['category_id']] += 1

    print("\n   Annotations per category:")
    for cat_id in sorted(cat_counts.keys()):
        name = categories.get(cat_id, "UNKNOWN!")
        print(f"   id={cat_id}  name={name:30s}  count={cat_counts[cat_id]}")

    # ── Check for out-of-bounds IDs ───────────────────────────────
    print(f"\n CHECKING FOR ISSUES:")

    # Check if 0-indexed or 1-indexed
    if min(cat_ids) == 0:
        print("   Categories are 0-indexed (starts at 0)")
    elif min(cat_ids) == 1:
        print("     Categories are 1-INDEXED (starts at 1) — common COCO convention")
        print("      DETR expects 0-indexed labels!")
        print(f"      Your IDs: {cat_ids}")
        print(f"      → You need num_object_classes = {max(cat_ids) + 1}")
        print(f"        (to accommodate the highest ID = {max(cat_ids)})")

    # Check for gaps
    expected = list(range(min(cat_ids), max(cat_ids) + 1))
    missing  = [i for i in expected if i not in cat_ids]
    if missing:
        print(f"     GAP in category IDs: {missing}")
    else:
        print(f"     No gaps in category IDs")

    # ── Recommendation ────────────────────────────────────────────
    print(f"\n RECOMMENDATION:")
    max_id = max(cat_ids)
    if min(cat_ids) == 1:
        print(f"""
   Your annotations use category IDs: {cat_ids}
   
   You have TWO options:
   
   Option A — Set num_object_classes correctly (EASIEST):
     num_object_classes: {max_id + 1}
     class_names:
       - "background"   # index 0 (unused but needed as padding)
       - "{categories.get(1, 'class_1')}"  # index 1
       - "{categories.get(2, 'class_2')}"  # index 2
       ...up to index {max_id}
   
   Option B — Remap category IDs to 0-indexed (CLEANER):
     Run remap_category_ids() below to create a fixed annotation file.
""")
    else:
        print(f"    Category IDs look correct for DETR")

    return categories, cat_counts, cat_ids


def remap_category_ids(ann_file: str, output_file: str):
    """
    Remaps 1-indexed category IDs to 0-indexed.
    Creates a new annotation file — does NOT modify original.
    
    e.g. category_id 1 → 0, 2 → 1, 3 → 2, etc.
    
    WARNING: Only use this if you want 0-indexed IDs.
    The easier fix is just setting num_object_classes correctly (Option A).
    """
    with open(ann_file, 'r') as f:
        data = json.load(f)

    old_ids  = sorted(cat['id'] for cat in data['categories'])
    id_remap = {old: new for new, old in enumerate(old_ids)}  # e.g. {1:0, 2:1, ...}

    print(f"\nRemapping category IDs: {id_remap}")

    # Remap categories
    for cat in data['categories']:
        cat['id'] = id_remap[cat['id']]

    # Remap annotations
    for ann in data['annotations']:
        ann['category_id'] = id_remap[ann['category_id']]

    with open(output_file, 'w') as f:
        json.dump(data, f)

    print(f" Remapped annotation file saved to: {output_file}")
    return id_remap


def check_config_vs_annotations(ann_file: str, num_object_classes: int, class_names: list):
    """
    Check if your config matches your annotation file
    """
    print(f"\n{'='*60}")
    print("Checking config vs annotations")
    print(f"{'='*60}")

    with open(ann_file, 'r') as f:
        data = json.load(f)

    cat_ids  = sorted(cat['id'] for cat in data.get('categories', []))
    max_id   = max(cat_ids)
    
    print(f"\n  Annotation max category_id : {max_id}")
    print(f"  Config num_object_classes  : {num_object_classes}")
    print(f"  Config class_names count   : {len(class_names)}")

    if max_id >= num_object_classes:
        print(f"""
     MISMATCH DETECTED!
     category_id {max_id} is OUT OF BOUNDS for num_object_classes={num_object_classes}
     (valid indices are 0 to {num_object_classes - 1})

  FIX: Change your config to:
     num_object_classes: {max_id + 1}
""")
    else:
        print(f"\n   Config looks correct!")


if __name__ == "__main__":
    import yaml

    # Load your config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    ann_files = [
        config['data']['train_ann_file'],
        config['data']['val_ann_file'],
        config['data']['test_ann_file'],
    ]

    # Analyze each split
    for ann_file in ann_files:
        if os.path.exists(ann_file):
            categories, cat_counts, cat_ids = analyze_annotations(ann_file)

            # Check against config
            check_config_vs_annotations(
                ann_file,
                num_object_classes=config['model']['num_object_classes'],
                class_names=config['model'].get('class_names', [])
            )
        else:
            print(f"File not found: {ann_file}")