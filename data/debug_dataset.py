# debug_dataset.py — run from your project root
import sys, json
sys.path.insert(0, '.')
import yaml
from turbine_processing.dataset import TurbineCocoDataset

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Check raw annotation file first
ann_file = config['data']['train_ann_file']
with open(ann_file) as f:
    raw = json.load(f)

print(f"=== RAW ANNOTATION FILE ===")
print(f"Images       : {len(raw['images'])}")
print(f"Annotations  : {len(raw['annotations'])}")
images_with_anns = len(set(a['image_id'] for a in raw['annotations']))
print(f"Images with anns: {images_with_anns}")
print(f"Background images: {len(raw['images']) - images_with_anns}")

# Sample a few bboxes
print(f"\nSample bboxes (first 5):")
for a in raw['annotations'][:5]:
    print(f"  image_id={a['image_id']} bbox={a['bbox']}")

# Now check what the dataset actually returns after filtering
print(f"\n=== AFTER DATASET FILTERING ===")
ds = TurbineCocoDataset(
    images_dir=config['data']['train_images_dir'],
    ann_file=ann_file,
    transforms=None
)
empty = 0
has_boxes = 0
for i in range(min(200, len(ds))):
    _, target = ds[i]
    if target['boxes'].shape[0] == 0:
        empty += 1
    else:
        has_boxes += 1

print(f"Checked 200 samples:")
print(f"  With boxes    : {has_boxes}")
print(f"  Empty (no boxes): {empty}")
print(f"  Empty ratio   : {empty/200*100:.1f}%")