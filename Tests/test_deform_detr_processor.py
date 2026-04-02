import torch
import numpy as np
from PIL import Image
from transformers import DeformableDetrImageProcessor

def main():
    print("Loading Deformable DETR Processor...")
    # Initialize the processor with conversion turned on (default behavior)
    processor = DeformableDetrImageProcessor.from_pretrained(
        "SenseTime/deformable-detr", 
        do_convert_annotations=True
    )

    # 1. Create a dummy image (e.g., 800 pixels wide, 600 pixels high)
    img_w, img_h = 800, 600
    dummy_image = Image.new('RGB', (img_w, img_h), color='red')

    # 2. Create a dummy bounding box in COCO Format [x_min, y_min, width, height]
    # Let's put a 200x100 box near the top left.
    coco_bbox = [100.0, 50.0, 200.0, 100.0] 

    dummy_annotation = {
        "image_id": 1,
        "annotations": [
            {
                "bbox": coco_bbox,
                "category_id": 1,
                "area": coco_bbox[2] * coco_bbox[3], # w * h
                "iscrowd": 0
            }
        ]
    }

    print(f"\n--- INPUT TO PROCESSOR ---")
    print(f"Image Size : {img_w}x{img_h} (W x H)")
    print(f"Input Box  : {coco_bbox} (Format: [x_min, y_min, width, height] in PIXELS)")

    # 3. Run the processor
    inputs = processor(images=dummy_image, annotations=dummy_annotation, return_tensors="pt")

    # 4. Extract the processed labels
    processed_labels = inputs["labels"][0]
    output_box = processed_labels["boxes"][0].tolist()

    print(f"\n--- OUTPUT FROM PROCESSOR ---")
    print(f"Output Box : {[round(x, 4) for x in output_box]} (Format: [center_x, center_y, width, height] NORMALIZED)")

    # 5. Let's do the manual math to prove the processor did it right!
    print(f"\n--- VERIFYING THE MATH ---")
    expected_cx = (coco_bbox[0] + (coco_bbox[2] / 2)) / img_w  # (100 + 100) / 800 = 0.25
    expected_cy = (coco_bbox[1] + (coco_bbox[3] / 2)) / img_h  # (50 + 50) / 600 = 0.1666...
    expected_w = coco_bbox[2] / img_w                          # 200 / 800 = 0.25
    expected_h = coco_bbox[3] / img_h                          # 100 / 600 = 0.1666...

    print(f"Expected Center X : {expected_cx:.4f} | Processor Output: {output_box[0]:.4f}")
    print(f"Expected Center Y : {expected_cy:.4f} | Processor Output: {output_box[1]:.4f}")
    print(f"Expected Norm W   : {expected_w:.4f} | Processor Output: {output_box[2]:.4f}")
    print(f"Expected Norm H   : {expected_h:.4f} | Processor Output: {output_box[3]:.4f}")

    if all(abs(a - b) < 1e-4 for a, b in zip(output_box, [expected_cx, expected_cy, expected_w, expected_h])):
        print("\nSUCCESS! The processor correctly converted COCO pixels to Normalized Center coordinates.")
    else:
        print("\nMISMATCH DETECTED!")

if __name__ == "__main__":
    main()