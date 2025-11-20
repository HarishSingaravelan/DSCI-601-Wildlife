# 📷 Dataset Structure and Annotation Formats

This directory contains the raw aerial imagery and the specific annotation files used for labeling wildlife near turbines. Understanding the format of these files is critical, as the JSON format is proprietary, while the XML is standard Pascal VOC.

## 1. File Structure and Core Content

The images (`.JPG`) are paired with individual `.json` and `.xml` files that contain the annotations. These individual files are typically parsed by a script to generate the single, aggregate COCO JSON file required by the `TurbineCocoDataset` class in the training pipeline.

| File Type | Example | Format Details |
|-----------|---------|----------------|
| Raw Image | `DJI_0676.JPG` | Aerial image. |
| Proprietary JSON | `DJI_0677.json` | Contains unique inspection metadata and normalized point locations for bounding boxes/objects. |
| Pascal VOC XML | `DJI_0676.xml` | Contains the standard absolute pixel bounding boxes and object metadata. |

---

## 2. Proprietary JSON Format (`.json`)

The JSON files use a custom, proprietary structure that includes metadata about the inspection and object data under the `"carcasses"` key.

### Key Observations

- **Coordinate System**: The `"location"` data uses normalized coordinates (floats between 0.0 and 1.0) and appears to define two points (e.g., center points or corners) rather than a direct `[x, y, w, h]` bounding box.
- **Object Metadata**: Rich proprietary metadata is provided, including `group`, `class` (e.g., `"red_winged_blackbird"`), `sex`, and `condition`.
- **Source**: This format is likely used by an internal system or specific annotation tool for tracking unique inspection data and detailed object attributes.

### JSON Structure Example

```json
{
    "meta": {
        "wildlife": "unique_inspection_id",
        "inspection_utc": "2022-03-18 22:25:05.987023"
    },
    "carcasses": [
        {
            "src": "D:\\...",
            "type": "rec",
            "location": [ /* Normalized Coordinates [0.0 - 1.0] */
                {"x": "0.67803...", "y": "0.76617..."}, 
                {"x": "0.65928...", "y": "0.75379..."}
            ],
            "group": "songbird",
            "class": "red_winged_blackbird",
            /* ... additional attributes ... */
        }
    ]
}
```

---

## 3. Pascal VOC XML Format (`.xml`)

The XML files follow the standard Pascal VOC format, which is commonly used in object detection and is essential because it contains the explicit, four-corner bounding box coordinates.

### Key Observations

- **Coordinate System**: The `<bndbox>` tag contains absolute pixel coordinates (`xmin`, `ymin`, `xmax`, `ymax`). This is the definitive bounding box data that must be converted to COCO format for the PyTorch pipeline.
- **Image Size**: The `<size>` tag provides the original image dimensions (`width`, `height`), necessary for validating the absolute bounding box coordinates.
- **Object Class**: The `<name>` tag provides the object class label (e.g., `red_winged_blackbird`).

### XML Structure Example

```xml
<annotation>
    <folder>20210912_OT_OF_01_NA_30_RGB</folder>
    <filename>DJI_0676.JPG</filename>
    <size>
        <width>5280</width>
        <height>3956</height>
        <depth>3</depth>
    </size>
    <object>
        <name>red_winged_blackbird</name>
        <bndbox>
            <xmin>3438</xmin>
            <ymin>547</ymin>
            <xmax>3541</xmax>
            <ymax>595</ymax>
        </bndbox>
    </object>
</annotation>
```

---

## 4. Pipeline Integration

For the object detection training pipeline, a separate script must be used to:

1. **Read the Pascal VOC XML** (`.xml`) files to extract the absolute pixel bounding boxes (`xmin`, `ymin`, `xmax`, `ymax`) and the `name`.
2. **Read the image file dimensions** from the XML to ensure correctness.
3. **Format this information into the aggregate COCO JSON format** (`[x, y, width, height]`), which is then consumed by the `TurbineCocoDataset`. 

> **Note**: The proprietary JSON is likely ignored or used only for auditing.