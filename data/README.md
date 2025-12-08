# COCO Dataset Creation Tool 🧩

This project provides a Python utility for creating and managing datasets in **COCO (Common Objects in Context)** format.  
It includes functionality to generate COCO-style annotations from custom datasets and supports automated testing using **pytest**.

---

## 📂 Project Structure
```
Wild-Life Detection Project/
│
├── coco_dataset_creation/
│── coco_dataset_generator.py
│
├── tests/
│ └── test_coco_dataset_generator.py
│
├── .gitignore
├── requirements.txt
└── README.md
```



## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/HarishSingaravelan/DSCI-601-Wildlife.git
   cd coco-dataset-creation

2. (Optional) Create a virtual environment:
    ```
    conda create -n dsci601 python=3.10
    conda activate dsci601
    ```

3. Install dependencies:

    `pip install -r requirements.txt`

## 🚀 Usage

The main script for generating a COCO-style dataset is:
```coco_dataset_creation/coco_dataset_generator.py```

Example usage inside Python:

from coco_dataset_creation.coco_dataset_generator import generate_coco_json

#### Example paths
```image_dir = "path/to/images"
output_json = "path/to/output/coco_annotations.json"

generate_coco_json(image_dir, output_json) 
```

This will generate a COCO-style JSON annotation file based on your input dataset.

## 🧪 Running Tests

All unit tests are located in the tests/ directory.
To run tests, execute:

`pytest -v`

## 🧱 Features

- ✅ Generate COCO-style annotation JSON automatically

- 🖼️ Supports multiple image formats

- 🧪 Includes Pytest-based unit tests

- 🧰 Modular design for dataset manipulation and extension

## 🧾 Example COCO JSON Output
```.json
{
  "images": [
    {"id": 1, "file_name": "image1.jpg", "height": 480, "width": 640}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 3, "bbox": [100, 120, 50, 60], "area": 3000}
  ],
  "categories": [
    {"id": 3, "name": "person"}
  ]
}
```


```
python data/split_dataset.py \
  --source_root ../../../shared/rc/turbine \
  --annotation_file ../../../shared/rc/turbine/coco_annotations_updated.json \
  --output_root ../../../shared/rc/turbine/turbine_split \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15
```

For windows
```
python data/split_dataset.py `
    --source_root . `
    --annotation_file coco_annotations.json `
    --output_root turbine_split `
    --train_ratio 0.7 `
    --val_ratio 0.15 `
    --test_ratio 0.15
```
