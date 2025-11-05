# To find the coco_dataset file
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pytest
from PIL import Image
from data.coco_dataset_generator import generate_coco_json, CATEGORY_MAPPING

@pytest.fixture
def mock_dataset(tmp_path):
    """
    Create a mock directory with:
    - One image
    - One JSON annotation
    """
    root_dir = tmp_path / "Test_2021_SY"
    root_dir.mkdir()

    # Create an image
    image_path = root_dir / "test_img.jpg"
    img = Image.new("RGB", (100, 80), color="red")
    img.save(image_path)

    # Create corresponding JSON annotation
    json_path = root_dir / "test_img.json"
    annotation_data = {
        "carcasses": [
            {
                "class": "red_winged_blackbird",
                "location": [{"x": 0.1, "y": 0.1}, {"x": 0.4, "y": 0.5}]
            }
        ]
    }
    with open(json_path, "w") as f:
        json.dump(annotation_data, f)

    return root_dir


def test_generate_coco_json_json_annotations(mock_dataset, tmp_path):
    """
    Test that a valid JSON annotation correctly generates a COCO JSON file.
    """
    output_file = tmp_path / "output.json"

    generate_coco_json(str(mock_dataset), str(output_file))

    # --- Validate output ---
    assert output_file.exists(), "Output JSON file should be created"

    with open(output_file, "r") as f:
        data = json.load(f)

    # Check top-level structure
    for key in ["info", "licenses", "categories", "images", "annotations"]:
        assert key in data, f"Missing '{key}' section in COCO output"

    # Check category structure
    assert isinstance(data["categories"], list)
    assert len(data["categories"]) == len(CATEGORY_MAPPING)

    # Check image entry
    assert len(data["images"]) == 1
    img_entry = data["images"][0]
    assert img_entry["file_name"].endswith(".jpg")
    assert img_entry["width"] == 100
    assert img_entry["height"] == 80

    # Check annotation entry
    assert len(data["annotations"]) == 1
    ann = data["annotations"][0]
    assert "bbox" in ann
    assert ann["category_id"] == CATEGORY_MAPPING["red_winged_blackbird"]
    assert ann["image_id"] == img_entry["id"]
    assert ann["area"] > 0


def test_generate_coco_json_skips_missing_annotations(tmp_path):
    """
    Ensure that images without annotation files are still listed,
    but no annotations are added.
    """
    root_dir = tmp_path / "Test_2021_SY"
    root_dir.mkdir()

    # Create an image without annotation
    image_path = root_dir / "unannotated_img.jpg"
    Image.new("RGB", (120, 100), color="blue").save(image_path)

    output_file = tmp_path / "output_no_ann.json"
    generate_coco_json(str(root_dir), str(output_file))

    with open(output_file, "r") as f:
        data = json.load(f)

    assert len(data["images"]) == 1, "Image should still appear in output"
    assert len(data["annotations"]) == 0, "No annotations should be added"
