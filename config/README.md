# ⚙️ Configuration File (`config/config.yaml`)

This YAML file centralizes all hyperparameters, device settings, and file paths used across the training and inference pipelines.

## 🏃 Training Parameters

These control the optimization process and execution of the training loop.

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_epochs` | `int` | Total number of passes over the training dataset. |
| `batch_size` | `int` | Batch size for DataLoaders (e.g., `1` for trial). |
| `learning_rate` | `float` | Initial learning rate. |
| `weight_decay` | `float` | L2 regularization coefficient. |
| `device` | `str` | Primary device for training (`"cuda"` or `"cpu"`). |
| `output_model_path` | `str` | Filename for saving the best model weights. |

---

## 📁 Data Paths

These define the location of the dataset files.

| Parameter | Type | Description |
|-----------|------|-------------|
| `root_dir` | `str` | Base directory for the entire dataset (paths below are relative to this). |
| `images_root` | `str` | Image file location (set to `.` if COCO paths include subfolders). |
| `annotation_file` | `str` | Name of the aggregate COCO JSON file. |

---

## 🧠 Model & Inference

These control the model architecture and loading behavior.

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_object_classes` | `int` | Number of foreground classes (background is class 0). |
| `model_path` | `str` | Path to the saved model file (`.pth`) for inference. |
| `images_path` | `str` | Path to the directory containing images for inference testing. |