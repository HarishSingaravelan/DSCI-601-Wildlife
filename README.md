# 🦌 Wildlife Detection Capstone Project

## 🌟 Overview

This repository contains the code for a **Wildlife Detection and Segmentation** project, leveraging a deep learning pipeline (e.g., FasterRCNN, DETR and its variants) built using PyTorch. The goal is to train a model to accurately identify and localize various wildlife species within camera trap imagery using the **COCO data format**.

The project is structured modularly, separating data handling, modeling, optimization, and utility functions for clarity and maintainability.

---

## 🏗️ Project Structure

The codebase is organized into logical directories following the separation of concerns principle.

| Directory | Purpose | Key Files | Detailed Docs |
| :--- | :--- | :--- | :--- |
| **`data/`** | Source data and scripts for converting to the COCO format. | `generate_coco.py` | [Dataset Guide](data/README.md) |
| **`config/`** | Configuration files for hyperparameters and paths. | `config.yaml` | [Config Guide](config/README.md) |
| **`processing/`** | Components for data preparation, loading, and batching. | `dataset.py`, `dataloader.py`, `transforms.py` | [Data Pipeline](turbine_processing/README.md) |
| **`modeling/`** | Core neural network architecture and training logic. | `model.py`, `trainer.py` | [Model & Trainer](modeling/README.md) |
| **`inference/`** | Model evaluation and analysis scripts. | `mAPGenerator.py`, `confusionMatrixGenerator.py`,`inference.py` | [inference & Evaluation Guide](inference/README.md) |
| **`tests/`** | Unit and integration tests for component validation. | `test_*.py` | [Testing Guide](Tests/README.md) |
| **Root** | Main entry point for running the pipeline. | `main.py` | — |
---

📚 Documentation
Comprehensive API documentation is available via Sphinx:
Local Documentation: Open doc\_build\html\index.html in your browser for complete API reference and module documentation.

## 🧪 Core Components

### Main Entry Point (main.py)

main.py is the orchestration script responsible for setting up and executing the end-to-end training pipeline.
| Step              | Description |
|-------------------|-------------|
| **Configuration** | Loads hyperparameters (learning rate, batch size, device, paths, etc.) from `config/config.yaml`. |
| **Model Setup**   | Initializes Faster R-CNN (ResNet-50 FPN backbone) and configures the correct number of output classes. |
| **Optimization**  | Uses Adam optimizer and applies Gradient Clipping (implemented in `modeling/trainer.py`) to prevent exploding gradients and NaN losses. |
| **Real-Time Logging** | Creates a TensorBoard `SummaryWriter` in `runs/` with timestamped folders for monitoring |
| **Training Loop** | Executes epoch iterations, calling `trainer.train_one_epoch()` for training and `trainer.validate_metrics()` for evaluating validation mAP metrics. |


## 📊 Datasets and Annotation

This project utilizes a large, proprietary dataset for wildlife detection. For demonstration and testing purposes, this repository contains a small sample of that data.

### 1. Raw Image Dataset (The Source Data)

| Detail | Description |
|--------|-------------|
| **Full Dataset Size** | ~500 GB (not publicly hosted or downloadable). |
| **Sample Location** | `Sample_images/` |
| **Sample Content** | This folder contains 10 sample images and their original annotations, used strictly for demonstrating the functionality of the data pipeline and ensuring the PyTorch data loaders can initialize correctly. |
| **Download** | **Note:** The full 500 GB dataset is cannot be downloaded. Users must supply their own annotated image collection to perform full model training. |
| **Purpose** | The small sample set ensures that all scripts (`coco_dataset_generator.py`, `main.py`, etc.) are runnable immediately after cloning the repository. |

## ⚙️ Setup and Installation

### Prerequisites

* Python 3.8+
* NVIDIA GPU (recommended for training)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HarishSingaravelan/DSCI-601-Wildlife.git
    cd DSCI-601-Wildlife
    ```

2.  **Create & Activate Environment**
    ```bash
    python -m venv wildlife-env
    wildlife-env\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
---


## 🏃 Getting Started

### 1. Generate COCO Dataset (from sample images)

This project includes sample images to help you test the dataset creation pipeline.

Run the COCO generator script:


```bash
python data/coco_dataset_generator.py
```

This script will:
 - read images from the sample directory
 - generate COCO annotations
 - save output JSON files in root folder
 - prepare the dataset structure used by the model


```bash
# Linux/macOS
python data/split_dataset.py \
    --source_root . \
    --annotation_file coco_annotations.json \
    --output_root turbine_split \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

or

```bash
# Windows PowerShell
python data/split_dataset.py `
    --source_root . `
    --annotation_file coco_annotations.json `
    --output_root turbine_split `
    --train_ratio 0.7 `
    --val_ratio 0.15 `
    --test_ratio 0.15
```
### 2. Training
Execute the main training script. This script orchestrates the entire pipeline: loading data, instantiating the model, setting up the optimizer/scheduler, and running the Trainer.

```Bash
python main.py
```
### 3. View Training Progress in TensorBoard
#### 1. Start TensorBoard
 Inside your project directory:

 ```
 tensorboard --logdir runs --port 6006
 ```

 #### 2. Open Browser
 ```
 http://localhost:6006
 ```

