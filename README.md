# 🦌 Wildlife Detection Capstone Project

## 🌟 Overview

This repository contains the code for a **Wildlife Detection and Segmentation** project, leveraging a deep learning pipeline (e.g., FasterRCNN, DETR and its variants) built using PyTorch. The goal is to train a model to accurately identify and localize various wildlife species within camera trap imagery using the COCO data format.

The project is structured modularly, separating data handling, modeling, optimization, and utility functions for clarity and maintainability.

---

## 📚 Code Documentation

This project is properly documented using standard Python docstrings. The automated HTML documentation was generated using [pdoc](https://pdoc.dev/).

**To view the documentation:**
1. Navigate to the `docs/` folder in this repository.
2. Download and open `index.html` in any standard web browser to explore the module and function-level documentation.

**Regenerating the Documentation:**

If you modify the source code and need to regenerate the HTML documentation locally, ensure you have `pdoc` installed and run the following command from the root directory:

```bash
pip install pdoc
PYTHONPATH=. pdoc data/ inference/ modeling/ turbine_processing/ main.py -o docs
```

---

## 🏗️ Project Structure

The codebase is organized into logical directories following the separation of concerns principle.

| Directory | Purpose | Key Files | Detailed Docs |
|---|---|---|---|
| `data/` | Source data and scripts for converting to the COCO format. | `generate_coco.py` | Dataset Guide |
| `config/` | Configuration files for hyperparameters and paths. | `config.yaml` | Config Guide |
| `turbine_processing/` | Components for data preparation, loading, and batching. | `dataset.py`, `dataloader.py`, `transforms_detr.py` | Data Pipeline |
| `modeling/` | Core neural network architecture and training logic. | `model.py`, `trainer.py` | Model & Trainer |
| `inference/` | Model evaluation and analysis scripts. | `mAPGenerator.py`, `confusionMatrixGenerator.py`, `inference.py` | Inference & Evaluation Guide |
| `Tests/` | Unit and integration tests for component validation. | `test_*.py` | Testing Guide |
| Root | Main entry point for running the pipeline. | `main.py` | — |

---

## 🧪 Core Components

### Main Entry Point (`main.py`)

`main.py` is the orchestration script responsible for setting up and executing the end-to-end training pipeline.

| Step | Description |
|---|---|
| **Configuration** | Loads hyperparameters (learning rate, batch size, device, paths, etc.) from `config/config.yaml`. |
| **Model Setup** | Initializes the dynamic architecture (Faster R-CNN, DETR, Deformable DETR, or D-FINE) and configures the correct number of output classes. |
| **Optimization** | Uses AdamW optimizer and applies Gradient Clipping to prevent exploding gradients and NaN losses. |
| **Real-Time Logging** | Creates a TensorBoard `SummaryWriter` in `runs/` with timestamped folders for monitoring. |
| **Training Loop** | Executes epoch iterations using the custom Trainer loops, handling adaptive sampling, backpropagation, and validation metrics. |

---

## 📊 Datasets and Annotation

This project utilizes a large, proprietary dataset for wildlife detection. For demonstration and testing purposes, this repository contains a small sample of that data.

### 1. Raw Image Dataset (The Source Data)

| Detail | Description |
|---|---|
| **Full Dataset Size** | ~500 GB (not publicly hosted or downloadable). |
| **Sample Location** | `Sample_images/` |
| **Sample Content** | This folder contains sample images and their original annotations, used strictly for demonstrating the functionality of the data pipeline and ensuring the PyTorch data loaders can initialize correctly. |
| **Download** | Note: The full 500 GB dataset cannot be downloaded. Users must supply their own annotated image collection to perform full model training. |
| **Purpose** | The small sample set ensures that all scripts (`coco_dataset_generator.py`, `main.py`, etc.) are runnable immediately after cloning the repository. |

---

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU (highly recommended for training)

### Installation

**1. Clone the repository:**

```bash
git clone https://github.com/HarishSingaravelan/DSCI-601-Wildlife.git
cd DSCI-601-Wildlife
```

**2. Create & Activate Environment:**

```bash
# Linux/macOS
python -m venv wildlife-env
source wildlife-env/bin/activate

# Windows
python -m venv wildlife-env
wildlife-env\Scripts\activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 🏃 Getting Started

### 1. Generate COCO Dataset (from sample images)

This project includes sample images to help you test the dataset creation pipeline. Run the COCO generator script:

```bash
python data/coco_dataset_generator.py
```

This script will read images from the sample directory, generate COCO annotations, save output JSON files, and prepare the dataset structure used by the model.

Next, split the dataset:

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

### 2. Changing the Model Architecture

The system features a **dynamic architecture toggler**. You can easily switch the pipeline between Standard DETR, Deformable DETR, and D-FINE without rewriting code.

Open `config/config.yaml`, scroll to the bottom, and update the `architecture` and `pretrained_model` fields. For example, to run D-FINE:

```yaml
architecture: "dfine"                          # Options: "standard_detr", "deformable_detr", or "dfine"
pretrained_model: "ustc-community/dfine_x_coco" # Options: "facebook/detr-resnet-50", "SenseTime/deformable-detr", or "ustc-community/dfine_x_coco"
```

### 3. Training

Execute the main training script. This script orchestrates the entire pipeline: loading data, instantiating the selected model, setting up the optimizer/scheduler, and running the adaptive feedback loop.

```bash
python3 modeling/detr/train_detr.py
```

### 4. View Training Progress in TensorBoard

To monitor your training loss, validation mAP, and Adaptive Sampler class weights in real time:

**Start TensorBoard** (run this inside your project directory):

```bash
tensorboard --logdir runs --port 6006
```

**Open Browser:**

Navigate to [http://localhost:6006](http://localhost:6006) to view the dashboards.