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
| **`tests/`** | Unit and integration tests for component validation. | `test_*.py` | [Testing Guide](Tests/README.md) |
| **Root** | Main entry point for running the pipeline. | `main.py` | — |
---

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

