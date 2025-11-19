# 🦌 Wildlife Detection Capstone Project

## 🌟 Overview

This repository contains the code for a **Wildlife Detection and Segmentation** project, leveraging a deep learning pipeline (e.g., RetinaNet, DeepLabV3) built using PyTorch. The goal is to train a model to accurately identify and localize various wildlife species within camera trap imagery using the **COCO data format**.

The project is structured modularly, separating data handling, modeling, optimization, and utility functions for clarity and maintainability.

---

## 🏗️ Project Structure

The codebase is organized into logical directories following the separation of concerns principle.

| Directory | Purpose | Key Files |
| :--- | :--- | :--- |
| **`data/`** | Source data and scripts for converting to the COCO format. | `generate_coco.py` |
| **`config/`** | Configuration files for hyperparameters and paths. | `config.yaml` |
| **`processing/`** | Components for data preparation and batching. | `dataset.py`, `augmenter.py`, `collate_function.py` |
| **`modeling/`** | Core neural network architecture and training logic. | `model.py`, `trainer.py` |
| **`optim/`** | Configuration for optimization algorithms. | `optimizer_setup.py`, `scheduler.py` |
| **`utils/`** | Helper functions for metrics, logging, and error calculation. | `metrics_tracker.py`, `loss_function.py` |
| **`tests/`** | Unit and integration tests for component validation. | `test_*.py` |
| **Root** | Main entry point for running the pipeline. | `main.py` |

---

## 🧪 Core Components

1. Dynamic Balanced Sampler (turbine_processing/sampler.py)

This custom PyTorch Sampler is critical for addressing the common challenge of class and background imbalance in detection datasets.

| Feature          | Description |
|------------------|-------------|
| **50/50 Split**  | Ensures each training epoch contains 50% wildlife images (positive samples) and 50% background images (negative samples). |
| **Class Balancing** | Within the positive samples, minority wildlife classes (rare species) are oversampled with replacement, while majority classes are sampled without replacement. This ensures equal exposure to all species each epoch. |
| **GPU Efficiency** | The `TurbineDataLoader` combines this sampler with `pin_memory=True` and multiple `num_workers` to keep the GPU fully utilized and avoid data-loading bottlenecks. |


2. Main Entry Point (main.py)

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

2.  **Create and activate the environment:**
    ```bash
    conda create -n wildlife python=3.9
    conda activate wildlife
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
---

## 🏃 Getting Started

### 1. Data Preparation

Place your raw imagery in `data/images/` and raw annotations in `data/annotations/`.

Run the generation script to create the standardized COCO JSON files (for train, validation, and test splits):
```bash
python data/coco_dataset_generator.py
```

### 2. Training
Execute the main training script. This script orchestrates the entire pipeline: loading data, instantiating the model, setting up the optimizer/scheduler, and running the Trainer.

```Bash
python main.py
```
✅ Testing
To ensure all components are working correctly, run the test suite:

```Bash
pytest tests/
```
