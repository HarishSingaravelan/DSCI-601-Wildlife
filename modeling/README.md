# 🧠 Core Model and Training Pipeline

This section documents the main components of the object detection project: the model architecture definition (`get_model`) and the training control loop (`Trainer` class).

## 1. Model Definition (`modeling/model.py`)

The `get_model` function provides a standardized way to initialize the object detection architecture, specifically the Faster R-CNN with a ResNet-50 Feature Pyramid Network (FPN) backbone.

### Key Features

- **Architecture**: Faster R-CNN with a ResNet-50-FPN backbone, a highly effective standard for object detection.

- **Customization**: The final classification head is automatically replaced using `FastRCNNPredictor` to match the custom `num_classes` required by the turbine dataset.

- **Pre-trained Weights Control**: The `load_weights` flag enables fast initialization for testing (`load_weights=False`) or loads COCO pre-trained weights for production/training (`load_weights=True`) to leverage transfer learning.

### Function Overview

The `get_model` function:
- Loads a Faster R-CNN model with ResNet-50 FPN backbone
- Optionally loads pre-trained COCO weights
- Replaces the classifier head to match custom number of classes
- Returns the configured model ready for training

---

## 2. Trainer Class (`trainer/trainer.py`)

The `Trainer` class encapsulates the entire training and validation procedure, handling device management, optimization, logging, and metric calculation for object detection tasks.

### Core Components

- **Model/Optimizer Setup**: Initializes the model, optimizer, and moves the model to the specified device (CPU/GPU).

- **TensorBoard Logging**: Uses `torch.utils.tensorboard.SummaryWriter` to log losses and metrics per batch and per epoch to a specified `log_dir`.

- **Metrics**: Initializes `torchmetrics.detection.MeanAveragePrecision` for calculating industry-standard object detection metrics.

### Key Methods

#### `train_one_epoch()`

Handles the primary training loop, including:
- Moving data to the specified device
- Calculating total loss (`loss = sum(v for v in loss_dict.values())`)
- Backpropagation and optimizer step
- Progress feedback via `tqdm` with live updates of batch loss and average loss
- Logging all individual loss components (`loss_objectness`, `loss_rpn_box_reg`, etc.) to TensorBoard at the batch level
- Calls `validate_loss` at the end of the epoch

#### `validate_loss()`

Calculates the validation loss.

**Crucial implementation note**: Faster R-CNN models only return loss values when set to training mode (`model.train()`) and passed targets. Therefore, this method calls `model.train()` but wraps the loop in `@torch.no_grad()` to prevent gradient calculation, achieving a true loss measurement without updating weights.

#### `validate_metrics(val_loader, epoch)`

Computes the final validation metrics for the epoch:
- Switches the model to evaluation mode (`model.eval()`) to enable non-maximum suppression (NMS) and get predictions (boxes, scores, labels) instead of loss components
- Accumulates predictions and targets using `self.metric_calculator.update()`
- Computes and logs Mean Average Precision (mAP) metrics (`mAP@0.5`, `mAP@0.75`, `mAP`) to TensorBoard

### Trainer Initialization

The `Trainer` class is initialized with:
- `model`: PyTorch model to train
- `optimizer`: Optimizer for weight updates
- `device`: Target device (CPU/GPU)
- `log_dir`: Directory for TensorBoard logs (default: `"runs/faster_rcnn_experiment"`)