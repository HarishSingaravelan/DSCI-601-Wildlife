# 🗃️ Data Pipeline and Preprocessing

This document outlines the core components of the data loading, augmentation, and preprocessing pipeline for the object detection model, including custom dataset handling for COCO-style annotations, robust image loading, specialized transformations, and efficient PyTorch data loaders.

## 1. Custom Dataset: TurbineCocoDataset

Defined in `turbine_processing/dataset.py`, this class extends `torchvision.datasets.CocoDetection` and is responsible for loading the image files and their associated COCO annotations.

### Key Features

- **PyTorch/COCO Bridge**: Integrates standard COCO annotations with the PyTorch detection model requirements.

- **Robust Loading**: Implements the `_safe_load` method to detect and automatically skip corrupted or truncated images, preventing training crashes.

- **Target Formatting**: Converts raw COCO bounding box format (`[x_min, y_min, width, height]`) to the PyTorch standard (`[x_min, y_min, x_max, y_max]`) and ensures all target fields (`boxes`, `labels`, `area`, `iscrowd`, `image_id`) are correctly formatted as Tensors.

### Implementation

The `__getitem__` method handles:
- Safe image loading through `_safe_load`
- Target building in PyTorch detection format
- Bounding box conversion and Tensor building
- Optional transformation application

Final target dictionary includes:
- `boxes`: Bounding box coordinates
- `labels`: Class labels
- `area`: Box areas
- `iscrowd`: Crowd annotations
- `image_id`: Unique image identifier

---

## 2. Data Loading Wrapper: TurbineDataLoader

This class wraps the creation of `torch.utils.data.DataLoader` instances for both training and validation sets, ensuring proper configuration for object detection and high-performance loading.

### Key Features

- **Detection Collation**: Uses a custom `collate_fn` to correctly handle the variable number of objects and the list-of-dictionaries structure required by PyTorch detection models (like Faster R-CNN).

- **Performance**: Enables `pin_memory=True` for faster host-to-device (CPU-to-GPU) data transfer.

- **Balanced Sampling**: Supports using a custom `DynamicBalancedSampler` (when `use_balanced_sampler=True`) to dynamically address class imbalance during training by controlling the image sampling frequency.

### DataLoader Configuration

The `get_loaders()` method returns two DataLoader instances:

**Training Loader**:
- Uses `DynamicBalancedSampler` if balanced sampling is enabled
- Standard shuffled DataLoader otherwise
- Custom collate function for detection format

**Validation Loader**:
- No shuffling for consistent evaluation
- Same collate function as training
- Pin memory enabled for performance

---

## 3. Transformations and Augmentation

Defined in `turbine_processing/transforms.py`, this module handles all image and bounding box transformations using the **albumentations** library, which is highly efficient for geometric augmentations.

### AlbumentationsDetectionTransform (Adapter)

This adapter class allows the albumentations pipeline to be seamlessly integrated into the PyTorch `Dataset.__getitem__` method, handling all necessary type and format conversions between PIL/NumPy and PyTorch Tensors.

**Key Logic**: 
- Converts the image and target dict to NumPy/list
- Runs the `A.Compose` transformation
- Converts results back to `torch.FloatTensor` (C, H, W)
- Correctly recalculates bounding box area

### Transformation Pipelines

The module provides two distinct pipelines, built using `albumentations.Compose`:

#### `get_train_transform()`
Includes robust geometric and photometric augmentations for training:
- **Affine transformations**: Scaling, Rotation, Shear
- **ColorJitter**: Color variations
- **HorizontalFlip**: Random horizontal flipping
- Uses `min_visibility=0.3` to filter out severely clipped bounding boxes

#### `get_val_transform()`
A minimal pipeline used for validation and inference:
- Only includes `ToTensorV2()` to format the data for the model
- Ensures fair, non-augmented evaluation
- No data augmentation applied

---

## 4. Dynamic Balanced Sampler 

This custom PyTorch Sampler is critical for addressing the common challenge of class and background imbalance in detection datasets. which is defined in `turbine_processing/sampler.py`

| Feature          | Description |
|------------------|-------------|
| **50/50 Split**  | Ensures each training epoch contains 50% wildlife images (positive samples) and 50% background images (negative samples). |
| **Class Balancing** | Within the positive samples, minority wildlife classes (rare species) are oversampled with replacement, while majority classes are sampled without replacement. This ensures equal exposure to all species each epoch. |
| **GPU Efficiency** | The `TurbineDataLoader` combines this sampler with `pin_memory=True` and multiple `num_workers` to keep the GPU fully utilized and avoid data-loading bottlenecks. |
