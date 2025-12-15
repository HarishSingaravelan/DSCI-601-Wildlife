# 📈 Model Evaluation and Analysis

The final set of scripts in the `inference/` directory is used for rigorous evaluation of the trained model (`.pth` file) on the dedicated test set.

## 1. Mean Average Precision (mAP) Generator

The `mAPGenerator.py` script executes the standard object detection evaluation pipeline, including IoU matching, to produce the quantitative mAP metrics (e.g., mAP₅₀, mAP₇₅) required for scientific reporting.

**Script:** `python inference/mAPGenerator.py`

### Command Options

| Command | Description |
|---------|-------------|
| **Basic Evaluation (Required)** | Runs mAP calculation using parameters defined in the configuration file. |
| **Custom Batch/Workers** | Overrides the batch size and number of workers for environments with more or less memory/cores. |
| **Custom Run Name** | Assigns a unique name to the evaluation run, useful for logging or output files. |

### Usage Examples

```bash
# Basic evaluation on TEST set
python inference/mAPGenerator.py \
    --model_path fasterrcnn_turbine_adam_map.pth \
    --config config/config.yaml

# With custom batch size and workers
python inference/mAPGenerator.py \
    --model_path fasterrcnn_turbine_adam_map.pth \
    --config config/config.yaml \
    --batch_size 8 \
    --num_workers 8

# With custom run name
python inference/mAPGenerator.py \
    --model_path fasterrcnn_turbine_adam_map.pth \
    --config config/config.yaml \
    --run_name "final_test_baseline"
```

### Argument Descriptions

| Argument | Description | Example Value |
|----------|-------------|---------------|
| `--model_path` | Path to the trained model file (.pth) | `fasterrcnn_turbine_adam_map.pth` |
| `--config` | Path to the configuration YAML file | `config/config.yaml` |
| `--batch_size` | (Optional) Number of images to process per batch | `8` |
| `--num_workers` | (Optional) Number of worker threads for data loading | `8` |
| `--run_name` | (Optional) Custom name for the evaluation run | `"final_test_baseline"` |

## 2. Confusion Matrix Generator

The `confusionMatrixGenerator.py` script utilizes the saved model and test data to perform IoU-based matching, generating the True/Predicted label pairs necessary to visualize misclassification errors and confusion patterns between different classes.

**Script:** `python inference/confusionMatrixGenerator.py`

### Command Purpose

| Command | Purpose |
|---------|---------|
| **Run Confusion Matrix** | Executes the pipeline to generate the Normalized Confusion Matrix plot (saved as an image file) and the detailed Classification Report for per-class analysis. |

### Usage Example

```bash
# Run confusion matrix
python inference/confusionMatrixGenerator.py
```

## 📊 Output Files

The evaluation scripts typically generate the following outputs:

- **mAP Metrics:** Console output and/or log files containing mAP scores at various IoU thresholds
- **Confusion Matrix:** PNG/JPG image file showing the normalized confusion matrix
- **Classification Report:** Text or JSON file with per-class precision, recall, and F1-score

## 📝 Notes

- Ensure your configuration file (`config/config.yaml`) contains the correct paths to your test dataset
- The model file (`.pth`) should be from a completed training run
- Sufficient GPU/CPU memory is required for batch processing; adjust `--batch_size` if needed
- Results are typically saved in an `outputs/` or `results/` directory