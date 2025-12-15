# 🧪 Testing Guide

This project includes a set of automated tests located inside the `Tests/` directory. All tests use pytest and are designed to validate the core components of the wildlife detection pipeline, ensuring robustness and correctness before and during training.

## 📁 Directory Structure and Coverage

Our testing suite covers the entire pipeline, from data preparation to the training loop.

| File | Pipeline Component Tested | Key Assertions |
|------|---------------------------|----------------|
| `test_coco_dataset_generator.py` | COCO Data Preparation | Verifies correct structure, ID generation, and JSON format. |
| `test_dataloader.py` | PyTorch Data Loaders | Confirms correct batch size, target structure, and `collate_fn` output. |
| `test_model.py` | Model Initialization | Checks that `get_model()` returns a valid PyTorch detection module. |
| `test_sampler.py` | Dynamic Sampler Logic | Critical test for 50/50 background split and minority class oversampling. |
| `test_trainer.py` | Training Logic | Confirms `train_one_epoch` runs without crashing and calculates positive losses. |
| `test_sample_images.py` | Environment Check | Verifies that the `Sample_images` folder exists and is not empty (required for inference tests). |

## 🚀 Running All Tests

To execute the entire test suite, run the following command from the project root directory:

```bash
pytest
```

For more detailed output showing passed/failed tests and assertion values:

```bash
pytest -vv
```

## 🎯 Running a Specific Test File

You can target any individual test file to isolate failures or check specific component logic.

### Example 1 — Test the Dynamic Sampler Logic (Crucial for Imbalance)

```bash
pytest Tests/test_sampler.py
```

### Example 2 — Test the Training Loop and Trainer Class

```bash
pytest Tests/test_trainer.py
```

### Example 3 — Test the COCO Dataset Generator

```bash
pytest Tests/test_coco_dataset_generator.py
```

## 📝 Notes

- Ensure all dependencies are installed before running tests: `pip install pytest pytest-cov`
- Tests require the `Sample_images/` directory to be present for environment validation
- Some tests may require a valid configuration file at `config/config.yaml`
- If tests fail, check that your Python environment matches the project requirements
