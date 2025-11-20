# 🧪 Testing Guide

This project includes a set of automated tests located inside the `Tests/` directory. All tests use pytest, and each test file validates one part of the pipeline such as:
* coco_dataset_generator
* Dataset generation
* Model initialization
* Training loop


## 📁 Directory Structure

```
Tests/
├── test_coco_dataset_generator.py
├── test_dataloader.py
├── test_model.py
└── test_training.py
```

Each file contains independent unit tests that can be run separately.

## 🚀 Running All Tests

From the project root directory:

```bash
pytest
```

Or with more detailed output:

```bash
pytest -vv
```

## 🎯 Running a Specific Test File

You can run any one test file individually:

### Example — run sampler tests

```bash
pytest Tests/test_sampler.py
```

### Example — run COCO dataset generator tests

```bash
pytest Tests/test_coco_generator.py
```