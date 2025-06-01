# Technical Documentation: RT-DETR Distillation Pipeline

## Environment Setup

### Base Environment
This pipeline runs on the **Kaggle Python Docker image** (`kaggle/python`) which provides pre-installed analytics libraries. For local deployment, replicate the environment using:

```bash
# Core dependencies
pip install lightly-train nibabel opencv-python pandas pyyaml tqdm

# PyTorch (version compatible with lightly_train)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Directory Structure
```
project/
├── src/
│   └── core.py              # Contains YAMLConfig class
├── data/
│   ├── images/              # Input NIfTI/DICOM files
│   ├── labels/              # Segmentation masks
│   └── yolo/                # Converted YOLO format output
│       ├── images/
│       └── labels/
└── config.yaml              # Training configuration
```

## Dependencies

| Package | Purpose | Version Constraint |
|---------|---------|-------------------|
| `lightly_train` | RT-DETR wrapper & distillation framework | Latest |
| `nibabel` | Medical imaging I/O (NIfTI support) | ≥5.0 |
| `opencv-python` | Image preprocessing & resizing | ≥4.8 |
| `pandas` | CSV metadata processing | ≥2.0 |
| `pyyaml` | Configuration management | ≥6.0 |
| `tqdm` | Progress tracking | ≥4.65 |

## API Usage

### Configuration (`YAMLConfig`)
```python
from src.core import YAMLConfig

config = YAMLConfig("config.yaml")
# Expected keys: model_path, data_yaml, epochs, batch_size, distill_params
```

### Model Wrapper (`RTDETRModelWrapper`)
```python
from lightly_train.model_wrappers import RTDETRModelWrapper

model = RTDETRModelWrapper(config.model_path)
model.distill(
    train_data=config.data_yaml,
    epochs=config.epochs,
   