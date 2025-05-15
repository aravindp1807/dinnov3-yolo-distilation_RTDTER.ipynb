# Medical Image Segmentation to YOLO Format Converter & Distillation Pipeline

> A production-ready pipeline for converting 3D medical imaging data (NIfTI) into YOLO-format datasets and performing knowledge distillation with RT-DETR models using the Lightly framework.

---

## 📋 Overview

This project provides a complete workflow for preparing medical imaging datasets for object detection training and deploying efficient models via knowledge distillation. The pipeline handles:

- **NIfTI to YOLO conversion**: Transforms 3D volumetric segmentations (`.nii.gz`) into 2D YOLO-format annotations suitable for detection training
- **Configurable preprocessing**: Slice extraction, normalization, and train/val splitting via YAML configuration
- **Knowledge distillation**: Leverages `lightly_train` with RT-DETR wrappers to distill large teacher models into efficient student architectures

Designed for Kaggle/colab environments with GPU acceleration, the codebase is modular, reproducible, and ready for integration into MLOps pipelines.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **NIfTI Processing** | Reads `.nii.gz` volumes via `nibabel`, extracts 2D slices with configurable windowing |
| **Mask → YOLO Conversion** | `mask_to_yolo()` converts binary/multi-class segmentation masks to YOLO txt format (class, x_center, y_center, width, height normalized) |
| **Batch Pipeline** | `process(files, img_out, lbl_out)` orchestrates end-to-end conversion with progress tracking via `tqdm` |
| **YAML-Driven Config** | `YAMLConfig` from `src.core` manages paths, splits, class maps, and preprocessing params |
| **RT-DETR Distillation** | Integrates `lightly_train.model_wrappers.RTDETRModelWrapper` for teacher-student training |
| **Reproducible Splits** | Deterministic train/val/test splits using `random` with fixed seeds |
| **OpenCV Acceleration** | Uses `cv2` for fast image I/O, resizing, and normalization |

---

## 🛠 Installation

### Requirements
- Python 3.9+
- CUDA 11.7+ (for GPU distillation)
- 16 GB+ RAM recommended for 3D volume loading

### Quick Start
```bash
# Clone repository
git clone https://github.com/your-org/medical-yolo-distillation.git
cd medical-yolo-distillation

# Create environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`
```text
lightly-train>=0.3.0
nibabel>=5.2.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.65.0
```

> **Note**: `lightly_train` requires a valid license for distillation features. Install via `pip install lightly-train` or follow [Lightly docs](https://docs.lightly.ai/).

---

## 🚀 Usage

### 1. Configure the Pipeline
Edit `config.yaml` (loaded via `YAMLConfig`):

```yaml
# config.yaml
data:
  input_dir: "/kaggle/input/medical-segmentation/"
  image_output: "/kaggle/working/images/"
  label_output: "/kaggle/working/labels/"
  split_ratio: [0.8, 0.1, 0.1]  # train/val/test
  seed: 42

preprocessing:
  slice_axis: 2          # axial