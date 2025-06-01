# Technical Documentation: Medical Imaging Data Conversion & Distillation Pipeline

## System Overview

This pipeline implements a **medical imaging data preprocessing and model distillation workflow** designed for object detection tasks using RT-DETR (Real-Time DEtection TRansformer). The system converts 3D NIfTI medical volumes with segmentation masks into 2D YOLO-format training data, enabling knowledge distillation from a teacher model to a student detector.

---

## Architecture Components

### 1. Data Ingestion Layer (`nibabel`, `cv2`, `numpy`)
- **NIfTI Parser**: Loads 3D volumetric medical scans (`.nii/.nii.gz`) and corresponding segmentation masks
- **Slice Extraction**: Converts 3D volumes into 2D axial/coronal/sagittal slices via NumPy array slicing
- **Image Normalization**: Applies windowing/normalization (HU range clipping, z-score) for consistent intensity distributions

### 2. Annotation Conversion Module (`mask_to_yolo`)
```python
def mask_to_yolo(mask):  # Converts binary segmentation masks → YOLO normalized bbox format
```
- **Input**: 2D binary mask (H×W) per class
- **Processing**: Connected component analysis → bounding box extraction (x_min, y_min, x_max, y_max)
- **Output**: YOLO format `<class_id> <x_center> <y_center> <width> <height>` (normalized 0–1)

### 3. Dataset Organization (`process`)
```python
def process(files, img_out, lbl_out):  # Batch processes file lists → structured YOLO dataset
```
- **Splits**: Train/val/test directory structure creation
- **Pairing**: Matches image slices with corresponding label files
- **Manifest Generation**: Creates `data.yaml` with class names, paths, and split ratios

### 4. Configuration Management (`YAMLConfig`, `yaml`)
-