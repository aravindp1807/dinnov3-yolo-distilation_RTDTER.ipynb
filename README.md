# 🏥 Medical Image Segmentation to YOLO Format Converter & RT-DETR Knowledge Distillation Pipeline

> A scalable medical imaging pipeline that converts volumetric NIfTI segmentation datasets into YOLO object detection format and performs teacher-student knowledge distillation using RT-DETR and Lightly Train for efficient deployment.

---

# 📖 Overview

Medical imaging datasets are commonly distributed as volumetric segmentation masks in **NIfTI (.nii/.nii.gz)** format. While these datasets are ideal for segmentation tasks, many modern deployment scenarios require lightweight object detection models.

This project bridges that gap by providing a complete end-to-end workflow that:

1. Converts 3D medical segmentation datasets into YOLO detection datasets.
2. Extracts and preprocesses 2D slices from volumetric scans.
3. Generates normalized YOLO bounding-box annotations from segmentation masks.
4. Creates reproducible train/validation/test splits.
5. Trains efficient RT-DETR student models through knowledge distillation.
6. Produces lightweight deployment-ready object detectors while retaining teacher-level performance.

The pipeline is designed for:

* Medical AI Research
* Dataset Engineering
* Model Compression
* Edge Deployment
* Clinical Computer Vision Applications

---

# 🚀 Key Features

## 🧠 NIfTI Volume Processing

Supports:

```text
.nii
.nii.gz
```

using:

* Nibabel
* NumPy
* OpenCV

for efficient volumetric image handling.

---

## 🩺 Medical Slice Extraction

Extracts 2D slices from:

* Axial Plane
* Coronal Plane
* Sagittal Plane

with configurable slice selection strategies.

---

## 🎯 Mask-to-YOLO Conversion

Automatically converts segmentation masks into YOLO object detection labels.

Generated format:

```text
class_id x_center y_center width height
```

All coordinates are normalized according to YOLO specifications.

---

## ⚙️ YAML Configuration System

The entire workflow is controlled via YAML configuration files.

Configurable options include:

* Dataset Paths
* Class Mapping
* Slice Axis
* Split Ratios
* Random Seeds
* Image Resolution
* Preprocessing Parameters

---

## 📦 Automated Dataset Generation

Produces a complete YOLO dataset structure:

```text
dataset/
├── images/
│   ├── train
│   ├── val
│   └── test
│
├── labels/
│   ├── train
│   ├── val
│   └── test
│
└── data.yaml
```

ready for training.

---

## 🚀 RT-DETR Knowledge Distillation

Uses:

* RT-DETR Teacher Models
* RT-DETR Student Models
* Lightly Train Framework

to transfer knowledge from larger models into compact architectures.

Benefits include:

* Faster Inference
* Reduced Memory Usage
* Lower Compute Cost
* Edge Deployment Readiness

---

## 🔄 Reproducible Data Splits

Deterministic dataset splitting using fixed seeds ensures:

* Experiment Reproducibility
* Consistent Benchmarking
* Reliable Validation

---

# 🏗 System Architecture

```text
Raw Medical Dataset (.nii.gz)
              │
              ▼
       NIfTI Loader
              │
              ▼
      Slice Extraction
              │
              ▼
    Segmentation Masks
              │
              ▼
      YOLO Conversion
              │
              ▼
      Dataset Builder
              │
              ▼
      Teacher Training
              │
              ▼
 Knowledge Distillation
              │
              ▼
     RT-DETR Student
              │
              ▼
 Deployment-Ready Model
```

---

# ⚙️ Technology Stack

| Component              | Technology     |
| ---------------------- | -------------- |
| Medical Imaging        | Nibabel        |
| Deep Learning          | PyTorch        |
| Object Detection       | RT-DETR        |
| Distillation           | Lightly Train  |
| Image Processing       | OpenCV         |
| Configuration          | YAML           |
| Data Handling          | Pandas         |
| Numerical Computing    | NumPy          |
| Progress Tracking      | tqdm           |
| Deployment Environment | Kaggle / Colab |

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/medical-yolo-distillation.git

cd medical-yolo-distillation
```

---

## Create Virtual Environment

```bash
python -m venv .venv

source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

```text
lightly-train>=0.3.0
nibabel>=5.2.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.65.0
torch>=2.0.0
torchvision>=0.15.0
```

---

# 🚀 Usage

## Step 1 — Configure Dataset

Edit:

```yaml
data:
  input_dir: "/kaggle/input/medical-segmentation"

  image_output: "/kaggle/working/images"

  label_output: "/kaggle/working/labels"

  split_ratio: [0.8, 0.1, 0.1]

  seed: 42
```

---

## Step 2 — Configure Preprocessing

```yaml
preprocessing:
  slice_axis: 2

  image_size: 640

  normalize: true

  remove_empty_masks: true
```

---

## Step 3 — Convert Dataset

```bash
python convert_dataset.py
```

Pipeline performs:

* Volume Loading
* Slice Extraction
* Bounding Box Generation
* YOLO Annotation Creation
* Dataset Splitting

---

## Step 4 — Verify YOLO Dataset

Generated structure:

```text
images/train
images/val
images/test

labels/train
labels/val
labels/test
```

---

## Step 5 — Train Teacher Model

```bash
yolo detect train \
    model=rtdetr-l.pt \
    data=data.yaml \
    epochs=100
```

---

## Step 6 — Knowledge Distillation

```bash
lightly-train \
    train.yaml
```

This transfers knowledge from:

```text
Teacher → RT-DETR-Large
Student → RT-DETR-Small
```

---

# 📂 Project Structure

```text
Medical-YOLO-Distillation/
│
├── configs/
│   ├── config.yaml
│   └── train.yaml
│
├── data/
│
├── src/
│   ├── converter/
│   ├── preprocessing/
│   ├── distillation/
│   └── core/
│
├── notebooks/
│   └── dino_yolo_distillation.ipynb
│
├── outputs/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🔬 Medical Conversion Workflow

### Load NIfTI Volume

```python
volume = nib.load(path).get_fdata()
```

---

### Extract Slice

```python
slice_img = volume[:, :, idx]
```

---

### Generate Bounding Boxes

```python
boxes = mask_to_yolo(mask)
```

---

### Save YOLO Labels

```text
0 0.51 0.48 0.18 0.27
1 0.63 0.52 0.14 0.21
```

---

# 🧠 Knowledge Distillation Workflow

### Teacher Network

```text
RT-DETR Large
```

High accuracy model.

---

### Student Network

```text
RT-DETR Small
```

Lightweight deployment model.

---

### Distillation Losses

Common objectives:

* Classification Distillation
* Feature Matching
* Bounding Box Regression
* Response Mimicking

---

# 🎯 Applications

## Medical Object Detection

* Tumor Detection
* Lesion Localization
* Organ Detection

## Radiology AI

* CT Scan Analysis
* MRI Analysis
* Ultrasound Detection

## Edge Healthcare Devices

Deploy lightweight models on:

* Jetson Nano
* Edge TPU
* Embedded GPUs

## Clinical Decision Support

Assist radiologists through automated detection pipelines.

---

# 📊 Advantages

| Benefit               | Description                    |
| --------------------- | ------------------------------ |
| Annotation Automation | Eliminates manual box creation |
| Dataset Conversion    | Segmentation → Detection       |
| Faster Training       | Optimized YOLO format          |
| Efficient Deployment  | Distilled RT-DETR models       |
| Reproducibility       | YAML-driven pipeline           |

---

# 🔮 Future Improvements

* 3D YOLO Support
* DINOv2 Feature Distillation
* Multi-Class Medical Detection
* ONNX Export
* TensorRT Optimization
* Federated Learning Support
* MONAI Integration
* Active Learning Pipeline

---

# 🤝 Contributing

Contributions are welcome.

1. Fork Repository
2. Create Feature Branch
3. Commit Changes
4. Push Changes
5. Open Pull Request

---

# 📜 License

Released under the MIT License.

---

# 👨‍💻 Author

**Aravind Pyli**

Machine Learning Researcher | Medical AI Engineer | Computer Vision Enthusiast

### Research Interests

* Medical Image Analysis
* Knowledge Distillation
* Computer Vision
* Object Detection
* Healthcare AI
* Foundation Models

---

⭐ If this repository helped your research or deployment workflow, consider giving it a star.
