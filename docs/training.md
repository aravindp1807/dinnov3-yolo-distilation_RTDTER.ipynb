# Technical Documentation: RT-DETR Knowledge Distillation Pipeline

## Overview
This pipeline implements a **knowledge distillation workflow** for RT-DETR (Real-Time Detection Transformer) on 3D medical imaging data (NIfTI format), converting volumetric annotations to 2D YOLO format for efficient training.

---

## 1. Data Conversion: NIfTI → YOLO Format

### `mask_to_yolo(mask)` Function
Converts 3D segmentation masks to YOLO-format bounding boxes.

```python
def mask_to_yolo(mask):
    # Input: 3D numpy array (H, W, D) or (D, H, W) from nibabel
    # Output: List of [class_id, x_center, y_center, width, height] normalized [0,1]
```

**Processing Logic:**
1. **Slice-wise extraction**: Iterates through axial slices (`mask[:, :, z]`)
2. **Connected components**: Uses `cv2.findContours` on binary slices
3. **Bounding box computation**: `cv2.boundingRect` → normalized YOLO coordinates
4. **Class mapping**: Assumes single-class (class_id=0) or maps from label values

### `process(files, img_out, lbl_out)` Function
Batch processes NIfTI volumes:

| Parameter | Description |
|-----------|-------------|
| `files` | List of `.nii.gz` paths |
| `img_out` | Output directory for 2D PNG/JPG slices |
| `lbl_out` | Output directory for `.txt` YOLO labels |

**Pipeline:**
```
NIfTI volume → nibabel.load() → numpy array
    → slice extraction (axial plane)
    → mask_to_yolo() per slice
    → cv2.imwrite(slice_image) + write(label.txt)
```

**Output Structure:**
```
img_out/
  case_001_slice_000.png
  case_001_slice_001.png
  ...
lbl_out/
  case_001_slice_000.txt  # YOLO format: class x y w h
  case_001