# Stage 6: feat: add mask_to_yolo conversion for segmentation masks
# ==================================================

import os
import cv2
import yaml
import random
import numpy as np
import nibabel as nib
from tqdm import tqdm

# ===============================
# INPUT PATHS
# ===============================
IMG_NII = "/kaggle/input/3d-ircad20/Dataset001_Liver/imagesTr"
MSK_NII = "/kaggle/input/3d-ircad20/Dataset001_Liver/labelsTr"

# ===============================
# OUTPUT PATHS
# ===============================
YOLO_ROOT = "/kaggle/working/yolo_dataset"
IMG_TRAIN = f"{YOLO_ROOT}/images/train"
IMG_VAL   = f"{YOLO_ROOT}/images/val"
LBL_TRAIN = f"{YOLO_ROOT}/labels/train"
LBL_VAL   = f"{YOLO_ROOT}/labels/val"

for p in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    os.makedirs(p, exist_ok=True)

# ===============================
# PARAMETERS
# ===============================
VAL_RATIO = 0.2
CLASS_ID = 0

# ===============================
# IMAGE FILES
# ===============================
image_files = sorted(
    [f for f in os.listdir(IMG_NII) if f.endswith(".nii")]
)

random.shuffle(image_files)

split_idx = int(len(image_files) * (1 - VAL_RATIO))
train_files = image_files[:split_idx]
val_files   = image_files[split_idx:]

print(f"Train patients: {len(train_files)} | Val patients: {len(val_files)}")

# ===============================
# MASK → YOLO SEG
# ===============================
def mask_to_yolo(mask):
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = []

    for cnt in contours:
        if len(cnt) < 3:
            continue

        cnt = cnt.squeeze().astype(float)
        cnt[:, 0] /= w
        cnt[:, 1] /= h

        labels.append(
            " ".join([str(CLASS_ID)] + [f"{v:.6f}" for v in cnt.flatten()])
        )

    return labels

# ===============================
# PROCESS
# ===============================
def process(files, img_out, lbl_out):
    for img_fname in tqdm(files):
        # liver_000_0000.nii → 000
        patient_id = img_fname.split("_")[1]

        img_path  = os.path.join(IMG_NII, img_fname)
        mask_path = os.path.join(MSK_NII, f"liver_{patient_id}.nii")

        img = nib.load(img_path).get_fdata()
        msk = nib.load(mask_path).get_fdata()

        # mask already 0/1
        msk = msk.astype(np.uint8)

        # ✅ CORRECT AXIS: Z = axis 2
        for z in range(img.shape[2]):
            mask_slice = msk[:, :, z]
            if mask_slice.sum() == 0:
                continue

            img_slice = img[:, :, z]
            img_slice = cv2.normalize(
                img_slice, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            img_name = f"liver_{patient_id}_z{z}.png"
            lbl_name = img_name.replace(".png", ".txt")

            cv2.imwrite(os.path.join(img_out, img_name), img_slice)

            labels = mask_to_yolo(mask_slice * 255)
            with open(os.path.join(lbl_out, lbl_name), "w") as f:
                for line in labels:
                    f.write(line + "\n")

# ===============================
# RUN
# ===============================
process(train_files, IMG_TRAIN, LBL_TRAIN)
process(val_files, IMG_VAL, LBL_VAL)

# ===============================
# WRITE data.yaml
# ===============================
yaml_data = {
    "path": YOLO_ROOT,
    "train": "images/train",
    "val": "images/val",
    "names": {0: "organ"}
}

with open(f"{YOLO_ROOT}/data.yaml", "w") as f:
    yaml.dump(yaml_data, f)

print("✅ YOLO dataset created successfully")