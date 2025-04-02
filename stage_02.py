# Stage 2: docs: add README with 3D-IRCADb dataset description
# ==================================================

import nibabel as nib
import numpy as np
import os

IMG_NII = "/kaggle/input/3d-ircad20/Dataset001_Liver/imagesTr"
MSK_NII = "/kaggle/input/3d-ircad20/Dataset001_Liver/labelsTr"

img_file = sorted(os.listdir(IMG_NII))[0]
patient_id = img_file.split("_")[1]
mask_file = f"liver_{patient_id}.nii"

img = nib.load(os.path.join(IMG_NII, img_file)).get_fdata()
msk = nib.load(os.path.join(MSK_NII, mask_file)).get_fdata()

print("Image shape:", img.shape)
print("Mask shape :", msk.shape)
print("Mask unique values:", np.unique(msk))
print("Foreground voxels:", np.sum(msk > 0))