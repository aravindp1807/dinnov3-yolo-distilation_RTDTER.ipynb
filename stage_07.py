# Stage 7: refactor: optimize mask_to_yolo with vectorized operations
# ==================================================

# import os

# ROOT = "/kaggle/working/yolo_dataset"

# paths = [
#     "images/train",
#     "images/val",
#     "labels/train",
#     "labels/val",
#     "data.yaml"
# ]

# print("📂 Checking directory structure:\n")
# for p in paths:
#     full = os.path.join(ROOT, p)
#     print(f"{p}: {'✅ EXISTS' if os.path.exists(full) else '❌ MISSING'}")