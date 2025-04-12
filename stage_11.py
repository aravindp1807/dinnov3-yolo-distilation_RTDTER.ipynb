# Stage 11: chore: create dataset.yaml for YOLO format
# ==================================================

import os
output_path = "/kaggle/working/distilation_RTDTER_results"
os.makedirs(output_path, exist_ok=True)
print(f"{p}: {'✅ EXISTS' if os.path.exists(output_path) else '❌ MISSING'}")