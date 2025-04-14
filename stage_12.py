# Stage 12: feat: integrate lightly_train RTDETRModelWrapper
# ==================================================

from src.core import YAMLConfig

import lightly_train
from lightly_train.model_wrappers import RTDETRModelWrapper
config = YAMLConfig("configs/rtdetr/rtdetr_r50vd_6x_coco.yml")
model = config.model