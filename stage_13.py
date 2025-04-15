# Stage 13: fix: resolve YAMLConfig import from src.core
# ==================================================

import lightly_train

data_path = "/kaggle/working/yolo_dataset/images"
if __name__ == "__main__":
    wrapped_model = RTDETRModelWrapper(model)
    lightly_train.pretrain(
        out=output_path,
        data=data_path,
        model=wrapped_model,
        method="distillationv1",
        overwrite=True,
        method_args={
            "teacher": "dinov3/vits16",
            "teacher_url":"https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoidHZpMngydGx1cjN0eHNtMWd1YWdpYjF3IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0OTM0MDZ9fX1dfQ__&Signature=MojlptxwiEPfpltfODdWHPsRCUpErxgMhJzpeQb2RkXlb-w1GMbc8J-rFEY7X2PzAr0CmWpWhHuIHP%7E48a16P6cYkRMx2t%7ETrGb-WA7xw6paYYSoprXi8n5wQ7W1CmgfJwN7aLps1Ypg8qVyJgcG7DMoVVi1kgCpsj9wVNBEbQUlQm3vCmZzEq10wsDFHiL18XZthx7noWBKmC0NZvFFya5LmzpMHHKXh0ZRI0PVYEHRmWI8sC4Ia2AWRg8ZcyvC4zZr5KtocXawDLY3PQx6KBvT6rjVQahkD%7EXWNCXMzXLi5HRVLQYIAJgx8Cc%7EmTIm2NWVAwMSdo7YZjjmJFM5XQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=813015768081005"

        }
    )