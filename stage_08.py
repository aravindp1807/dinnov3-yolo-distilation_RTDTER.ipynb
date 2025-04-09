# Stage 8: feat: implement process() for batch file conversion
# ==================================================

# def count_files(img_dir, lbl_dir):
#     imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
#     lbls = sorted([f for f in os.listdir(lbl_dir) if f.endswith(".txt")])

#     print(f"Images: {len(imgs)}")
#     print(f"Labels: {len(lbls)}")

#     missing = set(imgs) - set(l.replace(".txt", ".png") for l in lbls)
#     if missing:
#         print("❌ Images without labels:", list(missing)[:5])
#     else:
#         print("✅ All images have labels")

# print("\n🔍 TRAIN SET")
# count_files(
#     f"{ROOT}/images/train",
#     f"{ROOT}/labels/train"
# )

# print("\n🔍 VAL SET")
# count_files(
#     f"{ROOT}/images/val",
#     f"{ROOT}/labels/val"
# )