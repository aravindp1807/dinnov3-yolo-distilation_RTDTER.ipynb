# Stage 4: feat: implement NIfTI to 2D slice extraction
# ==================================================

# import cv2

# OUT = "/kaggle/working/debug_slices"
# os.makedirs(OUT, exist_ok=True)

# saved = 0

# for z in range(msk.shape[0]):  # CHANGE AXIS IF NEEDED
#     if np.sum(msk[z]) > 0:
#         img_slice = img[z]
#         img_slice = cv2.normalize(img_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         cv2.imwrite(f"{OUT}/slice_{z}.png", img_slice)
#         saved += 1

#     if saved == 5:
#         break

# print("Saved slices:", saved)