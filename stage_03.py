# Stage 3: feat: add data loading utilities for NIfTI files
# ==================================================

# counts = {
#     "axis0": 0,
#     "axis1": 0,
#     "axis2": 0
# }

# for z in range(msk.shape[0]):
#     if np.sum(msk[z, :, :]) > 0:
#         counts["axis0"] += 1

# for z in range(msk.shape[1]):
#     if np.sum(msk[:, z, :]) > 0:
#         counts["axis1"] += 1

# for z in range(msk.shape[2]):
#     if np.sum(msk[:, :, z]) > 0:
#         counts["axis2"] += 1

# print(counts)