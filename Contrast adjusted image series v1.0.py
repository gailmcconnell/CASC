# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:10:16 2026

@author: Gail McConnell, SIPBS, University of Strathclyde
"""

import cv2
import numpy as np
import os

# ----------------------------
# Paths
# ----------------------------
input_path = r"C:\CASC\crop.tif"
input_dir = os.path.dirname(input_path)
output_folder = os.path.join(input_dir, "reduced_contrast_outputs")
os.makedirs(output_folder, exist_ok=True)

# ----------------------------
# Load image
# ----------------------------
image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
if image is None:
    raise ValueError(f"Could not read image: {input_path}")

# Ensure 8-bit workflow
if image.dtype != np.uint8:
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

img = image.astype(np.float32)

# ----------------------------
# Parameters
# ----------------------------
num_steps = 16

# Final output range should be 0 to 1 counts
final_min = 0.0
final_max = 1.0

# Original image min/max
if img.ndim == 2:
    orig_min = np.min(img)
    orig_max = np.max(img)
else:
    orig_min = np.min(img, axis=(0, 1), keepdims=True)
    orig_max = np.max(img, axis=(0, 1), keepdims=True)

if np.any(orig_max == orig_min):
    raise ValueError("Image has no intensity range to compress.")

# ----------------------------
# Generate reduced-contrast sequence
# ----------------------------
for i in range(num_steps):
    alpha = i / (num_steps - 1)

    # Target output range shrinks smoothly:
    # step 0: original range
    # step 15: range [0, 1]
    target_min = orig_min * (1.0 - alpha) + final_min * alpha
    target_max = orig_max * (1.0 - alpha) + final_max * alpha

    # Linearly remap original intensities into the shrinking target range
    reduced = target_min + (img - orig_min) * (target_max - target_min) / (orig_max - orig_min)

    # Clip and convert to uint8 for saving
    reduced_8bit = np.clip(reduced, 0, 255).astype(np.uint8)

    output_path = os.path.join(output_folder, f"contrast_{i:02d}.tif")
    cv2.imwrite(output_path, reduced_8bit)

    current_min = reduced_8bit.min()
    current_max = reduced_8bit.max()
    print(f"Saved: {output_path} | output range: [{current_min}, {current_max}]")

print("Done.")