# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:16:31 2026

@author: Gail McConnell, SIPBS, University of Strathclyde
"""

import cv2
import numpy as np
import os

# ----------------------------
# Paths
# ----------------------------
input_path = r"C:\crop.tif"
output_folder = r"C:\noisy_outputs"
os.makedirs(output_folder, exist_ok=True)

# ----------------------------
# Load image
# ----------------------------
image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
if image is None:
    raise ValueError(f"Could not read image: {input_path}")

# Ensure 8-bit
if image.dtype != np.uint8:
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

img = image.astype(np.float32)

# ----------------------------
# Parameters (tuned down)
# ----------------------------
num_steps = 16
max_sigma = 140.0              # ↓ reduced from 220 → softer noise
min_signal_retention = 0.15    # ↑ keep more of original in final frame

np.random.seed(42)

h, w = img.shape[:2]
is_color = (img.ndim == 3)

# ----------------------------
# Generate sequence
# ----------------------------
for i in range(num_steps):
    alpha = i / (num_steps - 1)

    # Slightly gentler curve
    noise_strength = alpha ** 2.0
    sigma = noise_strength * max_sigma

    # Monochromatic noise
    mono_noise = np.random.normal(0, sigma, (h, w)).astype(np.float32)

    if is_color:
        mono_noise = np.repeat(mono_noise[:, :, np.newaxis], img.shape[2], axis=2)

    # Preserve more signal
    signal_retention = 1.0 - alpha * (1.0 - min_signal_retention)

    noisy = img * signal_retention + mono_noise

    noisy_8bit = np.clip(noisy, 0, 255).astype(np.uint8)

    output_path = os.path.join(output_folder, f"noisy_{i:02d}.tif")
    cv2.imwrite(output_path, noisy_8bit)

    print(f"Saved: {output_path} | sigma={sigma:.2f} | signal={signal_retention:.2f}")

print("Done.")