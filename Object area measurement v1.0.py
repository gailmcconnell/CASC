# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:29:08 2026

@author: Gail McConnell, SIPBS, University of Strathclyde
"""

import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, segmentation, feature
from scipy import ndimage as ndi
from scipy.stats import ks_2samp


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# -------------------------
# User settings
# -------------------------
original_path = r"C:\Users\acp98112\OneDrive - University of Strathclyde\Documents\Research\Compress\Object area measurement\HeLa crop.tif (red) contadj.tif"
compacted_path = r"C:\Users\acp98112\OneDrive - University of Strathclyde\Documents\Research\Compress\Object area measurement\output.tif"

pixel_size_um = 0.224
min_area_px = 20

# -------------------------
# Output directory
# -------------------------
output_dir = os.path.dirname(original_path)

histogram_png = os.path.join(output_dir, "object_area_histogram_logx_arial.png")
histogram_svg = os.path.join(output_dir, "object_area_histogram_logx_arial.svg")
histogram_pdf = os.path.join(output_dir, "object_area_histogram_logx_arial.pdf")

cdf_png = os.path.join(output_dir, "object_area_ecdf_arial.png")
cdf_svg = os.path.join(output_dir, "object_area_ecdf_arial.svg")
cdf_pdf = os.path.join(output_dir, "object_area_ecdf_arial.pdf")

# -------------------------
# Segmentation function
# -------------------------
def segment_objects(img, min_area_px=20):
    if img.ndim == 3:
        img = img[..., 0]

    img = img.astype(float)

    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    else:
        raise ValueError("Image has no intensity variation.")

    threshold = filters.threshold_otsu(img)
    mask = img > threshold

    mask = morphology.remove_small_objects(mask, min_size=min_area_px)
    mask = ndi.binary_fill_holes(mask)

    distance = ndi.distance_transform_edt(mask)

    coords = feature.peak_local_max(
        distance,
        labels=mask,
        min_distance=5
    )

    markers = np.zeros_like(mask, dtype=int)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    labels = segmentation.watershed(
        -distance,
        markers,
        mask=mask
    )

    return labels

# -------------------------
# Measure areas
# -------------------------
def measure_areas(labels, pixel_size_um=1.0):
    props = measure.regionprops_table(
        labels,
        properties=["label", "area"]
    )

    areas_px = np.array(props["area"])
    areas_um2 = areas_px * pixel_size_um**2

    return areas_px, areas_um2

# -------------------------
# Load images
# -------------------------
original = tifffile.imread(original_path)
compacted = tifffile.imread(compacted_path)

# -------------------------
# Segment objects
# -------------------------
labels_original = segment_objects(original, min_area_px)
labels_compacted = segment_objects(compacted, min_area_px)

# -------------------------
# Measure object areas
# -------------------------
areas_original_px, areas_original_um2 = measure_areas(
    labels_original,
    pixel_size_um
)

areas_compacted_px, areas_compacted_um2 = measure_areas(
    labels_compacted,
    pixel_size_um
)

# -------------------------
# Remove zero/negative values
# required for log plots
# -------------------------
areas_original_um2 = areas_original_um2[areas_original_um2 > 0]
areas_compacted_um2 = areas_compacted_um2[areas_compacted_um2 > 0]

all_areas = np.concatenate([
    areas_original_um2,
    areas_compacted_um2
])

# -------------------------
# Compare distributions
# -------------------------
ks_stat, ks_p = ks_2samp(
    areas_original_um2,
    areas_compacted_um2
)

count_retention = (
    100 * len(areas_compacted_um2) / len(areas_original_um2)
)

median_shift = (
    100
    * (
        np.median(areas_compacted_um2)
        - np.median(areas_original_um2)
    )
    / np.median(areas_original_um2)
)

print(f"Original object count: {len(areas_original_um2)}")
print(f"Compacted object count: {len(areas_compacted_um2)}")
print(f"Object count retention: {count_retention:.2f}%")
print(f"Median area original: {np.median(areas_original_um2):.2f} µm²")
print(f"Median area compacted: {np.median(areas_compacted_um2):.2f} µm²")
print(f"Median area shift: {median_shift:.2f}%")
print(f"KS statistic: {ks_stat:.4f}")
print(f"KS p-value: {ks_p:.4g}")

# -------------------------
# Histogram clipped at 99.5th percentile
# -------------------------
xmax = np.percentile(all_areas, 99.5)

areas_original_plot = areas_original_um2[
    areas_original_um2 <= xmax
]

areas_compacted_plot = areas_compacted_um2[
    areas_compacted_um2 <= xmax
]

all_areas_plot = np.concatenate([
    areas_original_plot,
    areas_compacted_plot
])

fig, ax = plt.subplots(figsize=(3.2, 2.4))

log_bins = np.logspace(
    np.log10(all_areas_plot.min()),
    np.log10(all_areas_plot.max()),
    60
)

ax.hist(
    areas_original_plot,
    bins=log_bins,
    alpha=0.5,
    density=True,
    label="Input image"
)

ax.hist(
    areas_compacted_plot,
    bins=log_bins,
    alpha=0.5,
    density=True,
    label="Compacted image"
)

ax.set_xscale("log")
ax.set_xlabel("Object area (µm²)")
ax.set_ylabel("Probability density")
ax.legend(frameon=True)

fig.tight_layout()

fig.savefig(histogram_png, dpi=600, bbox_inches="tight")
fig.savefig(histogram_svg, bbox_inches="tight")
fig.savefig(histogram_pdf, bbox_inches="tight")

plt.show()

print(f"Histogram clipped at 99.5th percentile ({xmax:.2f} µm²)")
print(f"Histogram saved to:\n{histogram_png}\n{histogram_svg}\n{histogram_pdf}")

# -------------------------
# ECDF plot
# -------------------------
fig, ax = plt.subplots(figsize=(3.2, 2.4))

for areas, label in [
    (areas_original_um2, "Input image"),
    (areas_compacted_um2, "Compacted image")
]:
    sorted_areas = np.sort(areas)
    cumulative = (
        np.arange(1, len(sorted_areas) + 1)
        / len(sorted_areas)
    )

    ax.plot(
        sorted_areas,
        cumulative,
        linewidth=1.2,
        label=label
    )

ax.set_xscale("log")
ax.set_xlabel("Object area (µm²)")
ax.set_ylabel("Cumulative fraction")
ax.legend(frameon=True)

fig.tight_layout()

fig.savefig(cdf_png, dpi=600, bbox_inches="tight")
fig.savefig(cdf_svg, bbox_inches="tight")
fig.savefig(cdf_pdf, bbox_inches="tight")

plt.show()

print(f"ECDF plot saved to:\n{cdf_png}\n{cdf_svg}\n{cdf_pdf}")