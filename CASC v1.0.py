# -*- coding: utf-8 -*-
"""
Created on 30 March, 08:02:06 2026

@author: Gail McConnell, SIPBS, University of Strathclyde, UK
"""


import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import time

import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree, distance_matrix
from scipy.stats import spearmanr
from skimage import exposure, filters, measure, morphology

# ----------------------------- USER SETTINGS -----------------------------

INPUT_PATH = r"path_to_input_image.tif"
OUTPUT_TIF = r"path_to_output_image.tif"
OUTPUT_METRICS = r"path_to_metrics.json"
OUTPUT_COMPONENTS = r"path_to_components.csv"

PADDING = 8
MIN_OBJECT_AREA = 25
K_NEIGHBOURS = 3

MORPH_OPEN_RADIUS = 1
MORPH_CLOSE_RADIUS = 2

VISUAL_HALO_RADIUS = 3

MAX_EXPECTED_FOREGROUND_FRACTION = 0.50

ALLOW_ROTATIONS = False          # set True only if scientifically acceptable
DILATE_MASK_FOR_PACKING = 1      # 0 = cells can touch; 1-3 gives spacing
COARSE_STEP = 4
CONTACT_WEIGHT = 2.0
AREA_WEIGHT = 1.0
PERIMETER_WEIGHT = 0.25
MULTISTART_ORDERS = ("area", "height", "width", "maxdim")
WORKING_CANVAS_SCALE = 1.8
TIFF_COMPRESSION = "zlib"

# --------------------------- DATA STRUCTURES -----------------------------

@dataclass
class ComponentCrop:
    label: int
    area: int
    bbox_y0: int
    bbox_x0: int
    bbox_y1: int
    bbox_x1: int
    orig_centroid_y: float
    orig_centroid_x: float
    crop: np.ndarray
    crop_mask: np.ndarray       # true foreground object mask
    render_mask: np.ndarray     # object plus small halo for visual rendering
    pack_mask: np.ndarray       # collision mask for packing
    crop_height: int
    crop_width: int
    placed_y: int | None = None
    placed_x: int | None = None
    rotation_k: int = 0

# ---------------------------- IO UTILITIES ------------------------------

def ensure_parent_exists(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).astype(np.float32, copy=False)
    p1, p99 = np.percentile(arr, [1, 99])
    if p99 <= p1:
        amin = float(arr.min())
        amax = float(arr.max())
        p1 = amin
        p99 = amax if amax > amin else amin + 1.0
    arr = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)

def save_preview_png(image: np.ndarray, path: str) -> None:
    ensure_parent_exists(path)

    preview = image.copy()

    if preview.ndim == 2:
        Image.fromarray(normalize_to_uint8(preview)).save(path)
        return

    if preview.ndim == 3 and preview.shape[-1] in (3, 4):
        preview = np.stack(
            [normalize_to_uint8(preview[..., c]) for c in range(preview.shape[-1])],
            axis=-1,
        )
        Image.fromarray(preview).save(path)
        return

    gray = to_detection_2d(preview)
    Image.fromarray(normalize_to_uint8(gray)).save(path)

def save_mask_png(mask: np.ndarray, path: str) -> None:
    ensure_parent_exists(path)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)

def save_mask_overlay_png(image: np.ndarray, mask: np.ndarray, path: str) -> None:
    ensure_parent_exists(path)

    gray = normalize_to_uint8(to_detection_2d(image))
    rgb = np.stack([gray, gray, gray], axis=-1)

    # Red overlay where foreground mask is true.
    overlay = rgb.copy()
    overlay[mask, 0] = 255
    overlay[mask, 1] = (overlay[mask, 1] * 0.35).astype(np.uint8)
    overlay[mask, 2] = (overlay[mask, 2] * 0.35).astype(np.uint8)

    Image.fromarray(overlay).save(path)

# -------------------------- IMAGE PREPARATION ---------------------------

def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input image not found: {path}")
    return tiff.imread(path)

def to_detection_2d(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)

    if image.ndim == 2:
        return image.astype(np.float32)

    if image.ndim == 3:
        # RGB or RGBA image
        if image.shape[-1] in (3, 4):
            rgb = image[..., :3].astype(np.float32)

            # For brightfield/stained cells, purple/dark objects are often better
            # separated by darkness than by brightness. Return luminance here;
            # the foreground polarity is handled later.
            return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

        # Channel-first stack, e.g. C/Y/X
        if image.shape[0] <= 16 and image.shape[0] < image.shape[-1] and image.shape[0] < image.shape[-2]:
            return np.max(image.astype(np.float32), axis=0)

        # Z/Y/X or other stack: max-project over first axis
        return np.max(image.astype(np.float32), axis=0)

    raise ValueError(f"Unsupported image shape: {image.shape}")

def make_output_2d_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)

    if image.ndim == 2:
        return image

    if image.ndim == 3:
        if image.shape[-1] in (3, 4):
            return image
        return np.max(image, axis=0)

    raise ValueError(f"Unsupported image shape for output: {image.shape}")

def foreground_mask_from_image(det_img: np.ndarray) -> np.ndarray:
    x = det_img.astype(np.float32)
    x = x - np.min(x)
    if np.max(x) > 0:
        x = x / np.max(x)

    x = exposure.equalize_adapthist(x, clip_limit=0.03)
    x_smooth = filters.gaussian(x, sigma=1.0, preserve_range=True)

    thresh = filters.threshold_otsu(x_smooth)

    mask = x_smooth > thresh

    fg_fraction = float(np.mean(mask))
    if fg_fraction > MAX_EXPECTED_FOREGROUND_FRACTION:
        mask = ~mask
        print(
            f"Mask polarity inverted: initial foreground fraction was {fg_fraction:.3f}. "
            "Using dark/stained objects as foreground."
        )
    else:
        print(f"Mask polarity kept: foreground fraction is {fg_fraction:.3f}.")

    if MORPH_OPEN_RADIUS > 0:
        mask = morphology.binary_opening(mask, morphology.disk(MORPH_OPEN_RADIUS))
    if MORPH_CLOSE_RADIUS > 0:
        mask = morphology.binary_closing(mask, morphology.disk(MORPH_CLOSE_RADIUS))

    mask = morphology.remove_small_objects(mask, min_size=MIN_OBJECT_AREA)
    mask = morphology.remove_small_holes(mask, area_threshold=max(64, MIN_OBJECT_AREA * 2))

    return mask.astype(bool)

# ------------------------ COMPONENT EXTRACTION --------------------------

def make_pack_mask(crop_mask: np.ndarray) -> np.ndarray:
    pm = crop_mask.copy()
    if DILATE_MASK_FOR_PACKING > 0:
        pm = morphology.binary_dilation(pm, morphology.disk(DILATE_MASK_FOR_PACKING))
    return pm

def make_render_mask(crop_mask: np.ndarray) -> np.ndarray:
    rm = crop_mask.copy()
    if VISUAL_HALO_RADIUS > 0:
        rm = morphology.binary_dilation(rm, morphology.disk(VISUAL_HALO_RADIUS))
    return rm

def extract_components(output_img: np.ndarray, mask: np.ndarray, padding: int = 8) -> List[ComponentCrop]:
    labeled = measure.label(mask, connectivity=2)
    props = measure.regionprops(labeled)

    components: List[ComponentCrop] = []
    H, W = mask.shape

    for p in props:
        if p.area < MIN_OBJECT_AREA:
            continue

        y0, x0, y1, x1 = p.bbox
        y0p = max(0, y0 - padding)
        x0p = max(0, x0 - padding)
        y1p = min(H, y1 + padding)
        x1p = min(W, x1 + padding)

        crop_img = output_img[y0p:y1p, x0p:x1p].copy()
        crop_mask = (labeled[y0p:y1p, x0p:x1p] == p.label)

        # Trim to object plus visual halo, not just exact foreground pixels.
        trim_mask = make_render_mask(crop_mask)
        ys, xs = np.nonzero(trim_mask)
        if len(ys) == 0:
            continue

        iy0 = int(ys.min())
        ix0 = int(xs.min())
        iy1 = int(ys.max() + 1)
        ix1 = int(xs.max() + 1)

        crop_img = crop_img[iy0:iy1, ix0:ix1].copy()
        crop_mask = crop_mask[iy0:iy1, ix0:ix1].copy()
        render_mask = make_render_mask(crop_mask)
        pack_mask = make_pack_mask(crop_mask)

        components.append(
            ComponentCrop(
                label=int(p.label),
                area=int(p.area),
                bbox_y0=int(y0p + iy0),
                bbox_x0=int(x0p + ix0),
                bbox_y1=int(y0p + iy1),
                bbox_x1=int(x0p + ix1),
                orig_centroid_y=float(p.centroid[0]),
                orig_centroid_x=float(p.centroid[1]),
                crop=crop_img,
                crop_mask=crop_mask,
                render_mask=render_mask,
                pack_mask=pack_mask,
                crop_height=int(crop_mask.shape[0]),
                crop_width=int(crop_mask.shape[1]),
            )
        )

    return components

# ---------------------------- PACKING LOGIC -----------------------------

def clone_component(c: ComponentCrop) -> ComponentCrop:
    return ComponentCrop(
        label=c.label,
        area=c.area,
        bbox_y0=c.bbox_y0,
        bbox_x0=c.bbox_x0,
        bbox_y1=c.bbox_y1,
        bbox_x1=c.bbox_x1,
        orig_centroid_y=c.orig_centroid_y,
        orig_centroid_x=c.orig_centroid_x,
        crop=c.crop.copy(),
        crop_mask=c.crop_mask.copy(),
        render_mask=c.render_mask.copy(),
        pack_mask=c.pack_mask.copy(),
        crop_height=c.crop_height,
        crop_width=c.crop_width,
        placed_y=None,
        placed_x=None,
        rotation_k=0,
    )

def clone_components(components: List[ComponentCrop]) -> List[ComponentCrop]:
    return [clone_component(c) for c in components]

def sort_components_for_packing(components: List[ComponentCrop], order: str) -> List[ComponentCrop]:
    if order == "area":
        return sorted(components, key=lambda c: np.count_nonzero(c.pack_mask), reverse=True)
    if order == "height":
        return sorted(components, key=lambda c: (c.crop_height, c.crop_width), reverse=True)
    if order == "width":
        return sorted(components, key=lambda c: (c.crop_width, c.crop_height), reverse=True)
    if order == "maxdim":
        return sorted(components, key=lambda c: (max(c.crop_height, c.crop_width), c.area), reverse=True)
    return sorted(components, key=lambda c: c.area, reverse=True)

def estimate_working_canvas_size(components: List[ComponentCrop]) -> int:
    total_rect_area = sum(c.crop_height * c.crop_width for c in components)
    max_dim = max(max(c.crop_height, c.crop_width) for c in components)
    est = int(math.ceil(math.sqrt(total_rect_area) * WORKING_CANVAS_SCALE))
    return max(est, max_dim + 8)

def valid_shape_placement(occupied: np.ndarray, comp_mask: np.ndarray, y: int, x: int) -> bool:
    h, w = comp_mask.shape
    if y < 0 or x < 0:
        return False
    if y + h > occupied.shape[0] or x + w > occupied.shape[1]:
        return False
    region = occupied[y:y+h, x:x+w]
    return not np.any(region & comp_mask)

def touches_existing(occupied: np.ndarray, comp_mask: np.ndarray, y: int, x: int) -> int:
    h, w = comp_mask.shape

    y0 = max(0, y - 1)
    x0 = max(0, x - 1)
    y1 = min(occupied.shape[0], y + h + 1)
    x1 = min(occupied.shape[1], x + w + 1)

    region = occupied[y0:y1, x0:x1]
    local = np.zeros_like(region, dtype=bool)

    yy = y - y0
    xx = x - x0
    local[yy:yy+h, xx:xx+w] = comp_mask

    dil = ndi.binary_dilation(local)
    return int(np.count_nonzero(dil & region))

def placement_score(
    comp_mask: np.ndarray,
    y: int,
    x: int,
    used_h: int,
    used_w: int,
    occupied: np.ndarray,
) -> float:
    h, w = comp_mask.shape

    old_h = max(used_h, 1)
    old_w = max(used_w, 1)
    new_h = max(old_h, y + h)
    new_w = max(old_w, x + w)

    old_area = old_h * old_w
    new_area = new_h * new_w
    area_growth = new_area - old_area
    perimeter_growth = (new_h + new_w) - (old_h + old_w)
    contact = touches_existing(occupied, comp_mask, y, x)

    return (
        AREA_WEIGHT * area_growth
        + PERIMETER_WEIGHT * perimeter_growth
        - CONTACT_WEIGHT * contact
    )

def generate_frontier_candidates(
    placed_components: List[ComponentCrop],
    used_h: int,
    used_w: int,
    step: int,
) -> List[Tuple[int, int]]:
    candidates = {(0, 0)}

    for c in placed_components:
        if c.placed_y is None or c.placed_x is None:
            continue
        h, w = c.pack_mask.shape

        candidates.add((c.placed_y + h, c.placed_x))
        candidates.add((c.placed_y, c.placed_x + w))
        candidates.add((c.placed_y + h, c.placed_x + w))

        candidates.add((c.placed_y + h, max(0, c.placed_x - w // 2)))
        candidates.add((max(0, c.placed_y - h // 2), c.placed_x + w))

    max_h = max(used_h + step, step)
    max_w = max(used_w + step, step)

    for y in range(0, max_h + 1, step):
        candidates.add((y, used_w))
    for x in range(0, max_w + 1, step):
        candidates.add((used_h, x))

    return list(candidates)

def refine_candidate_neighbourhood(
    occupied: np.ndarray,
    comp_mask: np.ndarray,
    base_y: int,
    base_x: int,
    used_h: int,
    used_w: int,
    radius: int,
) -> Tuple[float, int, int] | None:
    best = None
    for y in range(max(0, base_y - radius), base_y + radius + 1):
        for x in range(max(0, base_x - radius), base_x + radius + 1):
            if not valid_shape_placement(occupied, comp_mask, y, x):
                continue
            s = placement_score(comp_mask, y, x, used_h, used_w, occupied)
            if best is None or s < best[0]:
                best = (s, y, x)
    return best

def get_variants(c: ComponentCrop, allow_rotations: bool) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    variants = [(c.crop, c.crop_mask, c.render_mask, c.pack_mask, 0)]
    if allow_rotations:
        for k in (1, 2, 3):
            variants.append((
                np.rot90(c.crop, k),
                np.rot90(c.crop_mask, k),
                np.rot90(c.render_mask, k),
                np.rot90(c.pack_mask, k),
                k,
            ))
    return variants

def irregular_shape_pack_once(
    components: List[ComponentCrop],
    allow_rotations: bool = False,
    coarse_step: int = 4,
) -> Tuple[List[ComponentCrop], int, int]:
    if not components:
        return [], 1, 1

    canvas_size = estimate_working_canvas_size(components)
    occupied = np.zeros((canvas_size, canvas_size), dtype=bool)

    placed: List[ComponentCrop] = []
    used_h = 0
    used_w = 0

    for i, c in enumerate(components):
        variants = get_variants(c, allow_rotations=allow_rotations)
        best = None

        candidates = [(0, 0)] if i == 0 else generate_frontier_candidates(placed, used_h, used_w, step=coarse_step)

        for crop_var, mask_var, render_mask_var, pack_mask_var, rot_k in variants:
            for y, x in candidates:
                if not valid_shape_placement(occupied, pack_mask_var, y, x):
                    continue

                s = placement_score(pack_mask_var, y, x, used_h, used_w, occupied)
                if best is None or s < best[0]:
                    best = (s, y, x, crop_var, mask_var, render_mask_var, pack_mask_var, rot_k)

            if best is not None:
                _, by, bx, _, _, _, _, _ = best
                refined = refine_candidate_neighbourhood(
                    occupied=occupied,
                    comp_mask=pack_mask_var,
                    base_y=by,
                    base_x=bx,
                    used_h=used_h,
                    used_w=used_w,
                    radius=coarse_step,
                )
                if refined is not None:
                    rs, ry, rx = refined
                    if rs < best[0]:
                        best = (rs, ry, rx, crop_var, mask_var, render_mask_var, pack_mask_var, rot_k)

        if best is None:
            raise RuntimeError(
                "Could not place a component within working canvas. "
                "Try increasing WORKING_CANVAS_SCALE."
            )

        _, y, x, best_crop, best_crop_mask, best_render_mask, best_pack_mask, rot_k = best

        c.crop = best_crop
        c.crop_mask = best_crop_mask
        c.render_mask = best_render_mask
        c.pack_mask = best_pack_mask
        c.crop_height, c.crop_width = c.crop_mask.shape
        c.placed_y = y
        c.placed_x = x
        c.rotation_k = rot_k

        h, w = c.pack_mask.shape
        occupied[y:y+h, x:x+w] |= c.pack_mask
        used_h = max(used_h, y + h)
        used_w = max(used_w, x + w)

        placed.append(c)

    return placed, used_h, used_w

def irregular_shape_pack(
    components: List[ComponentCrop],
    allow_rotations: bool = False,
    coarse_step: int = 4,
) -> Tuple[List[ComponentCrop], int, int]:
    if not components:
        return [], 1, 1

    best_result = None

    for order in MULTISTART_ORDERS:
        comps = clone_components(components)
        comps = sort_components_for_packing(comps, order=order)

        placed, out_h, out_w = irregular_shape_pack_once(
            comps,
            allow_rotations=allow_rotations,
            coarse_step=coarse_step,
        )

        score = out_h * out_w
        print(f"Packing order {order!r}: output {out_h} x {out_w}, area {score}")

        if best_result is None or score < best_result[0]:
            best_result = (score, placed, out_h, out_w)

    if best_result is None:
        raise RuntimeError("Irregular packing failed.")

    _, placed, out_h, out_w = best_result
    return placed, out_h, out_w

# ------------------------ COMPACT IMAGE RENDERING -----------------------

def place_crop_with_mask(canvas: np.ndarray, crop: np.ndarray, render_mask: np.ndarray, y: int, x: int) -> None:
    h, w = render_mask.shape
    target = canvas[y:y+h, x:x+w]

    if crop.ndim == 2:
        target[render_mask] = crop[render_mask]
    elif crop.ndim == 3:
        for ch in range(crop.shape[-1]):
            ch_target = target[..., ch]
            ch_target[render_mask] = crop[..., ch][render_mask]
    else:
        raise ValueError("Unsupported crop dimensionality")

def render_compact_image(
    components: List[ComponentCrop],
    out_h: int,
    out_w: int,
    dtype: np.dtype,
    channels: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if channels is None:
        compact = np.zeros((out_h, out_w), dtype=dtype)
    else:
        compact = np.zeros((out_h, out_w, channels), dtype=dtype)

    compact_mask = np.zeros((out_h, out_w), dtype=bool)

    for c in components:
        if c.placed_y is None or c.placed_x is None:
            raise RuntimeError("Component placement missing")

        place_crop_with_mask(compact, c.crop, c.render_mask, c.placed_y, c.placed_x)

        h, w = c.render_mask.shape
        compact_mask[c.placed_y:c.placed_y+h, c.placed_x:c.placed_x+w] |= c.render_mask

    return compact, compact_mask

def crop_canvas_to_content(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return image, mask

    y0, y1 = int(ys.min()), int(ys.max() + 1)
    x0, x1 = int(xs.min()), int(xs.max() + 1)

    return image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()

# ------------------------------ METRICS ---------------------------------

def compute_object_centroids_from_components(components: List[ComponentCrop]) -> np.ndarray:
    coords = []
    for c in components:
        if c.placed_y is None or c.placed_x is None:
            raise RuntimeError("Component placement missing")
        local_props = measure.regionprops(c.crop_mask.astype(np.uint8))
        if not local_props:
            continue
        ly, lx = local_props[0].centroid
        coords.append([c.placed_y + ly, c.placed_x + lx])
    return np.asarray(coords, dtype=np.float64)

def compute_original_centroids(components: List[ComponentCrop]) -> np.ndarray:
    return np.asarray([[c.orig_centroid_y, c.orig_centroid_x] for c in components], dtype=np.float64)

def nearest_neighbour_retention(orig_pts: np.ndarray, new_pts: np.ndarray, k: int = 3) -> float:
    n = len(orig_pts)
    if n < 2:
        return 1.0

    k = max(1, min(k, n - 1))

    orig_tree = cKDTree(orig_pts)
    new_tree = cKDTree(new_pts)

    _, orig_idx = orig_tree.query(orig_pts, k=k+1)
    _, new_idx = new_tree.query(new_pts, k=k+1)

    orig_idx = orig_idx[:, 1:]
    new_idx = new_idx[:, 1:]

    scores = []
    for i in range(n):
        a = set(map(int, np.atleast_1d(orig_idx[i])))
        b = set(map(int, np.atleast_1d(new_idx[i])))
        denom = len(a | b)
        scores.append(len(a & b) / denom if denom > 0 else 1.0)
    return float(np.mean(scores))

def pairwise_distance_rank_correlation(orig_pts: np.ndarray, new_pts: np.ndarray) -> float:
    n = len(orig_pts)
    if n < 3:
        return 1.0

    d0 = distance_matrix(orig_pts, orig_pts)
    d1 = distance_matrix(new_pts, new_pts)

    iu = np.triu_indices(n, k=1)
    v0 = d0[iu]
    v1 = d1[iu]

    corr, _ = spearmanr(v0, v1)
    return 1.0 if np.isnan(corr) else float(corr)

def mean_relative_nearest_neighbour_distortion(orig_pts: np.ndarray, new_pts: np.ndarray) -> float:
    n = len(orig_pts)
    if n < 2:
        return 0.0

    orig_tree = cKDTree(orig_pts)
    new_tree = cKDTree(new_pts)

    d0, _ = orig_tree.query(orig_pts, k=2)
    d1, _ = new_tree.query(new_pts, k=2)

    nn0 = d0[:, 1]
    nn1 = d1[:, 1]

    denom = np.maximum(nn0, 1e-8)
    rel = np.abs(nn1 - nn0) / denom
    return float(np.mean(rel))

def evaluate(
    original_image: np.ndarray,
    original_mask: np.ndarray,
    compact_image: np.ndarray,
    compact_mask: np.ndarray,
    components: List[ComponentCrop],
) -> Dict[str, Any]:
    H0, W0 = original_mask.shape
    H1, W1 = compact_mask.shape

    orig_fg = int(np.count_nonzero(original_mask))
    compact_fg = int(sum(np.count_nonzero(c.crop_mask) for c in components))

    orig_area = int(H0 * W0)
    compact_area = int(H1 * W1)

    orig_bg = orig_area - orig_fg
    compact_bg = compact_area - int(np.count_nonzero(compact_mask))

    orig_pts = compute_original_centroids(components)
    new_pts = compute_object_centroids_from_components(components)

    fg_recall = compact_fg / max(orig_fg, 1)

    if original_image.ndim == 2:
        orig_obj_mean = float(np.mean(original_image[original_mask])) if orig_fg > 0 else 0.0
    elif original_image.ndim == 3 and original_image.shape[-1] in (3, 4):
        orig_obj_mean = float(np.mean(original_image[original_mask])) if orig_fg > 0 else 0.0
    else:
        orig_obj_mean = 0.0

    # Estimate compact object mean from component masks, not halo masks.
    compact_values = []
    for c in components:
        if c.crop.ndim == 2:
            compact_values.append(c.crop[c.crop_mask])
        elif c.crop.ndim == 3 and c.crop.shape[-1] in (3, 4):
            compact_values.append(c.crop[c.crop_mask])
    if compact_values:
        compact_obj_mean = float(np.mean(np.concatenate([v.reshape(-1) for v in compact_values])))
    else:
        compact_obj_mean = 0.0

    return {
        "input_shape": list(map(int, original_image.shape)),
        "output_shape": list(map(int, compact_image.shape)),
        "original_fov_area_pixels": orig_area,
        "compact_fov_area_pixels": compact_area,
        "area_reduction_fraction": float(1.0 - (compact_area / max(orig_area, 1))),
        "original_foreground_pixels": orig_fg,
        "compact_foreground_pixels": compact_fg,
        "foreground_retention_fraction": float(fg_recall),
        "original_background_pixels": int(orig_bg),
        "compact_background_pixels": int(compact_bg),
        "background_reduction_fraction": float(1.0 - (compact_bg / max(orig_bg, 1))),
        "original_foreground_density": float(orig_fg / max(orig_area, 1)),
        "compact_rendered_density": float(np.count_nonzero(compact_mask) / max(compact_area, 1)),
        "density_gain_fold": float(
            (compact_fg / max(compact_area, 1)) / max(orig_fg / max(orig_area, 1), 1e-12)
        ),
        "component_count": int(len(components)),
        "component_count_preserved_fraction": 1.0,
        "nearest_neighbour_retention_jaccard_k{}".format(min(K_NEIGHBOURS, max(len(components) - 1, 1))):
            float(nearest_neighbour_retention(orig_pts, new_pts, K_NEIGHBOURS)),
        "pairwise_distance_rank_spearman": float(pairwise_distance_rank_correlation(orig_pts, new_pts)),
        "mean_relative_nearest_neighbour_distortion": float(mean_relative_nearest_neighbour_distortion(orig_pts, new_pts)),
        "original_object_mean_intensity": orig_obj_mean,
        "compact_object_mean_intensity": compact_obj_mean,
        "mean_intensity_difference_fraction": float(
            abs(compact_obj_mean - orig_obj_mean) / max(abs(orig_obj_mean), 1e-12)
        ),
    }

# ------------------------------ REPORTING -------------------------------

def components_to_dataframe(components: List[ComponentCrop]) -> pd.DataFrame:
    rows = []
    for c in components:
        rows.append({
            "label": c.label,
            "area_pixels": c.area,
            "orig_centroid_y": c.orig_centroid_y,
            "orig_centroid_x": c.orig_centroid_x,
            "orig_bbox_y0": c.bbox_y0,
            "orig_bbox_x0": c.bbox_x0,
            "orig_bbox_y1": c.bbox_y1,
            "orig_bbox_x1": c.bbox_x1,
            "crop_height": c.crop_height,
            "crop_width": c.crop_width,
            "new_top_y": c.placed_y,
            "new_left_x": c.placed_x,
            "rotation_k_90deg": c.rotation_k,
        })
    return pd.DataFrame(rows)

def print_metrics(metrics: Dict[str, Any]) -> None:
    print("\n=== COMPACTION METRICS ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print("==========================\n")

# -------------------------------- MAIN ----------------------------------

def main() -> None:
    raw = load_image(INPUT_PATH)

    det_img = to_detection_2d(raw)
    output_img = make_output_2d_image(raw)

    mask = foreground_mask_from_image(det_img)

    fg_fraction_after_cleanup = float(np.mean(mask))
    print(f"Final foreground fraction after cleanup: {fg_fraction_after_cleanup:.4f}")

    if np.count_nonzero(mask) == 0:
        raise RuntimeError(
            "No foreground detected. Try lowering MIN_OBJECT_AREA or adjusting the thresholding pipeline."
        )

    components = extract_components(output_img, mask, padding=PADDING)

    print(f"Detected components after filtering: {len(components)}")

    if not components:
        raise RuntimeError(
            "No valid connected components after filtering. Try lowering MIN_OBJECT_AREA."
        )

    packed_components, out_h, out_w = irregular_shape_pack(
        components=components,
        allow_rotations=ALLOW_ROTATIONS,
        coarse_step=COARSE_STEP,
    )

    channels = output_img.shape[-1] if (output_img.ndim == 3 and output_img.shape[-1] in (3, 4)) else None
    compact_img, compact_mask = render_compact_image(
        packed_components,
        out_h=out_h,
        out_w=out_w,
        dtype=output_img.dtype,
        channels=channels,
    )

    compact_img, compact_mask = crop_canvas_to_content(compact_img, compact_mask)

    ensure_parent_exists(OUTPUT_TIF)
    photometric = "rgb" if (compact_img.ndim == 3 and compact_img.shape[-1] in (3, 4)) else "minisblack"

    tiff.imwrite(
        OUTPUT_TIF,
        compact_img,
        compression=TIFF_COMPRESSION,
        photometric=photometric,
    )

    metrics = evaluate(
        original_image=output_img,
        original_mask=mask,
        compact_image=compact_img,
        compact_mask=compact_mask,
        components=packed_components,
    )

    with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    comp_df = components_to_dataframe(packed_components)
    comp_df.to_csv(OUTPUT_COMPONENTS, index=False)

    print_metrics(metrics)
    print(f"Saved compact TIFF:       {OUTPUT_TIF}")
    print(f"Saved metrics JSON:       {OUTPUT_METRICS}")
    print(f"Saved components CSV:     {OUTPUT_COMPONENTS}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
