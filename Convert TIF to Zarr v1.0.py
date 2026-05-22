# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:43:57 2026

@author: Gail McConnell, SIPBS, University of Strathclyde
"""

import tifffile
import zarr
from ome_zarr.writer import write_image
from ome_zarr.io import parse_url
from numcodecs import Blosc

# Read TIFF
img = tifffile.imread(r"C:\CASC\HeLa.tif")

# Output OME-Zarr location
store = parse_url(r"C:\CASC\HeLa.zarr", mode="w").store

# Create zarr group
root = zarr.group(store=store)

# Write OME-Zarr
write_image(
    image=img,
    group=root,
    axes="yx",   # change if needed
    storage_options=dict(
        chunks=(512, 512),
        compressor=Blosc(cname="zstd", clevel=5)
    )
)

print("Conversion complete.")