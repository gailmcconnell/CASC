# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tifffile

img = tifffile.imread(r"C:\HeLa.tif")

tifffile.imwrite(
    r"C:\HeLa.jp2",
    img,
    compression="jpeg2000"
)

print("Done")