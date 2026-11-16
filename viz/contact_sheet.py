#!/usr/bin/env python3
"""Tile the first 12 PNGs found in a folder (f_*.png) into a 4x3 contact sheet.

usage: contact_sheet.py <frames_dir> <out.png>
"""
import glob
import sys

from PIL import Image

fs = sorted(glob.glob(sys.argv[1] + "/f_*.png"))[:12]
W, H = 320, 180
im = Image.new("RGB", (4 * W, 3 * H))
for i, f in enumerate(fs):
    im.paste(Image.open(f).convert("RGB"), ((i % 4) * W, (i // 4) * H))
im.save(sys.argv[2])
print(f"{len(fs)} frames -> {sys.argv[2]}")
