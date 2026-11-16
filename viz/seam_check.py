#!/usr/bin/env python3
"""Report the biggest frame-to-frame jumps of a draft video, next to the planned hard-cut times.

usage: seam_check.py <video.mp4> [trim_plan.json]   (needs ffmpeg on PATH; run on a compute node)
"""
import json
import subprocess
import sys

import numpy as np

vid = sys.argv[1]
plan = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
ORDER = ["G0_title", "G1_mode_orchestra", "G2_theta_desk", "G3_nearest_star", "G4_crystallise", "G5_frequency_brush", "F10_front_sweep",
         "G8_harmonic_peel", "G7_four_oceans", "G9_complexity_bloom", "G10_same_destination", "G12_galaxy_dolly", "G13_outro"]
X, FPS, w, h = 0.6, 24, 240, 135

raw = subprocess.run(["ffmpeg", "-v", "error", "-cpuflags", "0", "-i", vid, "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                     capture_output=True, check=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)
d = np.abs(fr[1:] - fr[:-1]).mean((1, 2))
print(f"{len(fr)} frames, median step diff {np.median(d):.3f}, 99th pct {np.percentile(d, 99):.3f}")

# planned hard cuts (output time)
cuts, L = [], 0.0
for k, name in enumerate(ORDER):
    segs = plan.get(name)
    if not segs:
        continue
    start = 0.0 if k == 0 else L
    t = start
    for i, (a, b, sp) in enumerate(segs):
        t += (b - a) / sp
        if i < len(segs) - 1:
            cuts.append((round(t, 2), name))
    # clip length in output
    pass
# recompute offsets properly
L = 0.0
cuts = []
for k, name in enumerate(ORDER):
    segs = plan.get(name, None)
    clip_len = sum((b - a) / sp for a, b, sp in segs) if segs else None
    if clip_len is None:
        continue
    base = 0.0 if k == 0 else L - X
    t = base
    for i, (a, b, sp) in enumerate(segs[:-1]):
        t += (b - a) / sp
        cuts.append((t, name))
    L = (clip_len if k == 0 else L + clip_len - X)
print("planned hard cuts (s):", [(round(c, 1), n.split('_')[0]) for c, n in cuts])
top = np.argsort(-d)[:10]
print("largest frame jumps:")
for i in sorted(top):
    print(f"  t={(i + 1) / FPS:6.2f}s  diff={d[i]:.2f}")
for c, n in cuts:
    i = int(round(c * FPS)) - 1
    lo, hi = max(0, i - 2), min(len(d), i + 3)
    print(f"cut {n:22s} at {c:6.2f}s: diff across cut {d[lo:hi].max():.2f}  (local median {np.median(d[max(0, i - 24):i + 24]):.2f})")
