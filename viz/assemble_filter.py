#!/usr/bin/env python3
"""Build the ffmpeg filter_complex for the showcase draft: per-clip trims/speed (viz/trim_plan.json), then
cross-fades between clips, then a dip to the background colour at both ends.

usage: assemble_filter.py <order, space separated> <durations, space separated> <xfade_s> [trim_plan.json|none]
prints two lines: the filter string, then the total length in seconds.
"""
import json
import sys

order = sys.argv[1].split()
durs = [float(x) for x in sys.argv[2].split()]
X = float(sys.argv[3])
plan = json.load(open(sys.argv[4])) if len(sys.argv) > 4 and sys.argv[4] != "none" else {}
FPS = 24

parts, labels, clip_len = [], [], []
for k, (name, d) in enumerate(zip(order, durs)):
    segs = plan.get(name, [[0, d, 1]])
    outs = []
    for i, (a, b, sp) in enumerate(segs):
        b = min(b, d)
        lab = f"c{k}s{i}"
        # speed != 1: blend neighbouring frames (framerate) instead of dropping them (fps), which stutters
        resample = f"framerate=fps={FPS},settb=1/{FPS}" if sp != 1 else f"fps={FPS}"
        parts.append(f"[{k}:v]trim=start={a}:end={b},setpts=(PTS-STARTPTS)/{sp},{resample}[{lab}]")
        outs.append(f"[{lab}]")
    if len(outs) == 1:
        labels.append(outs[0][1:-1])
    else:
        parts.append("".join(outs) + f"concat=n={len(outs)}:v=1:a=0,fps={FPS}[c{k}]")
        labels.append(f"c{k}")
    clip_len.append(sum((min(b, d) - a) / sp for a, b, sp in segs))

L = clip_len[0]
prev = labels[0]
for k in range(1, len(order)):
    off = L - X
    parts.append(f"[{prev}][{labels[k]}]xfade=transition=fade:duration={X}:offset={off:.3f}[x{k}]")
    prev = f"x{k}"
    L = L + clip_len[k] - X
fo = L - X
parts.append(f"[{prev}]fade=t=in:st=0:d={X}:color=0x05101A,fade=t=out:st={fo:.3f}:d={X}:color=0x05101A,format=yuv420p[out]")
print(";".join(parts))
print(f"{L:.3f}")
