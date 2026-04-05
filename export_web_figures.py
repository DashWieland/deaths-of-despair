"""
Export publication figures as web-optimized PNGs.

Resizes the three fig_pub_*.png files to max 1400px wide (preserving aspect
ratio) and saves with maximum PNG compression to web/.
"""

import os
from pathlib import Path
from PIL import Image

FIGURES = [
    "fig_pub_bivariate.png",
    "fig_pub_trajectory.png",
    "fig_pub_regimes.png",
]

MAX_WIDTH = 1400
OUT_DIR = Path("web")
OUT_DIR.mkdir(exist_ok=True)

for fname in FIGURES:
    src = Path(fname)
    dst = OUT_DIR / fname

    img = Image.open(src).convert("RGBA")
    w, h = img.size

    if w > MAX_WIDTH:
        new_h = round(h * MAX_WIDTH / w)
        img = img.resize((MAX_WIDTH, new_h), Image.LANCZOS)

    img.save(dst, format="PNG", optimize=True, compress_level=9)

    src_kb = src.stat().st_size // 1024
    dst_kb = dst.stat().st_size // 1024
    print(f"{fname}: {w}x{h}px → {img.size[0]}x{img.size[1]}px  "
          f"{src_kb}KB → {dst_kb}KB")

print(f"\nWeb figures written to {OUT_DIR.resolve()}/")
