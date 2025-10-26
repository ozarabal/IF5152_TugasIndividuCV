
"""
02_edge/run_edge.py
Compute Sobel and Canny edges; explore threshold & sampling; save maps and parameter CSV.
"""
from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import numpy as np
from skimage.filters import sobel
from skimage.feature import canny
from skimage.transform import rescale
from utils.common import load_std_images, to_gray, save_image, RunLog, ensure_dir

OUTDIR = os.path.join(os.path.dirname(__file__), "out")

def detect_edges():
    ensure_dir(OUTDIR)
    log = RunLog(headers=["image", "method", "param", "sampling", "notes"], rows=[])
    imgs = load_std_images({"pribadi":"images/futsal.jpg"})
    for name, im in imgs.items():
        g = to_gray(im)
        for s in [1.0, 0.5]:  # sampling scales
            g_s = g if s == 1.0 else rescale(g, s, channel_axis=None, anti_aliasing=True)
            # Sobel (no params)
            e_sobel = sobel(g_s)
            save_image(os.path.join(OUTDIR, f"{name}_sobel_s{s}.png"), e_sobel / e_sobel.max())
            log.add(name, "sobel", "-", s, "skimage.filters.sobel")
            # Canny with different high thresholds
            for high in [0.1, 0.2, 0.3]:
                e_canny = canny(g_s, sigma=1.2, low_threshold=high*0.5, high_threshold=high)
                save_image(os.path.join(OUTDIR, f"{name}_canny_h{high}_s{s}.png"), e_canny.astype(float))
                log.add(name, "canny", f"sigma=1.2, low={high*0.5:.2f}, high={high:.2f}", s, "skimage.feature.canny")
        # Save original for reference
        save_image(os.path.join(OUTDIR, f"{name}_original.png"), g)
    log.to_csv(os.path.join(OUTDIR, "edge_params.csv"))

if __name__ == "__main__":
    detect_edges()