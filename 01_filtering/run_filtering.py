
"""
01_filtering/run_filtering.py
Apply Gaussian and Median filters; save before/after images and a parameter table.
"""
from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from typing import Dict
import numpy as np
from skimage.filters import gaussian, median
from skimage.morphology import disk
from utils.common import load_std_images, to_gray, save_image, RunLog, ensure_dir

OUTDIR = os.path.join(os.path.dirname(__file__), "out")

def apply_filters(imgs: Dict[str, np.ndarray]) -> None:
    ensure_dir(OUTDIR)
    log = RunLog(headers=["image", "filter", "sigma_or_radius", "notes"], rows=[])
    for name, im in imgs.items():
        g = to_gray(im)
        # Gaussian with multiple sigma to show effect
        for sigma in [0.8, 2.0]:
            g_gauss = gaussian(g, sigma=sigma, channel_axis=None)
            save_image(os.path.join(OUTDIR, f"{name}_gaussian_sigma{sigma}.png"), g_gauss)
            log.add(name, "gaussian", sigma, "skimage.filters.gaussian")
        # Median with two radii
        for r in [1, 3]:
            g_med = median(g, footprint=disk(r))
            save_image(os.path.join(OUTDIR, f"{name}_median_r{r}.png"), g_med)
            log.add(name, "median", r, "skimage.filters.median with disk footprint")
        # Save original for before/after comparison
        save_image(os.path.join(OUTDIR, f"{name}_original.png"), g)
    log.to_csv(os.path.join(OUTDIR, "filter_params.csv"))

if __name__ == "__main__":
    imgs = load_std_images({"pribadi":"images/futsal.jpg"})
    apply_filters(imgs)