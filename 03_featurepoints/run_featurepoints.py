
"""
03_featurepoints/run_featurepoints.py
Detect Harris corners and ORB keypoints; mark on images; write statistics CSV.
"""
from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import numpy as np
from skimage.feature import corner_harris, corner_peaks, ORB
from skimage.color import gray2rgb
from skimage.draw import circle_perimeter
from utils.common import load_std_images, to_gray, save_image, RunLog, ensure_dir

OUTDIR = os.path.join(os.path.dirname(__file__), "out")

def draw_points(img_gray: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Return RGB image with red circles on given coordinates."""
    vis = gray2rgb(img_gray)
    h, w = img_gray.shape
    for (r, c) in coords:
        rr, cc = circle_perimeter(int(r), int(c), 3, shape=img_gray.shape)
        vis[rr, cc] = (1.0, 0.0, 0.0)
    return vis

def detect_and_save():
    ensure_dir(OUTDIR)
    log = RunLog(headers=["image", "method", "num_features", "params"], rows=[])
    imgs = load_std_images({"pribadi":"images/futsal.jpg"})
    for name, im in imgs.items():
        g = to_gray(im)
        # Harris
        response = corner_harris(g, method='k', k=0.05, sigma=1.2)
        coords = corner_peaks(response, min_distance=3, threshold_rel=0.02)
        vis_harris = draw_points(g, coords)
        save_image(os.path.join(OUTDIR, f"{name}_harris.png"), vis_harris)
        log.add(name, "harris", len(coords), "k=0.05, sigma=1.2, min_dist=3, thr=0.02")
        # ORB keypoints
        orb = ORB(n_keypoints=300, fast_threshold=0.08)
        orb.detect_and_extract(g)
        kps = orb.keypoints if orb.keypoints is not None else np.empty((0,2))
        vis_orb = draw_points(g, kps)
        save_image(os.path.join(OUTDIR, f"{name}_orb.png"), vis_orb)
        log.add(name, "orb", len(kps), "n_keypoints=300, fast_threshold=0.08")
        # Save original
        save_image(os.path.join(OUTDIR, f"{name}_original.png"), g)
    log.to_csv(os.path.join(OUTDIR, "feature_stats.csv"))

if __name__ == "__main__":
    detect_and_save()