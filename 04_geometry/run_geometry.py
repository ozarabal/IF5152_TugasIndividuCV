
"""
04_geometry/run_geometry.py
Simulate a simple camera/projective transform on a checkerboard and overlay results.
Also estimate a homography from four point correspondences to demonstrate parameter matrix.
"""
from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import numpy as np
from skimage.transform import ProjectiveTransform, warp, AffineTransform
from skimage.draw import polygon
from utils.common import load_std_images, save_image, to_gray, ensure_dir, RunLog
from skimage import data, img_as_float

OUTDIR = os.path.join(os.path.dirname(__file__), "out")

def simulate_projection():
    ensure_dir(OUTDIR)
    log = RunLog(headers=["transform", "matrix_3x3_flat"], rows=[])

    img = img_as_float(data.checkerboard())
    h, w = img.shape
    # Define corners of the source rectangle (image corners)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    # Define a plausible camera projection (keystone)
    dst = np.array([[50, 40], [w-30, 20], [w-10, h-20], [30, h-10]], dtype=float)

    # Estimate & apply projective transform
    tform = ProjectiveTransform()
    tform.estimate(src, dst)
    warped = warp(img, tform.inverse, output_shape=img.shape)

    save_image(os.path.join(OUTDIR, "checkerboard_original.png"), img)
    save_image(os.path.join(OUTDIR, "checkerboard_projected.png"), warped)
    log.add("ProjectiveTransform", list(tform.params.ravel()))

    # Also show a simple affine (scale+shear) for comparison
    aff = AffineTransform(scale=(0.9, 1.1), shear=0.1, translation=(20, -10))
    aff_img = warp(img, aff.inverse, output_shape=img.shape)
    save_image(os.path.join(OUTDIR, "checkerboard_affine.png"), aff_img)
    log.add("AffineTransform", list(np.pad(aff.params, ((0,0),(0,0))).ravel()))

    # Save matrix parameters
    log.to_csv(os.path.join(OUTDIR, "transform_params.csv"))

if __name__ == "__main__":
    simulate_projection()