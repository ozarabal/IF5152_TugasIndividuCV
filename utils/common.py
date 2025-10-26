
"""
utils/common.py
Utility functions shared across modules: loading images, saving outputs, and small helpers.
All functions have clear docstrings and are import-safe.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import numpy as np
from skimage import data, img_as_float, color, io as skio
from skimage.transform import rescale

DATASET_NAMES = ["camera", "coins", "checkerboard", "astronaut"]

def load_std_images(personal_images: dict[str, str] | None = None) -> Dict[str, np.ndarray]:
    """Load standard images from skimage.data as float in [0,1].
    Returns a dict mapping name->image ndarray.
    """
    imgs = {
        "camera": img_as_float(data.camera()),
        "coins": img_as_float(data.coins()),
        "checkerboard": img_as_float(data.checkerboard()),
        "astronaut": img_as_float(data.astronaut()),
    }

    if personal_images:
        for name, path in personal_images.items():
            try:
                imgs[name] = img_as_float(skio.imread(path))
                print(f"[INFO] Loaded personal image '{name}' from {path}")
            except Exception as e:
                print(f"[WARN] Cannot load {path}: {e}")
    return imgs

def to_gray(img: np.ndarray) -> np.ndarray:
    """Ensure image is grayscale float in [0,1]."""
    if img.ndim == 3:
        return color.rgb2gray(img)
    return img

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_image(path: str, image: np.ndarray) -> None:
    """Save image to path, auto-creating parent dir."""
    ensure_dir(os.path.dirname(path))
    # Convert to uint8 for portability
    skio.imsave(path, (np.clip(image, 0, 1) * 255).astype(np.uint8))

def percentile_threshold(img: np.ndarray, p: float) -> float:
    """Return intensity threshold at percentile p (0-100)."""
    return float(np.percentile(img, p))

@dataclass
class RunLog:
    """Collects parameter rows to be written as CSV later."""
    headers: List[str]
    rows: List[List[Any]]

    def add(self, *values: Any) -> None:
        self.rows.append(list(values))

    def to_csv(self, path: str) -> None:
        ensure_dir(os.path.dirname(path))
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            for r in self.rows:
                writer.writerow(r)
