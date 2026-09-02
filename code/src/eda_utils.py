from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def masked_image_statistics(image_path, mask_path, sample_size=(256, 256), threshold=0.5):
    """Resolution-normalized descriptive statistics; source files are unchanged."""
    with Image.open(image_path) as opened:
        width, height = opened.size
        image = np.asarray(opened.convert("RGB").resize(sample_size, Image.Resampling.BILINEAR), dtype=np.float32)
    with Image.open(mask_path) as opened:
        mask = np.asarray(opened.convert("L").resize(sample_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    foreground = mask >= threshold
    if not foreground.any():
        raise ValueError(f"Empty foreground mask: {mask_path}")
    pixels = image[foreground]
    gray = image.mean(axis=2)
    gx, gy = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
    gx_ok, gy_ok = foreground[:, 1:] & foreground[:, :-1], foreground[1:, :] & foreground[:-1, :]
    edges = np.concatenate([gx[gx_ok], gy[gy_ok]])
    return {
        "width": width, "height": height, "aspect_ratio": width / height,
        "foreground_fraction": float(foreground.mean()),
        "brightness": float(pixels.mean(axis=1).mean()),
        "contrast": float(pixels.mean(axis=1).std()),
        "sharpness_proxy": float(edges.mean()),
        "red_mean": float(pixels[:, 0].mean()),
        "green_mean": float(pixels[:, 1].mean()),
        "blue_mean": float(pixels[:, 2].mean()),
    }

