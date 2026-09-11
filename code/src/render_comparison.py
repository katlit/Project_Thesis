"""Shared metrics and presentation helpers for reconstruction comparisons."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

from .supervision_losses import ssim_map


def foreground_metrics(prediction, target, foreground, opacity, threshold=0.5):
    """Measure RGB fit on known foreground and silhouette agreement."""
    prediction = np.clip(np.asarray(prediction, dtype=np.float32), 0, 1)
    target = np.clip(np.asarray(target, dtype=np.float32), 0, 1)
    foreground = np.asarray(foreground, dtype=bool)
    opacity = np.asarray(opacity, dtype=np.float32).squeeze()
    if not foreground.any():
        raise ValueError("Foreground mask is empty.")
    difference = prediction[foreground] - target[foreground]
    mse = float(np.mean(difference ** 2))
    prediction_tensor = torch.from_numpy(prediction).permute(2, 0, 1)[None]
    target_tensor = torch.from_numpy(target).permute(2, 0, 1)[None]
    score_map = ssim_map(prediction_tensor, target_tensor)[0, 0].numpy()
    predicted_mask = opacity >= threshold
    intersection = np.logical_and(predicted_mask, foreground).sum()
    union = np.logical_or(predicted_mask, foreground).sum()
    return {
        "masked_mae": float(np.mean(np.abs(difference))),
        "masked_psnr": float(-10 * np.log10(max(mse, 1e-10))),
        "masked_ssim": float(score_map[foreground].mean()),
        "silhouette_iou": float(intersection / max(union, 1)),
    }


def labelled_pair(left, right, left_label, right_label, separator=8):
    """Join two uint8 RGB frames with permanent labels for an MP4."""
    left = np.asarray(left, dtype=np.uint8)
    right = np.asarray(right, dtype=np.uint8)
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + separator + right.shape[1]
    canvas = Image.new("RGB", (width, height + 30), "white")
    canvas.paste(Image.fromarray(left), (0, 30))
    canvas.paste(Image.fromarray(right), (left.shape[1] + separator, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), left_label, fill="black")
    draw.text((left.shape[1] + separator + 8, 8), right_label, fill="black")
    return np.asarray(canvas)
