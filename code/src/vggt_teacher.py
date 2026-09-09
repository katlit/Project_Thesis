"""Dense confidence-aware pseudo-view teacher built from VGGT geometry."""

from __future__ import annotations

import numpy as np
import torch
from gsplat.rendering import rasterization


def build_teacher_gaussians(points, colors, confidence, max_points=150_000,
                            scale_divisor=350.0, scale_multiplier=2.5,
                            opacity=0.95, seed=42, device="cuda"):
    points = np.asarray(points).reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3)
    confidence = np.asarray(confidence).reshape(-1)
    valid = np.isfinite(points).all(1) & np.isfinite(colors).all(1) & np.isfinite(confidence)
    valid &= confidence > 0
    points, colors, confidence = points[valid], colors[valid], confidence[valid]
    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        probability = np.maximum(confidence, 0.02)
        probability /= probability.sum()
        chosen = rng.choice(len(points), max_points, replace=False, p=probability)
        points, colors, confidence = points[chosen], colors[chosen], confidence[chosen]
    center = np.median(points, axis=0)
    scene_radius = max(float(np.percentile(np.linalg.norm(points - center, axis=1), 90)), 1e-4)
    scale = scene_radius / scale_divisor * scale_multiplier
    count = len(points)
    feature = np.concatenate([np.clip(colors, 0, 1), confidence[:, None]], axis=1)
    return {
        "means": torch.as_tensor(points, dtype=torch.float32, device=device),
        "scales": torch.full((count, 3), float(scale), device=device),
        "quats": torch.tensor([1., 0., 0., 0.], device=device).repeat(count, 1),
        "opacities": torch.full((count,), float(opacity), device=device),
        "features": torch.as_tensor(feature, dtype=torch.float32, device=device),
        "scene_radius": scene_radius,
    }


@torch.inference_mode()
def render_teacher(teacher, extrinsic, intrinsic, height, width,
                   alpha_min=0.55, confidence_min=0.10):
    device = teacher["means"].device
    view = torch.eye(4, dtype=torch.float32, device=device)[None]
    view[0, :3, :4] = torch.as_tensor(extrinsic, dtype=torch.float32, device=device)
    intrinsic = torch.as_tensor(intrinsic, dtype=torch.float32, device=device)[None]
    features, alpha, _ = rasterization(
        means=teacher["means"], quats=teacher["quats"], scales=teacher["scales"],
        opacities=teacher["opacities"], colors=teacher["features"],
        viewmats=view, Ks=intrinsic, width=int(width), height=int(height),
        sh_degree=None, packed=False, render_mode="RGB",
    )
    features = features[0].float().cpu().numpy()
    alpha = alpha[0, ..., 0].float().cpu().numpy()
    normalized = features / np.maximum(alpha[..., None], 1e-6)
    rgb = np.clip(normalized[..., :3], 0, 1)
    confidence = np.clip(normalized[..., 3], 0, 1)
    validity = (alpha >= alpha_min) & (confidence >= confidence_min)
    return rgb, validity, confidence, alpha


def coverage_summary(validity, confidence, alpha):
    validity = np.asarray(validity, dtype=bool)
    return {
        "valid_fraction": float(validity.mean()),
        "mean_valid_confidence": float(np.asarray(confidence)[validity].mean()) if validity.any() else np.nan,
        "mean_valid_alpha": float(np.asarray(alpha)[validity].mean()) if validity.any() else np.nan,
    }
