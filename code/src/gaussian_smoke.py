"""Minimal gsplat optimization used to validate the custom supervision path."""

from __future__ import annotations

import numpy as np
import torch
from gsplat.rendering import rasterization

from .supervision_losses import real_view_loss, synthetic_view_loss


def initialize_gaussians(points, colors, max_gaussians=30000, seed=42, device="cuda"):
    valid = np.isfinite(points).all(1) & np.isfinite(colors).all(1)
    points, colors = points[valid], colors[valid]
    rng = np.random.default_rng(seed)
    if len(points) > max_gaussians:
        chosen = rng.choice(len(points), max_gaussians, replace=False)
        points, colors = points[chosen], colors[chosen]
    radius = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    scale = max(float(np.percentile(radius, 75)) / 150.0, 1e-5)
    count = len(points)
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.as_tensor(points, dtype=torch.float32, device=device)),
        "log_scales": torch.nn.Parameter(torch.full((count, 3), np.log(scale), device=device)),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(count, 1)),
        "opacity_logits": torch.nn.Parameter(torch.full((count,), -1.5, device=device)),
        "color_logits": torch.nn.Parameter(torch.logit(torch.as_tensor(colors, dtype=torch.float32, device=device).clamp(.001, .999))),
    })


def render_gaussians(parameters, extrinsic, intrinsic, height, width):
    view = torch.eye(4, dtype=torch.float32, device=parameters["means"].device)[None]
    view[0, :3, :4] = extrinsic
    rendered, alpha, _ = rasterization(
        means=parameters["means"], quats=parameters["quats"],
        scales=torch.exp(parameters["log_scales"]),
        opacities=torch.sigmoid(parameters["opacity_logits"]),
        colors=torch.sigmoid(parameters["color_logits"]),
        viewmats=view, Ks=intrinsic[None], width=int(width), height=int(height),
        packed=False, render_mode="RGB",
    )
    return rendered.permute(0, 3, 1, 2), alpha.permute(0, 3, 1, 2)


def train_smoke(parameters, real_views, synthetic_views, steps=1000,
                synthetic_probability=0.25, synthetic_weight=0.25, seed=42):
    """Optimize a fixed-size Gaussian set; validates losses, not a full densifying 3DGS baseline."""
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam([
        {"params": [parameters["means"]], "lr": 1e-4},
        {"params": [parameters["log_scales"], parameters["quats"], parameters["opacity_logits"]], "lr": 5e-3},
        {"params": [parameters["color_logits"]], "lr": 2.5e-3},
    ])
    history = []
    device = parameters["means"].device
    for step in range(steps):
        use_synthetic = bool(synthetic_views) and rng.random() < synthetic_probability
        pool = synthetic_views if use_synthetic else real_views
        sample = pool[int(rng.integers(len(pool)))]
        rgb, alpha = render_gaussians(
            parameters,
            torch.as_tensor(sample["extrinsic"], dtype=torch.float32, device=device),
            torch.as_tensor(sample["intrinsic"], dtype=torch.float32, device=device),
            sample["height"], sample["width"],
        )
        target = torch.as_tensor(sample["rgb"], dtype=torch.float32, device=device).permute(2, 0, 1)[None]
        if use_synthetic:
            loss, parts = synthetic_view_loss(
                rgb, alpha, target,
                torch.as_tensor(sample["validity"], device=device)[None],
                torch.as_tensor(sample["confidence"], device=device)[None],
            )
            loss = synthetic_weight * loss
        else:
            loss, parts = real_view_loss(
                rgb, alpha, target,
                torch.as_tensor(sample["mask"], device=device)[None],
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            parameters["quats"].div_(parameters["quats"].norm(dim=-1, keepdim=True).clamp_min(1e-8))
        if step % 25 == 0 or step == steps - 1:
            history.append({
                "step": step, "kind": "synthetic" if use_synthetic else "real",
                "loss": float(loss.detach()),
                **{name: float(value.detach()) for name, value in parts.items()},
            })
    return history
