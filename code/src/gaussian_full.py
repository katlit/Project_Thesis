"""Adaptive gsplat trainer for real + confidence-weighted synthetic supervision."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from gsplat import DefaultStrategy
from gsplat.rendering import rasterization

from .supervision_losses import real_view_loss, synthetic_view_loss


SH_C0 = 0.28209479177387814


def initialize_full_gaussians(points, colors, max_initial=50_000, sh_degree=3,
                              seed=42, device="cuda"):
    valid = np.isfinite(points).all(1) & np.isfinite(colors).all(1)
    points, colors = np.asarray(points)[valid], np.asarray(colors)[valid]
    rng = np.random.default_rng(seed)
    if len(points) > max_initial:
        chosen = rng.choice(len(points), max_initial, replace=False)
        points, colors = points[chosen], colors[chosen]
    center = np.median(points, axis=0)
    scene_scale = max(float(np.percentile(np.linalg.norm(points - center, axis=1), 90)), 1e-4)
    initial_scale = scene_scale / 200.0
    count = len(points)
    sh0 = (np.clip(colors, 0, 1) - 0.5) / SH_C0
    sh_rest = np.zeros((count, (sh_degree + 1) ** 2 - 1, 3), dtype=np.float32)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.as_tensor(points, dtype=torch.float32, device=device)),
        "scales": torch.nn.Parameter(torch.full((count, 3), math.log(initial_scale), device=device)),
        "quats": torch.nn.Parameter(torch.tensor([1., 0., 0., 0.], device=device).repeat(count, 1)),
        "opacities": torch.nn.Parameter(torch.full((count,), math.log(0.1 / 0.9), device=device)),
        "sh0": torch.nn.Parameter(torch.as_tensor(sh0[:, None], dtype=torch.float32, device=device)),
        "shN": torch.nn.Parameter(torch.as_tensor(sh_rest, dtype=torch.float32, device=device)),
    })
    return params, scene_scale


def create_optimizers(params, scene_scale):
    rates = {
        "means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
        "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20,
    }
    return {name: torch.optim.Adam([parameter], lr=rates[name], eps=1e-15)
            for name, parameter in params.items()}


def render_full(params, extrinsic, intrinsic, height, width, sh_degree=3, packed=False):
    device = params["means"].device
    view = torch.eye(4, dtype=torch.float32, device=device)[None]
    view[0, :3, :4] = extrinsic
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    rendered, alpha, info = rasterization(
        means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]), colors=colors,
        viewmats=view, Ks=intrinsic[None], width=int(width), height=int(height),
        sh_degree=int(sh_degree), packed=packed, render_mode="RGB",
    )
    return rendered.permute(0, 3, 1, 2), alpha.permute(0, 3, 1, 2), info


def checkpoint(path, step, params, optimizers, strategy_state, history, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step, "params": {key: value.detach().cpu() for key, value in params.items()},
        "optimizers": {key: value.state_dict() for key, value in optimizers.items()},
        "strategy_state": strategy_state, "history": history, "config": config,
    }, path)


def load_checkpoint(path, device="cuda"):
    saved = torch.load(path, map_location=device, weights_only=False)
    params = torch.nn.ParameterDict({
        key: torch.nn.Parameter(value.to(device)) for key, value in saved["params"].items()
    })
    scene_scale = float(saved["config"]["scene_scale"])
    optimizers = create_optimizers(params, scene_scale)
    for key, optimizer in optimizers.items():
        optimizer.load_state_dict(saved["optimizers"][key])

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value

    return params, scene_scale, optimizers, move(saved["strategy_state"]), saved["history"], int(saved["step"])


def train_adaptive(params, scene_scale, real_views, synthetic_views, output_dir,
                   steps=30_000, synthetic_probability=0.25, synthetic_weight=0.25,
                   sh_degree=3, checkpoint_every=5_000, seed=42,
                   optimizers=None, strategy_state=None, history=None, start_step=0):
    if not synthetic_views:
        raise ValueError("This trainer requires synthetic views; real-only training is intentionally unsupported.")
    device = params["means"].device
    optimizers = optimizers or create_optimizers(params, scene_scale)
    strategy = DefaultStrategy(
        prune_opa=0.005, grow_grad2d=0.0002, grow_scale3d=0.01,
        prune_scale3d=0.1, refine_start_iter=500,
        refine_stop_iter=min(15_000, steps - 1), refine_every=100,
        reset_every=3_000, verbose=True,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy_state or strategy.initialize_state(scene_scale=scene_scale)
    rng = np.random.default_rng(seed)
    history = list(history or [])
    output_dir = Path(output_dir)
    config = {
        "steps": steps, "synthetic_probability": synthetic_probability,
        "synthetic_weight": synthetic_weight, "sh_degree": sh_degree,
        "scene_scale": scene_scale,
    }

    for step in range(start_step, steps):
        use_synthetic = rng.random() < synthetic_probability
        pool = synthetic_views if use_synthetic else real_views
        sample = pool[int(rng.integers(len(pool)))]
        active_sh = min(sh_degree, step // 1_000)
        rendered, alpha, info = render_full(
            params,
            torch.as_tensor(sample["extrinsic"], dtype=torch.float32, device=device),
            torch.as_tensor(sample["intrinsic"], dtype=torch.float32, device=device),
            sample["height"], sample["width"], active_sh,
        )
        target = torch.as_tensor(sample["rgb"], dtype=torch.float32, device=device).permute(2, 0, 1)[None]
        if use_synthetic:
            raw_loss, parts = synthetic_view_loss(
                rendered, alpha, target,
                torch.as_tensor(sample["validity"], device=device)[None],
                torch.as_tensor(sample["confidence"], device=device)[None],
            )
            loss = synthetic_weight * raw_loss
        else:
            loss, parts = real_view_loss(
                rendered, alpha, target,
                torch.as_tensor(sample["mask"], device=device)[None],
            )

        strategy.step_pre_backward(params, optimizers, state, step, info)
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)

        # Exponential position learning-rate decay.
        fraction = step / max(steps - 1, 1)
        optimizers["means"].param_groups[0]["lr"] = scene_scale * 1.6e-4 * (0.01 ** fraction)
        if step % 50 == 0 or step == steps - 1:
            history.append({
                "step": step, "kind": "synthetic" if use_synthetic else "real",
                "loss": float(loss.detach()), "gaussians": len(params["means"]),
                "active_sh_degree": active_sh,
                **{name: float(value.detach()) for name, value in parts.items()},
            })
        if (step + 1) % checkpoint_every == 0 or step == steps - 1:
            checkpoint(output_dir / f"checkpoint_{step + 1:06d}.pt", step + 1,
                       params, optimizers, state, history, config)
    return params, history
