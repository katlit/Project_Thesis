"""Experimental foreground-aware losses for a later 3DGS ablation.

This module is intentionally not wired into the current trainer.  It preserves
three proposed changes for later comparison with ``supervision_losses.py``:

1. class-balanced real-view silhouette BCE;
2. an eroded real foreground mask for local-window SSIM;
3. separate synthetic RGB-validity and foreground-occupancy masks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .supervision_losses import _mask4, ssim_map


def erode_binary_mask(mask, kernel_size=5):
    """Erode a BCHW binary mask using only PyTorch pooling operations."""
    mask = _mask4(mask).clamp(0, 1)
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd so erosion stays pixel-aligned.")
    padding = kernel_size // 2
    return 1 - F.max_pool2d(1 - mask, kernel_size, stride=1, padding=padding)


def balanced_silhouette_bce(render_alpha, foreground_mask, eps=1e-8):
    """Give foreground and background equal authority regardless of area."""
    target = _mask4(foreground_mask).clamp(0, 1)
    alpha = render_alpha.clamp(1e-6, 1 - 1e-6)
    foreground = target
    background = 1 - target
    foreground_loss = -(foreground * torch.log(alpha)).sum() / (foreground.sum() + eps)
    background_loss = -(background * torch.log1p(-alpha)).sum() / (background.sum() + eps)
    return 0.5 * (foreground_loss + background_loss)


def real_view_loss_v2(
    render_rgb,
    render_alpha,
    target_rgb,
    foreground_mask,
    lambda_ssim=0.2,
    lambda_silhouette=0.1,
    ssim_mask_erosion=5,
    eps=1e-8,
):
    """Real-view loss with balanced silhouette BCE and boundary-safe SSIM."""
    foreground = _mask4(foreground_mask).clamp(0, 1)
    ssim_foreground = erode_binary_mask(foreground, ssim_mask_erosion)

    rgb = (torch.abs(render_rgb - target_rgb) * foreground).sum() / (
        3 * foreground.sum() + eps
    )
    ssim = ((1 - ssim_map(render_rgb, target_rgb)) * ssim_foreground).sum() / (
        ssim_foreground.sum() + eps
    )
    silhouette = balanced_silhouette_bce(render_alpha, foreground, eps)
    total = (
        (1 - lambda_ssim) * rgb
        + lambda_ssim * ssim
        + lambda_silhouette * silhouette
    )
    return total, {
        "rgb": rgb,
        "ssim": ssim,
        "silhouette": silhouette,
        "ssim_valid_fraction": ssim_foreground.mean(),
    }


def synthetic_view_loss_v2(
    render_rgb,
    render_alpha,
    target_rgb,
    rgb_validity_mask,
    foreground_occupancy_mask,
    confidence,
    lambda_ssim=0.2,
    lambda_silhouette=0.02,
    eps=1e-8,
):
    """Synthetic loss with independent colour and opacity supervision masks.

    ``rgb_validity_mask`` says where the synthetic RGB target is usable.
    ``foreground_occupancy_mask`` says where opacity is known to be foreground.
    Pixels outside the occupancy mask receive no synthetic opacity target; zero
    does not mean confidently known background.

    Both terms retain absolute confidence attenuation by dividing by their
    respective binary-mask counts rather than by confidence sums.
    """
    rgb_validity = _mask4(rgb_validity_mask).clamp(0, 1)
    foreground_occupancy = _mask4(foreground_occupancy_mask).clamp(0, 1)
    confidence = _mask4(confidence).detach().clamp(0, 1)

    rgb_weight = rgb_validity * confidence
    rgb_count = rgb_validity.sum()
    rgb = (torch.abs(render_rgb - target_rgb) * rgb_weight).sum() / (
        3 * rgb_count + eps
    )
    ssim = ((1 - ssim_map(render_rgb, target_rgb)) * rgb_weight).sum() / (
        rgb_count + eps
    )

    occupancy_weight = foreground_occupancy * confidence
    occupancy_count = foreground_occupancy.sum()
    opacity_map = -torch.log(render_alpha.clamp(1e-6, 1 - 1e-6))
    silhouette = (opacity_map * occupancy_weight).sum() / (occupancy_count + eps)

    total = (
        (1 - lambda_ssim) * rgb
        + lambda_ssim * ssim
        + lambda_silhouette * silhouette
    )
    return total, {
        "rgb": rgb,
        "ssim": ssim,
        "silhouette": silhouette,
        "mean_rgb_confidence": rgb_weight.sum() / (rgb_count + eps),
        "mean_occupancy_confidence": occupancy_weight.sum() / (occupancy_count + eps),
    }
