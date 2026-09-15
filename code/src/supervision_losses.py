"""Foreground-aware real and confidence-aware synthetic supervision for 3DGS."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mask4(mask):
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    return mask.float()


def ssim_map(prediction, target, window=11):
    """Differentiable channel-averaged SSIM map for BCHW tensors in [0,1]."""
    padding = window // 2
    mu_x = F.avg_pool2d(prediction, window, 1, padding)
    mu_y = F.avg_pool2d(target, window, 1, padding)
    sigma_x = F.avg_pool2d(prediction * prediction, window, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, window, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, window, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return score.clamp(-1, 1).mean(dim=1, keepdim=True)


def real_view_loss(render_rgb, render_alpha, target_rgb, foreground_mask,
                   lambda_ssim=0.2, lambda_silhouette=0.1, eps=1e-8):
    mask = _mask4(foreground_mask)
    rgb = (torch.abs(render_rgb - target_rgb) * mask).sum() / (3 * mask.sum() + eps)
    ssim = ((1 - ssim_map(render_rgb, target_rgb)) * mask).sum() / (mask.sum() + eps)
    silhouette = F.binary_cross_entropy(render_alpha.clamp(1e-6, 1 - 1e-6), mask)
    total = (1 - lambda_ssim) * rgb + lambda_ssim * ssim + lambda_silhouette * silhouette
    return total, {"rgb": rgb, "ssim": ssim, "silhouette": silhouette}


def synthetic_view_loss(render_rgb, render_alpha, target_rgb, validity_mask,
                        confidence, lambda_ssim=0.2, lambda_silhouette=0.02,
                        eps=1e-8):
    """Apply absolute confidence attenuation over the valid synthetic pixels.

    The denominator is the valid-pixel count, not the confidence sum. Thus a
    uniformly low-confidence pseudo-view produces a proportionally smaller
    loss and gradient. Confidence is detached so pseudo-targets cannot change
    their own authority.
    """
    validity = _mask4(validity_mask)
    confidence = _mask4(confidence).detach().clamp(0, 1)
    weight = validity * confidence
    valid_count = validity.sum()
    rgb = (torch.abs(render_rgb - target_rgb) * weight).sum() / (3 * valid_count + eps)
    ssim = ((1 - ssim_map(render_rgb, target_rgb)) * weight).sum() / (valid_count + eps)
    # Only supported pseudo-foreground is supervised; unknown holes receive no opacity target.
    silhouette_map = F.binary_cross_entropy(
        render_alpha.clamp(1e-6, 1 - 1e-6), torch.ones_like(render_alpha), reduction="none"
    )
    silhouette = (silhouette_map * weight).sum() / (valid_count + eps)
    total = (1 - lambda_ssim) * rgb + lambda_ssim * ssim + lambda_silhouette * silhouette
    mean_valid_confidence = weight.sum() / (valid_count + eps)
    return total, {
        "rgb": rgb,
        "ssim": ssim,
        "silhouette": silhouette,
        "mean_valid_confidence": mean_valid_confidence,
    }
