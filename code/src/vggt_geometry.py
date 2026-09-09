"""Geometry-guided pseudo-view synthesis from VGGT predictions."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def confidence_to_unit_interval(confidence, foreground=None, low=5.0, high=95.0):
    """Robustly rank-scale VGGT confidence; this is not probability calibration."""
    confidence = np.asarray(confidence, dtype=np.float32)
    valid = np.isfinite(confidence)
    if foreground is not None:
        valid &= np.asarray(foreground, dtype=bool)
    values = confidence[valid]
    if not values.size:
        return np.zeros_like(confidence)
    lo, hi = np.percentile(values, [low, high])
    return np.clip((confidence - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)


def multiview_depth_support(point_maps, depths, extrinsics, intrinsics, foreground_masks,
                            relative_tolerance=0.05, absolute_tolerance=0.01):
    """Count views supporting each world point by mask and predicted depth."""
    points = np.asarray(point_maps, dtype=np.float64)
    depth = np.asarray(depths)
    if depth.ndim == 4:
        depth = depth.squeeze(-1)
    masks = np.asarray(foreground_masks, dtype=bool)
    views, height, width = masks.shape
    support = masks.astype(np.uint8)
    for source in range(views):
        flat_points = points[source].reshape(-1, 3)
        source_valid = masks[source].reshape(-1) & np.isfinite(flat_points).all(axis=1)
        for target in range(views):
            if target == source:
                continue
            camera = flat_points @ extrinsics[target, :3, :3].T + extrinsics[target, :3, 3]
            z = camera[:, 2]
            pixels = camera @ intrinsics[target].T
            denominator = np.where(np.abs(pixels[:, 2]) > 1e-12, pixels[:, 2], 1e-12)
            u = np.rint(pixels[:, 0] / denominator).astype(np.int64)
            v = np.rint(pixels[:, 1] / denominator).astype(np.int64)
            inside = source_valid & (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            indices = np.flatnonzero(inside)
            if not len(indices):
                continue
            target_depth = depth[target, v[indices], u[indices]]
            tolerance = absolute_tolerance + relative_tolerance * np.maximum(
                np.abs(target_depth), np.abs(z[indices])
            )
            agreed = (
                masks[target, v[indices], u[indices]]
                & np.isfinite(target_depth)
                & (np.abs(z[indices] - target_depth) <= tolerance)
            )
            support[source].reshape(-1)[indices[agreed]] += 1
    return support


def _as_homogeneous(extrinsic):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = np.asarray(extrinsic, dtype=np.float64)
    return matrix


def interpolate_camera(extrinsic_a, intrinsic_a, extrinsic_b, intrinsic_b, alpha):
    """Interpolate camera-to-world rotation/translation, return OpenCV world-to-camera."""
    c2w_a = np.linalg.inv(_as_homogeneous(extrinsic_a))
    c2w_b = np.linalg.inv(_as_homogeneous(extrinsic_b))
    rotations = Rotation.from_matrix(np.stack([c2w_a[:3, :3], c2w_b[:3, :3]]))
    rotation = Slerp([0.0, 1.0], rotations)([float(alpha)]).as_matrix()[0]
    translation = (1.0 - alpha) * c2w_a[:3, 3] + alpha * c2w_b[:3, 3]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = translation
    w2c = np.linalg.inv(c2w)
    intrinsic = (1.0 - alpha) * np.asarray(intrinsic_a) + alpha * np.asarray(intrinsic_b)
    return w2c[:3, :4].astype(np.float32), intrinsic.astype(np.float32)


def interpolate_closed_orbit(extrinsics, intrinsics, views_between=10):
    """Create only interior cameras for every consecutive pair, including last-to-first."""
    records = []
    count = len(extrinsics)
    for left in range(count):
        right = (left + 1) % count
        for step in range(1, views_between + 1):
            alpha = step / (views_between + 1)
            extrinsic, intrinsic = interpolate_camera(
                extrinsics[left], intrinsics[left], extrinsics[right], intrinsics[right], alpha
            )
            records.append({
                "left_view": left, "right_view": right, "step": step,
                "alpha": alpha, "extrinsic": extrinsic, "intrinsic": intrinsic,
            })
    return records


def forward_splat(point_maps, colors, confidences, foreground_masks, target_extrinsic,
                  target_intrinsic, output_size, depth_tolerance=0.02,
                  min_support=2, confidence_threshold=0.25):
    """Project VGGT world points, z-filter, and confidence-weight RGB samples.

    Returns RGB, binary validity, confidence weight, support count, and depth.
    Confidence is a geometric weighting proxy, not a calibrated correctness probability.
    """
    height, width = map(int, output_size)
    rotation = np.asarray(target_extrinsic)[:3, :3]
    translation = np.asarray(target_extrinsic)[:3, 3]
    intrinsic = np.asarray(target_intrinsic)

    projected = []
    for points, rgb, conf, foreground in zip(point_maps, colors, confidences, foreground_masks):
        points = np.asarray(points).reshape(-1, 3)
        rgb = np.asarray(rgb).reshape(-1, 3)
        conf = np.asarray(conf).reshape(-1)
        foreground = np.asarray(foreground).reshape(-1).astype(bool)
        camera = points @ rotation.T + translation
        z = camera[:, 2]
        pixels = camera @ intrinsic.T
        x = np.rint(pixels[:, 0] / np.maximum(pixels[:, 2], 1e-8)).astype(np.int64)
        y = np.rint(pixels[:, 1] / np.maximum(pixels[:, 2], 1e-8)).astype(np.int64)
        keep = foreground & np.isfinite(camera).all(1) & np.isfinite(conf) & (z > 1e-6)
        keep &= (x >= 0) & (x < width) & (y >= 0) & (y < height) & (conf > 0)
        projected.append((y[keep] * width + x[keep], z[keep], rgb[keep], conf[keep]))

    pixel = np.concatenate([item[0] for item in projected])
    depth = np.concatenate([item[1] for item in projected])
    rgb = np.concatenate([item[2] for item in projected])
    conf = np.concatenate([item[3] for item in projected])
    nearest = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(nearest, pixel, depth)
    near_surface = depth <= nearest[pixel] * (1.0 + depth_tolerance)
    pixel, depth, rgb, conf = pixel[near_surface], depth[near_surface], rgb[near_surface], conf[near_surface]

    weight_sum = np.zeros(height * width, dtype=np.float32)
    color_sum = np.zeros((height * width, 3), dtype=np.float32)
    confidence_sum = np.zeros(height * width, dtype=np.float32)
    support = np.zeros(height * width, dtype=np.int16)
    np.add.at(weight_sum, pixel, conf)
    np.add.at(color_sum, pixel, rgb * conf[:, None])
    np.add.at(confidence_sum, pixel, conf)
    np.add.at(support, pixel, 1)

    occupied = weight_sum > 0
    output_rgb = np.ones((height * width, 3), dtype=np.float32)
    output_rgb[occupied] = color_sum[occupied] / weight_sum[occupied, None]
    output_conf = np.zeros(height * width, dtype=np.float32)
    output_conf[occupied] = confidence_sum[occupied] / np.maximum(support[occupied], 1)
    validity = occupied & (support >= min_support) & (output_conf >= confidence_threshold)
    output_depth = nearest
    output_depth[~np.isfinite(output_depth)] = 0
    return (
        output_rgb.reshape(height, width, 3), validity.reshape(height, width),
        output_conf.reshape(height, width), support.reshape(height, width),
        output_depth.reshape(height, width),
    )


def confidence_error_table(confidence, error, validity, bins=10):
    """Bin confidence against observed error for held-out-view reliability analysis."""
    confidence = np.asarray(confidence)[validity]
    error = np.asarray(error)[validity]
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for index in range(bins):
        chosen = (confidence >= edges[index]) & (confidence < edges[index + 1] if index < bins - 1 else confidence <= 1)
        if chosen.any():
            rows.append({
                "bin": index, "confidence_low": edges[index], "confidence_high": edges[index + 1],
                "pixels": int(chosen.sum()), "mean_confidence": float(confidence[chosen].mean()),
                "mean_absolute_error": float(error[chosen].mean()),
            })
    return rows
