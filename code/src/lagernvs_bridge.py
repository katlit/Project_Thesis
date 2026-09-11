"""Camera and artifact helpers connecting LagerNVS to the thesis 3DGS pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def w2c_to_c2w(extrinsics):
    extrinsics = torch.as_tensor(extrinsics, dtype=torch.float32)
    bottom = torch.tensor([0, 0, 0, 1], dtype=extrinsics.dtype, device=extrinsics.device)
    matrices = torch.cat([extrinsics, bottom.view(1, 1, 4).expand(len(extrinsics), -1, -1)], dim=1)
    return torch.linalg.inv(matrices)


def c2w_to_w2c(c2w):
    return torch.linalg.inv(c2w)[..., :3, :4]


def normalize_lagernvs_cameras(extrinsics, intrinsics, scale_factor=1.35):
    """Match LagerNVS camera normalization and retain an exact inverse mapping."""
    c2w_original = w2c_to_c2w(extrinsics)
    first_camera = c2w_original[0].clone()
    relative = torch.linalg.inv(first_camera)[None] @ c2w_original
    scene_scale = scale_factor * torch.linalg.norm(relative[:, :3, 3], dim=-1).max().clamp_min(1e-6)
    normalized = relative.clone()
    normalized[:, :3, 3] /= scene_scale
    return normalized, torch.as_tensor(intrinsics, dtype=torch.float32), first_camera, scene_scale


def restore_lagernvs_trajectory(normalized_c2w, first_camera, scene_scale):
    relative = torch.as_tensor(normalized_c2w, dtype=torch.float32).clone()
    relative[..., :3, 3] *= float(scene_scale)
    original_c2w = torch.as_tensor(first_camera, dtype=torch.float32)[None] @ relative
    return c2w_to_w2c(original_c2w)


def save_pseudo_view_set(root, rgb, validity, confidence, extrinsics, intrinsics):
    """Save learned RGB, geometric trust maps, and matching cameras."""
    from PIL import Image

    root = Path(root)
    for name in ["images", "validity", "confidence"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(len(rgb)):
        filename = f"view_{index:03d}.png"
        Image.fromarray(np.uint8(np.clip(rgb[index], 0, 1) * 255)).save(root / "images" / filename)
        Image.fromarray(np.uint8(validity[index]) * 255).save(root / "validity" / filename)
        Image.fromarray(np.uint8(np.clip(confidence[index], 0, 1) * 255)).save(root / "confidence" / filename)
        records.append({
            "index": index,
            "image": str(root / "images" / filename),
            "validity": str(root / "validity" / filename),
            "confidence": str(root / "confidence" / filename),
            "extrinsic": np.asarray(extrinsics[index]).tolist(),
            "intrinsic": np.asarray(intrinsics[index]).tolist(),
            "valid_fraction": float(np.asarray(validity[index]).mean()),
        })
    (root / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    np.savez_compressed(root / "cameras.npz", extrinsics=extrinsics, intrinsics=intrinsics)
    return records


def load_pseudo_view_set(root):
    from PIL import Image

    root = Path(root)
    records = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cameras = np.load(root / "cameras.npz")
    views = []
    for record, extrinsic, intrinsic in zip(records, cameras["extrinsics"], cameras["intrinsics"]):
        views.append({
            "rgb": np.asarray(Image.open(record["image"]).convert("RGB"), dtype=np.float32) / 255,
            "validity": np.asarray(Image.open(record["validity"]).convert("L")) >= 128,
            "confidence": np.asarray(Image.open(record["confidence"]).convert("L"), dtype=np.float32) / 255,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
        })
    return views
