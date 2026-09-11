"""Small helpers for sending a COLMAP camera orbit to CityGaussian."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def patch_portable_knn_initialization(city_root):
    """Replace CityGaussian's CUDA-compiled simple-knn scale initializer.

    The replacement computes the same nearest-neighbour squared-distance
    quantity with SciPy.  This code runs once when the initial point cloud is
    converted to Gaussians; rendering and MCMC optimization remain unchanged.
    """
    source_path = Path(city_root) / "internal/models/vanilla_gaussian.py"
    source = source_path.read_text(encoding="utf-8")
    old = """        # TODO: replace `simple_knn`
        from simple_knn._C import distCUDA2
        # the parameter device may be "cpu", so tensor must move to cuda before calling distCUDA2()
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.cuda()), 0.0000001).to(fused_point_cloud.device)
"""
    new = """        # Portable initialization for runtimes whose CUDA toolkit does not
        # match this environment's PyTorch build.  This replaces only the
        # one-time simple-knn call; the renderer and optimization are unchanged.
        from scipy.spatial import cKDTree
        xyz_numpy = fused_point_cloud.detach().cpu().numpy()
        neighbour_count = min(4, len(xyz_numpy))
        if neighbour_count < 2:
            dist2 = torch.full((len(xyz_numpy),), 1e-7, dtype=fused_point_cloud.dtype,
                               device=fused_point_cloud.device)
        else:
            distances, _ = cKDTree(xyz_numpy).query(xyz_numpy, k=neighbour_count, workers=-1)
            distance_squared = np.mean(np.square(distances[:, 1:]), axis=1)
            dist2 = torch.as_tensor(distance_squared, dtype=fused_point_cloud.dtype,
                                    device=fused_point_cloud.device).clamp_min_(1e-7)
"""
    if old in source:
        source_path.write_text(source.replace(old, new, 1), encoding="utf-8")
        status = "applied"
    elif "Portable initialization for runtimes" in source:
        status = "already_applied"
    else:
        raise RuntimeError(
            "CityGaussian's scale-initialization code changed; refusing to patch an unknown version."
        )
    return {
        "adaptation": "portable_scipy_knn_gaussian_scale_initialization",
        "status": status,
        "file": str(source_path),
        "scope": "one-time initial Gaussian scales only",
    }


def patch_cuda128_cstdint(rasterizer_root):
    """Add the standard integer header omitted by the legacy rasterizer."""
    header = Path(rasterizer_root) / "cuda_rasterizer/rasterizer_impl.h"
    source = header.read_text(encoding="utf-8")
    marker = "#include <cstdint>  // Colab CUDA 12.8 compatibility"
    if marker in source:
        status = "already_applied"
    else:
        source = marker + "\n" + source
        header.write_text(source, encoding="utf-8")
        status = "applied"
    return {
        "adaptation": "legacy_rasterizer_cuda128_cstdint_header",
        "status": status,
        "file": str(header),
        "scope": "compile-only dependency imported by CityGaussian renderer registry",
    }


def _camera_matrix(image):
    """Return an OpenCV camera-to-world matrix from a PyCOLMAP image."""
    world_to_camera = np.eye(4, dtype=np.float64)
    pose_value = image.cam_from_world
    pose = pose_value() if callable(pose_value) else pose_value
    world_to_camera[:3, :3] = np.asarray(pose.rotation.matrix())
    world_to_camera[:3, 3] = np.asarray(pose.translation)
    return np.linalg.inv(world_to_camera)


def closed_colmap_camera_path(
    reconstruction,
    output_path,
    frames_between=10,
    fps=15,
    background=(1.0, 1.0, 1.0),
):
    """Create a closed CityGaussian camera-path JSON from ordered COLMAP images."""
    images = sorted(reconstruction.images.values(), key=lambda item: item.name)
    if len(images) < 2:
        raise ValueError("At least two registered COLMAP images are required.")

    c2w = np.stack([_camera_matrix(image) for image in images])
    cameras = [reconstruction.cameras[image.camera_id] for image in images]
    widths = np.asarray([camera.width for camera in cameras], dtype=int)
    heights = np.asarray([camera.height for camera in cameras], dtype=int)
    fx = np.asarray([camera.params[0] for camera in cameras], dtype=float)
    width, height = int(np.median(widths)), int(np.median(heights))

    frames = []
    for start in range(len(images)):
        end = (start + 1) % len(images)
        rotations = Rotation.from_matrix(np.stack([c2w[start, :3, :3], c2w[end, :3, :3]]))
        rotation_curve = Slerp([0.0, 1.0], rotations)
        for step in range(frames_between):
            t = step / frames_between
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = rotation_curve([t]).as_matrix()[0]
            matrix[:3, 3] = (1.0 - t) * c2w[start, :3, 3] + t * c2w[end, :3, 3]
            # CityGaussian's renderer expects an OpenGL camera in the JSON and
            # converts its Y/Z axes back internally.
            matrix[:3, 1:3] *= -1
            focal = (1.0 - t) * fx[start] + t * fx[end]
            fov = math.degrees(2.0 * math.atan(width / (2.0 * focal)))
            frames.append({
                "camera_to_world": matrix.reshape(-1).tolist(),
                "fov": float(fov),
                "aspect": float(width / height),
                "model_poses": None,
            })

    payload = {
        "camera_path": frames,
        "orientation_transform": np.eye(4).tolist(),
        "render_width": width,
        "render_height": height,
        "fps": int(fps),
        "background_color": list(background),
        "sh_degree": 3,
        "enable_transform": False,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"frames": len(frames), "width": width, "height": height}
