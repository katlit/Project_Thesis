"""Small helpers for sending a COLMAP camera orbit to CityGaussian."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


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
