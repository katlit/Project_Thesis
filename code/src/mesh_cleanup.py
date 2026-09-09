"""Safe utilities for extracting car-only HQ200 reference meshes."""

from pathlib import Path
import numpy as np
import pandas as pd
import trimesh


def load_triangle_mesh(path):
    mesh = trimesh.load_mesh(Path(path), force="mesh", process=False)
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError(f"Empty triangle mesh: {path}")
    return mesh


def mesh_summary(mesh, scene=None, path=None):
    extent = np.asarray(mesh.bounds[1] - mesh.bounds[0], float)
    return {
        "scene": scene, "path": str(path) if path else None,
        "vertices": len(mesh.vertices), "faces": len(mesh.faces),
        "x_min": mesh.bounds[0, 0], "x_max": mesh.bounds[1, 0], "x_extent": extent[0],
        "y_min": mesh.bounds[0, 1], "y_max": mesh.bounds[1, 1], "y_extent": extent[1],
        "z_min": mesh.bounds[0, 2], "z_max": mesh.bounds[1, 2], "z_extent": extent[2],
    }


def normalized_to_absolute_box(mesh, normalized_box):
    box = np.asarray(normalized_box, dtype=float)
    if box.shape != (3, 2) or np.any(box < 0) or np.any(box > 1) or np.any(box[:, 0] >= box[:, 1]):
        raise ValueError("Box must be [[xmin,xmax],[ymin,ymax],[zmin,zmax]] in [0,1].")
    return mesh.bounds[0, :, None] + box * (mesh.bounds[1] - mesh.bounds[0])[:, None]


def crop_mesh(mesh, normalized_box, minimum_component_faces=500):
    """Keep faces whose centroids are inside a normalized XYZ box."""
    bounds = normalized_to_absolute_box(mesh, normalized_box)
    centers = mesh.triangles_center
    keep = np.all((centers >= bounds[:, 0]) & (centers <= bounds[:, 1]), axis=1)
    if not np.any(keep):
        raise ValueError("Crop box contains no mesh faces.")
    cropped = mesh.submesh([keep], append=True, repair=False)
    components = [part for part in cropped.split(only_watertight=False)
                  if len(part.faces) >= minimum_component_faces]
    return trimesh.util.concatenate(components) if components else cropped


def component_table(mesh, minimum_faces=50):
    rows = []
    for index, part in enumerate(mesh.split(only_watertight=False)):
        if len(part.faces) >= minimum_faces:
            rows.append({"component": index, **mesh_summary(part)})
    return pd.DataFrame(rows)


def sample_for_display(mesh, count=30_000, seed=42):
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return points


def export_clean_reference(mesh, output_path, overwrite=False):
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    return output_path
