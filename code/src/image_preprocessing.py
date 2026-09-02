import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIEW_LABELS = [
    "FRONT", "FRONT_LEFT", "SIDE_LEFT", "REAR_LEFT",
    "REAR", "REAR_RIGHT", "SIDE_RIGHT", "FRONT_RIGHT",
]


def _images_in(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _median_area(paths):
    areas = []
    for path in paths:
        with Image.open(path) as image:
            areas.append(image.width * image.height)
    return float(np.median(areas)) if areas else -1.0


def select_highest_resolution_industrial_images(root):
    """Choose images/ or images_raw/ independently for every car by median area."""
    rows = []
    car_dirs = sorted(
        path for path in Path(root).rglob("*")
        if path.is_dir() and path.name.casefold().startswith(("car_", "carid_"))
    )
    for car_dir in car_dirs:
        options = []
        for folder_name in ("images_raw", "images"):
            paths = _images_in(car_dir / folder_name)
            if paths:
                options.append((folder_name, paths, _median_area(paths)))
        if not options:
            continue
        folder_name, paths, median_area = max(options, key=lambda item: item[2])
        for path in paths:
            with Image.open(path) as image:
                rows.append({
                    "dataset": "IndustrialInventory", "scene": car_dir.name,
                    "source": path, "source_variant": folder_name,
                    "width": image.width, "height": image.height,
                    "median_folder_area": median_area,
                })
    return pd.DataFrame(rows)


def collect_original_inventory(industrial_root, hq200_root):
    industrial = select_highest_resolution_industrial_images(industrial_root)
    rows = industrial.to_dict("records") if not industrial.empty else []
    for scene in sorted(path for path in Path(hq200_root).iterdir() if path.is_dir()):
        for path in sorted(scene.glob("frame_*.jpg")):
            with Image.open(path) as image:
                rows.append({
                    "dataset": "3DRealCar", "scene": scene.name,
                    "source": path, "source_variant": "frame_jpg",
                    "width": image.width, "height": image.height,
                    "median_folder_area": np.nan,
                })
    return pd.DataFrame(rows)


def output_paths(output_root, dataset, scene, source, source_root):
    relative = Path(source).relative_to(source_root)
    token = hashlib.sha1(str(relative).encode()).hexdigest()[:8]
    stem = f"{Path(source).stem}_{token}"
    base = Path(output_root) / dataset / scene
    return base / "images" / f"{stem}.png", base / "masks" / f"{stem}.png"


def mask_bbox(mask, threshold=128):
    array = np.asarray(mask.convert("L")) >= threshold
    ys, xs = np.where(array)
    if not len(xs):
        raise ValueError("Foreground mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def normalized_bbox(mask, threshold=128):
    x0, y0, x1, y1 = mask_bbox(mask, threshold)
    width, height = mask.size
    return x0 / width, y0 / height, x1 / width, y1 / height


def shared_normalized_crop(mask_paths, margin_fraction=0.10, threshold=128):
    """Union foreground boxes and add margin in normalized image coordinates."""
    boxes = []
    for path in mask_paths:
        with Image.open(path) as mask:
            boxes.append(normalized_bbox(mask, threshold))
    boxes = np.asarray(boxes, dtype=float)
    x0, y0 = boxes[:, :2].min(axis=0)
    x1, y1 = boxes[:, 2:].max(axis=0)
    margin_x = (x1 - x0) * margin_fraction
    margin_y = (y1 - y0) * margin_fraction
    return max(0.0, x0 - margin_x), max(0.0, y0 - margin_y), min(1.0, x1 + margin_x), min(1.0, y1 + margin_y)


def pixel_crop_box(normalized_box, size):
    width, height = size
    x0, y0, x1, y1 = normalized_box
    return (
        max(0, int(np.floor(x0 * width))), max(0, int(np.floor(y0 * height))),
        min(width, int(np.ceil(x1 * width))), min(height, int(np.ceil(y1 * height))),
    )


def contain_without_upscale(image, target_size, fill):
    """Resize proportionally into a canvas, never enlarging source pixels."""
    target_width, target_height = map(int, target_size)
    scale = min(target_width / image.width, target_height / image.height, 1.0)
    resized_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(resized_size, Image.Resampling.LANCZOS) if resized_size != image.size else image.copy()
    canvas = Image.new(image.mode, (target_width, target_height), fill)
    pad_x = (target_width - resized.width) // 2
    pad_y = (target_height - resized.height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def method_canvas(crop_size, method):
    """Return a model-compatible landscape canvas while preserving aspect ratio."""
    width, height = crop_size
    aspect = width / height
    if method == "dust3r_mast3r":
        allowed_heights = np.array([384, 336, 288, 256, 160])
        desired = 512 / aspect
        return 512, int(allowed_heights[np.argmin(np.abs(allowed_heights - desired))])
    if method == "vggt":
        desired = int(round((518 / aspect) / 14) * 14)
        return 518, int(np.clip(desired, 14, 518))
    raise ValueError(f"Unknown method: {method}")


def summarize_resolutions(inventory):
    return (
        inventory.groupby(["dataset", "source_variant", "width", "height"])
        .size().rename("images").reset_index().sort_values(["dataset", "width", "height"])
    )

