"""Input construction and point-cloud utilities for the preprocessing ablation."""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image

from .image_preprocessing import (
    contain_without_upscale, method_canvas, pixel_crop_box, shared_normalized_crop,
)


VARIANT_ORDER = ("full", "full_ref", "loose", "loose_ref", "current", "current_ref")


def _save_canvas(image_path, mask_path, output_image, output_mask, crop_box, method):
    with Image.open(image_path) as opened:
        image = opened.convert("RGB").crop(crop_box)
    with Image.open(mask_path) as opened:
        mask = opened.convert("L").crop(crop_box)
    canvas_size = method_canvas(image.size, method)
    image, scale, pad_x, pad_y = contain_without_upscale(image, canvas_size, (0, 0, 0))
    mask, _, _, _ = contain_without_upscale(mask, canvas_size, 0)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_image, compress_level=3)
    mask.save(output_mask, compress_level=3)
    return scale, pad_x, pad_y


def build_standard_variants(scene_rows, output_root, loose_margin=0.25, overwrite=False):
    """Build full and loose model inputs from BiRefNet originals; copy current inputs."""
    output_root = Path(output_root)
    base = scene_rows.sort_values("view_order").drop_duplicates("source")
    source_images = [Path(value) for value in base.output_image]
    source_masks = [Path(value) for value in base.output_mask]
    loose_box = shared_normalized_crop(source_masks, margin_fraction=loose_margin)
    records = []
    for method in ("vggt", "dust3r_mast3r"):
        method_rows = scene_rows[scene_rows.method.eq(method)].sort_values("view_order")
        for index, row in enumerate(method_rows.itertuples()):
            for variant, normalized_box in (("full", (0, 0, 1, 1)), ("loose", loose_box)):
                source_image, source_mask = source_images[index], source_masks[index]
                with Image.open(source_image) as opened:
                    crop_box = pixel_crop_box(normalized_box, opened.size)
                image_out = output_root / "inputs" / variant / method / "images" / f"view_{index:02d}.png"
                mask_out = output_root / "inputs" / variant / method / "masks" / f"view_{index:02d}.png"
                if overwrite or not (image_out.is_file() and mask_out.is_file()):
                    scale, pad_x, pad_y = _save_canvas(
                        source_image, source_mask, image_out, mask_out, crop_box, method,
                    )
                else:
                    scale = pad_x = pad_y = np.nan
                records.append({"variant": variant, "method": method, "view": index,
                                "image": str(image_out), "mask": str(mask_out),
                                "source_image": str(source_image), "source_mask": str(source_mask),
                                "crop_box": repr(crop_box), "resize_scale": scale,
                                "pad_x": pad_x, "pad_y": pad_y})
            for variant, image_value, mask_value in (
                ("current", row.method_image, row.method_mask),
            ):
                image_out = output_root / "inputs" / variant / method / "images" / f"view_{index:02d}.png"
                mask_out = output_root / "inputs" / variant / method / "masks" / f"view_{index:02d}.png"
                image_out.parent.mkdir(parents=True, exist_ok=True)
                mask_out.parent.mkdir(parents=True, exist_ok=True)
                if overwrite or not image_out.is_file(): shutil.copy2(image_value, image_out)
                if overwrite or not mask_out.is_file(): shutil.copy2(mask_value, mask_out)
                records.append({"variant": variant, "method": method, "view": index,
                                "image": str(image_out), "mask": str(mask_out),
                                "source_image": str(image_value), "source_mask": str(mask_value),
                                "crop_box": "existing", "resize_scale": np.nan,
                                "pad_x": np.nan, "pad_y": np.nan})
    return pd.DataFrame(records), loose_box


def finite_masked_points(points, masks, confidence=None, threshold=None):
    keep = np.asarray(masks, bool) & np.isfinite(points).all(axis=-1)
    if confidence is not None and threshold is not None:
        keep &= np.asarray(confidence) >= threshold
    return np.asarray(points)[keep]

