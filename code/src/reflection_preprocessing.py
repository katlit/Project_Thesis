"""Reusable helpers for the optional UnReflectAnything preprocessing branch."""

from pathlib import Path
import shutil

import numpy as np
from PIL import Image


def composite_foreground(diffuse, source, mask, threshold=128):
    """Use the inferred diffuse RGB only on foreground pixels."""
    diffuse = diffuse.convert("RGB")
    source = source.convert("RGB")
    mask = mask.convert("L")
    if diffuse.size != source.size:
        diffuse = diffuse.resize(source.size, Image.Resampling.BICUBIC)
    if mask.size != source.size:
        raise ValueError(f"Mask {mask.size} does not match image {source.size}")
    binary = mask.point(lambda value: 255 if value >= threshold else 0)
    return Image.composite(diffuse, source, binary)


def change_statistics(before, after, mask, threshold=128, changed_threshold=2 / 255):
    """Describe changes inside the foreground; these are not quality metrics."""
    before = np.asarray(before.convert("RGB"), dtype=np.float32) / 255.0
    after = np.asarray(after.convert("RGB"), dtype=np.float32) / 255.0
    foreground = np.asarray(mask.convert("L")) >= threshold
    if before.shape != after.shape or before.shape[:2] != foreground.shape:
        raise ValueError("Before, after, and mask must have matching dimensions")
    difference = np.abs(after - before)
    values = difference[foreground]
    if not len(values):
        raise ValueError("Foreground mask is empty")
    pixel_mae = values.mean(axis=1)
    return {
        "foreground_mae": float(values.mean()),
        "foreground_rmse": float(np.sqrt(np.mean(values ** 2))),
        "changed_foreground_fraction": float(np.mean(pixel_mae >= changed_threshold)),
        "foreground_max_change": float(values.max()),
    }


def reflection_difference_image(before, after, gain=4.0):
    """Return an amplified RGB absolute-difference diagnostic."""
    a = np.asarray(before.convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(after.convert("RGB"), dtype=np.float32) / 255.0
    return Image.fromarray(np.uint8(np.clip(np.abs(b - a) * gain, 0, 1) * 255))


def process_with_unreflectanything(
    source_path,
    mask_path,
    output_image_path,
    output_mask_path,
    model,
    threshold=0.30,
    dilation=40,
    overwrite=False,
):
    """Run official file inference, restore the original canvas, and copy the mask."""
    source_path, mask_path = Path(source_path), Path(mask_path)
    output_image_path, output_mask_path = Path(output_image_path), Path(output_mask_path)
    if output_image_path.is_file() and output_mask_path.is_file() and not overwrite:
        with Image.open(source_path) as opened:
            before = opened.convert("RGB")
        with Image.open(output_image_path) as opened:
            after = opened.convert("RGB")
        with Image.open(mask_path) as opened:
            mask = opened.convert("L")
        return {"status": "skipped", **change_statistics(before, after, mask)}

    import unreflectanything

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_image_path.with_name(output_image_path.stem + ".diffuse.tmp.png")
    try:
        unreflectanything.inference(
            source_path,
            output=temporary,
            model=model,
            threshold=float(threshold),
            dilation=int(dilation),
            resize_output=True,
            verbose=False,
        )
        with Image.open(source_path) as opened:
            before = opened.convert("RGB")
        with Image.open(mask_path) as opened:
            mask = opened.convert("L")
        with Image.open(temporary) as opened:
            diffuse = opened.convert("RGB")
        after = composite_foreground(diffuse, before, mask)
        after.save(output_image_path, compress_level=3)
        shutil.copy2(mask_path, output_mask_path)
        return {"status": "written", **change_statistics(before, after, mask)}
    finally:
        temporary.unlink(missing_ok=True)
