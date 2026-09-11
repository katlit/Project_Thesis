"""Build the Colab notebooks in this directory.

Run after editing notebook cell sources so the committed .ipynb files remain
deterministic and free of execution outputs.
"""

import json
from pathlib import Path
from textwrap import dedent


HERE = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


def write_notebook(name, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (HERE / name).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


COMMON_PATHS = r'''
from pathlib import Path
from google.colab import drive

DRIVE_MOUNT = Path("/content/drive")
if not (DRIVE_MOUNT / "MyDrive").is_dir():
    drive.mount(str(DRIVE_MOUNT))

# Change only this value if the Drive project is moved.
PROJECT_ROOT = DRIVE_MOUNT / "MyDrive" / "ITU" / "3D" / "Thesis"

def find_unique_dir(names, search_roots):
    direct = [root / name for root in search_roots for name in names]
    matches = [p for p in direct if p.is_dir()]
    if not matches:
        for root in search_roots:
            if root.is_dir():
                matches.extend(p for p in root.rglob("*") if p.is_dir() and p.name.casefold() in {n.casefold() for n in names})
    unique = list(dict.fromkeys(p.resolve() for p in matches))
    if len(unique) != 1:
        raise FileNotFoundError(f"Expected one of {names}; found {len(unique)}: {unique}")
    return unique[0]

SEARCH_ROOTS = [PROJECT_ROOT / "data", PROJECT_ROOT]
INDUSTRIAL_ROOT = find_unique_dir(["IndustrialInventory"], SEARCH_ROOTS)
HQ200_ROOT = find_unique_dir(["3DrealCarHQ200", "HQ200"], SEARCH_ROOTS)

# If HQ200 is a wrapper folder, descend to the folder containing capture scenes.
if (HQ200_ROOT / "3DrealCarHQ200").is_dir():
    HQ200_ROOT = HQ200_ROOT / "3DrealCarHQ200"

print("Project:   ", PROJECT_ROOT)
print("Industrial:", INDUSTRIAL_ROOT)
print("3DRealCar: ", HQ200_ROOT)
'''


CPU_RUNTIME_GUARD = r'''
import torch

CPU_ONLY_NOTEBOOK = True
GPU_ATTACHED = torch.cuda.is_available()

print("Runtime check")
print("-" * 50)
print("GPU attached:", GPU_ATTACHED)

if GPU_ATTACHED:
    print("GPU:", torch.cuda.get_device_name(0))
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"GPU memory: {free_bytes / 1024**3:.2f} GB free / {total_bytes / 1024**3:.2f} GB total")

if CPU_ONLY_NOTEBOOK and GPU_ATTACHED:
    raise RuntimeError(
        "This notebook is CPU-only. To conserve Colab GPU availability, "
        "change the Colab hardware accelerator to None, reconnect the CPU "
        "kernel in VS Code, and run the notebook again."
    )

print("Correct CPU runtime: continue with the notebook.")
'''


SRC_BOOTSTRAP = r'''
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path("/content/Project_Thesis_code")
REPOSITORY = "https://github.com/katlit/Project_Thesis.git"
BRANCH = "codex/hq200-example-notebook"

if not (CODE_ROOT / "code" / "src").is_dir():
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CODE_ROOT)],
        check=True,
    )
else:
    subprocess.run(["git", "-C", str(CODE_ROOT), "pull", "--ff-only"], check=True)

CODE_PACKAGE_ROOT = CODE_ROOT / "code"
if str(CODE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_PACKAGE_ROOT))

print("Reusable code:", CODE_ROOT / "code" / "src")
'''


write_notebook("00_drive_access_test.ipynb", [
    md('''
    # Test Google Drive data access

    Run this first with the hosted Colab kernel. It mounts Drive, locates both datasets, counts representative files, and opens one image from each dataset. It does not modify data.
    '''),
    code('''
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    from PIL import Image

    print("Python:", sys.version)
    try:
        import torch
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except ImportError:
        print("PyTorch is not installed.")
    '''),
    code(COMMON_PATHS),
    code('''
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    def image_files(root):
        return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

    industrial_images = image_files(INDUSTRIAL_ROOT)
    hq_images = sorted(HQ200_ROOT.rglob("frame_*.jpg"))
    if not hq_images:
        hq_images = image_files(HQ200_ROOT)

    summary = {
        "Industrial images": len(industrial_images),
        "3DRealCar RGB images": len(hq_images),
        "3DRealCar camera JSON": len(list(HQ200_ROOT.rglob("frame_*.json"))),
        "3DRealCar depth maps": len(list(HQ200_ROOT.rglob("depth_*.png"))),
        "3DRealCar meshes": len(list(HQ200_ROOT.rglob("*.obj"))),
    }
    for label, count in summary.items():
        print(f"{label:28s}: {count:,}")

    if not industrial_images or not hq_images:
        raise FileNotFoundError("At least one dataset has no discoverable RGB images. Check the printed roots.")
    '''),
    code('''
    examples = [("IndustrialInventory", industrial_images[0]), ("3DRealCar", hq_images[0])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (dataset, path) in zip(axes, examples):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            ax.imshow(rgb)
            ax.set_title(f"{dataset}\\n{rgb.width} × {rgb.height}\\n{path.name}")
        ax.axis("off")
        print(dataset, "read OK:", path)
    plt.tight_layout()
    plt.show()
    print("Drive access test passed.")
    '''),
])


write_notebook("01_birefnet_original_resolution.ipynb", [
    md('''
    # 01 — BiRefNet foreground extraction at original resolution

    This notebook removes image **backgrounds** with BiRefNet for both datasets. Raw files are never overwritten. Each output contains:

    - `images/`: original-resolution RGB foreground composited on white;
    - `masks/`: matching grayscale foreground masks;
    - `manifest.csv`: provenance, original size, output size, and status.

    BiRefNet receives a temporary 1024×1024 tensor, but its mask is returned to the raw RGB dimensions. The saved RGB is never resized, padded, or upscaled. For each Industrial car, `images_raw/` and `images/` are compared and the folder with greater median pixel area is selected. Method-specific resizing happens in notebook 03.
    '''),
    code('''
    !pip -q install "transformers>=4.39" safetensors kornia timm
    '''),
    code('''
    import gc
    import hashlib
    from collections import Counter
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import kornia
    import timm
    import torch
    from PIL import Image, ImageOps
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    torch.set_float32_matmul_precision("high")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)
    if DEVICE != "cuda":
        raise RuntimeError("Connect a Colab GPU runtime before running BiRefNet. CPU would be unnecessarily slow.")
    print("kornia:", kornia.__version__, "| timm:", timm.__version__)
    '''),
    code(COMMON_PATHS),
    code(SRC_BOOTSTRAP),
    code('''
    from src.image_preprocessing import collect_original_inventory, output_paths, summarize_resolutions
    '''),
    md('''
    ## Configuration

    The default is a smoke test: three deterministic images from each dataset (six total). Smoke-test files are written below `_smoke_test` and cannot be confused with a complete run. When the results look correct, set `SMOKE_TEST=False` to process everything. Completed outputs are skipped, so rerunning after a Colab disconnect continues rather than starting over.
    '''),
    code('''
    OUTPUT_ROOT = PROJECT_ROOT / "data_processed" / "birefnet_original_resolution"
    RUN_PREPROCESSING = True
    SMOKE_TEST = True
    SMOKE_IMAGES_PER_DATASET = 3
    OVERWRITE = False
    MODEL_ID = "ZhengPeng7/BiRefNet"
    MODEL_INPUT_SIZE = (1024, 1024)
    WHITE_BACKGROUND = (255, 255, 255)

    inventory = collect_original_inventory(INDUSTRIAL_ROOT, HQ200_ROOT)
    display(summarize_resolutions(inventory))
    display(
        inventory.groupby(["dataset", "scene", "source_variant"])
        .agg(images=("source", "size"), width=("width", "first"), height=("height", "first"))
    )
    print("Industrial source folder selected per car by greatest median pixel area.")

    def deterministic_sample(group, count):
        group = group.sort_values("source").reset_index(drop=True)
        if len(group) <= count:
            return group
        positions = np.linspace(0, len(group) - 1, count, dtype=int)
        return group.iloc[positions]

    if SMOKE_TEST:
        work_inventory = pd.concat(
            [deterministic_sample(group, SMOKE_IMAGES_PER_DATASET) for _, group in inventory.groupby("dataset", sort=True)],
            ignore_index=True,
        )
        ACTIVE_OUTPUT_ROOT = OUTPUT_ROOT / "_smoke_test"
    else:
        work_inventory = inventory.copy()
        ACTIVE_OUTPUT_ROOT = OUTPUT_ROOT

    print(f"Mode: {'SMOKE TEST' if SMOKE_TEST else 'FULL RUN'}")
    print(f"Images scheduled: {len(work_inventory):,}")
    display(work_inventory[["dataset", "source", "width", "height"]])
    '''),
    code('''
    print("Output policy: foreground RGB and mask retain each raw source width and height.")
    print("No crop, shared canvas, downscale, or upscale is applied in notebook 01.")
    '''),
    code('''
    transform_image = transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = None
    if RUN_PREPROCESSING:
        model = AutoModelForImageSegmentation.from_pretrained(MODEL_ID, trust_remote_code=True)
        model.to(DEVICE).eval()
        if DEVICE == "cuda":
            model.half()
        print("Loaded", MODEL_ID)
    else:
        print("Model not loaded. Set RUN_PREPROCESSING=True when ready.")
    '''),
    code('''
    def predict_mask(image):
        tensor = transform_image(image).unsqueeze(0).to(DEVICE)
        if DEVICE == "cuda":
            tensor = tensor.half()
        with torch.inference_mode():
            prediction = model(tensor)[-1].sigmoid()[0, 0].float().cpu().numpy()
        mask = Image.fromarray(np.uint8(np.clip(prediction, 0, 1) * 255), mode="L")
        return mask.resize(image.size, Image.Resampling.LANCZOS)

    def destination_paths(row):
        source, dataset = row.source, row.dataset
        root = HQ200_ROOT if dataset == "3DRealCar" else INDUSTRIAL_ROOT
        image_path, mask_path = output_paths(ACTIVE_OUTPUT_ROOT, dataset, row.scene, source, root)
        return image_path, mask_path, row.scene

    def process_one(row):
        output_image, output_mask, scene = destination_paths(row)
        if output_image.exists() and output_mask.exists() and not OVERWRITE:
            return {
                "status": "skipped", "scene": scene,
                "output_image": output_image, "output_mask": output_mask,
                "output_width": row.width, "output_height": row.height,
            }
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output_mask.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(row.source) as opened:
            image = opened.convert("RGB")
        mask = predict_mask(image)
        foreground = Image.composite(image, Image.new("RGB", image.size, WHITE_BACKGROUND), mask)
        assert foreground.size == image.size and mask.size == image.size
        foreground.save(output_image, compress_level=3)
        mask.save(output_mask, compress_level=3)
        return {
            "status": "written", "scene": scene,
            "output_image": output_image, "output_mask": output_mask,
            "output_width": image.width, "output_height": image.height,
        }
    '''),
    code('''
    results = []
    if RUN_PREPROCESSING:
        for number, row in enumerate(work_inventory.itertuples(index=False), start=1):
            try:
                result = process_one(row)
                result.update({
                    "dataset": row.dataset, "source": row.source,
                    "source_variant": row.source_variant,
                    "source_width": row.width, "source_height": row.height,
                })
            except Exception as error:
                result = {"dataset": row.dataset, "source": row.source, "status": "error", "error": repr(error)}
            results.append(result)
            if number % 25 == 0 or number == len(work_inventory):
                print(f"{number:,}/{len(work_inventory):,}", Counter(r["status"] for r in results))
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

        manifest = pd.DataFrame(results)
        ACTIVE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(ACTIVE_OUTPUT_ROOT / "manifest.csv", index=False)
        complete = manifest.status.isin(["written", "skipped"])
        assert (manifest.loc[complete, "source_width"] == manifest.loc[complete, "output_width"]).all()
        assert (manifest.loc[complete, "source_height"] == manifest.loc[complete, "output_height"]).all()
        display(manifest.groupby(["dataset", "status"]).size().rename("images"))
        print("Manifest:", ACTIVE_OUTPUT_ROOT / "manifest.csv")
    else:
        print(f"Dry run only: {len(work_inventory):,} images scheduled. Set RUN_PREPROCESSING=True to start/resume.")
    '''),
    code('''
    if RUN_PREPROCESSING and results:
        successful = pd.DataFrame(results)
        successful = successful[successful.status.isin(["written", "skipped"])]
        successful = pd.concat([
            deterministic_sample(group, 3) for _, group in successful.groupby("dataset", sort=True)
        ], ignore_index=True)
        fig, axes = plt.subplots(len(successful), 3, figsize=(13, 4 * len(successful)), squeeze=False)
        for row_axes, (_, row) in zip(axes, successful.iterrows()):
            with Image.open(row.source) as source_image:
                row_axes[0].imshow(source_image.convert("RGB"))
            row_axes[1].imshow(Image.open(row.output_image).convert("RGB"))
            row_axes[2].imshow(Image.open(row.output_mask).convert("L"), cmap="gray", vmin=0, vmax=255)
            row_axes[0].set_ylabel(row.dataset)
            for ax, title in zip(row_axes, ["Original", "BiRefNet foreground", "Mask"]):
                ax.set_title(title)
                ax.axis("off")
        plt.tight_layout()
        plt.show()
        print("These are at most three examples per dataset. If they look correct, set SMOKE_TEST=False for the full run.")
    '''),
    md('''
    ## Important reconstruction note

    BiRefNet masks the object but does not improve camera poses or add image detail. This notebook preserves the original pixel grid, so it does not require an intrinsic-coordinate change. Notebook 03 records every later crop, scale, and padding offset needed to transform known intrinsics consistently.
    '''),
])


write_notebook("02_hq200_view_selection.ipynb", [
    md('''
    # 0.2 — HQ200 complete-car view selection

    HQ200 contains several capture circles and some frames crop the car. This notebook uses camera poses to arrange frames by azimuth, divides the orbit into eight rotational sectors, and displays the six best complete-car candidates per sector. The ranking uses the BiRefNet mask boundary margin, foreground coverage, and camera-height consistency.

    The pose sequence gives rotational order, but it cannot identify which side is the physical front of every car. Use the per-scene offset/reverse controls to align the displayed order, and override any candidate that still looks incomplete.
    '''),
    code('''
    import io
    import json
    import math
    import re
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    '''),
    code(CPU_RUNTIME_GUARD),
    code(COMMON_PATHS),
    code('''
    PROCESSED_ROOT = PROJECT_ROOT / "data_processed" / "birefnet_original_resolution"
    MANIFEST_PATH = PROCESSED_ROOT / "manifest.csv"
    SPLIT_ROOT = PROJECT_ROOT / "splits" / "sparse8"
    MANUAL_HQ_SELECTION_PATH = SPLIT_ROOT / "hq200_manual_8views.csv"
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    CANDIDATES_PER_SECTOR = 6
    N_SECTORS = 8

    manifest = pd.read_csv(MANIFEST_PATH)
    hq = manifest[
        manifest.dataset.eq("3DRealCar")
        & manifest.status.isin(["written", "skipped"])
    ].copy()
    # Do not call Path.is_file() for every Drive file. Thousands of individual
    # FUSE metadata requests can abort the mounted Drive connection. The
    # preprocessing manifest is the source of truth; files are opened lazily.
    hq = hq[hq.output_image.notna() & hq.output_mask.notna()].copy()
    print("HQ200 scenes:", hq.scene.nunique(), "RGB images:", len(hq))
    '''),
    code('''
    def frame_number(path):
        match = re.search(r"frame_(\\d+)", Path(path).stem)
        return int(match.group(1)) if match else None

    def camera_center(source_path):
        source = Path(source_path)
        metadata_path = source.with_suffix(".json")
        if not metadata_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pose = np.asarray(metadata["cameraPoseARFrame"], dtype=float).reshape(4, 4)
        return pose[:3, 3]

    def mask_quality(mask_path, sample_size=(256, 256), threshold=128):
        with Image.open(mask_path) as opened:
            mask = np.asarray(opened.convert("L").resize(sample_size, Image.Resampling.BILINEAR)) >= threshold
        ys, xs = np.where(mask)
        if not len(xs):
            return {"foreground_fraction": 0.0, "margin_fraction": 0.0, "border_contact": 1.0}
        height, width = mask.shape
        margins = np.array([xs.min(), ys.min(), width - 1 - xs.max(), height - 1 - ys.max()], dtype=float)
        border_band = max(2, round(min(width, height) * 0.015))
        border = np.zeros_like(mask)
        border[:border_band] = border[-border_band:] = True
        border[:, :border_band] = border[:, -border_band:] = True
        return {
            "foreground_fraction": float(mask.mean()),
            "margin_fraction": float(margins.min() / min(width, height)),
            "border_contact": float((mask & border).sum() / max(mask.sum(), 1)),
        }

    pose_rows = []
    for row in hq.itertuples(index=False):
        center = camera_center(row.source)
        if center is None:
            continue
        quality = mask_quality(row.output_mask)
        pose_rows.append({**row._asdict(), "frame_id": frame_number(row.source), "cx": center[0], "cy": center[1], "cz": center[2], **quality})
    posed = pd.DataFrame(pose_rows)
    print("Images with matching camera poses:", len(posed), "/", len(hq))
    '''),
    code('''
    def add_orbit_coordinates(group):
        group = group.copy()
        centers = group[["cx", "cy", "cz"]].to_numpy(float)
        centered = centers - np.median(centers, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        plane = vh[:2].T
        coordinates = centered @ plane
        group["orbit_x"] = coordinates[:, 0]
        group["orbit_y"] = coordinates[:, 1]
        group["azimuth"] = np.mod(np.arctan2(coordinates[:, 1], coordinates[:, 0]), 2 * np.pi)
        group["raw_sector"] = np.floor(group.azimuth / (2 * np.pi / N_SECTORS)).astype(int).clip(0, N_SECTORS - 1)
        # Distance away from the median capture plane favours one consistent camera-height circle.
        group["height_deviation"] = np.abs(centered @ vh[2])
        scale = max(float(group.height_deviation.median()), 1e-6)
        group["quality_score"] = (
            6.0 * group.margin_fraction
            - 12.0 * group.border_contact
            - 1.5 * np.abs(group.foreground_fraction - 0.38)
            - 0.15 * group.height_deviation / scale
        )
        return group

    posed = pd.concat([add_orbit_coordinates(group) for _, group in posed.groupby("scene", sort=True)], ignore_index=True)

    def circular_angle_distance(values, target):
        return np.abs((values - target + np.pi) % (2 * np.pi) - np.pi)

    # Targets spread across each sector prevent candidates from clustering at
    # almost the same angle. Six candidates provide fine angular choices.
    sector_width = 2 * np.pi / N_SECTORS
    angular_offsets = np.linspace(-0.30, 0.30, CANDIDATES_PER_SECTOR) * sector_width

    candidate_rows = []
    for scene, scene_group in posed.groupby("scene", sort=True):
        used_indices = set()
        for sector in range(N_SECTORS):
            sector_center = (sector + 0.5) * sector_width
            selected = []
            for rank, offset in enumerate(angular_offsets, start=1):
                target_angle = np.mod(sector_center + offset, 2 * np.pi)
                pool = scene_group.loc[~scene_group.index.isin(used_indices)].copy()
                pool["target_angle_distance"] = circular_angle_distance(pool.azimuth.to_numpy(), target_angle)

                # Prefer frames close to this angular target. A small overlap
                # beyond the sector boundary gives better diagonal choices.
                nearby = pool[pool.target_angle_distance <= 0.62 * sector_width].copy()
                if nearby.empty:
                    nearby = pool.nsmallest(12, "target_angle_distance").copy()

                # Angle is primary; mask quality chooses between nearby frames.
                nearby["candidate_score"] = (
                    nearby.quality_score
                    - 3.0 * nearby.target_angle_distance / sector_width
                )
                chosen = nearby.sort_values(["candidate_score", "frame_id"], ascending=[False, True]).iloc[0].copy()
                chosen["raw_sector"] = sector
                chosen["candidate_rank"] = rank
                chosen["target_angle"] = target_angle
                chosen["angle_from_target_deg"] = np.degrees(chosen.target_angle_distance)
                chosen["candidate_id"] = f"{scene}__s{sector}__r{rank}__f{int(chosen.frame_id)}"
                selected.append(chosen)
                used_indices.add(chosen.name)
            candidate_rows.append(pd.DataFrame(selected))
    candidates = pd.concat(candidate_rows, ignore_index=True)
    candidates.to_csv(SPLIT_ROOT / "hq200_view_candidates.csv", index=False)
    sector_counts = candidates.groupby(["scene", "raw_sector"]).size().unstack(fill_value=0)
    display(sector_counts)
    if not (sector_counts > 0).all().all():
        print("WARNING: a scene has an empty pose sector; inspect its trajectory before selection.")
    '''),
    md('''
    ## Candidate sheets

    Each figure is one car. Columns follow the camera orbit; rows are candidate ranks 1–6. Prefer an image where the entire car has clear white space around it. The dropdown annotator below is the authoritative selection tool.
    '''),
    code('''
    def display_candidate_sheet(scene_candidates):
        scene = scene_candidates.scene.iloc[0]
        fig, axes = plt.subplots(CANDIDATES_PER_SECTOR, N_SECTORS, figsize=(28, 3.2 * CANDIDATES_PER_SECTOR), squeeze=False)
        for sector in range(N_SECTORS):
            sector_rows = scene_candidates[scene_candidates.raw_sector.eq(sector)].sort_values("candidate_rank")
            for rank_index in range(CANDIDATES_PER_SECTOR):
                ax = axes[rank_index, sector]
                if rank_index < len(sector_rows):
                    row = sector_rows.iloc[rank_index]
                    with Image.open(row.output_image) as opened:
                        ax.imshow(opened.convert("RGB"))
                    ax.set_title(
                        f"sector {sector}, candidate {rank_index + 1}\\nframe {int(row.frame_id)}\\n"
                        f"margin={row.margin_fraction:.3f}, border={row.border_contact:.3f}\\n"
                        f"target difference={row.angle_from_target_deg:.1f}°", fontsize=8
                    )
                    ax.text(0.5, -0.08, row.candidate_id, transform=ax.transAxes, ha="center", va="top", fontsize=6)
                ax.axis("off")
        fig.suptitle(scene, fontsize=15)
        plt.tight_layout()
        plt.show()

    SHOW_STATIC_CANDIDATE_SHEETS = False

    if SHOW_STATIC_CANDIDATE_SHEETS:
        for scene, scene_candidates in candidates.groupby("scene", sort=True):
            display_candidate_sheet(scene_candidates)
    else:
        print("Static 480-image preview skipped. Use the interactive annotator below.")
    '''),
    md('''
    ## Interactive orientation annotator

    For each scene, inspect sectors 0–7. Only one sector's six candidates are loaded at a time so the widget remains reliable in VS Code. Label exactly one candidate as each canonical orientation and leave all unused candidates as `IGNORE`. Labels remain in memory while switching sectors; click **Save this scene** before switching cars. The annotations are written to Drive and can be resumed after a disconnect.
    '''),
    code('''
    VIEW_LABELS = [
        "FRONT", "FRONT_LEFT", "SIDE_LEFT", "REAR_LEFT",
        "REAR", "REAR_RIGHT", "SIDE_RIGHT", "FRONT_RIGHT",
    ]
    ANNOTATION_PATH = SPLIT_ROOT / "hq200_candidate_annotations.csv"

    if ANNOTATION_PATH.is_file():
        saved_annotations = pd.read_csv(ANNOTATION_PATH)
        annotation_state = dict(zip(saved_annotations.candidate_id, saved_annotations.view_label))
        print("Loaded existing annotations:", ANNOTATION_PATH)
    else:
        annotation_state = {}

    scene_names = sorted(candidates.scene.unique())

    def scene_is_complete(scene):
        scene_ids = set(candidates.loc[candidates.scene.eq(scene), "candidate_id"])
        labels = [annotation_state.get(candidate_id, "IGNORE") for candidate_id in scene_ids]
        return all(labels.count(view_label) == 1 for view_label in VIEW_LABELS)

    incomplete_scenes = [scene for scene in scene_names if not scene_is_complete(scene)]
    initial_scene = incomplete_scenes[0] if incomplete_scenes else scene_names[0]
    print(f"Completed scenes: {len(scene_names) - len(incomplete_scenes)}/{len(scene_names)}")
    if incomplete_scenes:
        print("Opening first incomplete scene:", initial_scene)
    else:
        print("All scenes already have eight saved orientation labels.")

    scene_selector = widgets.Dropdown(options=scene_names, value=initial_scene, description="Scene:", layout=widgets.Layout(width="65%"))
    sector_selector = widgets.Dropdown(options=list(range(N_SECTORS)), value=0, description="Sector:", layout=widgets.Layout(width="180px"))
    candidate_area = widgets.GridBox(layout=widgets.Layout(
        grid_template_columns="repeat(3, minmax(240px, 1fr))",
        grid_gap="12px",
        width="100%",
    ))
    status_output = widgets.Output()
    save_button = widgets.Button(description="Save this scene", button_style="success", icon="save")
    previous_button = widgets.Button(description="Previous", icon="arrow-left")
    next_button = widgets.Button(description="Next", icon="arrow-right")
    def thumbnail_bytes(path, size=(320, 220)):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            image.thumbnail(size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=88)
        return buffer.getvalue()

    def remember_label(candidate_id, change):
        if change["name"] == "value":
            annotation_state[candidate_id] = change["new"]

    def render_page(scene, sector):
        cards = []
        rows = candidates[
            candidates.scene.eq(scene) & candidates.raw_sector.eq(int(sector))
        ].sort_values("candidate_rank")
        for row in rows.itertuples(index=False):
            image_widget = widgets.Image(value=thumbnail_bytes(row.output_image), format="jpeg", layout=widgets.Layout(width="100%"))
            label_widget = widgets.HTML(
                value=f"<b>sector {int(row.raw_sector)} · candidate {int(row.candidate_rank)}</b><br>"
                      f"frame {int(row.frame_id)} · margin {row.margin_fraction:.3f}<br>"
                      f"angle from target {row.angle_from_target_deg:.1f}°<br>"
                      f"<small>{row.candidate_id}</small>"
            )
            dropdown = widgets.Dropdown(
                options=["IGNORE"] + VIEW_LABELS,
                value=annotation_state.get(row.candidate_id, "IGNORE"),
                description="Label:",
                layout=widgets.Layout(width="100%"),
            )
            dropdown.observe(
                lambda change, candidate_id=row.candidate_id: remember_label(candidate_id, change),
                names="value",
            )
            cards.append(widgets.VBox([image_widget, label_widget, dropdown], layout=widgets.Layout(border="1px solid #bbb", padding="6px")))
        candidate_area.children = tuple(cards)
        with status_output:
            clear_output()
            selected = [
                annotation_state.get(candidate_id, "IGNORE")
                for candidate_id in candidates.loc[candidates.scene.eq(scene), "candidate_id"]
            ]
            print(f"{scene} — sector {sector}: showing {len(rows)} candidates.")
            print(f"Orientations currently selected in this scene: {sum(label != 'IGNORE' for label in selected)}/8")
            print("Labels are remembered while changing sectors. Click Save this scene after all 8 sectors are checked.")

    def save_current_scene(_=None):
        scene = scene_selector.value
        scene_ids = candidates.loc[candidates.scene.eq(scene), "candidate_id"].tolist()
        labels = {candidate_id: annotation_state.get(candidate_id, "IGNORE") for candidate_id in scene_ids}
        selected = [label for label in labels.values() if label != "IGNORE"]
        missing = [label for label in VIEW_LABELS if selected.count(label) == 0]
        duplicates = [label for label in VIEW_LABELS if selected.count(label) > 1]
        with status_output:
            clear_output()
            if missing or duplicates:
                print("NOT SAVED")
                print("Missing:", missing or "none")
                print("Duplicated:", duplicates or "none")
                return
            annotation_state.update(labels)
            rows = [{"candidate_id": cid, "view_label": label} for cid, label in annotation_state.items()]
            pd.DataFrame(rows).sort_values("candidate_id").to_csv(ANNOTATION_PATH, index=False)
            print("Saved:", scene)
            print("File:", ANNOTATION_PATH)

    def move_scene(step):
        index = scene_names.index(scene_selector.value)
        scene_selector.value = scene_names[(index + step) % len(scene_names)]

    def scene_changed(change):
        if change["name"] == "value":
            sector_selector.value = 0
            render_page(change["new"], 0)

    def sector_changed(change):
        if change["name"] == "value":
            render_page(scene_selector.value, change["new"])

    scene_selector.observe(scene_changed, names="value")
    sector_selector.observe(sector_changed, names="value")
    save_button.on_click(save_current_scene)
    previous_button.on_click(lambda _: move_scene(-1))
    next_button.on_click(lambda _: move_scene(1))

    print(f"Annotation UI: {len(scene_names)} scenes, {len(candidates)} candidates total.")
    display(widgets.VBox([
        widgets.HBox([scene_selector, sector_selector, previous_button, next_button, save_button]),
        status_output,
        candidate_area,
    ]))
    # Only six images are embedded at once; 48-image widget payloads can fail
    # to render through a hosted Colab connection in VS Code.
    render_page(scene_selector.value, sector_selector.value)
    '''),
    md('''
    ## Build the final labeled selection

    This cell reads the dropdown annotations. It shows progress without failing when some scenes are unfinished. The final eight-view CSV is written automatically only after every scene has exactly one image for each canonical orientation.
    '''),
    code('''
    ANNOTATION_PATH = SPLIT_ROOT / "hq200_candidate_annotations.csv"
    OUTPUT_PATH = SPLIT_ROOT / "hq200_manual_8views.csv"
    expected_scenes = sorted(candidates.scene.unique())
    view_order_map = {label: index for index, label in enumerate(VIEW_LABELS)}

    if not ANNOTATION_PATH.is_file():
        selection = candidates.iloc[0:0].copy()
        selection["view_label"] = pd.Series(dtype=str)
        selection["view_order"] = pd.Series(dtype=int)
        print("No saved dropdown annotations yet.")
        print("Use the annotation program above and click 'Save this scene'.")
    else:
        annotations = pd.read_csv(ANNOTATION_PATH)
        annotated = annotations[annotations.view_label.isin(VIEW_LABELS)].copy()
        selection = candidates.merge(annotated, on="candidate_id", how="inner")

        label_counts = (
            selection.groupby(["scene", "view_label"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=expected_scenes, columns=VIEW_LABELS, fill_value=0)
        )
        display(label_counts.assign(valid=label_counts.eq(1).all(axis=1)))

        duplicate_scenes = label_counts.index[label_counts.gt(1).any(axis=1)].tolist()
        if duplicate_scenes:
            raise ValueError(f"Duplicate orientation labels found in scenes: {duplicate_scenes}")

        completed_scenes = label_counts.index[label_counts.eq(1).all(axis=1)].tolist()
        incomplete_scenes = sorted(set(expected_scenes) - set(completed_scenes))
        selection = selection[selection.scene.isin(completed_scenes)].copy()
        selection["view_order"] = selection.view_label.map(view_order_map)
        selection = selection.sort_values(["scene", "view_order"])

        if not selection.empty:
            display(selection[["scene", "view_label", "view_order", "candidate_id", "frame_id", "margin_fraction", "border_contact", "output_image"]])

        if incomplete_scenes:
            print(f"Completed {len(completed_scenes)}/{len(expected_scenes)} scenes.")
            print("Still annotate and save:")
            for scene in incomplete_scenes:
                print(" -", scene)
            print("The final selection file was not written yet.")
        else:
            final_counts = selection.groupby("scene").size()
            assert final_counts.eq(8).all(), "Every scene must have exactly eight labeled views."
            selection.to_csv(OUTPUT_PATH, index=False)
            print("All scenes complete. Saved final selection:", OUTPUT_PATH)
    '''),
    code('''
    def display_final_selection(selection):
        if selection.empty:
            print("No completed scenes to display yet. Save annotations in the dropdown program first.")
            return
        scenes = sorted(selection.scene.unique())
        fig, axes = plt.subplots(len(scenes), 8, figsize=(28, 3.2 * len(scenes)), squeeze=False)
        for row_index, scene in enumerate(scenes):
            rows = selection[selection.scene.eq(scene)].sort_values("view_order")
            for column_index, (_, row) in enumerate(rows.iterrows()):
                with Image.open(row.output_image) as opened:
                    axes[row_index, column_index].imshow(opened.convert("RGB"))
                orientation = row.get("view_label", f"view {column_index + 1}")
                axes[row_index, column_index].set_title(f"{orientation} | frame {int(row.frame_id)}", fontsize=8)
                axes[row_index, column_index].axis("off")
            axes[row_index, 0].set_ylabel(scene, rotation=0, ha="right", va="center", labelpad=85, fontsize=8)
        plt.suptitle("Final HQ200 selection — 8 rotationally ordered complete-car views per scene")
        plt.tight_layout()
        plt.show()

    display_final_selection(selection)
    '''),
])


write_notebook("03_scene_crop_method_inputs.ipynb", [
    md('''
    # 03 — Shared scene crops and reconstruction inputs

    This CPU notebook makes the vehicle larger without modifying the eight-view geometry independently. For each car it computes **one union crop from its eight selected training masks**, adds a configurable margin, and applies that same normalized rectangle to every view. Original BiRefNet files remain unchanged.

    It exports three reproducible derivatives: high-resolution cropped inputs for 3DGS, one shared 512-based set for DUSt3R/MASt3R, and a 14-pixel-grid set for VGGT. Small inputs are never enlarged. The manifest records crop, scale, and padding values for later camera-intrinsic transformation.
    '''),
    code('''
    import math
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image
    from tqdm.auto import tqdm
    '''),
    code(CPU_RUNTIME_GUARD),
    code(COMMON_PATHS),
    code(SRC_BOOTSTRAP),
    code('''
    from src.image_preprocessing import (
        VIEW_LABELS, contain_without_upscale, method_canvas,
        pixel_crop_box, shared_normalized_crop,
    )

    BIREFNET_ROOT = PROJECT_ROOT / "data_processed" / "birefnet_original_resolution"
    BIREFNET_MANIFEST = BIREFNET_ROOT / "manifest.csv"
    SELECTION_PATH = PROJECT_ROOT / "splits" / "sparse8" / "hq200_manual_8views.csv"
    OUTPUT_ROOT = PROJECT_ROOT / "data_processed" / "method_inputs"
    RUN_EXPORT = False
    OVERWRITE = False
    CROP_MARGIN = 0.10
    EXCLUDED_SCENES = {"car_12104835"}

    if not BIREFNET_MANIFEST.is_file():
        raise FileNotFoundError(f"Run notebook 01 in full mode first: {BIREFNET_MANIFEST}")
    if not SELECTION_PATH.is_file():
        raise FileNotFoundError(f"Finish and save notebook 02 annotations first: {SELECTION_PATH}")
    '''),
    code('''
    manifest = pd.read_csv(BIREFNET_MANIFEST)
    manifest = manifest[manifest.status.isin(["written", "skipped"])].copy()
    manifest = manifest[manifest.output_image.notna() & manifest.output_mask.notna()].copy()
    manifest = manifest[~manifest.scene.astype(str).str.casefold().isin({x.casefold() for x in EXCLUDED_SCENES})]
    manual = pd.read_csv(SELECTION_PATH)

    rows = []
    for (dataset, scene), group in manifest.groupby(["dataset", "scene"], sort=True):
        group = group.copy()
        if dataset == "3DRealCar":
            selected = manual[manual.scene.astype(str).eq(str(scene))].copy()
            if len(selected) != 8:
                raise ValueError(f"{scene}: expected 8 saved HQ annotations, found {len(selected)}")
            order_map = dict(zip(selected.source.astype(str), selected.view_order.astype(int)))
            group["split"] = np.where(group.source.astype(str).isin(order_map), "train", "test")
            group["view_order"] = group.source.astype(str).map(order_map)
        else:
            if len(group) != 8:
                print(f"Skipping {scene}: Industrial scene has {len(group)} rather than 8 images")
                continue
            group["split"] = "train"
            group["view_order"] = range(8)
        rows.append(group)

    inputs = pd.concat(rows, ignore_index=True)
    train_counts = inputs.query("split == 'train'").groupby(["dataset", "scene"]).size()
    assert train_counts.eq(8).all(), "Every retained scene must have exactly eight reconstruction inputs."
    display(train_counts.rename("training_views").to_frame())
    '''),
    code('''
    crop_rows = []
    for (dataset, scene), group in inputs.groupby(["dataset", "scene"], sort=True):
        training = group[group.split.eq("train")].sort_values("view_order")
        box = shared_normalized_crop(training.output_mask, margin_fraction=CROP_MARGIN)
        crop_sizes = []
        for row in group.itertuples(index=False):
            with Image.open(row.output_image) as image:
                px = pixel_crop_box(box, image.size)
            crop_sizes.append((px[2] - px[0], px[3] - px[1]))
        crop_rows.append({
            "dataset": dataset, "scene": scene,
            "crop_x0_norm": box[0], "crop_y0_norm": box[1],
            "crop_x1_norm": box[2], "crop_y1_norm": box[3],
            "scene_width": min(x[0] for x in crop_sizes),
            "scene_height": min(x[1] for x in crop_sizes),
        })
    scene_crops = pd.DataFrame(crop_rows)
    display(scene_crops)
    '''),
    code('''
    def export_row(row, crop):
        box_norm = tuple(crop[x] for x in ["crop_x0_norm", "crop_y0_norm", "crop_x1_norm", "crop_y1_norm"])
        with Image.open(row.output_image) as opened:
            rgb = opened.convert("RGB")
            crop_box = pixel_crop_box(box_norm, rgb.size)
            cropped_rgb = rgb.crop(crop_box)
        with Image.open(row.output_mask) as opened:
            cropped_mask = opened.convert("L").crop(crop_box)

        scene_size = (int(crop.scene_width), int(crop.scene_height))
        high_rgb, high_scale, high_pad_x, high_pad_y = contain_without_upscale(cropped_rgb, scene_size, (255, 255, 255))
        high_mask, _, _, _ = contain_without_upscale(cropped_mask, scene_size, 0)
        outputs = [("3dgs", high_rgb, high_mask, high_scale, high_pad_x, high_pad_y)]
        for method in ["dust3r_mast3r", "vggt"]:
            canvas = method_canvas(scene_size, method)
            out_rgb, second_scale, second_pad_x, second_pad_y = contain_without_upscale(high_rgb, canvas, (255, 255, 255))
            out_mask, _, _, _ = contain_without_upscale(high_mask, canvas, 0)
            total_scale = second_scale * high_scale
            total_pad_x = second_pad_x + second_scale * high_pad_x
            total_pad_y = second_pad_y + second_scale * high_pad_y
            outputs.append((method, out_rgb, out_mask, total_scale, total_pad_x, total_pad_y))

        records = []
        name = Path(row.output_image).name
        for method, out_rgb, out_mask, scale, pad_x, pad_y in outputs:
            base = OUTPUT_ROOT / method / row.dataset / row.scene
            image_path, mask_path = base / "images" / name, base / "masks" / name
            if OVERWRITE or not (image_path.is_file() and mask_path.is_file()):
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                out_rgb.save(image_path, compress_level=3)
                out_mask.save(mask_path, compress_level=3)
            records.append({
                **row._asdict(), "method": method,
                "crop_x0": crop_box[0], "crop_y0": crop_box[1],
                "crop_x1": crop_box[2], "crop_y1": crop_box[3],
                "scale_after_crop": scale, "pad_x": pad_x, "pad_y": pad_y,
                "method_width": out_rgb.width, "method_height": out_rgb.height,
                "method_image": str(image_path), "method_mask": str(mask_path),
            })
        return records

    if RUN_EXPORT:
        records = []
        crop_lookup = scene_crops.set_index(["dataset", "scene"])
        for row in tqdm(inputs.itertuples(index=False), total=len(inputs), desc="Exporting method inputs"):
            records.extend(export_row(row, crop_lookup.loc[(row.dataset, row.scene)]))
        method_manifest = pd.DataFrame(records)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        method_manifest.to_csv(OUTPUT_ROOT / "manifest.csv", index=False)
        scene_crops.to_csv(OUTPUT_ROOT / "scene_crops.csv", index=False)
        print("Saved:", OUTPUT_ROOT / "manifest.csv")
    else:
        print("Dry run complete. Set RUN_EXPORT=True to create method inputs (GPU is not needed).")
    '''),
    md('''
    ## Intrinsics after preprocessing

    If original intrinsics are $(f_x,f_y,c_x,c_y)$, use the recorded crop origin, uniform scale, and padding:

    $f'_x=s f_x$, $f'_y=s f_y$, $c'_x=s(c_x-x_0)+p_x$, and $c'_y=s(c_y-y_0)+p_y$.

    No new detail is invented: the 3DGS export never enlarges pixels, while the model-specific sets may only downsample. The shared per-scene crop keeps all eight views geometrically consistent.
    '''),
    code('''
    def show_all_3dgs_scenes(method_manifest):
        shown = method_manifest[(method_manifest.method == "3dgs") & (method_manifest.split == "train")].copy()
        for dataset in ["3DRealCar", "IndustrialInventory"]:
            part = shown[shown.dataset == dataset]
            scenes = sorted(part.scene.unique())
            fig, axes = plt.subplots(len(scenes), 8, figsize=(24, 3 * len(scenes)), squeeze=False)
            for r, scene in enumerate(scenes):
                group = part[part.scene == scene].sort_values(["view_order", "source"])
                for c, (_, item) in enumerate(group.head(8).iterrows()):
                    with Image.open(item.method_image) as image:
                        axes[r, c].imshow(image.convert("RGB"))
                    axes[r, c].axis("off")
                    if r == 0:
                        axes[r, c].set_title(VIEW_LABELS[c])
                axes[r, 0].set_ylabel(scene, rotation=0, ha="right", labelpad=90, fontsize=8)
            fig.suptitle(f"{dataset}: shared-crop 3DGS inputs")
            plt.tight_layout()
            plt.show()

    if RUN_EXPORT:
        show_all_3dgs_scenes(method_manifest)
    '''),
])


write_notebook("04_eda_8view_comparison.ipynb", [
    md('''
    # 0.1 — Combined EDA and sparse-view split

    This notebook compares the processed IndustrialInventory and 3DRealCar data at the same canvas size.

    - **3DRealCar:** select 8 approximately uniform training views from each dense capture; every remaining RGB view is test data.
    - **IndustrialInventory:** retain only complete 8-view scenes and use all eight as reconstruction input. `car_12104835` is explicitly excluded because it has only seven views.

    The split unit is a view within each car. Cars are kept separate throughout. Descriptive EDA compares only the eight training inputs per retained scene; dense 3DRealCar test views remain in the manifest for later novel-view evaluation.
    '''),
    code('''
    import json
    import math
    import re
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image
    from tqdm.auto import tqdm

    pd.set_option("display.max_columns", 100)
    '''),
    code(CPU_RUNTIME_GUARD),
    code(COMMON_PATHS),
    code('''
    PROCESSED_ROOT = PROJECT_ROOT / "data_processed" / "birefnet_normalized"
    MANIFEST_PATH = PROCESSED_ROOT / "manifest.csv"
    SPLIT_ROOT = PROJECT_ROOT / "splits" / "sparse8"
    MANUAL_HQ_SELECTION_PATH = SPLIT_ROOT / "hq200_manual_8views.csv"
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Run 0_Preprocessing.ipynb first; missing {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest.status.isin(["written", "skipped"])].copy()
    # Trust the completed preprocessing manifest. Per-file Path.is_file()
    # checks create thousands of Drive FUSE requests and can disconnect Drive.
    manifest = manifest[manifest.output_image.notna() & manifest.output_mask.notna()].copy()
    EXCLUDED_SCENES = {"car_12104835"}
    manifest = manifest[~manifest.scene.astype(str).str.casefold().isin({s.casefold() for s in EXCLUDED_SCENES})].copy()
    print("Processed images:", len(manifest))
    print("Excluded scenes:", sorted(EXCLUDED_SCENES))
    manual_hq = pd.read_csv(MANUAL_HQ_SELECTION_PATH) if MANUAL_HQ_SELECTION_PATH.is_file() else None
    print("HQ200 selection:", "manual pose-ordered file" if manual_hq is not None else "uniform capture-order fallback")
    display(manifest.groupby(["dataset", "scene"]).size().rename("images").to_frame())
    '''),
    md('''
    ## Eight-view selection

    If `0_2_HQ200_view_selection.ipynb` has saved a manual selection, those eight complete-car, pose-ordered views are used. Otherwise the notebook reports that it is falling back to uniformly spaced capture-order frames.
    '''),
    code('''
    def frame_number(path):
        match = re.search(r"frame_(\\d+)", Path(path).stem)
        return int(match.group(1)) if match else None

    def uniform_positions(n, count=8):
        if n < count:
            return np.arange(n, dtype=int)
        return np.unique(np.linspace(0, n - 1, count, dtype=int))

    def select_hq_training(group, count=8):
        ordered = group.assign(frame_id=group.source.map(frame_number)).sort_values(["frame_id", "source"])
        positions = uniform_positions(len(ordered), count)
        selected = ordered.iloc[positions].copy()
        selected["selection_method"] = "uniform_capture_order"
        return selected

    split_rows = []
    rejected_scenes = []
    for (dataset, scene), group in manifest.groupby(["dataset", "scene"], sort=True):
        group = group.copy()
        if dataset == "3DRealCar":
            if len(group) < 8:
                rejected_scenes.append({"dataset": dataset, "scene": scene, "reason": "fewer than 8 RGB views", "images": len(group)})
                continue
            if manual_hq is not None:
                manual_scene = manual_hq[manual_hq.scene.astype(str).eq(str(scene))].copy()
                requested_sources = set(manual_scene.source.astype(str))
                training = group[group.source.astype(str).isin(requested_sources)].copy()
                if len(training) != 8 or len(requested_sources) != 8:
                    raise ValueError(f"Manual HQ200 selection for {scene} does not match exactly 8 manifest images.")
                order_map = dict(zip(manual_scene.source.astype(str), manual_scene.view_order.astype(int)))
                training["view_order"] = training.source.astype(str).map(order_map)
                selection_method = "manual_pose_ordered_complete_car"
            else:
                training = select_hq_training(group, 8)
                training["view_order"] = np.arange(8)
                selection_method = "uniform_capture_order"
            train_indices = set(training.index)
            group["split"] = ["train" if idx in train_indices else "test" for idx in group.index]
            group["selection_method"] = selection_method
            group["view_order"] = np.nan
            group.loc[training.index, "view_order"] = training["view_order"]
        else:
            if len(group) != 8:
                rejected_scenes.append({"dataset": dataset, "scene": scene, "reason": "Industrial scene is not exactly 8 views", "images": len(group)})
                continue
            # All eight industrial views are inputs. There is no independent dense-view test set.
            group["split"] = "train"
            group["selection_method"] = "complete_industrial_8_views"
            group["view_order"] = np.arange(8)
        split_rows.append(group)

    split_manifest = pd.concat(split_rows, ignore_index=True)
    split_manifest.to_csv(SPLIT_ROOT / "split_manifest.csv", index=False)
    split_counts = split_manifest.groupby(["dataset", "scene", "split"]).size().unstack(fill_value=0)
    display(split_counts)
    train_counts = split_manifest.query("split == 'train'").groupby(["dataset", "scene"]).size()
    assert train_counts.eq(8).all(), "Every retained scene must have exactly eight training inputs."
    print(f"Retained scenes: {len(train_counts)}; all have exactly 8 training views.")
    if rejected_scenes:
        print("Rejected incomplete scenes:")
        display(pd.DataFrame(rejected_scenes))
    print("Saved:", SPLIT_ROOT / "split_manifest.csv")
    '''),
    code('''
    def masked_image_statistics(image_path, mask_path, sample_size=(256, 256), mask_threshold=0.5):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            array = np.asarray(image.resize(sample_size, Image.Resampling.BILINEAR), dtype=np.float32)
        with Image.open(mask_path) as opened_mask:
            mask = np.asarray(opened_mask.convert("L").resize(sample_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

        foreground = mask >= mask_threshold
        if not foreground.any():
            raise ValueError(f"Mask contains no foreground pixels: {mask_path}")

        pixels = array[foreground]
        gray = array.mean(axis=2)
        gx_values = np.abs(np.diff(gray, axis=1))
        gy_values = np.abs(np.diff(gray, axis=0))
        gx_valid = foreground[:, 1:] & foreground[:, :-1]
        gy_valid = foreground[1:, :] & foreground[:-1, :]
        sharpness = np.mean(np.concatenate([gx_values[gx_valid], gy_values[gy_valid]]))
        return {
            "width": width,
            "height": height,
            "aspect_ratio": width / height,
            "foreground_fraction": foreground.mean(),
            "brightness": pixels.mean(axis=1).mean(),
            "contrast": pixels.mean(axis=1).std(),
            "sharpness_proxy": sharpness,
            "red_mean": pixels[:, 0].mean(),
            "green_mean": pixels[:, 1].mean(),
            "blue_mean": pixels[:, 2].mean(),
        }

    # Compare only the eight sparse reconstruction inputs per scene. Dense HQ200
    # test views remain in split_manifest.csv for later rendered-view metrics.
    eda_inputs = split_manifest.query("split == 'train'").reset_index(drop=True).copy()
    stats = pd.DataFrame([
        masked_image_statistics(row.output_image, row.output_mask)
        for row in tqdm(eda_inputs.itertuples(index=False), total=len(eda_inputs), desc="Masked image statistics")
    ])
    eda = pd.concat([eda_inputs, stats], axis=1)
    eda.to_csv(SPLIT_ROOT / "image_statistics.csv", index=False)

    comparison = eda.groupby("dataset").agg(
        scenes=("scene", "nunique"), images=("output_image", "size"),
        width_mean=("width", "mean"), height_mean=("height", "mean"),
        foreground_fraction_mean=("foreground_fraction", "mean"),
        brightness_mean=("brightness", "mean"), brightness_std=("brightness", "std"),
        contrast_mean=("contrast", "mean"), sharpness_mean=("sharpness_proxy", "mean"),
    ).round(2)
    display(comparison)

    scene_comparison = eda.groupby(["dataset", "scene"]).agg(
        input_views=("output_image", "size"),
        foreground_fraction_mean=("foreground_fraction", "mean"),
        brightness_mean=("brightness", "mean"),
        contrast_mean=("contrast", "mean"),
        sharpness_mean=("sharpness_proxy", "mean"),
    ).round(2).reset_index()
    assert scene_comparison.input_views.eq(8).all()
    display(scene_comparison)
    scene_comparison.to_csv(SPLIT_ROOT / "scene_statistics_8views.csv", index=False)
    '''),
    code('''
    metrics = ["brightness", "contrast", "sharpness_proxy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))
    for ax, metric in zip(axes, metrics):
        for dataset, group in eda.groupby("dataset"):
            ax.hist(group[metric], bins=25, alpha=0.45, label=f"{dataset}: 8 training views")
        ax.set_title(metric.replace("_", " ").title())
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    '''),
    code('''
    def show_scene_views(frame, dataset, scene, split=None, max_images=16):
        chosen = frame[(frame.dataset == dataset) & (frame.scene == scene)].copy()
        if split is not None:
            chosen = chosen[chosen.split == split]
        chosen = chosen.sort_values("source")
        if len(chosen) > max_images:
            chosen = chosen.iloc[uniform_positions(len(chosen), max_images)]
        columns = min(4, len(chosen))
        rows = math.ceil(len(chosen) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows), squeeze=False)
        for ax, (_, row) in zip(axes.flat, chosen.iterrows()):
            ax.imshow(Image.open(row.output_image).convert("RGB"))
            ax.set_title(f"{row.split}: {Path(row.source).name}", fontsize=8)
            ax.axis("off")
        for ax in axes.flat[len(chosen):]:
            ax.axis("off")
        plt.suptitle(f"{dataset} — {scene}")
        plt.tight_layout()
        plt.show()

    industrial_scene = eda.query("dataset == 'IndustrialInventory'").scene.iloc[0]
    hq_scene = eda.query("dataset == '3DRealCar'").scene.iloc[0]
    show_scene_views(eda, "IndustrialInventory", industrial_scene, split="train", max_images=8)
    show_scene_views(eda, "3DRealCar", hq_scene, split="train", max_images=8)
    show_scene_views(split_manifest, "3DRealCar", hq_scene, split="test", max_images=16)
    '''),
    md('''
    ## Interpretation for reconstruction

    The eight training images are the sparse reconstruction inputs. The held-out 3DRealCar images are evaluation viewpoints and must not be used for COLMAP fitting, 3DGS training, hyperparameter selection, or background-model tuning. Render the trained model at their known/refined camera poses and compute PSNR, SSIM and LPIPS against these held-out images.

    IndustrialInventory has only the eight sparse inputs. Report reconstruction success and geometry/visual quality there, but do not label training-view agreement as unseen-view performance. A separate leave-one-out experiment can be added later if unseen-view evaluation on IndustrialInventory is required.
    '''),
])

# Replace the legacy split-building EDA above with a focused analysis of notebook 03 outputs.
write_notebook("04_eda_8view_comparison.ipynb", [
    md('''
    # 04 — Eight-view reconstruction-input EDA

    This notebook reads the **3DGS inputs produced by notebook 03** and compares exactly eight selected training views per retained scene. It does not resize or overwrite reconstruction images. A temporary 256×256 representation is used only to make descriptive statistics fast and comparable; this analysis resize is never saved as model input.

    3DRealCar held-out views remain available in the method manifest for later novel-view evaluation, but they are excluded from this input EDA. IndustrialInventory has only its eight reconstruction inputs.
    '''),
    code('''
    from pathlib import Path
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image
    from tqdm.auto import tqdm
    '''),
    code(CPU_RUNTIME_GUARD),
    code(COMMON_PATHS),
    code(SRC_BOOTSTRAP),
    code('''
    from src.eda_utils import masked_image_statistics
    from src.image_preprocessing import VIEW_LABELS

    METHOD_ROOT = PROJECT_ROOT / "data_processed" / "method_inputs"
    MANIFEST_PATH = METHOD_ROOT / "manifest.csv"
    EDA_ROOT = PROJECT_ROOT / "splits" / "sparse8"
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Run notebook 03 with RUN_EXPORT=True first: {MANIFEST_PATH}")

    method_manifest = pd.read_csv(MANIFEST_PATH)
    eda_inputs = method_manifest.query("method == '3dgs' and split == 'train'").copy()
    counts = eda_inputs.groupby(["dataset", "scene"]).size()
    assert counts.eq(8).all(), "EDA requires exactly eight selected inputs for every scene."
    print(f"Scenes: {len(counts)} | reconstruction inputs: {len(eda_inputs)}")
    display(counts.rename("views").to_frame())
    '''),
    md('''
    ## Masked statistics

    RGB and masks are sampled temporarily at 256×256 for computationally light dataset characterization. Brightness, contrast, and the gradient-based sharpness proxy are computed only inside the foreground mask. These are descriptive proxies, not perceptual-quality scores and not reconstruction preprocessing.
    '''),
    code('''
    statistics = []
    for row in tqdm(eda_inputs.itertuples(index=False), total=len(eda_inputs), desc="Analysis-only 256px statistics"):
        statistics.append(masked_image_statistics(row.method_image, row.method_mask, sample_size=(256, 256)))
    eda = pd.concat([eda_inputs.reset_index(drop=True), pd.DataFrame(statistics)], axis=1)
    EDA_ROOT.mkdir(parents=True, exist_ok=True)
    eda.to_csv(EDA_ROOT / "image_statistics.csv", index=False)

    comparison = eda.groupby("dataset").agg(
        scenes=("scene", "nunique"), images=("method_image", "size"),
        native_width_mean=("method_width", "mean"), native_height_mean=("method_height", "mean"),
        foreground_fraction_mean=("foreground_fraction", "mean"),
        brightness_mean=("brightness", "mean"), brightness_std=("brightness", "std"),
        contrast_mean=("contrast", "mean"), sharpness_mean=("sharpness_proxy", "mean"),
    ).round(2)
    display(comparison)

    scene_comparison = eda.groupby(["dataset", "scene"]).agg(
        input_views=("method_image", "size"),
        foreground_fraction_mean=("foreground_fraction", "mean"),
        brightness_mean=("brightness", "mean"),
        contrast_mean=("contrast", "mean"),
        sharpness_mean=("sharpness_proxy", "mean"),
    ).round(2).reset_index()
    assert scene_comparison.input_views.eq(8).all()
    display(scene_comparison)
    scene_comparison.to_csv(EDA_ROOT / "scene_statistics_8views.csv", index=False)
    '''),
    code('''
    metrics = ["foreground_fraction", "brightness", "contrast", "sharpness_proxy"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, metric in zip(axes, metrics):
        for dataset, group in eda.groupby("dataset"):
            ax.hist(group[metric], bins=20, density=True, alpha=0.45, label=dataset)
        ax.set_title(metric.replace("_", " ").title())
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    '''),
    code('''
    def show_all_selected(frame, dataset):
        part = frame[frame.dataset == dataset]
        scenes = sorted(part.scene.unique())
        fig, axes = plt.subplots(len(scenes), 8, figsize=(24, 3 * len(scenes)), squeeze=False)
        for r, scene in enumerate(scenes):
            group = part[part.scene == scene].sort_values(["view_order", "source"])
            for c, (_, row) in enumerate(group.iterrows()):
                with Image.open(row.method_image) as image:
                    axes[r, c].imshow(image.convert("RGB"))
                axes[r, c].axis("off")
                if r == 0:
                    axes[r, c].set_title(VIEW_LABELS[c])
            axes[r, 0].set_ylabel(scene, rotation=0, ha="right", labelpad=90, fontsize=8)
        fig.suptitle(f"{dataset}: eight annotated reconstruction inputs per scene")
        plt.tight_layout()
        plt.show()

    show_all_selected(eda, "3DRealCar")
    show_all_selected(eda, "IndustrialInventory")
    '''),
    md('''
    ## Interpretation

    Both datasets contribute eight training inputs per retained car; the datasets differ in their number of car scenes. Therefore raw image counts should not be interpreted as balanced sampling. Use 3DRealCar non-training frames only for held-out rendering metrics. Industrial training-view agreement is not unseen-view performance.
    '''),
])

print("Built numbered notebooks in code/notebooks")
