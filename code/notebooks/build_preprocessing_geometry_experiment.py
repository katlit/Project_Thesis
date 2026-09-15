"""Generate the one-scene preprocessing x geometry ablation notebook."""
import json
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent


def md(value):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(value).strip() + "\n"}


def code(value):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": dedent(value).strip() + "\n"}


cells = [
md('''
# EXP — Preprocessing sensitivity of VGGT and MASt3R geometry

This one-scene pilot asks a narrow question: **does crop strength or reflection removal materially change geometry?**
It reconstructs the same eight selected views with two methods and six inputs:

1. `full` — original-resolution BiRefNet foreground canvas;
2. `full_ref` — reflection handling applied to that full canvas;
3. `loose` — a shared, generous crop built from the original-resolution masks;
4. `loose_ref` — reflection handling applied independently to the loose images;
5. `current` — the canonical crop exported by notebook 03_A;
6. `current_ref` — its output from notebook 04.

This is a controlled pilot, not evidence that generalizes to the complete dataset. A large, consistent change is useful
for deciding whether the full ablation is worthwhile. Small differences on one car are inconclusive.

**Methods.** VGGT is feed-forward geometry followed here by foreground/confidence filtering. MASt3R uses pairwise
matching plus sparse global alignment. “MUST3R” is corrected to the official name **MASt3R**. MASt3R code/model use
the repository's non-commercial CC BY-NC-SA license; check this before redistributing results.
'''),
code('''
# Keep Colab's mutually compatible NumPy/OpenCV stack. UnReflectAnything's
# published metadata pins newer core packages that conflict with Colab, even
# though inference does not require those exact patch versions.
%pip -q uninstall -y xformers diffusers gradio
%pip -q install "pandas==2.2.3" "pillow==12.1.1" "huggingface-hub==0.36.2" matplotlib scipy trimesh einops roma scikit-learn

print("Dependencies prepared. Continue without restarting unless Colab explicitly requests it.")
'''),
code('''
import gc, json, os, shutil, subprocess, sys, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import trimesh
from PIL import Image
from google.colab import drive

if not torch.cuda.is_available(): raise RuntimeError("Connect a GPU runtime.")
drive.mount("/content/drive", force_remount=False)
PROJECT_ROOT = Path("/content/drive/MyDrive/ITU/3D/Thesis")
CODE_ROOT = Path("/content/Project_Thesis_code")
REPOSITORY = "https://github.com/katlit/Project_Thesis.git"
BRANCH = "codex/hq200-example-notebook"
if CODE_ROOT.exists() and not (CODE_ROOT / ".git").is_dir(): shutil.rmtree(CODE_ROOT)
command = (["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CODE_ROOT)]
           if not CODE_ROOT.exists() else ["git", "-C", str(CODE_ROOT), "pull", "--ff-only", "origin", BRANCH])
subprocess.run(command, check=True)
sys.path.insert(0, str(CODE_ROOT / "code"))

from src.preprocessing_geometry_ablation import build_standard_variants, VARIANT_ORDER
from src.reflection_preprocessing import process_with_unreflectanything
from src.geometry_evaluation import align_similarity_icp, apply_similarity, geometry_metrics
from src.vggt_geometry import confidence_to_unit_interval
'''),
md('''
## Configuration

The defaults reproduce the previously studied scene. Every expensive stage resumes from saved files unless its
overwrite switch is enabled. `loose_margin=0.25` means 25% of the shared foreground-box width/height is added on each
side. Full and loose variants are derived from BiRefNet originals—not by padding the canonical crop.
'''),
code('''
DATASET = "3DRealCar"
SCENE = "2024_09_11_12_52_22"
LOOSE_CROP_MARGIN = 0.25
RUN_REFLECTION = True
RUN_MAST3R = True
OVERWRITE_INPUTS = False
OVERWRITE_GEOMETRY = False
VGGT_CONFIDENCE_MIN = 0.10
CONFIDENCE_PERCENTILES = (5, 95)
MAST3R_CONFIDENCE_MIN = 1.5
MAST3R_MATCHING_CONFIDENCE = 5.0
MAST3R_SHARED_INTRINSICS = True  # correct when all eight frames use the same physical camera
REFERENCE_SAMPLES = 100_000
DISPLAY_POINTS = 25_000

INPUT_ROOT = PROJECT_ROOT / "data_processed/preprocessing_geometry_ablation" / DATASET / SCENE
SUMMARY_ROOT = PROJECT_ROOT / "experiments/Geometry/Preprocessing_Ablation" / DATASET / SCENE
GEOMETRY_ROOT = PROJECT_ROOT / "experiments/Geometry"
for path in (INPUT_ROOT, SUMMARY_ROOT): path.mkdir(parents=True, exist_ok=True)
manifest = pd.read_csv(PROJECT_ROOT / "data_processed/method_inputs/manifest.csv")
scene_rows = manifest.query("dataset == @DATASET and scene == @SCENE and split == 'train'").copy()
assert set(scene_rows.method) >= {"vggt", "dust3r_mast3r"}
assert scene_rows[scene_rows.method.eq("vggt")].view_order.nunique() == 8
display(scene_rows[["method", "view_order", "output_image", "method_image"]].sort_values(["method", "view_order"]))
'''),
md('''
## 1 — Build standard inputs

All variants keep one shared crop definition across the eight views. Aspect ratio is preserved and blank padding is
added only to reach each model's required canvas. This changes the effective focal length in pixels, which is exactly
the preprocessing sensitivity being measured.
'''),
code('''
input_manifest, loose_box = build_standard_variants(
    scene_rows, INPUT_ROOT, loose_margin=LOOSE_CROP_MARGIN, overwrite=OVERWRITE_INPUTS,
)
print("Loose normalized crop:", np.round(loose_box, 4))
display(input_manifest.groupby(["variant", "method"]).size().rename("images").reset_index())
'''),
md('''
## 2 — Build reflection-handled inputs

Reflection removal is run on each standard variant itself. It never changes the mask, crop, padding, or resolution.
The current reflected inputs are copied from notebook 04 when available. Full and loose reflection variants are newly
computed. This stage is optional because it is expensive and its model is large.
'''),
code('''
if RUN_REFLECTION:
    # Exact dependency combination expected by UnReflectAnything 1.1.1.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"], check=False)
    # Install the runtime dependencies explicitly but do not let the package's
    # strict NumPy/OpenCV pins replace Colab's binary-compatible stack.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "dill==0.4.1", "dotmap==1.3.30", "fvcore==0.1.5.post20221221",
                    "natsort==8.4.0", "protobuf==6.33.5", "python-dotenv==1.2.1",
                    "pyyaml==6.0.3", "tqdm==4.67.3",
                    "git+https://github.com/huggingface/transformers.git@2fe43376cdde02b7ffcf117e6eb9aa4375fb2dd1"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "unreflectanything==1.1.1"], check=True)
    import unreflectanything
    weights = Path(unreflectanything.cache("weights"))
    if not any(weights.glob("*.pth")):
        subprocess.run(["unreflectanything", "download", "--weights"], check=True)
    reflection_model = unreflectanything.model(pretrained=True, device=torch.device("cuda"), verbose=False)
    reflection_model.eval().requires_grad_(False)

    reflected_records = []
    ref_manifest_path = PROJECT_ROOT / "data_processed/method_inputs_ref/manifest.csv"
    ref_manifest = pd.read_csv(ref_manifest_path) if ref_manifest_path.is_file() else pd.DataFrame()
    for row in input_manifest.itertuples():
        if row.variant not in ("full", "loose", "current"): continue
        variant = row.variant + "_ref"
        image_out = INPUT_ROOT / "inputs" / variant / row.method / "images" / f"view_{row.view:02d}.png"
        mask_out = INPUT_ROOT / "inputs" / variant / row.method / "masks" / f"view_{row.view:02d}.png"
        copied = False
        if row.variant == "current" and not ref_manifest.empty:
            candidates = ref_manifest[(ref_manifest.dataset.eq(DATASET)) & (ref_manifest.scene.eq(SCENE)) &
                                      (ref_manifest.method.eq(row.method)) & (ref_manifest.view_order.eq(row.view))]
            if len(candidates) == 1:
                source = candidates.iloc[0]
                image_out.parent.mkdir(parents=True, exist_ok=True); mask_out.parent.mkdir(parents=True, exist_ok=True)
                if OVERWRITE_INPUTS or not image_out.is_file(): shutil.copy2(source.method_image, image_out)
                if OVERWRITE_INPUTS or not mask_out.is_file(): shutil.copy2(source.method_mask, mask_out)
                copied = True
        report = ({"status": "copied"} if copied else process_with_unreflectanything(
            row.image, row.mask, image_out, mask_out, reflection_model,
            overwrite=OVERWRITE_INPUTS, use_amp=True,
        ))
        reflected_records.append({"variant": variant, "method": row.method, "view": row.view,
                                  "image": str(image_out), "mask": str(mask_out), **report})
    del reflection_model; gc.collect(); torch.cuda.empty_cache()
    input_manifest = pd.concat([input_manifest, pd.DataFrame(reflected_records)], ignore_index=True)

missing = set(VARIANT_ORDER) - set(input_manifest.variant)
if missing: raise RuntimeError(f"Missing input variants: {sorted(missing)}")
input_manifest["variant"] = pd.Categorical(input_manifest.variant, VARIANT_ORDER, ordered=True)
input_manifest = input_manifest.sort_values(["variant", "method", "view"])
input_manifest.to_csv(INPUT_ROOT / "manifest.csv", index=False)
print("Saved input manifest:", INPUT_ROOT / "manifest.csv")
'''),
md('''
## 3 — Visualize all six input types

Each row is one preprocessing variant and columns are the eight physical views. The gallery uses the VGGT canvases;
MASt3R receives the same content and mask with its own required resolution. The table verifies both canvas sizes.
'''),
code('''
gallery = input_manifest[input_manifest.method.eq("vggt")]
fig, axes = plt.subplots(6, 8, figsize=(24, 17), squeeze=False)
for row_index, variant in enumerate(VARIANT_ORDER):
    rows = gallery[gallery.variant.astype(str).eq(variant)].sort_values("view")
    for column, item in enumerate(rows.itertuples()):
        axes[row_index, column].imshow(Image.open(item.image))
        axes[row_index, column].axis("off")
        if row_index == 0: axes[row_index, column].set_title(f"view {column}")
    axes[row_index, 0].set_ylabel(variant, fontsize=12)
plt.suptitle("Six preprocessing variants — identical eight selected views", fontsize=16)
plt.tight_layout(); plt.show()

sizes = []
for item in input_manifest.itertuples():
    with Image.open(item.image) as opened: sizes.append((str(item.variant), item.method, opened.width, opened.height))
display(pd.DataFrame(sizes, columns=["variant", "method", "width", "height"]).drop_duplicates().sort_values(["method", "variant"]))
'''),
md('''
## 4 — VGGT reconstructions

VGGT predicts depth, cameras, and confidence jointly. Depth is unprojected through the camera it predicts. Only finite
foreground pixels above the normalized confidence threshold are retained. The raw prediction is also saved, so this
display threshold can be changed later without rerunning the network.
'''),
code('''
VGGT_ROOT = Path("/content/vggt")
if not (VGGT_ROOT / ".git").is_dir(): subprocess.run(["git", "clone", "https://github.com/facebookresearch/vggt.git", VGGT_ROOT], check=True)
sys.path.insert(0, str(VGGT_ROOT))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
for variant in VARIANT_ORDER:
    output = GEOMETRY_ROOT / "VGGT" / DATASET / SCENE / "preprocessing_ablation" / variant / "geometry.npz"
    if output.is_file() and not OVERWRITE_GEOMETRY:
        print("VGGT cached:", variant); continue
    rows = input_manifest[(input_manifest.method.eq("vggt")) & (input_manifest.variant.astype(str).eq(variant))].sort_values("view")
    images = load_and_preprocess_images(rows.image.tolist()).cuda()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype): prediction = vggt_model(images[None])
    extrinsic, intrinsic = pose_encoding_to_extri_intri(prediction["pose_enc"], images.shape[-2:])
    depth = prediction["depth"].detach().float().cpu().numpy().squeeze(0)
    extrinsic = extrinsic.detach().float().cpu().numpy().squeeze(0)
    intrinsic = intrinsic.detach().float().cpu().numpy().squeeze(0)
    points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
    raw_conf = prediction["depth_conf"].detach().float().cpu().numpy().squeeze(0)
    colors = prediction["images"].detach().float().cpu().numpy().squeeze(0).transpose(0, 2, 3, 1)
    h, w = colors.shape[1:3]
    masks = np.stack([np.asarray(Image.open(path).convert("L").resize((w, h), Image.Resampling.NEAREST)) >= 128 for path in rows["mask"]])
    conf = np.stack([confidence_to_unit_interval(raw_conf[i], masks[i], *CONFIDENCE_PERCENTILES) for i in range(8)])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, points=points, colors=colors, confidence=conf, raw_confidence=raw_conf,
                        masks=masks, extrinsics=extrinsic, intrinsics=intrinsic)
    del images, prediction; gc.collect(); torch.cuda.empty_cache(); print("VGGT saved:", variant)
del vggt_model; gc.collect(); torch.cuda.empty_cache()
'''),
md('''
## 5 — MASt3R reconstructions

MASt3R matches all view pairs, then globally optimizes cameras, intrinsics, depths, and 3D structure. Six complete-graph
runs are computationally expensive; each variant has its own cache and completed `.npz`, so the cell safely resumes.
'''),
code('''
MAST3R_ROOT = Path("/content/mast3r")
if not (MAST3R_ROOT / ".git").is_dir():
    subprocess.run(["git", "clone", "--recursive", "https://github.com/naver/mast3r.git", MAST3R_ROOT], check=True)
else:
    subprocess.run(["git", "-C", str(MAST3R_ROOT), "submodule", "update", "--init", "--recursive"], check=True)
sys.path[:0] = [str(MAST3R_ROOT), str(MAST3R_ROOT / "dust3r")]
from mast3r.model import AsymmetricMASt3R
from src.mast3r_geometry import reconstruct_mast3r

if RUN_MAST3R:
    mast3r_model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").cuda().eval()
    for variant in VARIANT_ORDER:
        output = GEOMETRY_ROOT / "MASt3R" / DATASET / SCENE / "preprocessing_ablation" / variant / "geometry.npz"
        if output.is_file() and not OVERWRITE_GEOMETRY:
            print("MASt3R cached:", variant); continue
        rows = input_manifest[(input_manifest.method.eq("dust3r_mast3r")) & (input_manifest.variant.astype(str).eq(variant))].sort_values("view")
        result = reconstruct_mast3r(rows.image.tolist(), mast3r_model,
            SUMMARY_ROOT / "mast3r_cache" / variant, shared_intrinsics=MAST3R_SHARED_INTRINSICS,
            matching_conf_thr=MAST3R_MATCHING_CONFIDENCE)
        h, w = result["points"].shape[1:3]
        masks = np.stack([np.asarray(Image.open(path).convert("L").resize((w, h), Image.Resampling.NEAREST)) >= 128 for path in rows["mask"]])
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **result, masks=masks)
        print("MASt3R saved:", variant)
    del mast3r_model; gc.collect(); torch.cuda.empty_cache()
'''),
md('''
## 6 — Twelve reconstruction visualizations

Rows are methods and columns are preprocessing variants. Each panel applies only robust centering and a common display
extent; it does not alter saved geometry or evaluation. Point subsampling is visualization-only.
'''),
code('''
methods = ("VGGT", "MASt3R")
clouds = {}
for method in methods:
    for variant in VARIANT_ORDER:
        path = GEOMETRY_ROOT / method / DATASET / SCENE / "preprocessing_ablation" / variant / "geometry.npz"
        saved = np.load(path)
        threshold = VGGT_CONFIDENCE_MIN if method == "VGGT" else MAST3R_CONFIDENCE_MIN
        keep = saved["masks"].astype(bool) & np.isfinite(saved["points"]).all(-1) & (saved["confidence"] >= threshold)
        points, colors = saved["points"][keep], saved["colors"][keep]
        rng = np.random.default_rng(42)
        if len(points) > DISPLAY_POINTS:
            chosen = rng.choice(len(points), DISPLAY_POINTS, replace=False); points, colors = points[chosen], colors[chosen]
        center = np.median(points, axis=0); points = points - center
        radius = np.percentile(np.linalg.norm(points, axis=1), 99); visible = np.linalg.norm(points, axis=1) <= radius
        clouds[(method, variant)] = (points[visible], colors[visible], len(saved["points"][keep]))

fig = plt.figure(figsize=(24, 9))
for r, method in enumerate(methods):
    for c, variant in enumerate(VARIANT_ORDER):
        ax = fig.add_subplot(2, 6, r * 6 + c + 1, projection="3d")
        points, colors, count = clouds[(method, variant)]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.clip(colors, 0, 1), s=.15)
        ax.set_title(f"{method} — {variant}\\n{count:,} retained")
        ax.set_box_aspect(np.ptp(points, axis=0).clip(min=1e-6)); ax.view_init(18, -65)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
plt.suptitle("All 12 geometry reconstructions", fontsize=16); plt.tight_layout(); plt.show()
'''),
md('''
## 7 — Reference-mesh evaluation

Each cloud is independently aligned to the clean car-only mesh with robust similarity ICP, because monocular methods
do not recover metric scale. We report accuracy (prediction→mesh), completeness (mesh→prediction), symmetric
Chamfer-L1, and precision/recall/F-score at 1, 2, and 5 cm. **Lower is better** for distances; **higher is better** for
precision, recall, and F-score. Similarity alignment makes this a shape-quality test, not a camera-scale test.
'''),
code('''
mesh_path = PROJECT_ROOT / "data_processed/reference_meshes" / DATASET / SCENE / "car_reference.obj"
if not mesh_path.is_file(): raise FileNotFoundError(f"Clean reference mesh required: {mesh_path}")
mesh = trimesh.load(mesh_path, force="mesh", process=False)
reference, _ = trimesh.sample.sample_surface(mesh, REFERENCE_SAMPLES, seed=42)
metric_rows = []
for method in methods:
    for variant in VARIANT_ORDER:
        path = GEOMETRY_ROOT / method / DATASET / SCENE / "preprocessing_ablation" / variant / "geometry.npz"
        saved = np.load(path)
        threshold = VGGT_CONFIDENCE_MIN if method == "VGGT" else MAST3R_CONFIDENCE_MIN
        keep = saved["masks"].astype(bool) & np.isfinite(saved["points"]).all(-1) & (saved["confidence"] >= threshold)
        predicted = saved["points"][keep]
        rng = np.random.default_rng(42)
        sample = predicted[rng.choice(len(predicted), min(len(predicted), REFERENCE_SAMPLES), replace=False)]
        transform = align_similarity_icp(sample, reference)
        aligned = apply_similarity(sample, transform)
        values, _, _ = geometry_metrics(aligned, reference, (0.01, 0.02, 0.05))
        metric_rows.append({"method": method, "variant": variant, "retained_points": len(predicted),
                            "similarity_scale": transform["scale"], **values})
metrics = pd.DataFrame(metric_rows)
metrics.to_csv(SUMMARY_ROOT / "geometry_metrics.csv", index=False)
display(metrics[["method", "variant", "retained_points", "accuracy_median", "completeness_median",
                 "chamfer_l1", "fscore@0.01", "fscore@0.02", "fscore@0.05"]].round(4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for method in methods:
    part = metrics[metrics.method.eq(method)].set_index("variant").loc[list(VARIANT_ORDER)]
    axes[0].plot(VARIANT_ORDER, 100 * part.chamfer_l1, "o-", label=method)
    axes[1].plot(VARIANT_ORDER, part["fscore@0.05"], "o-", label=method)
axes[0].set_ylabel("Chamfer-L1 (cm), lower is better"); axes[1].set_ylabel("F-score @ 5 cm, higher is better")
for ax in axes: ax.tick_params(axis="x", rotation=35); ax.grid(alpha=.25); ax.legend()
plt.suptitle("One-scene preprocessing sensitivity"); plt.tight_layout(); plt.show()
print("Inputs:", INPUT_ROOT); print("Geometry:", GEOMETRY_ROOT); print("Summary:", SUMMARY_ROOT)
'''),
md('''
## Interpretation checklist

- Prefer a preprocessing variant only if it improves both distance and F-score, and the reconstruction looks coherent.
- If reflection handling improves appearance but harms geometry, do not use it as the default geometry input; it can
  remain an appearance-training branch.
- A better result for both VGGT and MASt3R is stronger evidence than a change in only one method.
- Choose the next full-dataset experiment only after checking that gains exceed run-to-run/threshold sensitivity.
- This pilot uses training inputs and a reference mesh; it does not measure held-out novel-view rendering quality.
''')]

notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "EXP_Preprocessing_Geometry.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"}}, "nbformat": 4, "nbformat_minor": 5}
(HERE / "EXP_Preprocessing_Geometry.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print("Wrote", HERE / "EXP_Preprocessing_Geometry.ipynb")
