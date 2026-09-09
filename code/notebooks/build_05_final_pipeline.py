"""Build the unified VGGT-to-adaptive-3DGS notebook only."""

import json
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent


def md(value):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(value).strip() + "\n"}


def code(value):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": dedent(value).strip() + "\n"}


cells = [
    md(r'''
    # 05 — Final unified pipeline: VGGT → dense pseudo-views → adaptive 3DGS

    This single GPU notebook takes one annotated eight-view car scene through the complete selected method:

    1. verify and visualize the eight real inputs and BiRefNet masks;
    2. run VGGT once and visualize cameras, depth-based 3D points, and confidence;
    3. construct a frozen Gaussian-surfel teacher from foreground VGGT geometry;
    4. render ten dense pseudo-views between each adjacent pair, with continuous confidence and binary validity;
    5. diagnose pseudo-pixel reliability by reconstructing each real view without its own source points;
    6. train adaptive 3DGS using **real + confidence-weighted synthetic supervision only**;
    7. visualize real fits, pseudo-view fits, training curves, and a rotational MP4.

    The public VGGT model does not contain the paper's unreleased learned RGB-NVS head. Here “pseudo-view” means a dense VGGT-geometry teacher render. VGGT's official implementation notes that unprojecting predicted depth with predicted cameras usually gives more accurate points than the direct point-map branch, so that is the default.
    '''),
    code(r'''
    %pip -q install scipy pandas pillow matplotlib huggingface_hub einops safetensors opencv-python imageio imageio-ffmpeg "gsplat==1.3.0"
    !test -d /content/vggt/.git || git clone -q https://github.com/facebookresearch/vggt.git /content/vggt
    %pip -q install --no-deps -e /content/vggt
    '''),
    code(r'''
    import gc, shutil, subprocess, sys, time
    from pathlib import Path
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from IPython.display import Video, display
    from google.colab import drive

    if not torch.cuda.is_available():
        raise RuntimeError("Connect a Colab GPU runtime before running this notebook.")
    print("GPU:", torch.cuda.get_device_name(0))
    drive.mount("/content/drive", force_remount=False)

    CODE_ROOT = Path("/content/Project_Thesis_code")
    REPOSITORY = "ht" + "tps:" + chr(47)*2 + "github.com" + chr(47) + "katlit" + chr(47) + "Project_Thesis.git"
    BRANCH = "codex/hq200-example-notebook"
    if CODE_ROOT.exists() and not (CODE_ROOT / ".git").is_dir(): shutil.rmtree(CODE_ROOT)
    command = (["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CODE_ROOT)]
               if not CODE_ROOT.exists() else ["git", "-C", str(CODE_ROOT), "pull", "--ff-only", "origin", BRANCH])
    subprocess.run(command, check=True)
    sys.path[:0] = [str(CODE_ROOT / "code"), "/content/vggt"]
    '''),
    code(r'''
    from src.gaussian_full import initialize_full_gaussians, load_checkpoint, render_full, train_adaptive
    from src.vggt_geometry import confidence_error_table, confidence_to_unit_interval, interpolate_closed_orbit, multiview_depth_support
    from src.vggt_teacher import build_teacher_gaussians, coverage_summary, render_teacher
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    PROJECT_ROOT = Path("/content/drive/MyDrive/ITU/3D/Thesis")
    METHOD_MANIFEST = PROJECT_ROOT / "data_processed/method_inputs/manifest.csv"
    EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"

    # ---------- scene ----------
    DATASET = "3DRealCar"              # or IndustrialInventory
    SCENE = None                        # None chooses first available scene
    REFLECTION_HANDLED = False          # True selects the future *_ref input/output variant

    # ---------- VGGT geometry knobs ----------
    VGGT_MODEL = "facebook/VGGT-1B"
    USE_DEPTH_UNPROJECTION = True       # recommended; False uses direct world_points
    CONFIDENCE_PERCENTILES = (5.0, 95.0)
    POINT_CONFIDENCE_MIN = 0.05
    COMPUTE_MULTIVIEW_SUPPORT = True
    MIN_VIEW_SUPPORT = 1              # 1 preserves current behavior; inspect 2 and 3 in the sweep
    DEPTH_RELATIVE_TOLERANCE = 0.05
    DEPTH_ABSOLUTE_TOLERANCE = 0.01
    TUNING_CONFIDENCE_THRESHOLDS = (0.00, 0.05, 0.10, 0.20, 0.40)
    TUNING_MIN_VIEW_SUPPORT = (1, 2, 3)

    # ---------- dense pseudo-view teacher knobs ----------
    VIEWS_BETWEEN = 10                  # 10 => about 4.09 degrees for a 45-degree interval
    TEACHER_MAX_POINTS = 150_000
    TEACHER_SCALE_DIVISOR = 350.0
    TEACHER_SCALE_MULTIPLIER = 2.5      # increase if pseudo-views have small holes
    TEACHER_OPACITY = 0.95
    PSEUDO_ALPHA_MIN = 0.55             # decrease for more coverage, less certainty
    PSEUDO_CONFIDENCE_MIN = 0.10        # decrease for more coverage, less certainty

    # ---------- adaptive 3DGS knobs ----------
    STEPS = 30_000
    INITIAL_GAUSSIANS = 50_000
    SH_DEGREE = 3
    SYNTHETIC_PROBABILITY = 0.25
    SYNTHETIC_WEIGHT = 0.25
    CHECKPOINT_EVERY = 5_000
    RESUME = True

    manifest = pd.read_csv(METHOD_MANIFEST)
    available = manifest.query("method == 'vggt' and split == 'train' and dataset == @DATASET")
    SCENE = SCENE or sorted(available.scene.unique())[0]
    scene = available[available.scene.eq(SCENE)].sort_values(["view_order", "source"]).copy()
    assert len(scene) == 8 and pd.to_numeric(scene.view_order).astype(int).tolist() == list(range(8))
    dataset_name = {"3DRealCar": "3DRealCar", "IndustrialInventory": "Industrial"}[DATASET]
    DATASET_VARIANT = dataset_name + ("_ref" if REFLECTION_HANDLED else "")
    THREEDGS_EXPERIMENT = "VGGT_full_ref" if REFLECTION_HANDLED else "VGGT_full"
    GEOMETRY_ROOT = EXPERIMENT_ROOT / "Geometry" / "VGGT" / DATASET_VARIANT / SCENE
    SYNTHETIC_ROOT = EXPERIMENT_ROOT / "SyntheticViews" / "VGGT_NVS" / DATASET_VARIANT / SCENE
    SYNTHETIC_EVALUATION_ROOT = SYNTHETIC_ROOT / "evaluation"
    MESH_EVALUATION_ROOT = GEOMETRY_ROOT / "mesh_evaluation"
    RUN_ROOT = EXPERIMENT_ROOT / "3DGS" / THREEDGS_EXPERIMENT / DATASET_VARIANT / SCENE
    for directory in [GEOMETRY_ROOT, SYNTHETIC_EVALUATION_ROOT, MESH_EVALUATION_ROOT, RUN_ROOT]:
        directory.mkdir(parents=True, exist_ok=True)
    print("Selected:", DATASET_VARIANT, SCENE)
    print("Geometry:", GEOMETRY_ROOT)
    print("Synthetic views:", SYNTHETIC_ROOT)
    print("3DGS:", RUN_ROOT)
    '''),
    md('''## 1. Verify inputs and masks'''),
    code(r'''
    fig, axes = plt.subplots(2, 8, figsize=(28, 7), squeeze=False)
    for index, row in enumerate(scene.itertuples(index=False)):
        rgb = np.asarray(Image.open(row.method_image).convert("RGB"))
        mask = np.asarray(Image.open(row.method_mask).convert("L"))
        axes[0, index].imshow(rgb); axes[1, index].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[0, index].set_title(f"real {index}\n{Path(row.source).name}", fontsize=8)
        axes[1, index].set_title(f"mask · foreground {np.mean(mask >= 128):.2f}", fontsize=8)
        axes[0, index].axis("off"); axes[1, index].axis("off")
    axes[0, 0].set_ylabel("VGGT RGB"); axes[1, 0].set_ylabel("BiRefNet mask")
    plt.tight_layout(); plt.show()
    '''),
    md('''## 2. VGGT cameras, depth geometry, and confidence'''),
    code(r'''
    image_paths = scene.method_image.tolist()
    images = load_and_preprocess_images(image_paths).to("cuda")
    model = VGGT.from_pretrained(VGGT_MODEL).to("cuda").eval()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        prediction_torch = model(images)
        extrinsic_t, intrinsic_t = pose_encoding_to_extri_intri(prediction_torch["pose_enc"], images.shape[-2:])
    prediction = {key: value.detach().float().cpu().numpy().squeeze(0)
                  for key, value in prediction_torch.items() if isinstance(value, torch.Tensor)}
    extrinsics = extrinsic_t.detach().float().cpu().numpy().squeeze(0)
    intrinsics = intrinsic_t.detach().float().cpu().numpy().squeeze(0)
    colors = prediction["images"].transpose(0, 2, 3, 1)
    height, width = colors.shape[1:3]
    masks = np.stack([np.asarray(Image.open(path).convert("L").resize((width, height), Image.Resampling.NEAREST)) >= 128
                      for path in scene.method_mask])
    raw_confidence = prediction["depth_conf"] if USE_DEPTH_UNPROJECTION else prediction["world_points_conf"]
    confidence = np.stack([confidence_to_unit_interval(
        raw_confidence[index], masks[index], *CONFIDENCE_PERCENTILES
    ) for index in range(8)])
    points_by_view = (unproject_depth_map_to_point_map(
        prediction["depth"], extrinsics, intrinsics
    ) if USE_DEPTH_UNPROJECTION else prediction["world_points"])
    support = (multiview_depth_support(
        points_by_view, prediction["depth"], extrinsics, intrinsics, masks,
        relative_tolerance=DEPTH_RELATIVE_TOLERANCE,
        absolute_tolerance=DEPTH_ABSOLUTE_TOLERANCE,
    ) if COMPUTE_MULTIVIEW_SUPPORT else masks.astype(np.uint8))
    geometry_keep = (masks & np.isfinite(points_by_view).all(axis=-1) &
                     (confidence >= POINT_CONFIDENCE_MIN) & (support >= MIN_VIEW_SUPPORT))
    np.savez_compressed(GEOMETRY_ROOT / "vggt_geometry.npz", points=points_by_view,
                       raw_confidence=raw_confidence, confidence=confidence,
                       extrinsics=extrinsics, intrinsics=intrinsics, masks=masks, support=support)
    print("VGGT size:", width, "x", height, "| geometry:", "depth-unprojected" if USE_DEPTH_UNPROJECTION else "direct point map")
    del model, prediction_torch, images
    gc.collect(); torch.cuda.empty_cache()
    '''),
    code(r'''
    tuning_rows = []
    for threshold in TUNING_CONFIDENCE_THRESHOLDS:
        for minimum_support in TUNING_MIN_VIEW_SUPPORT:
            selected = masks & np.isfinite(points_by_view).all(axis=-1) & (confidence >= threshold) & (support >= minimum_support)
            tuning_rows.append({"confidence_min": threshold, "minimum_view_support": minimum_support,
                                "points": int(selected.sum()),
                                "retained_%_of_foreground": 100 * selected.sum() / max(masks.sum(), 1)})
    display(pd.DataFrame(tuning_rows).round(2))
    print("Active geometry:", int(geometry_keep.sum()), "points | confidence >=", POINT_CONFIDENCE_MIN,
          "| support >=", MIN_VIEW_SUPPORT)
    keep = geometry_keep
    points, point_colors, point_conf = points_by_view[keep], colors[keep], confidence[keep]
    rng = np.random.default_rng(42)
    if len(points) > 25_000:
        chosen = rng.choice(len(points), 25_000, replace=False)
        points, point_colors, point_conf = points[chosen], point_colors[chosen], point_conf[chosen]
    centered = points - np.median(points, axis=0)
    visible = np.linalg.norm(centered, axis=1) <= np.percentile(np.linalg.norm(centered, axis=1), 99)
    centered, point_colors, point_conf = centered[visible], point_colors[visible], point_conf[visible]
    fig = plt.figure(figsize=(15, 6)); ax1 = fig.add_subplot(121, projection="3d"); ax2 = fig.add_subplot(122, projection="3d")
    ax1.scatter(*centered.T, c=np.clip(point_colors, 0, 1), s=.7)
    plot = ax2.scatter(*centered.T, c=point_conf, cmap="viridis", vmin=0, vmax=1, s=.7)
    for axis, title in [(ax1, "Depth-unprojected foreground RGB"), (ax2, "Normalized VGGT confidence")]:
        axis.set_title(title); axis.set_box_aspect(np.ptp(centered, axis=0).clip(min=1e-6)); axis.view_init(18, -65)
    fig.colorbar(plot, ax=ax2, shrink=.65); plt.tight_layout(); plt.show()

    camera_to_world = np.linalg.inv(np.concatenate([
        extrinsics, np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (8, 1, 1))
    ], axis=1))
    centers = camera_to_world[:, :3, 3]
    fig = plt.figure(figsize=(7, 6)); axis = fig.add_subplot(111, projection="3d")
    axis.plot(*centers.T, "o-");
    for index, center in enumerate(centers): axis.text(*center, str(index))
    axis.set_title("VGGT camera orbit (0–7)"); axis.set_box_aspect(np.ptp(centers, axis=0).clip(min=1e-6)); plt.show()
    '''),
    md('''## 3. Dense Gaussian-surfel teacher and pseudo-views'''),
    code(r'''
    teacher_keep = geometry_keep
    teacher = build_teacher_gaussians(
        points_by_view[teacher_keep], colors[teacher_keep], confidence[teacher_keep],
        max_points=TEACHER_MAX_POINTS, scale_divisor=TEACHER_SCALE_DIVISOR,
        scale_multiplier=TEACHER_SCALE_MULTIPLIER, opacity=TEACHER_OPACITY,
    )
    print("Teacher surfels:", len(teacher["means"]), "| scene radius:", teacher["scene_radius"])
    cameras = interpolate_closed_orbit(extrinsics, intrinsics, views_between=VIEWS_BETWEEN)
    synthetic_views, coverage_rows = [], []
    pseudo_root = SYNTHETIC_ROOT
    for directory in [pseudo_root / "images", pseudo_root / "validity", pseudo_root / "confidence", pseudo_root / "alpha"]:
        directory.mkdir(parents=True, exist_ok=True)
    for index, camera in enumerate(cameras):
        rgb, validity, conf, alpha = render_teacher(
            teacher, camera["extrinsic"], camera["intrinsic"], height, width,
            alpha_min=PSEUDO_ALPHA_MIN, confidence_min=PSEUDO_CONFIDENCE_MIN,
        )
        synthetic_views.append({**camera, "rgb": rgb, "validity": validity, "confidence": conf,
                                "alpha": alpha, "height": height, "width": width})
        coverage_rows.append({"view": index, **coverage_summary(validity, conf, alpha)})
        display_rgb = np.ones_like(rgb); display_rgb[validity] = rgb[validity]
        Image.fromarray(np.uint8(display_rgb * 255)).save(pseudo_root / "images" / f"view_{index:03d}.png")
        Image.fromarray(np.uint8(validity) * 255).save(pseudo_root / "validity" / f"view_{index:03d}.png")
        Image.fromarray(np.uint8(conf * 255)).save(pseudo_root / "confidence" / f"view_{index:03d}.png")
        Image.fromarray(np.uint8(alpha * 255)).save(pseudo_root / "alpha" / f"view_{index:03d}.png")
    coverage = pd.DataFrame(coverage_rows); coverage.to_csv(SYNTHETIC_EVALUATION_ROOT / "coverage.csv", index=False)
    display(coverage.describe().round(3))
    '''),
    code(r'''
    # Eight rows: real view followed by ten dense teacher views toward the next real view.
    fig, axes = plt.subplots(8, VIEWS_BETWEEN + 1, figsize=(30, 23), squeeze=False)
    for real_index in range(8):
        axes[real_index, 0].imshow(colors[real_index]); axes[real_index, 0].set_title(f"real {real_index}")
        interval = sorted([view for view in synthetic_views if view["left_view"] == real_index], key=lambda view: view["step"])
        for column, view in enumerate(interval, start=1):
            shown = np.ones_like(view["rgb"]); shown[view["validity"]] = view["rgb"][view["validity"]]
            axes[real_index, column].imshow(shown)
            axes[real_index, column].set_title(f"step {column}\nvalid {view['validity'].mean():.2f}", fontsize=7)
        for axis in axes[real_index]: axis.axis("off")
        axes[real_index, 0].set_ylabel(f"{real_index} → {(real_index+1)%8}", rotation=0, ha="right", labelpad=45)
    fig.suptitle("Real input + ten dense VGGT-teacher interpolations per interval")
    plt.tight_layout(); plt.show()
    '''),
    md('''
    If coverage remains visibly incomplete, first increase `TEACHER_SCALE_MULTIPLIER` (for example 3.0), then consider lowering `PSEUDO_ALPHA_MIN` toward 0.40. Lowering confidence thresholds should be the last option because it admits less reliable geometry. Rerun from the teacher-construction cell after changing a knob; VGGT inference need not be repeated.
    '''),
    md('''## 4. Pseudo-confidence reliability on real targets'''),
    code(r'''
    reliability_rows = []
    fig, axes = plt.subplots(2, 8, figsize=(28, 7))
    for target in range(8):
        source = np.arange(8) != target
        source_keep = teacher_keep & source[:, None, None]
        heldout_teacher = build_teacher_gaussians(
            points_by_view[source_keep], colors[source_keep], confidence[source_keep],
            max_points=TEACHER_MAX_POINTS, scale_divisor=TEACHER_SCALE_DIVISOR,
            scale_multiplier=TEACHER_SCALE_MULTIPLIER, opacity=TEACHER_OPACITY,
        )
        rgb, valid, conf, alpha = render_teacher(
            heldout_teacher, extrinsics[target], intrinsics[target], height, width,
            alpha_min=PSEUDO_ALPHA_MIN, confidence_min=PSEUDO_CONFIDENCE_MIN,
        )
        evaluation = valid & masks[target]
        error = np.abs(rgb - colors[target]).mean(axis=-1)
        reliability_rows.extend({"target": target, **row} for row in confidence_error_table(conf, error, evaluation))
        shown = np.ones_like(rgb); shown[valid] = rgb[valid]
        axes[0, target].imshow(colors[target]); axes[1, target].imshow(shown)
        axes[0, target].set_title(f"real {target}"); axes[1, target].set_title(
            f"7-view teacher\ncoverage {evaluation.sum()/max(masks[target].sum(),1):.2f}", fontsize=8)
        axes[0, target].axis("off"); axes[1, target].axis("off")
        del heldout_teacher; torch.cuda.empty_cache()
    plt.tight_layout(); plt.show()
    reliability = pd.DataFrame(reliability_rows); reliability.to_csv(SYNTHETIC_EVALUATION_ROOT / "reliability.csv", index=False)
    curve = reliability.groupby("bin").agg(confidence=("mean_confidence", "mean"), error=("mean_absolute_error", "mean"), pixels=("pixels", "sum"))
    display(curve)
    curve.plot(x="confidence", y="error", marker="o", title="Does greater confidence correspond to lower held-out RGB error?")
    plt.show()
    print("This is still optimistic: VGGT jointly inferred cameras/depth from all eight inputs before source-point exclusion.")
    '''),
    md('''## 5. Adaptive 3DGS: real + confidence-weighted synthetic only'''),
    code(r'''
    real_views = [{"rgb": colors[index], "mask": masks[index], "extrinsic": extrinsics[index],
                   "intrinsic": intrinsics[index], "height": height, "width": width} for index in range(8)]
    checkpoints = sorted((RUN_ROOT / "checkpoints").glob("checkpoint_*.pt"))
    if RESUME and checkpoints:
        params, scene_scale, optimizers, strategy_state, history, start_step = load_checkpoint(checkpoints[-1])
        print("Resuming", checkpoints[-1], "at", start_step, "with", len(params["means"]), "Gaussians")
    else:
        params, scene_scale = initialize_full_gaussians(
            points_by_view[teacher_keep], colors[teacher_keep], max_initial=INITIAL_GAUSSIANS, sh_degree=SH_DEGREE
        )
        optimizers = strategy_state = None; history, start_step = [], 0
        print("Initial Gaussians:", len(params["means"]), "| scene scale:", scene_scale)
    del teacher; torch.cuda.empty_cache()
    '''),
    code(r'''
    start_time = time.time()
    params, history = train_adaptive(
        params, scene_scale, real_views, synthetic_views, RUN_ROOT / "checkpoints",
        steps=STEPS, synthetic_probability=SYNTHETIC_PROBABILITY,
        synthetic_weight=SYNTHETIC_WEIGHT, sh_degree=SH_DEGREE,
        checkpoint_every=CHECKPOINT_EVERY, optimizers=optimizers,
        strategy_state=strategy_state, history=history, start_step=start_step,
    )
    elapsed_hours = (time.time() - start_time) / 3600
    history = pd.DataFrame(history); history.to_csv(RUN_ROOT / "training_history.csv", index=False)
    print(f"Training time this session: {elapsed_hours:.2f} h | final Gaussians: {len(params['means']):,}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    history.plot(x="step", y="loss", logy=True, ax=axes[0], title="Training loss")
    history.plot(x="step", y="gaussians", ax=axes[1], title="Adaptive Gaussian count")
    plt.tight_layout(); plt.show()
    '''),
    md('''## 6. Final real-view and pseudo-view results'''),
    code(r'''
    metrics = []
    fig, axes = plt.subplots(2, 8, figsize=(28, 7))
    for index, sample in enumerate(real_views):
        with torch.inference_mode():
            rendered, alpha, _ = render_full(params,
                torch.as_tensor(sample["extrinsic"], dtype=torch.float32, device="cuda"),
                torch.as_tensor(sample["intrinsic"], dtype=torch.float32, device="cuda"),
                height, width, SH_DEGREE)
        rgb = rendered[0].permute(1,2,0).cpu().numpy(); foreground = sample["mask"]
        mse = np.mean((rgb - sample["rgb"])[foreground] ** 2); mae = np.mean(np.abs(rgb - sample["rgb"])[foreground])
        psnr = -10*np.log10(max(mse, 1e-10)); metrics.append({"view": index, "masked_mae": mae, "masked_psnr": psnr})
        axes[0,index].imshow(sample["rgb"]); axes[1,index].imshow(np.clip(rgb,0,1))
        axes[0,index].set_title(f"real {index}"); axes[1,index].set_title(f"3DGS · PSNR {psnr:.1f}")
        axes[0,index].axis("off"); axes[1,index].axis("off")
    plt.tight_layout(); plt.show()
    metrics = pd.DataFrame(metrics); metrics.to_csv(RUN_ROOT / "real_training_metrics.csv", index=False); display(metrics)
    print("Training-view metrics are diagnostics, not held-out NVS scores.")
    '''),
    code(r'''
    chosen = np.linspace(0, len(synthetic_views)-1, 16, dtype=int)
    fig, axes = plt.subplots(2, len(chosen), figsize=(32, 6))
    for column, index in enumerate(chosen):
        sample = synthetic_views[index]
        with torch.inference_mode():
            rendered, alpha, _ = render_full(params,
                torch.as_tensor(sample["extrinsic"], dtype=torch.float32, device="cuda"),
                torch.as_tensor(sample["intrinsic"], dtype=torch.float32, device="cuda"),
                height, width, SH_DEGREE)
        target = np.ones_like(sample["rgb"]); target[sample["validity"]] = sample["rgb"][sample["validity"]]
        rgb = rendered[0].permute(1,2,0).cpu().numpy(); opacity = alpha[0].permute(1,2,0).cpu().numpy()
        axes[0,column].imshow(target); axes[1,column].imshow(np.clip(rgb + 1-opacity,0,1))
        axes[0,column].set_title(f"pseudo {index}", fontsize=8); axes[1,column].set_title("3DGS", fontsize=8)
        axes[0,column].axis("off"); axes[1,column].axis("off")
    plt.tight_layout(); plt.show()
    '''),
    code(r'''
    video_path = RUN_ROOT / "final_closed_orbit.mp4"
    writer = imageio.get_writer(str(video_path), fps=15, codec="libx264", quality=8, macro_block_size=None)
    try:
        for camera in cameras:
            with torch.inference_mode():
                rendered, alpha, _ = render_full(params,
                    torch.as_tensor(camera["extrinsic"], dtype=torch.float32, device="cuda"),
                    torch.as_tensor(camera["intrinsic"], dtype=torch.float32, device="cuda"),
                    height, width, SH_DEGREE)
            rgb = rendered[0].permute(1,2,0).cpu().numpy(); opacity = alpha[0].permute(1,2,0).cpu().numpy()
            writer.append_data(np.uint8(np.clip(rgb + 1-opacity,0,1)*255))
    finally: writer.close()
    print("Saved:", video_path)
    display(Video(str(video_path), embed=True, html_attributes="controls autoplay loop muted"))
    '''),
    md(r'''
    ## Interpretation and tuning order

    Do not increase coverage by immediately accepting low-confidence geometry. Tune in this order: teacher splat scale, alpha threshold, then confidence threshold. Report pseudo-view coverage alongside reconstruction quality. Final thesis evaluation must use held-out real 3DRealCar images; neither training-view metrics nor agreement with synthetic targets is independent ground truth.
    '''),
]

notebook = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(HERE / "05_vggt_dense_pseudoview_adaptive_3dgs.ipynb").write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
)
print("Built 05_vggt_dense_pseudoview_adaptive_3dgs.ipynb")

# Keep the reference-mesh evaluation in regenerated copies while preserving
# the insertion logic separately from the already large notebook builder.
exec((HERE / "_insert_geometry_evaluation.py").read_text(encoding="utf-8"))
