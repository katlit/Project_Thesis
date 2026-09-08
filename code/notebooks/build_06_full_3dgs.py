"""Build only the full real-plus-synthetic 3DGS notebook."""

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
    # 06 — Full adaptive 3DGS: real + confidence-weighted synthetic views

    This notebook is the long-training version of notebook 05. It consumes that notebook's saved VGGT predictions and 80 geometry-guided pseudo-views and trains one car with only the selected condition:

    - eight real views with foreground RGB/SSIM and full silhouette supervision;
    - synthetic views with binary validity × detached continuous confidence;
    - 75% real sampling and 25% synthetic sampling;
    - synthetic loss multiplier 0.25;
    - spherical harmonics progressively activated to degree 3;
    - adaptive Gaussian cloning, splitting, pruning, and opacity resets;
    - 30,000 optimization steps and checkpoints every 5,000 steps;
    - training-view diagnostics and an MP4 closed-orbit render.

    Notebook 05 must have completed the same `DATASET` and `SCENE` first. Pseudo-views are VGGT geometry-guided reprojections, not outputs of the unreleased learned RGB NVS head.
    '''),
    code(r'''
    %pip -q install scipy pandas pillow huggingface_hub einops safetensors opencv-python imageio imageio-ffmpeg "gsplat==1.3.0"
    !test -d /content/vggt/.git || git clone -q https://github.com/facebookresearch/vggt.git /content/vggt
    %pip -q install --no-deps -e /content/vggt
    '''),
    code(r'''
    import shutil, subprocess, sys
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
        raise RuntimeError("Connect a Colab GPU runtime before running notebook 06.")
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
    from src.gaussian_full import (
        initialize_full_gaussians, load_checkpoint, render_full, train_adaptive,
    )
    from src.vggt_geometry import interpolate_closed_orbit
    from vggt.utils.load_fn import load_and_preprocess_images

    PROJECT_ROOT = Path("/content/drive/MyDrive/ITU/3D/Thesis")
    METHOD_MANIFEST = PROJECT_ROOT / "data_processed/method_inputs/manifest.csv"
    SMOKE_ROOT = PROJECT_ROOT / "experiments/vggt_pseudoview_3dgs_smoke"
    OUTPUT_ROOT = PROJECT_ROOT / "experiments/3dgs_real_plus_synthetic_full"

    DATASET = "3DRealCar"
    SCENE = None  # None uses the first scene; match notebook 05
    VIEWS_BETWEEN = 10
    STEPS = 30_000
    CHECKPOINT_EVERY = 5_000
    MAX_INITIAL_GAUSSIANS = 50_000
    SYNTHETIC_PROBABILITY = 0.25
    SYNTHETIC_WEIGHT = 0.25
    SH_DEGREE = 3
    RESUME = True

    manifest = pd.read_csv(METHOD_MANIFEST)
    selected = manifest.query("method == 'vggt' and split == 'train' and dataset == @DATASET")
    SCENE = SCENE or sorted(selected.scene.unique())[0]
    scene = selected[selected.scene.eq(SCENE)].sort_values(["view_order", "source"]).copy()
    assert len(scene) == 8
    SOURCE_ROOT = SMOKE_ROOT / DATASET / SCENE
    RUN_ROOT = OUTPUT_ROOT / DATASET / SCENE
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH = SOURCE_ROOT / "vggt_predictions.npz"
    if not PREDICTION_PATH.is_file():
        raise FileNotFoundError(f"Run notebook 05 for this scene first: {PREDICTION_PATH}")
    print("Scene:", DATASET, SCENE)
    '''),
    code(r'''
    saved = np.load(PREDICTION_PATH)
    world_points = saved["world_points"]
    confidence = saved["confidence_unit"]
    extrinsics, intrinsics, masks = saved["extrinsic"], saved["intrinsic"], saved["masks"].astype(bool)
    input_tensor = load_and_preprocess_images(scene.method_image.tolist())
    colors = input_tensor.numpy().transpose(0, 2, 3, 1)
    height, width = colors.shape[1:3]
    assert masks.shape[1:] == (height, width)

    real_views = [{
        "rgb": colors[index], "mask": masks[index], "extrinsic": extrinsics[index],
        "intrinsic": intrinsics[index], "height": height, "width": width,
    } for index in range(8)]

    pseudo_root = SOURCE_ROOT / "pseudo_views"
    cameras = interpolate_closed_orbit(extrinsics, intrinsics, views_between=VIEWS_BETWEEN)
    synthetic_views = []
    for index, camera in enumerate(cameras):
        image_path = pseudo_root / "images" / f"view_{index:03d}.png"
        validity_path = pseudo_root / "validity" / f"view_{index:03d}.png"
        confidence_path = pseudo_root / "confidence" / f"view_{index:03d}.png"
        if not all(path.is_file() for path in [image_path, validity_path, confidence_path]):
            raise FileNotFoundError(f"Missing notebook 05 pseudo-view files at index {index}")
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255
        validity = np.asarray(Image.open(validity_path).convert("L")) >= 128
        conf = np.asarray(Image.open(confidence_path).convert("L"), dtype=np.float32) / 255
        synthetic_views.append({
            **camera, "rgb": rgb, "validity": validity, "confidence": conf,
            "height": height, "width": width,
        })
    print("Loaded", len(real_views), "real and", len(synthetic_views), "synthetic views")
    '''),
    code(r'''
    checkpoints = sorted(RUN_ROOT.glob("checkpoint_*.pt"))
    if RESUME and checkpoints:
        latest = checkpoints[-1]
        params, scene_scale, optimizers, strategy_state, history, start_step = load_checkpoint(latest)
        print("Resuming:", latest, "from step", start_step, "with", len(params["means"]), "Gaussians")
    else:
        initialization = masks & (confidence >= 0.25)
        params, scene_scale = initialize_full_gaussians(
            world_points[initialization], colors[initialization],
            max_initial=MAX_INITIAL_GAUSSIANS, sh_degree=SH_DEGREE,
        )
        optimizers = strategy_state = None
        history, start_step = [], 0
        print("Initialized", len(params["means"]), "Gaussians")
    '''),
    code(r'''
    params, history = train_adaptive(
        params=params, scene_scale=scene_scale,
        real_views=real_views, synthetic_views=synthetic_views,
        output_dir=RUN_ROOT, steps=STEPS,
        synthetic_probability=SYNTHETIC_PROBABILITY,
        synthetic_weight=SYNTHETIC_WEIGHT,
        sh_degree=SH_DEGREE, checkpoint_every=CHECKPOINT_EVERY,
        optimizers=optimizers, strategy_state=strategy_state,
        history=history, start_step=start_step,
    )
    history_table = pd.DataFrame(history)
    history_table.to_csv(RUN_ROOT / "training_history.csv", index=False)
    display(history_table.tail())
    ax = history_table.plot(x="step", y="loss", logy=True, figsize=(10, 4), title="Full 3DGS loss")
    ax2 = history_table.plot(x="step", y="gaussians", figsize=(10, 4), title="Adaptive Gaussian count")
    plt.show()
    '''),
    code(r'''
    metrics = []
    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    for index, sample in enumerate(real_views):
        with torch.inference_mode():
            rendered, alpha, _ = render_full(
                params,
                torch.as_tensor(sample["extrinsic"], dtype=torch.float32, device="cuda"),
                torch.as_tensor(sample["intrinsic"], dtype=torch.float32, device="cuda"),
                height, width, sh_degree=SH_DEGREE,
            )
        prediction_rgb = rendered[0].permute(1, 2, 0).cpu().numpy()
        foreground = sample["mask"]
        mse = np.mean((prediction_rgb - sample["rgb"])[foreground] ** 2)
        mae = np.mean(np.abs(prediction_rgb - sample["rgb"])[foreground])
        psnr = -10 * np.log10(max(mse, 1e-10))
        metrics.append({"view": index, "masked_mae": mae, "masked_psnr": psnr})
        axes[0, index].imshow(sample["rgb"]); axes[1, index].imshow(np.clip(prediction_rgb, 0, 1))
        axes[0, index].set_title(f"real {index}"); axes[1, index].set_title(f"render PSNR {psnr:.1f}")
        axes[0, index].axis("off"); axes[1, index].axis("off")
    plt.tight_layout(); plt.show()
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(RUN_ROOT / "training_view_metrics.csv", index=False)
    display(metrics)
    print("These are training-view diagnostics, not held-out NVS metrics.")
    '''),
    code(r'''
    VIDEO_PATH = RUN_ROOT / "closed_orbit.mp4"
    writer = imageio.get_writer(str(VIDEO_PATH), fps=15, codec="libx264", quality=8, macro_block_size=None)
    try:
        for camera in cameras:
            with torch.inference_mode():
                rendered, alpha, _ = render_full(
                    params,
                    torch.as_tensor(camera["extrinsic"], dtype=torch.float32, device="cuda"),
                    torch.as_tensor(camera["intrinsic"], dtype=torch.float32, device="cuda"),
                    height, width, sh_degree=SH_DEGREE,
                )
            rgb = rendered[0].permute(1, 2, 0).cpu().numpy()
            opacity = alpha[0].permute(1, 2, 0).cpu().numpy()
            white_composite = np.clip(rgb + (1 - opacity), 0, 1)
            writer.append_data(np.uint8(white_composite * 255))
    finally:
        writer.close()
    print("Saved:", VIDEO_PATH)
    display(Video(str(VIDEO_PATH), embed=True, html_attributes="controls loop autoplay"))
    '''),
    md(r'''
    ## Required experiment reporting

    Report the eight real training views, number and construction of pseudo-views, confidence normalization, validity thresholds, real/synthetic sampling probability, synthetic multiplier, final Gaussian count, runtime, peak GPU memory, and training-view diagnostics. Final method claims still require evaluation against held-out real 3DRealCar views; pseudo-view agreement and training-view PSNR are not independent ground truth.
    '''),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
(HERE / "06_full_3dgs_real_plus_synthetic.ipynb").write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
)
print("Built 06_full_3dgs_real_plus_synthetic.ipynb")
