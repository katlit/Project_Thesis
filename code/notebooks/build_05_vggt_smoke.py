"""Build only notebook 05; never rewrites earlier user-executed notebooks."""

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
    # 05 — VGGT geometry, confidence-aware pseudo-views, and 3DGS smoke test

    This notebook runs one selected car scene from input to a trained Gaussian representation:

    1. infer VGGT cameras, depth/point maps, and confidence from eight real masked views;
    2. visualize the foreground point map;
    3. interpolate ten interior cameras between every adjacent pair (80 pseudo-views around the closed orbit);
    4. synthesize RGB by confidence-aware geometric reprojection and produce separate binary validity and continuous confidence maps;
    5. check confidence against observed error by reconstructing each real view from the other seven point maps;
    6. run a minimal gsplat optimization with separate real and synthetic supervision.

    **Terminology:** the public VGGT checkpoint does not contain the paper's unreleased fine-tuned Plücker-ray RGB head. The pseudo-views here are therefore *VGGT geometry-guided reprojections*, not direct learned VGGT NVS. They are deliberately sparse/conservative around disocclusions.

    With 8 real views, neighboring real cameras are approximately 45° apart. Ten interior views divide each interval into 11 steps, so the nominal step is approximately **4.09°**. Exactly 4.5° would require nine interior views.
    '''),
    code(r'''
    # Keep Colab's CUDA-enabled torch/torchvision. VGGT's legacy
    # requirements.txt pins torch 2.3.1, which has no Python 3.13 wheel.
    %pip -q install scipy pandas pillow huggingface_hub einops safetensors opencv-python "gsplat==1.3.0"
    !test -d /content/vggt/.git || git clone -q https://github.com/facebookresearch/vggt.git /content/vggt
    %pip -q install --no-deps -e /content/vggt
    '''),
    code(r'''
    import shutil, subprocess, sys
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from google.colab import drive

    if not torch.cuda.is_available():
        raise RuntimeError("Connect a Colab GPU runtime before running notebook 05.")
    print(torch.cuda.get_device_name(0))
    drive.mount("/content/drive", force_remount=False)

    CODE_ROOT = Path("/content/Project_Thesis_code")
    REPOSITORY = "ht" + "tps:" + chr(47)*2 + "github.com" + chr(47) + "katlit" + chr(47) + "Project_Thesis.git"
    BRANCH = "codex/hq200-example-notebook"
    # The project repository is public, so no GitHub token is required.
    if CODE_ROOT.exists() and not (CODE_ROOT / ".git").is_dir():
        shutil.rmtree(CODE_ROOT)
    command = (["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CODE_ROOT)]
               if not CODE_ROOT.exists() else
               ["git", "-C", str(CODE_ROOT), "pull", "--ff-only", "origin", BRANCH])
    subprocess.run(command, check=True)
    sys.path[:0] = [str(CODE_ROOT / "code"), "/content/vggt"]
    '''),
    code(r'''
    from src.vggt_geometry import (
        confidence_error_table, confidence_to_unit_interval,
        forward_splat, interpolate_closed_orbit,
    )
    from src.gaussian_smoke import initialize_gaussians, render_gaussians, train_smoke
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    PROJECT_ROOT = Path("/content/drive/MyDrive/ITU/3D/Thesis")
    METHOD_MANIFEST = PROJECT_ROOT / "data_processed/method_inputs/manifest.csv"
    RESULT_ROOT = PROJECT_ROOT / "experiments/vggt_pseudoview_3dgs_smoke"

    DATASET = "3DRealCar"                 # or "IndustrialInventory"
    SCENE = None                          # None selects the first available scene
    VIEWS_BETWEEN = 10
    MIN_SUPPORT = 2
    CONFIDENCE_THRESHOLD = 0.25
    MAX_GAUSSIANS = 30_000
    TRAIN_STEPS = 1_000
    RUN_SYNTHESIS = True
    RUN_3DGS = True

    manifest = pd.read_csv(METHOD_MANIFEST)
    available = manifest.query("method == 'vggt' and split == 'train' and dataset == @DATASET")
    SCENE = SCENE or sorted(available.scene.unique())[0]
    scene = available[available.scene.eq(SCENE)].sort_values(["view_order", "source"]).copy()
    assert len(scene) == 8, f"Expected eight inputs, found {len(scene)}"
    SCENE_ROOT = RESULT_ROOT / DATASET / SCENE
    SCENE_ROOT.mkdir(parents=True, exist_ok=True)
    display(scene[["dataset", "scene", "view_order", "method_image", "method_mask"]])
    '''),
    md(r'''
    ## VGGT inference

    VGGT confidence is an ordering/weighting signal, not a calibrated probability that a pixel is correct. We robustly map its 5th–95th foreground percentiles to [0,1], retain the raw confidence, and evaluate the mapped value later.
    '''),
    code(r'''
    image_paths = scene.method_image.tolist()
    images = load_and_preprocess_images(image_paths).to("cuda")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    prediction = {
        key: value.detach().float().cpu().numpy().squeeze(0)
        for key, value in predictions.items() if isinstance(value, torch.Tensor)
    }
    prediction["extrinsic"] = extrinsic.detach().float().cpu().numpy().squeeze(0)
    prediction["intrinsic"] = intrinsic.detach().float().cpu().numpy().squeeze(0)
    colors = prediction["images"].transpose(0, 2, 3, 1)
    height, width = colors.shape[1:3]
    masks = np.stack([
        np.asarray(Image.open(path).convert("L").resize((width, height), Image.Resampling.NEAREST)) >= 128
        for path in scene.method_mask
    ])
    confidence_raw = prediction["world_points_conf"]
    confidence = np.stack([
        confidence_to_unit_interval(confidence_raw[i], masks[i]) for i in range(8)
    ])
    np.savez_compressed(
        SCENE_ROOT / "vggt_predictions.npz",
        world_points=prediction["world_points"], world_points_conf=confidence_raw,
        confidence_unit=confidence, depth=prediction["depth"],
        depth_conf=prediction["depth_conf"], extrinsic=prediction["extrinsic"],
        intrinsic=prediction["intrinsic"], masks=masks,
    )
    print("VGGT tensor size:", width, "x", height)
    '''),
    code(r'''
    keep = masks & (confidence >= CONFIDENCE_THRESHOLD)
    points = prediction["world_points"][keep]
    point_colors = colors[keep]
    point_confidence = confidence[keep]
    finite = np.isfinite(points).all(axis=1) & np.isfinite(point_colors).all(axis=1)
    points, point_colors, point_confidence = points[finite], point_colors[finite], point_confidence[finite]
    print(f"Foreground/confident finite points: {len(points):,}")
    if not len(points):
        raise ValueError(
            "No points survived the mask/confidence filter. "
            "Inspect masks and confidence, or temporarily lower CONFIDENCE_THRESHOLD."
        )
    rng = np.random.default_rng(42)
    if len(points) > 20_000:
        chosen = rng.choice(len(points), 20_000, replace=False)
        points, point_colors, point_confidence = points[chosen], point_colors[chosen], point_confidence[chosen]

    # Static Matplotlib output is reliable through a VS Code-hosted Colab
    # connection; Plotly's interactive renderer may remain blank there.
    centered = points - np.median(points, axis=0)
    radius = np.linalg.norm(centered, axis=1)
    radius_limit = np.percentile(radius, 99)
    visible = radius <= radius_limit
    centered = centered[visible]
    display_colors = np.clip(point_colors[visible], 0, 1)

    fig = plt.figure(figsize=(15, 6))
    ax_rgb = fig.add_subplot(121, projection="3d")
    ax_conf = fig.add_subplot(122, projection="3d")
    ax_rgb.scatter(centered[:, 0], centered[:, 1], centered[:, 2], c=display_colors, s=0.7)
    confidence_scatter = ax_conf.scatter(
        centered[:, 0], centered[:, 1], centered[:, 2],
        c=point_confidence[visible], cmap="viridis", vmin=0, vmax=1, s=0.7,
    )
    ax_rgb.set_title("VGGT foreground point map — RGB")
    ax_conf.set_title("VGGT foreground point map — normalized confidence")
    for axis in [ax_rgb, ax_conf]:
        axis.set_box_aspect(np.ptp(centered, axis=0).clip(min=1e-6))
        axis.set_xlabel("X"); axis.set_ylabel("Y"); axis.set_zlabel("Z")
        axis.view_init(elev=18, azim=-65)
    fig.colorbar(confidence_scatter, ax=ax_conf, shrink=0.65, label="confidence")
    plt.tight_layout()
    plt.show()
    '''),
    md(r'''
    ## Confidence diagnostic on known views

    Each real target is reprojected using the other seven point maps. We report foreground coverage and confidence-binned absolute RGB error. This is an optimistic diagnostic because VGGT inferred the joint geometry and cameras using all eight images; a strict calibration experiment should rerun VGGT eight times with the target image removed.
    '''),
    code(r'''
    reliability_rows = []
    heldout_previews = []
    for target in range(8):
        source_ids = [index for index in range(8) if index != target]
        rgb, valid, conf, support, depth = forward_splat(
            prediction["world_points"][source_ids], colors[source_ids], confidence[source_ids], masks[source_ids],
            prediction["extrinsic"][target], prediction["intrinsic"][target], (height, width),
            min_support=MIN_SUPPORT, confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        evaluation_mask = valid & masks[target]
        error = np.abs(rgb - colors[target]).mean(axis=-1)
        reliability_rows.extend({"target_view": target, **row} for row in confidence_error_table(conf, error, evaluation_mask))
        heldout_previews.append((rgb, valid, conf, error, evaluation_mask))
        print(target, "foreground coverage:", evaluation_mask.sum() / max(masks[target].sum(), 1),
              "MAE:", error[evaluation_mask].mean() if evaluation_mask.any() else np.nan)
    reliability = pd.DataFrame(reliability_rows)
    reliability.to_csv(SCENE_ROOT / "confidence_reliability.csv", index=False)
    display(reliability.groupby("bin").agg(
        pixels=("pixels", "sum"), mean_confidence=("mean_confidence", "mean"),
        mean_absolute_error=("mean_absolute_error", "mean"),
    ))
    '''),
    md(r'''
    ## Closed-orbit pseudo-views

    A pseudo-pixel is valid only when at least `MIN_SUPPORT` projected samples agree with the visible surface and its normalized confidence exceeds the threshold. Invalid/disoccluded holes remain unsupervised; they are not filled with hallucinated RGB.
    '''),
    code(r'''
    synthetic_views = []
    cameras = interpolate_closed_orbit(
        prediction["extrinsic"], prediction["intrinsic"], views_between=VIEWS_BETWEEN
    )
    pseudo_root = SCENE_ROOT / "pseudo_views"
    if RUN_SYNTHESIS:
        for index, camera in enumerate(cameras):
            rgb, valid, conf, support, depth = forward_splat(
                prediction["world_points"], colors, confidence, masks,
                camera["extrinsic"], camera["intrinsic"], (height, width),
                min_support=MIN_SUPPORT, confidence_threshold=CONFIDENCE_THRESHOLD,
            )
            record = {**camera, "rgb": rgb, "validity": valid, "confidence": conf,
                      "height": height, "width": width, "index": index}
            synthetic_views.append(record)
            image_dir, valid_dir, conf_dir = pseudo_root / "images", pseudo_root / "validity", pseudo_root / "confidence"
            for directory in [image_dir, valid_dir, conf_dir]: directory.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).save(image_dir / f"view_{index:03d}.png")
            Image.fromarray(np.uint8(valid) * 255).save(valid_dir / f"view_{index:03d}.png")
            Image.fromarray(np.uint8(np.clip(conf, 0, 1) * 255)).save(conf_dir / f"view_{index:03d}.png")
        print("Synthetic views:", len(synthetic_views), "saved under", pseudo_root)

        show = np.linspace(0, len(synthetic_views)-1, 12, dtype=int)
        fig, axes = plt.subplots(3, len(show), figsize=(2.4*len(show), 7))
        for column, index in enumerate(show):
            item = synthetic_views[index]
            axes[0, column].imshow(item["rgb"]); axes[1, column].imshow(item["validity"], cmap="gray")
            axes[2, column].imshow(item["confidence"], vmin=0, vmax=1, cmap="viridis")
            axes[0, column].set_title(f"{index}")
            for axis in axes[:, column]: axis.axis("off")
        axes[0, 0].set_ylabel("RGB"); axes[1, 0].set_ylabel("valid"); axes[2, 0].set_ylabel("confidence")
        plt.tight_layout(); plt.show()
    '''),
    md(r'''
    ## Supervision design

    Real views use masked RGB, foreground-averaged SSIM, and full silhouette BCE. Synthetic RGB and SSIM use the detached weight $W=M^{syn}C^{syn}$ and normalize by the sum of weights, which keeps gradient magnitude stable when confidence changes. Synthetic silhouette supervision is weak and one-sided: it encourages opacity only at supported pseudo-foreground pixels and imposes no background/empty-space claim in holes.

    Synthetic samples are drawn only 25% of the time and their loss is multiplied by 0.25. Thus 80 correlated pseudo-views cannot numerically overwhelm eight real observations. For the thesis experiment, ablate real-only versus real+synthetic, confidence weighting, threshold, and pseudo-view density.
    '''),
    code(r'''
    real_views = [{
        "rgb": colors[index], "mask": masks[index],
        "extrinsic": prediction["extrinsic"][index], "intrinsic": prediction["intrinsic"][index],
        "height": height, "width": width,
    } for index in range(8)]

    if RUN_3DGS:
        initialization = masks & (confidence >= CONFIDENCE_THRESHOLD)
        initial_points = prediction["world_points"][initialization]
        initial_colors = colors[initialization]
        gaussians = initialize_gaussians(
            initial_points, initial_colors, max_gaussians=MAX_GAUSSIANS, device="cuda"
        )
        history = train_smoke(
            gaussians, real_views, synthetic_views, steps=TRAIN_STEPS,
            synthetic_probability=0.25, synthetic_weight=0.25,
        )
        history = pd.DataFrame(history)
        history.to_csv(SCENE_ROOT / "3dgs_training_history.csv", index=False)
        torch.save({"gaussians": gaussians.state_dict(), "scene": SCENE}, SCENE_ROOT / "3dgs_smoke.pt")
        display(history.tail())
        history.plot(x="step", y="loss", logy=True, title="3DGS smoke-training loss")
        plt.show()
    '''),
    code(r'''
    if RUN_3DGS:
        fig, axes = plt.subplots(2, 8, figsize=(24, 6))
        metric_rows = []
        for index, sample in enumerate(real_views):
            with torch.inference_mode():
                rendered, alpha = render_gaussians(
                    gaussians,
                    torch.as_tensor(sample["extrinsic"], dtype=torch.float32, device="cuda"),
                    torch.as_tensor(sample["intrinsic"], dtype=torch.float32, device="cuda"),
                    height, width,
                )
            rendered = rendered[0].permute(1, 2, 0).cpu().numpy()
            valid = sample["mask"]
            mae = np.abs(rendered - sample["rgb"])[valid].mean()
            psnr = -10 * np.log10(max(np.mean((rendered - sample["rgb"])[valid] ** 2), 1e-10))
            metric_rows.append({"view": index, "masked_mae": mae, "masked_psnr": psnr})
            axes[0, index].imshow(sample["rgb"]); axes[1, index].imshow(np.clip(rendered, 0, 1))
            axes[0, index].set_title(f"real {index}"); axes[1, index].set_title(f"render PSNR {psnr:.1f}")
            axes[0, index].axis("off"); axes[1, index].axis("off")
        plt.tight_layout(); plt.show()
        metrics = pd.DataFrame(metric_rows)
        display(metrics)
        metrics.to_csv(SCENE_ROOT / "real_view_metrics.csv", index=False)
    '''),
    md(r'''
    ## What this run establishes—and what it does not

    A successful run establishes that data loading, VGGT geometry, pseudo-view generation, uncertainty-aware losses, gsplat differentiation, checkpointing, and visualization connect end to end for one car. The included trainer keeps a fixed Gaussian set and is a smoke baseline; it does not implement adaptive densification/pruning or constitute the final publication-quality 3DGS comparison. Training-view PSNR is a pipeline diagnostic, not novel-view performance.
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
(HERE / "05_vggt_pseudoview_3dgs_smoke_test.ipynb").write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
)
print("Built 05_vggt_pseudoview_3dgs_smoke_test.ipynb")
