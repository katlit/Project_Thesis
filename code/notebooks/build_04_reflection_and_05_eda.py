"""Build the optional reflection-preprocessing notebook and three-way EDA."""

import json
from pathlib import Path
from textwrap import dedent


HERE = Path(__file__).resolve().parent


def md(value):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(value).strip() + "\n"}


def code(value):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": dedent(value).strip() + "\n"}


def write(name, cells, gpu=False):
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU" if gpu else "CPU",
            "colab": {"gpuType": "T4", "provenance": []} if gpu else {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (HERE / name).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


COMMON = r'''
from pathlib import Path
from google.colab import drive

DRIVE_MOUNT = Path("/content/drive")
if not (DRIVE_MOUNT / "MyDrive").is_dir():
    drive.mount(str(DRIVE_MOUNT))
PROJECT_ROOT = DRIVE_MOUNT / "MyDrive" / "ITU" / "3D" / "Thesis"
'''

BOOTSTRAP = r'''
import subprocess, sys
CODE_ROOT = Path("/content/Project_Thesis_code")
REPOSITORY = "https://github.com/katlit/Project_Thesis.git"
BRANCH = "codex/hq200-example-notebook"
if not (CODE_ROOT / "code" / "src").is_dir():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CODE_ROOT)], check=True)
else:
    subprocess.run(["git", "-C", str(CODE_ROOT), "pull", "--ff-only"], check=True)
sys.path.insert(0, str(CODE_ROOT / "code")) if str(CODE_ROOT / "code") not in sys.path else None
'''


def build():
    write("04_unreflectanything_reflection_handling.ipynb", [
        md(r'''
        # 04 — Optional reflection handling with UnReflectAnything

        This notebook creates the optional **reflection-handled (`_ref`) input branch**. It reads only the already foreground-removed, scene-cropped outputs of notebook 03_A. View selection, crop, canvas size, padding, split, ordering, and foreground mask remain unchanged.

        UnReflectAnything is an RGB-only **specular-highlight removal** model. A frozen DINOv3-Large encoder extracts image tokens; a reflection head predicts a soft highlight region; contaminated tokens are masked and filled by a learned token inpainter; a decoder reconstructs a diffuse-looking RGB image. Its synthetic training supervision uses MoGe-2 geometry with rendered Blinn–Phong/Fresnel highlights. It does not recover physically measured diffuse reflectance, and it may alter genuine paint/glass appearance. Therefore `_ref` is an ablation branch, not a replacement for the standard inputs.

        Outputs are written separately under `data_processed/method_inputs_ref`. Notebook 03_A files are never overwritten. The official model internally resizes for inference and restores the original dimensions; afterward this notebook restores every pixel outside the foreground mask from the standard input and copies the original mask unchanged.
        '''),
        code(r'''
        %pip -q install "unreflectanything>=1.0.3"

        import gc, subprocess
        from pathlib import Path
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        from PIL import Image
        from tqdm.auto import tqdm

        if not torch.cuda.is_available():
            raise RuntimeError("Select a Colab GPU runtime before running notebook 04.")
        print(torch.cuda.get_device_name(0))
        '''),
        code(COMMON), code(BOOTSTRAP),
        code(r'''
        from src.reflection_preprocessing import (
            process_with_unreflectanything, reflection_difference_image,
        )
        import unreflectanything

        INPUT_ROOT = PROJECT_ROOT / "data_processed" / "method_inputs"
        OUTPUT_ROOT = PROJECT_ROOT / "data_processed" / "method_inputs_ref"
        INPUT_MANIFEST = INPUT_ROOT / "manifest.csv"

        METHODS_TO_PROCESS = ["3dgs", "dust3r_mast3r", "vggt"]
        DATASETS_TO_PROCESS = None       # e.g. ["3DRealCar"] or None for all
        RUN_EXPORT = False              # inspect the preview first, then set True
        OVERWRITE = False               # False safely resumes an interrupted export
        PREVIEW_IMAGES = 1             # raise only after one image succeeds
        HIGHLIGHT_THRESHOLD = 0.30       # lower finds more suspected highlights
        MASK_DILATION = 40               # context removed around predicted highlights
        USE_MIXED_PRECISION = True       # substantially lowers activation memory

        if not INPUT_MANIFEST.is_file():
            raise FileNotFoundError(f"Run notebook 03_A export first: {INPUT_MANIFEST}")
        manifest = pd.read_csv(INPUT_MANIFEST)
        selected = manifest[manifest.method.isin(METHODS_TO_PROCESS)].copy()
        if DATASETS_TO_PROCESS is not None:
            selected = selected[selected.dataset.isin(DATASETS_TO_PROCESS)]
        selected = selected.sort_values(["method", "dataset", "scene", "split", "view_order", "source"], na_position="last")
        print("Rows:", len(selected), "| scenes:", selected[["dataset", "scene"]].drop_duplicates().shape[0])
        display(selected[["method", "dataset", "scene", "split", "view_order", "method_image", "method_mask"]].head(12))
        '''),
        md(r'''
        ## Load the released model

        The weights are downloaded once into Colab's package cache. `threshold` and `dilation` are the two official inference controls. Start with the defaults; compare alternatives on several glossy, glass, and low-highlight views before a full export.
        '''),
        code(r'''
        weights = Path(unreflectanything.cache("weights"))
        if not any(weights.glob("*.pth")):
            subprocess.run(["unreflectanything", "download", "--weights"], check=True)
        free_before, total_memory = torch.cuda.mem_get_info()
        print(f"Before model: {free_before / 1024**3:.1f}/{total_memory / 1024**3:.1f} GiB free")
        model = unreflectanything.model(pretrained=True, device=torch.device("cuda"), verbose=False)
        model.eval().requires_grad_(False)
        free_after, _ = torch.cuda.mem_get_info()
        print(f"After model:  {free_after / 1024**3:.1f}/{total_memory / 1024**3:.1f} GiB free")
        print("Weights:", weights)
        '''),
        code(r'''
        # A temporary preview does not write to Drive.
        preview_root = Path("/content/unreflectanything_preview")
        preview_rows = selected.query("method == '3dgs' and split == 'train'").head(PREVIEW_IMAGES)
        preview_records = []
        for row in tqdm(preview_rows.itertuples(index=False), total=len(preview_rows), desc="Preview"):
            out_image = preview_root / row.dataset / row.scene / Path(row.method_image).name
            out_mask = preview_root / row.dataset / row.scene / "masks" / Path(row.method_mask).name
            report = process_with_unreflectanything(
                row.method_image, row.method_mask, out_image, out_mask, model,
                threshold=HIGHLIGHT_THRESHOLD, dilation=MASK_DILATION, overwrite=True,
                use_amp=USE_MIXED_PRECISION,
            )
            preview_records.append((row, out_image, report))

        fig, axes = plt.subplots(len(preview_records), 3, figsize=(12, 4 * len(preview_records)), squeeze=False)
        for r, (row, output_path, report) in enumerate(preview_records):
            before = Image.open(row.method_image).convert("RGB")
            after = Image.open(output_path).convert("RGB")
            axes[r, 0].imshow(before); axes[r, 0].set_title("Standard cropped input")
            axes[r, 1].imshow(after); axes[r, 1].set_title("Reflection handled")
            axes[r, 2].imshow(reflection_difference_image(before, after)); axes[r, 2].set_title("|change| × 4 (diagnostic)")
            axes[r, 0].set_ylabel(f"{row.dataset}\n{row.scene}\nMAE {report['foreground_mae']:.3f}")
            for axis in axes[r]: axis.axis("off")
        plt.tight_layout(); plt.show()
        '''),
        md(r'''
        ## Export the `_ref` branch

        Set `RUN_EXPORT=True` only after the preview is acceptable. `OVERWRITE=False` resumes safely and retains finished files. The output manifest keeps all source metadata, adds provenance/settings/change diagnostics, and changes only `method_image` and `method_mask` to the parallel paths.
        '''),
        code(r'''
        records = []
        if RUN_EXPORT:
            for row in tqdm(selected.itertuples(index=False), total=len(selected), desc="Reflection handling"):
                relative_image = Path(row.method_image).relative_to(INPUT_ROOT)
                relative_mask = Path(row.method_mask).relative_to(INPUT_ROOT)
                output_image = OUTPUT_ROOT / relative_image
                output_mask = OUTPUT_ROOT / relative_mask
                report = process_with_unreflectanything(
                    row.method_image, row.method_mask, output_image, output_mask, model,
                    threshold=HIGHLIGHT_THRESHOLD, dilation=MASK_DILATION, overwrite=OVERWRITE,
                    use_amp=USE_MIXED_PRECISION,
                )
                record = row._asdict()
                record.update({
                    "source_method_image": record["method_image"],
                    "source_method_mask": record["method_mask"],
                    "method_image": str(output_image), "method_mask": str(output_mask),
                    "input_variant": "reflection_handled", "reflection_model": "UnReflectAnything",
                    "reflection_threshold": HIGHLIGHT_THRESHOLD, "reflection_dilation": MASK_DILATION,
                    **report,
                })
                records.append(record)
            output_manifest = pd.DataFrame(records)
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            output_manifest.to_csv(OUTPUT_ROOT / "manifest.csv", index=False)
            print("Saved:", OUTPUT_ROOT / "manifest.csv")
            display(output_manifest.groupby(["method", "dataset"])[["foreground_mae", "changed_foreground_fraction"]].agg(["mean", "median"]).round(4))
        else:
            print("Preview only. Set RUN_EXPORT=True to create the Drive outputs.")
        '''),
        code(r'''
        del model
        gc.collect(); torch.cuda.empty_cache()
        print("GPU memory released.")
        '''),
        md(r'''
        ## What to inspect

        Look for removal of compact white glare without loss of lamps, window borders, logos, or paint color. The amplified difference is only a change map—not the model's highlight mask and not a quality score. If real structure disappears, increase `HIGHLIGHT_THRESHOLD`, reduce `MASK_DILATION`, or keep the standard branch for that experiment.
        '''),
    ], gpu=True)

    write("05_eda_8view_comparison.ipynb", [
        md(r'''
        # 05 — Three-way input EDA

        This notebook compares the same selected eight training views in three representations:

        1. **Original:** the raw source photograph selected before foreground removal.
        2. **Standard input:** BiRefNet foreground removal followed by notebook 03_A's shared scene crop/canvas.
        3. **Reflection-handled input:** exactly the same crop, canvas, mask, split, and view order, with only foreground RGB processed by UnReflectAnything.

        Statistics use a temporary common 256×256 analysis grid and never modify model inputs. Original-versus-cropped distributions are descriptive because their framing differs. Standard-versus-reflection pixels are aligned, so their paired change measures are meaningful; they quantify change, not restoration quality (there is no diffuse ground truth).
        '''),
        code(r'''
        from pathlib import Path
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from PIL import Image
        from tqdm.auto import tqdm
        '''), code(COMMON), code(BOOTSTRAP),
        code(r'''
        from src.eda_utils import masked_image_statistics, paired_foreground_change
        from src.image_preprocessing import VIEW_LABELS

        STANDARD_ROOT = PROJECT_ROOT / "data_processed" / "method_inputs"
        REF_ROOT = PROJECT_ROOT / "data_processed" / "method_inputs_ref"
        STANDARD_MANIFEST = STANDARD_ROOT / "manifest.csv"
        REF_MANIFEST = REF_ROOT / "manifest.csv"
        EDA_ROOT = PROJECT_ROOT / "splits" / "sparse8"
        for path in [STANDARD_MANIFEST, REF_MANIFEST]:
            if not path.is_file(): raise FileNotFoundError(f"Required manifest missing: {path}")

        standard = pd.read_csv(STANDARD_MANIFEST).query("method == '3dgs' and split == 'train'").copy()
        reflected = pd.read_csv(REF_MANIFEST).query("method == '3dgs' and split == 'train'").copy()
        keys = ["dataset", "scene", "source"]
        reflected = reflected[keys + ["method_image", "method_mask"]].rename(columns={"method_image": "ref_image", "method_mask": "ref_mask"})
        paired = standard.merge(reflected, on=keys, how="inner", validate="one_to_one")
        counts = paired.groupby(["dataset", "scene"]).size()
        assert counts.eq(8).all(), "Each scene must have the same eight standard and reflection-handled inputs."
        print(f"Scenes: {len(counts)} | paired views: {len(paired)}")
        display(counts.rename("paired_views").to_frame())
        '''),
        code(r'''
        records = []
        for row in tqdm(paired.itertuples(index=False), total=len(paired), desc="Three-way statistics"):
            variants = {
                "original": (row.source, row.output_mask),
                "standard_input": (row.method_image, row.method_mask),
                "reflection_handled": (row.ref_image, row.ref_mask),
            }
            for variant, (image_path, mask_path) in variants.items():
                stats = masked_image_statistics(image_path, mask_path, sample_size=(256, 256))
                records.append({"dataset": row.dataset, "scene": row.scene, "source": row.source,
                                "view_order": row.view_order, "variant": variant, **stats})

        eda = pd.DataFrame(records)
        EDA_ROOT.mkdir(parents=True, exist_ok=True)
        eda.to_csv(EDA_ROOT / "image_statistics_three_way.csv", index=False)
        summary = eda.groupby(["dataset", "variant"]).agg(
            scenes=("scene", "nunique"), images=("source", "size"),
            width_mean=("width", "mean"), height_mean=("height", "mean"),
            foreground_fraction_mean=("foreground_fraction", "mean"),
            brightness_mean=("brightness", "mean"), contrast_mean=("contrast", "mean"),
            sharpness_mean=("sharpness_proxy", "mean"),
        ).round(3)
        display(summary)
        summary.to_csv(EDA_ROOT / "dataset_statistics_three_way.csv")
        '''),
        code(r'''
        paired_changes = []
        for row in paired.itertuples(index=False):
            paired_changes.append({"dataset": row.dataset, "scene": row.scene, "source": row.source,
                                   "view_order": row.view_order,
                                   **paired_foreground_change(row.method_image, row.ref_image, row.method_mask)})
        paired_changes = pd.DataFrame(paired_changes)
        paired_changes.to_csv(EDA_ROOT / "reflection_paired_changes.csv", index=False)
        display(paired_changes.groupby("dataset")[["ref_change_mae", "ref_change_rmse", "ref_change_psnr"]].agg(["mean", "median"]).round(4))
        '''),
        code(r'''
        metrics = ["foreground_fraction", "brightness", "contrast", "sharpness_proxy"]
        order = ["original", "standard_input", "reflection_handled"]
        colors = ["tab:gray", "tab:blue", "tab:orange"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for axis, metric in zip(axes.flat, metrics):
            values = [eda.loc[eda.variant.eq(variant), metric].to_numpy() for variant in order]
            axis.boxplot(values, tick_labels=["original", "standard", "reflection"], showfliers=False)
            for patch_color, x, samples in zip(colors, range(1, 4), values):
                jitter = np.random.default_rng(42 + x).normal(x, .035, len(samples))
                axis.scatter(jitter, samples, s=8, alpha=.25, color=patch_color)
            axis.set_title(metric.replace("_", " ").title())
        fig.suptitle("Same selected views: three input representations")
        plt.tight_layout(); plt.show()
        '''),
        code(r'''
        def show_scene(dataset, scene_name=None):
            part = paired[paired.dataset.eq(dataset)]
            scene_name = scene_name or sorted(part.scene.unique())[0]
            rows = part[part.scene.eq(scene_name)].sort_values(["view_order", "source"])
            fig, axes = plt.subplots(3, 8, figsize=(24, 9), squeeze=False)
            for column, row in enumerate(rows.itertuples(index=False)):
                paths = [row.source, row.method_image, row.ref_image]
                for r, path in enumerate(paths):
                    axes[r, column].imshow(Image.open(path).convert("RGB")); axes[r, column].axis("off")
                axes[0, column].set_title(VIEW_LABELS[column])
            for r, label in enumerate(["Original", "Standard input", "Reflection handled"]):
                axes[r, 0].set_ylabel(label, rotation=0, ha="right", labelpad=70)
            fig.suptitle(f"{dataset} / {scene_name}")
            plt.tight_layout(); plt.show()

        for dataset in sorted(paired.dataset.unique()):
            show_scene(dataset)
        '''),
        md(r'''
        ## Reading the comparison

        Foreground fraction changes strongly from original to cropped input because notebook 03_A deliberately removes unused canvas; that is expected. Brightness, contrast, and sharpness are computed inside the matching foreground mask on the same analysis resolution. A large standard-to-reflection MAE means the model changed much of the car, not necessarily that it improved it. Use the image grids to check whether glare was removed while lamps, glazing, logos, edges, and paint identity were preserved. Keep both standard and `_ref` branches for downstream ablations.
        '''),
    ], gpu=False)


if __name__ == "__main__":
    build()
    print("Built notebooks 04 and 05")
