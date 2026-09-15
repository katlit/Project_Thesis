"""Build the guarded HQ200 car-reference mesh cleaning notebook."""
import json
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": dedent(text).strip() + "\n"}

cells = [
    md('''
    # 03_B — HQ200 car-only reference mesh cleaning

    The supplied scanner mesh contains the car and background geometry. This notebook lets you inspect every scene,
    choose an explicit 3D crop, preview it, and approve it before export. Raw OBJ files are never modified.

    Output: `data_processed/reference_meshes/3DRealCar/<scene>/car_reference.obj`.
    Because automatic foreground extraction from a connected room/car mesh is unreliable, every exported crop requires
    manual approval. The saved JSON makes the process reproducible.
    '''),
    code('''
    %pip -q install trimesh scipy pandas matplotlib ipywidgets
    '''),
    code('''
    import gc, json, shutil, subprocess, sys
    from pathlib import Path
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from IPython.display import display, clear_output
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
    PROJECT_ROOT = Path("/content/drive/MyDrive/ITU/3D/Thesis")
    CODE_ROOT = Path("/content/Project_Thesis_code")
    REPOSITORY = "ht" + "tps:" + chr(47)*2 + "github.com" + chr(47) + "katlit" + chr(47) + "Project_Thesis.git"
    BRANCH = "codex/hq200-example-notebook"
    if CODE_ROOT.exists() and not (CODE_ROOT / ".git").is_dir(): shutil.rmtree(CODE_ROOT)
    command = (["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CODE_ROOT)]
               if not CODE_ROOT.exists() else ["git", "-C", str(CODE_ROOT), "pull", "--ff-only", "origin", BRANCH])
    subprocess.run(command, check=True)
    sys.path.insert(0, str(CODE_ROOT / "code"))

    from src.mesh_cleanup import crop_mesh_oriented, export_clean_reference, load_triangle_mesh, mesh_summary, sample_for_display
    '''),
    code('''
    def find_unique_dir(names, roots):
        matches = [root / name for root in roots for name in names if (root / name).is_dir()]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one HQ200 root, found: {matches}")
        return matches[0]

    HQ200_ROOT = find_unique_dir(["3DrealCarHQ200", "HQ200"], [PROJECT_ROOT / "data", PROJECT_ROOT])
    if (HQ200_ROOT / "3DrealCarHQ200").is_dir(): HQ200_ROOT /= "3DrealCarHQ200"
    OUTPUT_ROOT = PROJECT_ROOT / "data_processed/reference_meshes/3DRealCar"
    CONFIG_PATH = PROJECT_ROOT / "splits/reference_mesh_crop_boxes.json"
    mesh_paths = {path.parent.name: path for path in HQ200_ROOT.glob("*/textured_output.obj")}
    if not mesh_paths: raise FileNotFoundError(f"No textured_output.obj files below {HQ200_ROOT}")
    scenes = sorted(mesh_paths)
    rows = []
    for scene, path in mesh_paths.items():
        inspected_mesh = load_triangle_mesh(path)
        rows.append(mesh_summary(inspected_mesh, scene, path))
        del inspected_mesh
        gc.collect()
    display(pd.DataFrame(rows).sort_values("scene").round(3))
    print("Scenes:", len(scenes), "| raw meshes are read-only")
    '''),
    md('''
    ## Interactive crop and approval

    The sliders are normalized to each mesh's complete bounds: `0` is its minimum and `1` its maximum on that axis.
    Adjust the three ranges until every preview contains the complete car but no surrounding scan. The proposed crop is
    shown from front, rear, both sides, and two perspectives. These names assume the scanner axes follow the usual scene
    orientation; even if front/side labels are swapped, all four opposing horizontal directions are covered. Click
    **Approve and save** only when the complete car is retained. Saving the crop does not yet export a mesh.
    '''),
    code('''
    saved = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}
    default_box = [[0.25, 0.75], [0.05, 0.95], [0.25, 0.75]]
    scene_widget = widgets.Dropdown(options=scenes, description="Scene:", layout=widgets.Layout(width="750px"))
    sliders = [widgets.FloatRangeSlider(value=default_box[i], min=0, max=1, step=.01,
               description=axis, continuous_update=False, layout=widgets.Layout(width="700px"))
               for i, axis in enumerate(["X", "Y", "Z"])]
    rotation_sliders = [widgets.FloatSlider(value=0, min=-45, max=45, step=1,
               description=f"Rotate {axis}", continuous_update=False, readout_format=".0f",
               layout=widgets.Layout(width="700px")) for axis in ["X", "Y", "Z"]]
    preview_button = widgets.Button(description="Preview crop", button_style="info")
    approve_button = widgets.Button(description="Approve and save", button_style="success")
    output = widgets.Output()

    def current_box(): return [list(slider.value) for slider in sliders]
    def current_rotation(): return [float(slider.value) for slider in rotation_sliders]

    def load_saved_box(change=None):
        box = saved.get(scene_widget.value, {}).get("normalized_box", default_box)
        for slider, limits in zip(sliders, box): slider.value = tuple(limits)
        rotation = saved.get(scene_widget.value, {}).get("rotation_degrees", [0, 0, 0])
        for slider, value in zip(rotation_sliders, rotation): slider.value = float(value)

    def show_preview(_=None):
        with output:
            clear_output(wait=True)
            scene = scene_widget.value
            original = load_triangle_mesh(mesh_paths[scene])
            try: cropped = crop_mesh_oriented(original, current_box(), current_rotation())
            except Exception as error:
                print("Invalid crop:", error); return
            original_points = sample_for_display(original, 5_000)
            cropped_points = sample_for_display(cropped, 8_000)
            original_views = [
                (18, -65, "Perspective"), (0, 0, "Front"),
                (0, 180, "Rear"), (0, 90, "Left side"), (0, -90, "Right side"),
            ]
            original_figure = plt.figure(figsize=(24, 5.5))
            original_extent = np.ptp(original_points, axis=0).clip(min=1e-6)
            for index, (elevation, azimuth, title) in enumerate(original_views, start=1):
                axis = original_figure.add_subplot(1, 5, index, projection="3d")
                axis.scatter(*original_points.T, s=0.8, c="0.25", linewidths=0)
                axis.set_title(f"Original — {title}")
                axis.set_box_aspect(original_extent)
                axis.view_init(elevation, azimuth)
                axis.set_xticks([]); axis.set_yticks([]); axis.set_zticks([])
                axis.set_xlabel("X", color="crimson", fontweight="bold", labelpad=8)
                axis.set_ylabel("Y", color="forestgreen", fontweight="bold", labelpad=8)
                axis.set_zlabel("Z", color="royalblue", fontweight="bold", labelpad=8)
            original_figure.suptitle(f"Original mesh from five fixed directions — {scene}", fontsize=15)
            original_figure.subplots_adjust(left=.02, right=.98, bottom=.05, top=.87, wspace=.08)
            plt.show(); plt.close(original_figure)
            fig = plt.figure(figsize=(18, 8))
            for index, (points, title, azimuth) in enumerate([
                (original_points, "Original mesh including background", -65),
                (cropped_points, "Proposed car-only crop", -65),
                (cropped_points, "Proposed crop — second angle", 25),
            ], start=1):
                axis = fig.add_subplot(1, 3, index, projection="3d")
                axis.scatter(*points.T, s=0.8, c="0.25", linewidths=0)
                axis.set_title(title); axis.set_box_aspect(np.ptp(points, axis=0).clip(min=1e-6))
                axis.view_init(18, azimuth)
                axis.set_xlabel("X", color="crimson", fontweight="bold", labelpad=8)
                axis.set_ylabel("Y", color="forestgreen", fontweight="bold", labelpad=8)
                axis.set_zlabel("Z", color="royalblue", fontweight="bold", labelpad=8)
                axis.text2D(0.02, 0.96, "X red   Y green   Z blue",
                            transform=axis.transAxes, fontsize=9, fontweight="bold")
            plt.close(fig)  # obsolete three-panel overview; intentionally not displayed
            views = [
                (18, -65, "Perspective 1"), (18, 25, "Perspective 2"),
                (0, 0, "Front"), (0, 180, "Rear"),
                (0, 90, "Left side"), (0, -90, "Right side"),
                (90, -90, "Top"), (-90, -90, "Bottom"),
                (35, 115, "Elevated perspective 3"),
                (35, -155, "Elevated perspective 4"),
            ]
            detail = plt.figure(figsize=(24, 11))
            crop_extent = np.ptp(cropped_points, axis=0).clip(min=1e-6)
            for index, (elevation, azimuth, title) in enumerate(views, start=1):
                axis = detail.add_subplot(2, 5, index, projection="3d")
                axis.scatter(*cropped_points.T, s=0.65, c="0.25", linewidths=0)
                axis.set_title(title)
                axis.set_box_aspect(crop_extent)
                axis.view_init(elevation, azimuth)
                axis.set_xticks([]); axis.set_yticks([]); axis.set_zticks([])
                axis.set_xlabel("X", color="crimson", fontweight="bold", labelpad=8)
                axis.set_ylabel("Y", color="forestgreen", fontweight="bold", labelpad=8)
                axis.set_zlabel("Z", color="royalblue", fontweight="bold", labelpad=8)
                axis.text2D(0.02, 0.96, "X red   Y green   Z blue",
                            transform=axis.transAxes, fontsize=9, fontweight="bold")
            detail.suptitle(f"Proposed crop from eight fixed directions — {scene}", fontsize=15)
            detail.subplots_adjust(left=.03, right=.97, bottom=.05, top=.90, wspace=.10, hspace=.16)
            detail.suptitle(f"Proposed crop from ten fixed directions — {scene}", fontsize=15)
            plt.show(); plt.close(detail)
            display(pd.DataFrame([mesh_summary(cropped, scene)]).round(3))
            del original, cropped, original_points, cropped_points
            gc.collect()

    # Final preview implementation: one 3x5 grid directly below the controls.
    def show_preview(_=None):
        with output:
            clear_output(wait=True)
            scene = scene_widget.value
            original = load_triangle_mesh(mesh_paths[scene])
            try:
                cropped = crop_mesh_oriented(original, current_box(), current_rotation())
            except Exception as error:
                print("Invalid crop:", error)
                del original
                gc.collect()
                return

            original_points = sample_for_display(original, 8_000)
            cropped_points = sample_for_display(cropped, 15_000)
            original_views = [
                (18, -65, "Original — perspective"), (0, 0, "Original — front"),
                (0, 180, "Original — rear"), (0, 90, "Original — left side"),
                (0, -90, "Original — right side"),
            ]
            crop_views = [
                (18, -65, "Crop — perspective 1"), (18, 25, "Crop — perspective 2"),
                (35, 115, "Crop — perspective 3"), (35, -155, "Crop — perspective 4"),
                (55, -65, "Crop — high perspective"),
                (0, 0, "Crop — front"), (0, 180, "Crop — rear"),
                (0, 90, "Crop — left side"), (0, -90, "Crop — right side"),
                (90, -90, "Crop — top"),
            ]
            figure = plt.figure(figsize=(25, 16))
            original_extent = np.ptp(original_points, axis=0).clip(min=1e-6)
            crop_extent = np.ptp(cropped_points, axis=0).clip(min=1e-6)

            def draw(subplot_index, points, extent, elevation, azimuth, title, size):
                axis = figure.add_subplot(3, 5, subplot_index, projection="3d")
                axis.scatter(*points.T, s=size, c="0.20", linewidths=0, depthshade=True)
                axis.set_title(title, fontsize=11)
                axis.set_box_aspect(extent)
                axis.view_init(elevation, azimuth)
                axis.set_xticks([]); axis.set_yticks([]); axis.set_zticks([])
                axis.set_xlabel("X", color="crimson", fontweight="bold", labelpad=7)
                axis.set_ylabel("Y", color="forestgreen", fontweight="bold", labelpad=7)
                axis.set_zlabel("Z", color="royalblue", fontweight="bold", labelpad=7)

            for index, (elevation, azimuth, title) in enumerate(original_views, start=1):
                draw(index, original_points, original_extent, elevation, azimuth, title, 0.8)
            for index, (elevation, azimuth, title) in enumerate(crop_views, start=1):
                draw(5 + index, cropped_points, crop_extent, elevation, azimuth, title, 0.65)

            figure.suptitle(
                f"Oriented reference-mesh crop — {scene} | rotation XYZ = {np.round(current_rotation(), 1)}°",
                fontsize=16, y=.985,
            )
            figure.text(.01, .95, "Row 1: original mesh", fontsize=12, fontweight="bold")
            figure.text(.01, .625, "Rows 2–3: proposed crop", fontsize=12, fontweight="bold")
            figure.subplots_adjust(left=.025, right=.985, bottom=.035, top=.94, wspace=.08, hspace=.18)
            plt.show(); plt.close(figure)
            display(pd.DataFrame([mesh_summary(cropped, scene)]).round(3))
            del original, cropped, original_points, cropped_points
            gc.collect()

    def approve(_):
        scene = scene_widget.value
        original = load_triangle_mesh(mesh_paths[scene])
        cropped = crop_mesh_oriented(original, current_box(), current_rotation())
        saved[scene] = {"normalized_box": current_box(), "rotation_degrees": current_rotation(), "approved": True,
                        "raw_mesh": str(mesh_paths[scene]), "cropped_summary": mesh_summary(cropped, scene)}
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(saved, indent=2))
        with output: print("Approved and saved:", scene, "→", CONFIG_PATH)
        del original, cropped
        gc.collect()

    scene_widget.observe(load_saved_box, names="value")
    preview_button.on_click(show_preview); approve_button.on_click(approve)
    load_saved_box()
    display(widgets.VBox([
        scene_widget, widgets.HTML("<b>Crop limits in the rotated frame</b>"), *sliders,
        widgets.HTML("<b>Temporary crop-box rotation (degrees)</b>"), *rotation_sliders,
        widgets.HBox([preview_button, approve_button]), output,
    ]))
    show_preview()
    '''),
    md('''## Approval status and guarded batch export'''),
    code('''
    saved = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}
    status = pd.DataFrame([{"scene": scene, "approved": bool(saved.get(scene, {}).get("approved", False)),
                            "already_exported": (OUTPUT_ROOT / scene / "car_reference.obj").is_file()}
                           for scene in scenes])
    display(status)
    print("Approved:", int(status.approved.sum()), "/", len(status))
    '''),
    code('''
    RUN_EXPORT = False
    OVERWRITE = False

    if RUN_EXPORT:
        unapproved = [scene for scene in scenes if not saved.get(scene, {}).get("approved", False)]
        if unapproved:
            raise RuntimeError(f"Approve every scene before batch export. Missing: {unapproved}")
        export_rows = []
        for scene in scenes:
            original = load_triangle_mesh(mesh_paths[scene])
            cleaned = crop_mesh_oriented(
                original, saved[scene]["normalized_box"],
                saved[scene].get("rotation_degrees", [0, 0, 0]),
            )
            destination = OUTPUT_ROOT / scene / "car_reference.obj"
            export_clean_reference(cleaned, destination, overwrite=OVERWRITE)
            export_rows.append(mesh_summary(cleaned, scene, destination))
            del original, cleaned
            gc.collect()
        manifest = pd.DataFrame(export_rows)
        manifest.to_csv(OUTPUT_ROOT / "manifest.csv", index=False)
        display(manifest.round(3)); print("Saved:", OUTPUT_ROOT)
    else:
        print("Dry run. After approving every scene, set RUN_EXPORT=True.")
    '''),
]

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
(HERE / "03_B_hq200_reference_mesh_cleanup.ipynb").write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
)
