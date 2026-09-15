"""One-off mechanical insertion that preserves EXP_VGGT_full cells and outputs."""
import json
from pathlib import Path

path = Path(__file__).with_name("EXP_VGGT_full.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))

first = notebook["cells"][1]
source = "".join(first["source"])
if " trimesh" not in source:
    source = source.replace("scipy pandas", "scipy trimesh pandas")
    first["source"] = source.splitlines(keepends=True)

markdown = {
    "cell_type": "markdown", "metadata": {}, "source": [
        "### 2A. Reference-mesh evaluation (3DRealCar only)\n", "\n",
        "The plot above intentionally shows at most 25,000 points, so apparent display sparsity is not a density measurement. "
        "The first table below reports how many foreground points survive each filter.\n", "\n",
        "For 3DRealCar, this section uniformly samples the supplied scanner-derived triangle mesh, aligns VGGT geometry "
        "using a 7-DoF similarity transform (scale, rotation, and translation) followed by robust ICP, and reports "
        "accuracy, completeness, symmetric Chamfer-L1, and F-score. The scanner mesh is reference geometry rather "
        "than guaranteed metrology-grade ground truth. Metrics are invalid without alignment because VGGT scale and "
        "coordinates are arbitrary. IndustrialInventory has no corresponding reference mesh, so this section skips it.\n"
    ]
}

code = {
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": r'''
import trimesh
from src.geometry_evaluation import align_similarity_icp, apply_similarity, geometry_metrics

# Evaluation knobs. HQ200 meshes are expected to use metres; verify the printed dimensions.
REFERENCE_SAMPLE_POINTS = 100_000
ALIGNMENT_MAX_POINTS = 12_000
EVALUATION_MAX_PREDICTED_POINTS = 100_000
EVALUATION_OUTLIER_PERCENTILE = 99.5
REFERENCE_UNITS_PER_METRE = 1.0
F_SCORE_THRESHOLDS_METRES = (0.01, 0.02, 0.05)

finite = np.isfinite(points_by_view).all(axis=-1)
confident = confidence >= POINT_CONFIDENCE_MIN
retained = geometry_keep
density_rows = []
for view_index in range(len(points_by_view)):
    foreground_count = int(masks[view_index].sum())
    retained_count = int(retained[view_index].sum())
    density_rows.append({
        "view": view_index,
        "image_pixels": int(masks[view_index].size),
        "foreground_pixels": foreground_count,
        "finite_foreground": int((masks[view_index] & finite[view_index]).sum()),
        "median_view_support": float(np.median(support[view_index][masks[view_index]])),
        "retained_geometry": retained_count,
        "retained_%_of_foreground": 100 * retained_count / max(foreground_count, 1),
        "median_foreground_confidence": float(np.median(confidence[view_index][masks[view_index]])),
    })
density_statistics = pd.DataFrame(density_rows)
display(density_statistics.round(2))
print(f"Retained geometry: {retained.sum():,} points; earlier plot displayed at most 25,000.")

if DATASET != "3DRealCar":
    print("Reference-mesh evaluation skipped: IndustrialInventory has no supplied HQ200 mesh.")
else:
    mesh_path = PROJECT_ROOT / "data_processed/reference_meshes/3DRealCar" / SCENE / "car_reference.obj"
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"Clean car reference missing: {mesh_path}\n"
            "Run 03_B_hq200_reference_mesh_cleanup.ipynb and approve this scene first. "
            "The raw textured_output.obj is deliberately not used because its background invalidates the metrics."
        )

    reference_mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
    if reference_mesh.is_empty or len(reference_mesh.faces) == 0:
        raise ValueError(f"Reference mesh is empty or has no faces: {mesh_path}")
    reference_points, _ = trimesh.sample.sample_surface(
        reference_mesh, REFERENCE_SAMPLE_POINTS, seed=42
    )
    reference_extent = np.ptp(reference_points, axis=0)
    print("Reference mesh:", mesh_path)
    print("Reference vertices/faces:", len(reference_mesh.vertices), "/", len(reference_mesh.faces))
    print("Reference XYZ dimensions:", np.round(reference_extent, 4), "mesh units")
    print("Approximate XYZ dimensions (m):", np.round(reference_extent / REFERENCE_UNITS_PER_METRE, 4))

    predicted_points = points_by_view[retained]
    predicted_colors = colors[retained]
    predicted_confidence = confidence[retained]
    predicted_center = np.median(predicted_points, axis=0)
    radius = np.linalg.norm(predicted_points - predicted_center, axis=1)
    spatial_keep = radius <= np.percentile(radius, EVALUATION_OUTLIER_PERCENTILE)
    predicted_points = predicted_points[spatial_keep]
    predicted_colors = predicted_colors[spatial_keep]
    predicted_confidence = predicted_confidence[spatial_keep]
    print(f"Spatial filter retained {len(predicted_points):,} points "
          f"({EVALUATION_OUTLIER_PERCENTILE}% radius percentile).")

    transform = align_similarity_icp(
        predicted_points, reference_points,
        max_points=ALIGNMENT_MAX_POINTS, iterations=30, trim_fraction=0.80, seed=42,
    )
    predicted_aligned = apply_similarity(predicted_points, transform)
    rng = np.random.default_rng(42)
    if len(predicted_aligned) > EVALUATION_MAX_PREDICTED_POINTS:
        selected = rng.choice(len(predicted_aligned), EVALUATION_MAX_PREDICTED_POINTS, replace=False)
        predicted_aligned = predicted_aligned[selected]
        predicted_colors = predicted_colors[selected]
        predicted_confidence = predicted_confidence[selected]

    thresholds = tuple(value * REFERENCE_UNITS_PER_METRE for value in F_SCORE_THRESHOLDS_METRES)
    metric_values, pred_to_ref, ref_to_pred = geometry_metrics(
        predicted_aligned, reference_points, thresholds
    )
    metric_values.update({
        "scene": SCENE, "mesh": str(mesh_path), "similarity_scale": transform["scale"],
        "reference_units_per_metre": REFERENCE_UNITS_PER_METRE,
    })
    geometry_summary = pd.DataFrame([metric_values])
    geometry_summary.to_csv(MESH_EVALUATION_ROOT / "metrics.csv", index=False)
    np.savez_compressed(
        MESH_EVALUATION_ROOT / "alignment.npz",
        scale=transform["scale"], rotation=transform["rotation"],
        translation=transform["translation"], predicted_aligned=predicted_aligned,
        reference_points=reference_points, pred_to_ref=pred_to_ref, ref_to_pred=ref_to_pred,
    )

    distance_columns = ["accuracy_mean", "accuracy_median", "completeness_mean",
                        "completeness_median", "chamfer_l1"]
    readable = geometry_summary.copy()
    readable[distance_columns] = readable[distance_columns] / REFERENCE_UNITS_PER_METRE * 100
    readable = readable.rename(columns={column: column + "_cm" for column in distance_columns})
    display(readable.drop(columns=["mesh", "rotation", "translation"], errors="ignore").round(4))

    vis_count = min(30_000, len(predicted_aligned), len(reference_points))
    pred_vis = rng.choice(len(predicted_aligned), vis_count, replace=False)
    ref_vis = rng.choice(len(reference_points), vis_count, replace=False)
    combined = np.concatenate([predicted_aligned[pred_vis], reference_points[ref_vis]])
    lower, upper = np.percentile(combined, [1, 99], axis=0)
    extent = (upper - lower).clip(min=1e-6)
    error_cm = pred_to_ref[pred_vis] / REFERENCE_UNITS_PER_METRE * 100
    error_limit = max(np.percentile(error_cm, 95), 1e-6)

    fig = plt.figure(figsize=(20, 9))
    axes = [fig.add_subplot(241 + index, projection="3d") for index in range(4)]
    axes[0].scatter(*reference_points[ref_vis].T, s=.25, c="0.35")
    axes[0].set_title("Reference mesh samples")
    axes[1].scatter(*predicted_aligned[pred_vis].T, s=.25,
                    c=np.clip(predicted_colors[pred_vis], 0, 1))
    axes[1].set_title("Similarity-aligned VGGT")
    axes[2].scatter(*reference_points[ref_vis].T, s=.2, c="0.75", alpha=.35)
    axes[2].scatter(*predicted_aligned[pred_vis].T, s=.25, c="tab:blue", alpha=.65)
    axes[2].set_title("Overlay: reference gray / VGGT blue")
    error_plot = axes[3].scatter(*predicted_aligned[pred_vis].T, s=.3, c=error_cm,
                                 cmap="turbo", vmin=0, vmax=error_limit)
    axes[3].set_title("VGGT → mesh distance (cm)")
    for axis in axes:
        axis.set_box_aspect(extent)
        axis.set_xlim(lower[0], upper[0]); axis.set_ylim(lower[1], upper[1]); axis.set_zlim(lower[2], upper[2])
        axis.view_init(18, -65)
    fig.colorbar(error_plot, ax=axes[3], shrink=.65, label="cm (clipped at 95th percentile)")

    histogram_axes = [fig.add_subplot(245), fig.add_subplot(246), fig.add_subplot(247), fig.add_subplot(248)]
    histogram_axes[0].hist(pred_to_ref / REFERENCE_UNITS_PER_METRE * 100, bins=80, color="tab:blue")
    histogram_axes[0].set_title("Accuracy distances"); histogram_axes[0].set_xlabel("VGGT → mesh (cm)")
    histogram_axes[1].hist(ref_to_pred / REFERENCE_UNITS_PER_METRE * 100, bins=80, color="tab:orange")
    histogram_axes[1].set_title("Completeness distances"); histogram_axes[1].set_xlabel("mesh → VGGT (cm)")
    histogram_axes[2].scatter(predicted_confidence[pred_vis], error_cm, s=2, alpha=.2)
    histogram_axes[2].set_xlabel("VGGT confidence"); histogram_axes[2].set_ylabel("mesh distance (cm)")
    histogram_axes[2].set_title("Confidence vs geometry error")
    fscore_columns = [column for column in geometry_summary if column.startswith("fscore@")]
    histogram_axes[3].bar([f"{m*100:g} cm" for m in F_SCORE_THRESHOLDS_METRES],
                          geometry_summary.loc[0, fscore_columns])
    histogram_axes[3].set_ylim(0, 1); histogram_axes[3].set_title("Geometry F-score")
    plt.tight_layout(); plt.show()
    print("Saved metrics and aligned point samples under:", RUN_ROOT)
'''.strip().splitlines(keepends=True)
}

if not any("Reference-mesh evaluation" in "".join(cell.get("source", []))
           for cell in notebook["cells"]):
    insertion = next(index for index, cell in enumerate(notebook["cells"])
                     if "## 3. Dense Gaussian-surfel teacher" in "".join(cell.get("source", [])))
    notebook["cells"][insertion:insertion] = [markdown, code]

path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
