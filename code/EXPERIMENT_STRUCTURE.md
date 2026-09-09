# Experiment folder convention

This convention is the persistent source of truth for notebooks and scripts.

```text
experiments/
├── Geometry/<method>/<dataset_variant>/<scene>/
├── SyntheticViews/<method>/<dataset_variant>/<scene>/
└── 3DGS/<experiment>/<dataset_variant>/<scene>/
```

Dataset variants are `3DRealCar`, `3DRealCar_ref`, `Industrial`, and
`Industrial_ref`. The `_ref` suffix means reflection-handled preprocessing;
it never means reference-mesh evaluation.

Geometry methods currently include `VGGT` and later `MASt3R`. Synthetic-view
methods include `VGGT_NVS` and later `Backprojection`. Synthetic-view metrics
belong in an `evaluation/` directory below the corresponding synthetic-view
scene. HQ200 mesh comparison belongs in `Geometry/VGGT/.../mesh_evaluation/`.

Current 3DGS experiment names are:

- `VGGT_full`
- `VGGT_full_ref`
- `VGGT_backprojection`
- `VGGT_backprojection_ref`

Each 3DGS scene may contain `checkpoints/`, `renders/`, training histories,
metrics, configuration, and the closed-orbit video.
