"""Small, checkpoint-friendly wrapper around official MASt3R sparse global alignment."""

from pathlib import Path
import numpy as np


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
    return np.asarray(value)


def reconstruct_mast3r(image_paths, model, cache_dir, device="cuda", image_size=512,
                       scene_graph="complete", shared_intrinsics=True,
                       matching_conf_thr=5.0, coarse_iterations=300,
                       fine_iterations=300, optimize_depth=True):
    # MASt3R registers its bundled DUSt3R path through this import.
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from dust3r.utils.image import load_images
    from mast3r.image_pairs import make_pairs
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

    paths = [str(Path(path)) for path in image_paths]
    images = load_images(paths, size=image_size, verbose=False)
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    scene = sparse_global_alignment(
        paths, pairs, str(Path(cache_dir)), model,
        lr1=0.07, niter1=int(coarse_iterations),
        lr2=0.01, niter2=int(fine_iterations),
        device=device, opt_depth=bool(optimize_depth),
        shared_intrinsics=bool(shared_intrinsics),
        matching_conf_thr=float(matching_conf_thr),
    )
    points, depth, confidence = scene.get_dense_pts3d(clean_depth=True)
    points = np.stack([_numpy(value) for value in points])
    confidence = np.stack([_numpy(value) for value in confidence])
    color_values = [_numpy(value) for value in scene.imgs]
    colors = np.stack([value.transpose(1, 2, 0) if value.ndim == 3 and value.shape[0] in (1, 3) else value
                       for value in color_values])
    if colors.max() > 1.5: colors = colors / 255.0
    if colors.shape[1:3] != points.shape[1:3]:
        from PIL import Image
        height, width = points.shape[1:3]
        colors = np.stack([
            np.asarray(Image.fromarray(np.uint8(np.clip(value, 0, 1) * 255)).resize(
                (width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            for value in colors
        ])
    cameras_to_world = _numpy(scene.get_im_poses())
    focals = _numpy(scene.get_focals()).reshape(-1)
    return {"points": points, "confidence": confidence, "colors": colors,
            "depth": np.stack([_numpy(value) for value in depth]),
            "cameras_to_world": cameras_to_world, "focals": focals}
