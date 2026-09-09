"""Optional VGGT point-track bundle adjustment through PyCOLMAP."""

import numpy as np
import torch


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _foreground_track_mask(tracks, visibility, masks, visibility_threshold):
    """Combine tracker visibility with nearest-neighbour foreground lookup."""
    tracks = _to_numpy(tracks)
    valid = _to_numpy(visibility) > visibility_threshold
    views, _, _ = tracks.shape
    height, width = masks.shape[1:]
    for view in range(views):
        x = np.rint(tracks[view, :, 0]).astype(np.int64)
        y = np.rint(tracks[view, :, 1]).astype(np.int64)
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        foreground = np.zeros_like(inside)
        foreground[inside] = masks[view, y[inside], x[inside]]
        valid[view] &= foreground
    return valid


def _pose_matrix(image):
    pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
    matrix = pose.matrix() if callable(pose.matrix) else pose.matrix
    return np.asarray(matrix, dtype=np.float32)[:3, :4]


def _camera_matrix(camera):
    matrix = camera.calibration_matrix() if callable(camera.calibration_matrix) else camera.calibration_matrix
    return np.asarray(matrix, dtype=np.float32)


def run_vggt_bundle_adjustment(images, depth_confidence, point_maps, colors, masks,
                               extrinsics, intrinsics, max_reprojection_error=8.0,
                               visibility_threshold=0.2, max_query_points=4096,
                               query_frame_count=8, fine_tracking=True,
                               shared_camera=False, refine_focal_length=True,
                               refine_principal_point=False):
    """Build VGGT tracks, run one global BA, and return refined cameras.

    This follows the official VGGT ``demo_colmap.py`` path. Dense depth is not
    optimized by BA; callers must re-unproject it with the returned cameras.
    """
    import pycolmap
    from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
    from vggt.dependency.track_predict import predict_tracks

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        tracks, visibility, track_confidence, tracked_points, tracked_rgb = predict_tracks(
            images, conf=depth_confidence, points_3d=point_maps, masks=None,
            max_query_pts=max_query_points,
            query_frame_num=min(query_frame_count, len(images)),
            keypoint_extractor="aliked+sp", fine_tracking=fine_tracking,
        )
    tracks = _to_numpy(tracks)
    tracked_points = _to_numpy(tracked_points)
    tracked_rgb = _to_numpy(tracked_rgb) if tracked_rgb is not None else None
    track_mask = _foreground_track_mask(tracks, visibility, masks, visibility_threshold)
    if tracked_rgb is None:
        raise RuntimeError("VGGT tracker did not return colors for its sparse 3D points.")
    reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
        tracked_points, np.asarray(extrinsics), np.asarray(intrinsics), tracks,
        np.asarray(images.shape[-2:]), masks=track_mask,
        max_reproj_error=max_reprojection_error, shared_camera=shared_camera,
        camera_type="PINHOLE", points_rgb=tracked_rgb,
    )
    if reconstruction is None:
        raise RuntimeError("VGGT tracks could not form a COLMAP reconstruction.")
    before = reconstruction.compute_mean_reprojection_error()
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = refine_focal_length
    options.refine_principal_point = refine_principal_point
    options.refine_extra_params = False
    summary = pycolmap.bundle_adjustment(reconstruction, options)
    after = reconstruction.compute_mean_reprojection_error()

    refined_extrinsics, refined_intrinsics = [], []
    for image_id in sorted(reconstruction.images):
        image = reconstruction.images[image_id]
        refined_extrinsics.append(_pose_matrix(image))
        refined_intrinsics.append(_camera_matrix(reconstruction.cameras[image.camera_id]))
    if len(refined_extrinsics) != len(extrinsics):
        raise RuntimeError(f"BA registered {len(refined_extrinsics)}/{len(extrinsics)} cameras.")
    diagnostics = {
        "registered_cameras": len(refined_extrinsics),
        "candidate_tracks": int(track_mask.shape[1]),
        "valid_observations": int(track_mask.sum()),
        "valid_tracks": int(np.asarray(valid_track_mask).sum()),
        "mean_reprojection_error_before_px": float(before),
        "mean_reprojection_error_after_px": float(after),
        "summary": str(summary),
    }
    return (np.stack(refined_extrinsics), np.stack(refined_intrinsics),
            reconstruction, diagnostics)
