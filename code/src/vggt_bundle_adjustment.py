"""Optional VGGT point-track bundle adjustment through PyCOLMAP."""

import numpy as np
import torch


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _pad_tracker_field(value, padding, channels_last=False):
    """Letterbox-pad a dense VGGT field without changing its aspect ratio."""
    array = _to_numpy(value)
    left, right, top, bottom = padding
    if channels_last:
        return np.pad(array, ((0, 0), (top, bottom), (left, right), (0, 0)))
    if array.ndim == 3:
        return np.pad(array, ((0, 0), (top, bottom), (left, right)))
    return np.pad(array, ((0, 0), (0, 0), (top, bottom), (left, right)))


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


def _make_pycolmap_image(pycolmap, image_id, camera_id, pose=None):
    """Construct an Image across the pre-rig and rig-based PyCOLMAP APIs."""
    arguments = dict(name=f"image_{image_id}", camera_id=camera_id)
    if pose is not None:
        arguments["cam_from_world"] = pose
    try:
        return pycolmap.Image(image_id=image_id, **arguments)
    except (TypeError, AttributeError):
        return pycolmap.Image(id=image_id, **arguments)


def _tracks_to_pycolmap(pycolmap, points3d, extrinsics, intrinsics, tracks,
                        masks, image_width, image_height, points_rgb,
                        max_reprojection_error, shared_camera):
    """Version-tolerant replacement for VGGT's PyCOLMAP conversion helper."""
    from vggt.dependency.projection import project_3D_points_np

    projected, camera_points = project_3D_points_np(points3d, extrinsics, intrinsics)
    reprojection_error = np.linalg.norm(projected - tracks, axis=-1)
    masks = masks & (reprojection_error < max_reprojection_error)
    masks &= camera_points[:, -1] > 0
    if masks.sum(axis=1).min() < 64:
        return None, None

    reconstruction = pycolmap.Reconstruction()
    valid_tracks = masks.sum(axis=0) >= 2
    valid_indices = np.flatnonzero(valid_tracks)
    point_ids = []
    for index in valid_indices:
        point_ids.append(reconstruction.add_point3D(
            points3d[index], pycolmap.Track(), points_rgb[index]
        ))

    shared = None
    for view in range(len(extrinsics)):
        if shared is None or not shared_camera:
            parameters = np.array([
                intrinsics[view, 0, 0], intrinsics[view, 1, 1],
                intrinsics[view, 0, 2], intrinsics[view, 1, 2],
            ])
            shared = pycolmap.Camera(
                model="PINHOLE", width=int(image_width), height=int(image_height),
                params=parameters, camera_id=view + 1,
            )
            if hasattr(reconstruction, "add_camera_with_trivial_rig"):
                reconstruction.add_camera_with_trivial_rig(shared)
            else:
                reconstruction.add_camera(shared)
        pose = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[view, :3, :3]),
            extrinsics[view, :3, 3],
        )
        modern_rig_api = hasattr(reconstruction, "add_image_with_trivial_frame")
        image = _make_pycolmap_image(
            pycolmap, view + 1, shared.camera_id, None if modern_rig_api else pose
        )
        observations = []
        for point_id, original_index in zip(point_ids, valid_indices):
            if masks[view, original_index]:
                observation_index = len(observations)
                observations.append(pycolmap.Point2D(tracks[view, original_index], point_id))
                reconstruction.points3D[point_id].track.add_element(
                    view + 1, observation_index
                )
        point_list = getattr(pycolmap, "Point2DList", None)
        if point_list is None:
            point_list = pycolmap.ListPoint2D
        image.points2D = point_list(observations)
        if modern_rig_api:
            reconstruction.add_image_with_trivial_frame(image, pose)
        else:
            image.registered = True
            reconstruction.add_image(image)
    return reconstruction, valid_tracks


def run_vggt_bundle_adjustment(images, depth_confidence, point_maps, colors, masks,
                               extrinsics, intrinsics, max_reprojection_error=8.0,
                               visibility_threshold=0.2, max_query_points=4096,
                               query_frame_count=8, fine_tracking=True,
                               shared_camera=False, refine_focal_length=True,
                               refine_principal_point=False,
                               max_camera_center_drift=0.25):
    """Build VGGT tracks, run one global BA, and return refined cameras.

    This follows the official VGGT ``demo_colmap.py`` path. Dense depth is not
    optimized by BA; callers must re-unproject it with the returned cameras.
    """
    import pycolmap
    from vggt.dependency.track_predict import predict_tracks

    original_height, original_width = images.shape[-2:]
    tracker_images = images
    tracker_confidence = depth_confidence
    tracker_points = point_maps
    pad_left = pad_top = 0
    if original_height != original_width:
        # The optional tracker asserts H == W. Symmetric letterboxing preserves
        # object shape; anisotropic resizing corrupts feature geometry.
        tracker_size = max(original_height, original_width)
        pad_left = (tracker_size - original_width) // 2
        pad_right = tracker_size - original_width - pad_left
        pad_top = (tracker_size - original_height) // 2
        pad_bottom = tracker_size - original_height - pad_top
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        tracker_images = torch.nn.functional.pad(images, padding, value=1.0)
        tracker_confidence = _pad_tracker_field(depth_confidence, padding)
        tracker_points = _pad_tracker_field(point_maps, padding, channels_last=True)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        tracks, visibility, track_confidence, tracked_points, tracked_rgb = predict_tracks(
            tracker_images, conf=tracker_confidence, points_3d=tracker_points, masks=None,
            max_query_pts=max_query_points,
            query_frame_num=min(query_frame_count, len(images)),
            keypoint_extractor="aliked+sp", fine_tracking=fine_tracking,
        )
    tracks = _to_numpy(tracks)
    tracks[..., 0] -= pad_left
    tracks[..., 1] -= pad_top
    tracked_points = _to_numpy(tracked_points)
    tracked_rgb = _to_numpy(tracked_rgb) if tracked_rgb is not None else None
    track_mask = _foreground_track_mask(tracks, visibility, masks, visibility_threshold)
    if tracked_rgb is None:
        raise RuntimeError("VGGT tracker did not return colors for its sparse 3D points.")
    reconstruction, valid_track_mask = _tracks_to_pycolmap(
        pycolmap, tracked_points, np.asarray(extrinsics), np.asarray(intrinsics),
        tracks, track_mask, original_width, original_height, tracked_rgb,
        max_reprojection_error, shared_camera,
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
    refined_extrinsics = np.stack(refined_extrinsics)
    before_centers = np.linalg.inv(np.concatenate([
        np.asarray(extrinsics), np.tile([[[0, 0, 0, 1]]], (len(extrinsics), 1, 1))
    ], axis=1))[:, :3, 3]
    after_centers = np.linalg.inv(np.concatenate([
        refined_extrinsics, np.tile([[[0, 0, 0, 1]]], (len(extrinsics), 1, 1))
    ], axis=1))[:, :3, 3]
    orbit_radius = max(float(np.median(np.linalg.norm(
        before_centers - np.median(before_centers, axis=0), axis=1
    ))), 1e-8)
    relative_center_drift = float(np.max(
        np.linalg.norm(after_centers - before_centers, axis=1)
    ) / orbit_radius)
    if relative_center_drift > max_camera_center_drift:
        raise RuntimeError(
            f"Rejected unstable BA: maximum camera-center drift is "
            f"{relative_center_drift:.3f} orbit radii (limit {max_camera_center_drift:.3f})."
        )
    diagnostics = {
        "registered_cameras": len(refined_extrinsics),
        "tracker_square_size": int(max(original_height, original_width)),
        "tracker_letterboxed_from_non_square": bool(original_height != original_width),
        "maximum_camera_center_drift_orbit_radii": relative_center_drift,
        "candidate_tracks": int(track_mask.shape[1]),
        "valid_observations": int(track_mask.sum()),
        "valid_tracks": int(np.asarray(valid_track_mask).sum()),
        "mean_reprojection_error_before_px": float(before),
        "mean_reprojection_error_after_px": float(after),
        "summary": str(summary),
    }
    return (refined_extrinsics, np.stack(refined_intrinsics),
            reconstruction, diagnostics)
