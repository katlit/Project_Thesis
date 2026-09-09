"""Optional VGGT point-track bundle adjustment through PyCOLMAP."""

import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _resize_tracker_field(value, size, channels_last=False):
    """Resize a dense VGGT field for the square-only official tracker."""
    array = _to_numpy(value)
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if channels_last:
        tensor = tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 3:
        tensor = tensor[:, None]
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    if channels_last:
        return resized.permute(0, 2, 3, 1).cpu().numpy()
    resized = resized.cpu().numpy()
    return resized[:, 0] if array.ndim == 3 else resized


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


def _make_pycolmap_image(pycolmap, image_id, camera_id, pose):
    """Construct an Image across the PyCOLMAP 3.10 and newer APIs."""
    arguments = dict(
        name=f"image_{image_id}", camera_id=camera_id, cam_from_world=pose
    )
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
            reconstruction.add_camera(shared)
        pose = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[view, :3, :3]),
            extrinsics[view, :3, 3],
        )
        image = _make_pycolmap_image(pycolmap, view + 1, shared.camera_id, pose)
        observations = []
        for point_id, original_index in zip(point_ids, valid_indices):
            if masks[view, original_index]:
                observation_index = len(observations)
                observations.append(pycolmap.Point2D(tracks[view, original_index], point_id))
                reconstruction.points3D[point_id].track.add_element(
                    view + 1, observation_index
                )
        image.points2D = pycolmap.ListPoint2D(observations)
        if hasattr(image, "registered"):
            image.registered = True
        reconstruction.add_image(image)
    return reconstruction, valid_tracks


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
    from vggt.dependency.track_predict import predict_tracks

    original_height, original_width = images.shape[-2:]
    tracker_images = images
    tracker_confidence = depth_confidence
    tracker_points = point_maps
    scale_x = scale_y = 1.0
    if original_height != original_width:
        # VGGT's optional predict_tracks path currently asserts H == W. Resize
        # only its tracking inputs, then return tracks to original coordinates.
        tracker_size = max(original_height, original_width)
        tracker_images = F.interpolate(
            images, size=(tracker_size, tracker_size), mode="bilinear", align_corners=False
        )
        tracker_confidence = _resize_tracker_field(depth_confidence, tracker_size)
        tracker_points = _resize_tracker_field(point_maps, tracker_size, channels_last=True)
        scale_x = tracker_size / original_width
        scale_y = tracker_size / original_height

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        tracks, visibility, track_confidence, tracked_points, tracked_rgb = predict_tracks(
            tracker_images, conf=tracker_confidence, points_3d=tracker_points, masks=None,
            max_query_pts=max_query_points,
            query_frame_num=min(query_frame_count, len(images)),
            keypoint_extractor="aliked+sp", fine_tracking=fine_tracking,
        )
    tracks = _to_numpy(tracks)
    tracks[..., 0] /= scale_x
    tracks[..., 1] /= scale_y
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
    diagnostics = {
        "registered_cameras": len(refined_extrinsics),
        "tracker_square_size": int(max(original_height, original_width)),
        "tracker_resized_from_non_square": bool(original_height != original_width),
        "candidate_tracks": int(track_mask.shape[1]),
        "valid_observations": int(track_mask.sum()),
        "valid_tracks": int(np.asarray(valid_track_mask).sum()),
        "mean_reprojection_error_before_px": float(before),
        "mean_reprojection_error_after_px": float(after),
        "summary": str(summary),
    }
    return (np.stack(refined_extrinsics), np.stack(refined_intrinsics),
            reconstruction, diagnostics)
