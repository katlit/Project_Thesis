"""Similarity-aligned point-cloud evaluation against a reference triangle mesh."""

from itertools import permutations, product
import numpy as np
from scipy.spatial import cKDTree


def _pca_frame(points):
    _, _, vh = np.linalg.svd(points - np.mean(points, axis=0), full_matrices=False)
    return vh.T


def _umeyama(source, target):
    source_mean, target_mean = source.mean(0), target.mean(0)
    x, y = source - source_mean, target - target_mean
    u, singular, vt = np.linalg.svd(y.T @ x / len(source))
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(x * x, axis=1))
    scale = (singular * np.diag(correction)).sum() / max(variance, 1e-12)
    translation = target_mean - scale * (rotation @ source_mean)
    return {"scale": float(scale), "rotation": rotation, "translation": translation}


def apply_similarity(points, transform):
    points = np.asarray(points)
    return transform["scale"] * (points @ transform["rotation"].T) + transform["translation"]


def _trimmed_score(moved, reference, trim_fraction):
    forward = cKDTree(reference).query(moved, workers=-1)[0]
    backward = cKDTree(moved).query(reference, workers=-1)[0]
    nf = max(1, int(trim_fraction * len(forward)))
    nb = max(1, int(trim_fraction * len(backward)))
    return float(0.5 * (np.partition(forward, nf - 1)[:nf].mean() +
                        np.partition(backward, nb - 1)[:nb].mean()))


def align_similarity_icp(predicted, reference, max_points=12_000, iterations=30,
                         trim_fraction=0.8, seed=42):
    """PCA multi-start followed by robust bidirectional similarity ICP."""
    predicted, reference = np.asarray(predicted, float), np.asarray(reference, float)
    if min(len(predicted), len(reference)) < 20:
        raise ValueError("At least 20 predicted and reference points are required.")
    rng = np.random.default_rng(seed)
    p = predicted[rng.choice(len(predicted), min(len(predicted), max_points), replace=False)]
    r = reference[rng.choice(len(reference), min(len(reference), max_points), replace=False)]
    p_center, r_center = p.mean(0), r.mean(0)
    p_frame, r_frame = _pca_frame(p), _pca_frame(r)
    p_extent = np.linalg.norm(np.percentile(p, 95, axis=0) - np.percentile(p, 5, axis=0))
    r_extent = np.linalg.norm(np.percentile(r, 95, axis=0) - np.percentile(r, 5, axis=0))
    initial_scale = r_extent / max(p_extent, 1e-12)
    reference_tree = cKDTree(r)
    candidates = []
    for permutation in permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in product((-1.0, 1.0), repeat=3):
            rotation = r_frame @ permutation_matrix @ np.diag(signs) @ p_frame.T
            if np.linalg.det(rotation) < 0:
                continue
            transform = {
                "scale": float(initial_scale), "rotation": rotation,
                "translation": r_center - initial_scale * (rotation @ p_center),
            }
            candidates.append((_trimmed_score(apply_similarity(p, transform), r, trim_fraction), transform))

    best = None
    for _, transform in sorted(candidates, key=lambda item: item[0])[:4]:
        previous_score = np.inf
        for _ in range(iterations):
            moved = apply_similarity(p, transform)
            forward_distance, forward_index = reference_tree.query(moved, workers=-1)
            moved_tree = cKDTree(moved)
            backward_distance, backward_index = moved_tree.query(r, workers=-1)
            forward_keep = forward_distance <= np.quantile(forward_distance, trim_fraction)
            backward_keep = backward_distance <= np.quantile(backward_distance, trim_fraction)
            source_pairs = np.concatenate([p[forward_keep], p[backward_index[backward_keep]]])
            target_pairs = np.concatenate([r[forward_index[forward_keep]], r[backward_keep]])
            transform = _umeyama(source_pairs, target_pairs)
            score = _trimmed_score(apply_similarity(p, transform), r, trim_fraction)
            if np.isfinite(previous_score) and abs(previous_score - score) <= max(previous_score, 1e-12) * 1e-6:
                break
            previous_score = score
        if best is None or score < best[0]:
            best = score, transform
    return best[1]


def geometry_metrics(predicted_aligned, reference, thresholds):
    """Return accuracy, completeness, symmetric Chamfer-L1, and F-scores."""
    predicted_aligned, reference = np.asarray(predicted_aligned), np.asarray(reference)
    pred_to_ref = cKDTree(reference).query(predicted_aligned, workers=-1)[0]
    ref_to_pred = cKDTree(predicted_aligned).query(reference, workers=-1)[0]
    result = {
        "predicted_points": len(predicted_aligned), "reference_points": len(reference),
        "accuracy_mean": float(pred_to_ref.mean()), "accuracy_median": float(np.median(pred_to_ref)),
        "completeness_mean": float(ref_to_pred.mean()), "completeness_median": float(np.median(ref_to_pred)),
        "chamfer_l1": float(0.5 * (pred_to_ref.mean() + ref_to_pred.mean())),
    }
    for threshold in thresholds:
        precision = float(np.mean(pred_to_ref <= threshold))
        recall = float(np.mean(ref_to_pred <= threshold))
        result[f"precision@{threshold:g}"] = precision
        result[f"recall@{threshold:g}"] = recall
        result[f"fscore@{threshold:g}"] = 2 * precision * recall / max(precision + recall, 1e-12)
    return result, pred_to_ref, ref_to_pred
