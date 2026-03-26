"""GT-to-prediction alignment utilities using track name mapping."""

from typing import Dict, List, Tuple, Optional, Set
import sleap_io as sio
import numpy as np
import pandas as pd


def apply_track_map(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    track_map: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align GT and predicted labels using track name mapping.

    Converts both label sets to DataFrames with consistent frame/track structure
    for comparison. The track_map is used to translate GT track names to the
    corresponding predicted track names.

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.
        track_map: Mapping of GT track names to predicted track names,
            e.g., {"track_1": "A5859", "track_2": "B1234"}.

    Returns:
        Tuple of (gt_df, pred_df) DataFrames with columns:
        - frame_idx: Frame index
        - track_name: Track name (GT uses mapped names for alignment)
        - instance_idx: Instance index within the frame
        - centroid_x, centroid_y: Instance centroid coordinates
    """
    gt_rows = []
    for lf in gt_labels:
        for inst_idx, inst in enumerate(lf.instances):
            if inst.track is not None:
                gt_track_name = _get_track_name(inst.track)
                mapped_name = track_map.get(gt_track_name, gt_track_name)
                centroid = _get_centroid(inst)
                gt_rows.append(
                    {
                        "frame_idx": lf.frame_idx,
                        "track_name": mapped_name,
                        "original_track_name": gt_track_name,
                        "instance_idx": inst_idx,
                        "centroid_x": centroid[0] if centroid else None,
                        "centroid_y": centroid[1] if centroid else None,
                    }
                )

    pred_rows = []
    for lf in pred_labels:
        for inst_idx, inst in enumerate(lf.instances):
            if inst.track is not None:
                track_name = _get_track_name(inst.track)
                centroid = _get_centroid(inst)
                pred_rows.append(
                    {
                        "frame_idx": lf.frame_idx,
                        "track_name": track_name,
                        "instance_idx": inst_idx,
                        "centroid_x": centroid[0] if centroid else None,
                        "centroid_y": centroid[1] if centroid else None,
                    }
                )

    gt_df = pd.DataFrame(gt_rows) if gt_rows else pd.DataFrame()
    pred_df = pd.DataFrame(pred_rows) if pred_rows else pd.DataFrame()

    return gt_df, pred_df


def build_frame_instance_map(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    track_map: Dict[str, str],
    match_by: str = "spatial",
    distance_threshold: float = 50.0,
) -> Dict[int, List[Tuple[int, int, str, str]]]:
    """Build per-frame mapping of GT instances to predicted instances.

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.
        track_map: Mapping of GT track names to predicted track names.
        match_by: Method for matching instances - "spatial" (centroid distance)
            or "track" (by mapped track name).
        distance_threshold: Maximum distance for spatial matching (pixels).

    Returns:
        Dict mapping frame_idx to list of tuples:
        (gt_inst_idx, pred_inst_idx, gt_track_name, pred_track_name)
        pred_inst_idx is -1 if no match found.
    """
    frame_map = {}

    # Build index of pred frames
    pred_frame_dict = {lf.frame_idx: lf for lf in pred_labels}

    for gt_lf in gt_labels:
        frame_idx = gt_lf.frame_idx
        matches = []

        pred_lf = pred_frame_dict.get(frame_idx)
        if pred_lf is None:
            # No predictions for this frame
            for gt_inst_idx, gt_inst in enumerate(gt_lf.instances):
                if gt_inst.track is not None:
                    gt_track = _get_track_name(gt_inst.track)
                    matches.append((gt_inst_idx, -1, gt_track, None))
            frame_map[frame_idx] = matches
            continue

        if match_by == "track":
            matches = _match_by_track_name(gt_lf, pred_lf, track_map)
        else:  # spatial
            matches = _match_by_spatial(
                gt_lf, pred_lf, track_map, distance_threshold
            )

        frame_map[frame_idx] = matches

    return frame_map


def infer_track_map_from_overlap(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    overlap_threshold: float = 0.5,
    distance_threshold: float = 50.0,
) -> Dict[str, str]:
    """Automatically infer track_map based on spatial overlap.

    Matches GT tracks to predicted tracks by finding which predicted track
    most frequently appears near each GT track across all frames.

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.
        overlap_threshold: Minimum fraction of frames where tracks must co-occur.
        distance_threshold: Maximum distance to consider tracks as co-located.

    Returns:
        Dictionary mapping GT track names to predicted track names.
    """
    # Count co-occurrences of GT and pred tracks
    cooccurrence = {}  # (gt_track, pred_track) -> count
    gt_track_frames = {}  # gt_track -> total frame count

    pred_frame_dict = {lf.frame_idx: lf for lf in pred_labels}

    for gt_lf in gt_labels:
        frame_idx = gt_lf.frame_idx
        pred_lf = pred_frame_dict.get(frame_idx)
        if pred_lf is None:
            continue

        for gt_inst in gt_lf.instances:
            if gt_inst.track is None:
                continue
            gt_track = _get_track_name(gt_inst.track)
            gt_track_frames[gt_track] = gt_track_frames.get(gt_track, 0) + 1

            gt_centroid = _get_centroid(gt_inst)
            if gt_centroid is None:
                continue

            # Find closest pred instance
            min_dist = float("inf")
            closest_pred_track = None

            for pred_inst in pred_lf.instances:
                if pred_inst.track is None:
                    continue
                pred_centroid = _get_centroid(pred_inst)
                if pred_centroid is None:
                    continue

                dist = np.sqrt(
                    (gt_centroid[0] - pred_centroid[0]) ** 2
                    + (gt_centroid[1] - pred_centroid[1]) ** 2
                )
                if dist < min_dist and dist < distance_threshold:
                    min_dist = dist
                    closest_pred_track = _get_track_name(pred_inst.track)

            if closest_pred_track is not None:
                key = (gt_track, closest_pred_track)
                cooccurrence[key] = cooccurrence.get(key, 0) + 1

    # For each GT track, find the pred track with highest co-occurrence
    track_map = {}
    for gt_track, total_frames in gt_track_frames.items():
        best_pred = None
        best_count = 0

        for (gt, pred), count in cooccurrence.items():
            if gt == gt_track and count > best_count:
                best_count = count
                best_pred = pred

        if best_pred is not None:
            overlap_ratio = best_count / total_frames
            if overlap_ratio >= overlap_threshold:
                track_map[gt_track] = best_pred

    return track_map


def get_common_frames(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
) -> List[int]:
    """Get list of frame indices present in both GT and pred labels.

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.

    Returns:
        Sorted list of frame indices present in both.
    """
    gt_frames = {lf.frame_idx for lf in gt_labels}
    pred_frames = {lf.frame_idx for lf in pred_labels}
    return sorted(gt_frames & pred_frames)


def _get_track_name(track) -> str:
    """Extract track name from Track or TrackContext object.

    Args:
        track: Track, TrackContext, or similar object.

    Returns:
        Track name string.
    """
    # Handle TrackContext
    if hasattr(track, "name") and isinstance(track.name, str):
        return track.name
    # Handle sio.Track
    if hasattr(track, "name"):
        return str(track.name)
    return str(track)


def _get_centroid(inst: sio.Instance) -> Optional[Tuple[float, float]]:
    """Compute centroid of an instance.

    Args:
        inst: Instance object with points.

    Returns:
        (x, y) centroid tuple, or None if no valid points.
    """
    if inst.numpy() is None:
        return None

    points = inst.numpy()
    valid_mask = ~np.isnan(points).any(axis=1)
    if not valid_mask.any():
        return None

    valid_points = points[valid_mask]
    return (float(np.mean(valid_points[:, 0])), float(np.mean(valid_points[:, 1])))


def _match_by_track_name(
    gt_lf: sio.LabeledFrame,
    pred_lf: sio.LabeledFrame,
    track_map: Dict[str, str],
) -> List[Tuple[int, int, str, str]]:
    """Match GT and pred instances by mapped track name.

    Args:
        gt_lf: Ground truth labeled frame.
        pred_lf: Predicted labeled frame.
        track_map: GT to pred track name mapping.

    Returns:
        List of (gt_inst_idx, pred_inst_idx, gt_track, pred_track) tuples.
    """
    matches = []

    # Build pred track index
    pred_by_track = {}
    for pred_idx, pred_inst in enumerate(pred_lf.instances):
        if pred_inst.track is not None:
            pred_track = _get_track_name(pred_inst.track)
            pred_by_track[pred_track] = pred_idx

    for gt_idx, gt_inst in enumerate(gt_lf.instances):
        if gt_inst.track is None:
            continue

        gt_track = _get_track_name(gt_inst.track)
        expected_pred_track = track_map.get(gt_track)

        if expected_pred_track is not None and expected_pred_track in pred_by_track:
            pred_idx = pred_by_track[expected_pred_track]
            pred_track = expected_pred_track
        else:
            pred_idx = -1
            pred_track = None

        matches.append((gt_idx, pred_idx, gt_track, pred_track))

    return matches


def _match_by_spatial(
    gt_lf: sio.LabeledFrame,
    pred_lf: sio.LabeledFrame,
    track_map: Dict[str, str],
    distance_threshold: float,
) -> List[Tuple[int, int, str, str]]:
    """Match GT and pred instances by spatial proximity.

    Uses Hungarian algorithm for optimal assignment.

    Args:
        gt_lf: Ground truth labeled frame.
        pred_lf: Predicted labeled frame.
        track_map: GT to pred track name mapping.
        distance_threshold: Maximum matching distance.

    Returns:
        List of (gt_inst_idx, pred_inst_idx, gt_track, pred_track) tuples.
    """
    from scipy.optimize import linear_sum_assignment

    matches = []

    gt_instances = [
        (i, inst)
        for i, inst in enumerate(gt_lf.instances)
        if inst.track is not None
    ]
    pred_instances = [
        (i, inst)
        for i, inst in enumerate(pred_lf.instances)
        if inst.track is not None
    ]

    if not gt_instances or not pred_instances:
        for gt_idx, gt_inst in gt_instances:
            gt_track = _get_track_name(gt_inst.track)
            matches.append((gt_idx, -1, gt_track, None))
        return matches

    # Build cost matrix
    cost_matrix = np.full((len(gt_instances), len(pred_instances)), np.inf)

    for i, (gt_idx, gt_inst) in enumerate(gt_instances):
        gt_centroid = _get_centroid(gt_inst)
        if gt_centroid is None:
            continue

        for j, (pred_idx, pred_inst) in enumerate(pred_instances):
            pred_centroid = _get_centroid(pred_inst)
            if pred_centroid is None:
                continue

            dist = np.sqrt(
                (gt_centroid[0] - pred_centroid[0]) ** 2
                + (gt_centroid[1] - pred_centroid[1]) ** 2
            )
            if dist < distance_threshold:
                cost_matrix[i, j] = dist

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_gt = set()
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < np.inf:
            gt_idx, gt_inst = gt_instances[i]
            pred_idx, pred_inst = pred_instances[j]
            gt_track = _get_track_name(gt_inst.track)
            pred_track = _get_track_name(pred_inst.track)
            matches.append((gt_idx, pred_idx, gt_track, pred_track))
            matched_gt.add(i)

    # Add unmatched GT instances
    for i, (gt_idx, gt_inst) in enumerate(gt_instances):
        if i not in matched_gt:
            gt_track = _get_track_name(gt_inst.track)
            matches.append((gt_idx, -1, gt_track, None))

    return matches
