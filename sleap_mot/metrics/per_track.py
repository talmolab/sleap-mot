"""Per-track metrics for tracking evaluation."""

from typing import Dict, List, Set
import sleap_io as sio
import numpy as np
from collections import defaultdict

from sleap_mot.metrics.types import MetricResult
from sleap_mot.metrics.per_instance import compute_instance_accuracy


def compute_fragmentation(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    track_map: Dict[str, str],
    match_by: str = "spatial",
    distance_threshold: float = 50.0,
) -> MetricResult:
    """Compute fragmentation: how often GT tracks are split.

    Fragmentation measures how many contiguous segments with different
    predicted IDs exist for a single GT track. A perfectly tracked GT
    track has fragmentation = 1 (one segment).

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.
        track_map: GT to pred track name mapping.
        match_by: Method for matching instances.
        distance_threshold: Maximum distance for spatial matching.

    Returns:
        MetricResult with:
        - value: Mean fragmentation across all GT tracks
        - per_track: Dict mapping GT track names to segment counts
    """
    instance_results = compute_instance_accuracy(
        gt_labels,
        pred_labels,
        track_map,
        match_by=match_by,
        distance_threshold=distance_threshold,
    )

    # Group by GT track
    by_gt_track = defaultdict(list)
    for result in instance_results:
        by_gt_track[result["gt_track"]].append(result)

    per_gt_track = {}

    for gt_track, results in by_gt_track.items():
        # Sort by frame
        results.sort(key=lambda x: x["frame_idx"])

        if not results:
            per_gt_track[gt_track] = 0
            continue

        # Count segments (transitions between different pred tracks)
        segments = 1
        prev_pred = results[0]["pred_track"]

        for result in results[1:]:
            curr_pred = result["pred_track"]
            if curr_pred != prev_pred:
                segments += 1
            prev_pred = curr_pred

        per_gt_track[gt_track] = segments

    # Calculate mean
    values = list(per_gt_track.values())
    mean_frag = np.mean(values) if values else 0.0

    return MetricResult(
        name="fragmentation",
        value=float(mean_frag),
        per_track=per_gt_track,
        metadata={
            "description": "Number of contiguous segments per GT track",
            "ideal_value": 1,
            "total_gt_tracks": len(per_gt_track),
        },
    )


def compute_track_purity(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    track_map: Dict[str, str],
    match_by: str = "spatial",
    distance_threshold: float = 50.0,
) -> MetricResult:
    """Compute track purity: how many GT identities per predicted track.

    Purity measures how "pure" each predicted track is in terms of the
    GT identities it contains. A perfectly pure track contains only one
    GT identity (purity = 1.0).

    Purity = 1.0 / num_gt_identities_in_track

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.
        track_map: GT to pred track name mapping.
        match_by: Method for matching instances.
        distance_threshold: Maximum distance for spatial matching.

    Returns:
        MetricResult with:
        - value: Mean purity across all predicted tracks
        - per_track: Dict mapping pred track names to purity scores
    """
    instance_results = compute_instance_accuracy(
        gt_labels,
        pred_labels,
        track_map,
        match_by=match_by,
        distance_threshold=distance_threshold,
    )

    # Group by predicted track
    gt_identities_per_pred = defaultdict(set)

    for result in instance_results:
        pred_track = result["pred_track"]
        gt_track = result["gt_track"]

        if pred_track is not None:
            gt_identities_per_pred[pred_track].add(gt_track)

    per_pred_track = {}

    for pred_track, gt_identities in gt_identities_per_pred.items():
        num_identities = len(gt_identities)
        purity = 1.0 / num_identities if num_identities > 0 else 0.0
        per_pred_track[pred_track] = {
            "purity": purity,
            "num_gt_identities": num_identities,
            "gt_identities": list(gt_identities),
        }

    # Calculate mean purity
    purity_values = [v["purity"] for v in per_pred_track.values()]
    mean_purity = np.mean(purity_values) if purity_values else 0.0

    return MetricResult(
        name="track_purity",
        value=float(mean_purity),
        per_track=per_pred_track,
        metadata={
            "description": "Purity score (1/num_gt_identities) per predicted track",
            "ideal_value": 1.0,
            "total_pred_tracks": len(per_pred_track),
        },
    )


def compute_track_completeness(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    track_map: Dict[str, str],
    match_by: str = "spatial",
    distance_threshold: float = 50.0,
) -> MetricResult:
    """Compute track completeness: fraction of GT frames correctly tracked.

    For each GT track, measures what fraction of frames have the correct
    predicted identity assigned.

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.
        track_map: GT to pred track name mapping.
        match_by: Method for matching instances.
        distance_threshold: Maximum distance for spatial matching.

    Returns:
        MetricResult with:
        - value: Mean completeness across all GT tracks
        - per_track: Dict mapping GT track names to completeness scores
    """
    instance_results = compute_instance_accuracy(
        gt_labels,
        pred_labels,
        track_map,
        match_by=match_by,
        distance_threshold=distance_threshold,
    )

    # Group by GT track
    by_gt_track = defaultdict(list)
    for result in instance_results:
        by_gt_track[result["gt_track"]].append(result)

    per_gt_track = {}

    for gt_track, results in by_gt_track.items():
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        completeness = correct / total if total > 0 else 0.0

        per_gt_track[gt_track] = {
            "completeness": completeness,
            "correct_frames": correct,
            "total_frames": total,
        }

    # Calculate mean
    completeness_values = [v["completeness"] for v in per_gt_track.values()]
    mean_completeness = np.mean(completeness_values) if completeness_values else 0.0

    return MetricResult(
        name="track_completeness",
        value=float(mean_completeness),
        per_track=per_gt_track,
        metadata={
            "description": "Fraction of frames correctly tracked per GT track",
            "ideal_value": 1.0,
            "total_gt_tracks": len(per_gt_track),
        },
    )


def compute_id_switches_per_track(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    track_map: Dict[str, str],
    match_by: str = "spatial",
    distance_threshold: float = 50.0,
) -> MetricResult:
    """Compute number of ID switches per GT track.

    Counts how many times the predicted identity changes for each GT track.

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.
        track_map: GT to pred track name mapping.
        match_by: Method for matching instances.
        distance_threshold: Maximum distance for spatial matching.

    Returns:
        MetricResult with:
        - value: Total number of ID switches
        - per_track: Dict mapping GT track names to switch counts
    """
    instance_results = compute_instance_accuracy(
        gt_labels,
        pred_labels,
        track_map,
        match_by=match_by,
        distance_threshold=distance_threshold,
    )

    # Group by GT track
    by_gt_track = defaultdict(list)
    for result in instance_results:
        by_gt_track[result["gt_track"]].append(result)

    per_gt_track = {}
    total_switches = 0

    for gt_track, results in by_gt_track.items():
        # Sort by frame
        results.sort(key=lambda x: x["frame_idx"])

        switches = 0
        if len(results) > 1:
            for i in range(1, len(results)):
                if results[i]["pred_track"] != results[i - 1]["pred_track"]:
                    switches += 1

        per_gt_track[gt_track] = switches
        total_switches += switches

    return MetricResult(
        name="id_switches_per_track",
        value=float(total_switches),
        per_track=per_gt_track,
        metadata={
            "description": "Number of ID switches per GT track",
            "ideal_value": 0,
            "total_gt_tracks": len(per_gt_track),
        },
    )
