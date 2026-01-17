"""Per-instance metrics for tracking evaluation."""

from typing import Dict, List, Optional
import sleap_io as sio
import numpy as np

from sleap_mot.metrics.alignment import (
    build_frame_instance_map,
    _get_track_name,
    get_common_frames,
)


def compute_instance_accuracy(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    track_map: Dict[str, str],
    match_by: str = "spatial",
    distance_threshold: float = 50.0,
) -> List[Dict]:
    """Compute accuracy for each instance.

    Evaluates whether each GT instance has the correct predicted identity
    based on the track_map.

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.
        track_map: Mapping of GT track names to predicted track names.
        match_by: Method for matching instances - "spatial" or "track".
        distance_threshold: Maximum distance for spatial matching.

    Returns:
        List of dicts with keys:
        - frame_idx: Frame index
        - gt_track: Ground truth track name
        - pred_track: Predicted track name (or None if not matched)
        - expected_pred_track: Expected predicted track from track_map
        - correct: Whether prediction matches expected
        - gt_instance_idx: Instance index in GT frame
        - pred_instance_idx: Instance index in pred frame (-1 if unmatched)
    """
    results = []

    frame_map = build_frame_instance_map(
        gt_labels,
        pred_labels,
        track_map,
        match_by=match_by,
        distance_threshold=distance_threshold,
    )

    for frame_idx, matches in frame_map.items():
        for gt_inst_idx, pred_inst_idx, gt_track, pred_track in matches:
            expected_pred = track_map.get(gt_track)

            if pred_track is None:
                correct = False
            else:
                correct = pred_track == expected_pred

            results.append(
                {
                    "frame_idx": frame_idx,
                    "gt_track": gt_track,
                    "pred_track": pred_track,
                    "expected_pred_track": expected_pred,
                    "correct": correct,
                    "gt_instance_idx": gt_inst_idx,
                    "pred_instance_idx": pred_inst_idx,
                }
            )

    return results


def compute_instance_switch_events(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    track_map: Dict[str, str],
    match_by: str = "spatial",
    distance_threshold: float = 50.0,
) -> List[Dict]:
    """Identify frames where identity switches occur.

    A switch event occurs when a GT track is assigned a different predicted
    identity than it had in the previous frame.

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.
        track_map: Mapping of GT track names to predicted track names.
        match_by: Method for matching instances.
        distance_threshold: Maximum distance for spatial matching.

    Returns:
        List of switch events with:
        - frame_idx: Frame where switch occurred
        - gt_track: GT track that experienced the switch
        - old_pred_track: Previous predicted identity
        - new_pred_track: New predicted identity
        - switch_type: "incorrect_to_correct", "correct_to_incorrect",
          "incorrect_to_incorrect", or "appeared"/"disappeared"
    """
    # First get instance accuracy for all frames
    instance_results = compute_instance_accuracy(
        gt_labels,
        pred_labels,
        track_map,
        match_by=match_by,
        distance_threshold=distance_threshold,
    )

    # Group by GT track and sort by frame
    by_gt_track = {}
    for result in instance_results:
        gt_track = result["gt_track"]
        if gt_track not in by_gt_track:
            by_gt_track[gt_track] = []
        by_gt_track[gt_track].append(result)

    for gt_track in by_gt_track:
        by_gt_track[gt_track].sort(key=lambda x: x["frame_idx"])

    # Detect switches
    switch_events = []

    for gt_track, results in by_gt_track.items():
        expected_pred = track_map.get(gt_track)

        for i in range(len(results)):
            curr = results[i]

            if i == 0:
                # First appearance
                if curr["pred_track"] is not None and curr["pred_track"] != expected_pred:
                    switch_events.append(
                        {
                            "frame_idx": curr["frame_idx"],
                            "gt_track": gt_track,
                            "old_pred_track": None,
                            "new_pred_track": curr["pred_track"],
                            "expected_pred_track": expected_pred,
                            "switch_type": "appeared_incorrect",
                        }
                    )
                continue

            prev = results[i - 1]

            # Check for gaps (track disappeared and reappeared)
            if curr["frame_idx"] - prev["frame_idx"] > 1:
                # There's a gap - could be occlusion
                pass

            # Check for pred track change
            if curr["pred_track"] != prev["pred_track"]:
                prev_correct = prev["pred_track"] == expected_pred
                curr_correct = curr["pred_track"] == expected_pred

                if prev["pred_track"] is None and curr["pred_track"] is not None:
                    switch_type = (
                        "appeared_correct" if curr_correct else "appeared_incorrect"
                    )
                elif prev["pred_track"] is not None and curr["pred_track"] is None:
                    switch_type = "disappeared"
                elif prev_correct and not curr_correct:
                    switch_type = "correct_to_incorrect"
                elif not prev_correct and curr_correct:
                    switch_type = "incorrect_to_correct"
                else:
                    switch_type = "incorrect_to_incorrect"

                switch_events.append(
                    {
                        "frame_idx": curr["frame_idx"],
                        "gt_track": gt_track,
                        "old_pred_track": prev["pred_track"],
                        "new_pred_track": curr["pred_track"],
                        "expected_pred_track": expected_pred,
                        "switch_type": switch_type,
                    }
                )

    # Sort by frame
    switch_events.sort(key=lambda x: x["frame_idx"])

    return switch_events


def get_instance_timeline(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    track_map: Dict[str, str],
    gt_track_name: str,
    match_by: str = "spatial",
    distance_threshold: float = 50.0,
) -> List[Dict]:
    """Get the complete timeline of predictions for a specific GT track.

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.
        track_map: GT to pred track name mapping.
        gt_track_name: Name of the GT track to analyze.
        match_by: Method for matching instances.
        distance_threshold: Maximum distance for spatial matching.

    Returns:
        List of dicts sorted by frame with:
        - frame_idx: Frame index
        - pred_track: Predicted track at this frame
        - expected_pred_track: Expected predicted track
        - correct: Whether prediction is correct
    """
    instance_results = compute_instance_accuracy(
        gt_labels,
        pred_labels,
        track_map,
        match_by=match_by,
        distance_threshold=distance_threshold,
    )

    timeline = [r for r in instance_results if r["gt_track"] == gt_track_name]
    timeline.sort(key=lambda x: x["frame_idx"])

    return timeline
