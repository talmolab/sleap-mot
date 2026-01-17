"""Per-frame metrics for tracking evaluation."""

from typing import Dict, List, Optional
import sleap_io as sio
import numpy as np
from collections import defaultdict

from sleap_mot.metrics.types import MetricResult
from sleap_mot.metrics.per_instance import (
    compute_instance_accuracy,
    compute_instance_switch_events,
)


def compute_frame_accuracy(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    track_map: Dict[str, str],
    match_by: str = "spatial",
    distance_threshold: float = 50.0,
) -> MetricResult:
    """Compute accuracy per frame.

    For each frame, calculates the fraction of GT instances that have
    the correct predicted identity.

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.
        track_map: GT to pred track name mapping.
        match_by: Method for matching instances.
        distance_threshold: Maximum distance for spatial matching.

    Returns:
        MetricResult with:
        - value: Mean accuracy across all frames
        - per_frame: Dict mapping frame_idx to accuracy
    """
    instance_results = compute_instance_accuracy(
        gt_labels,
        pred_labels,
        track_map,
        match_by=match_by,
        distance_threshold=distance_threshold,
    )

    # Group by frame
    by_frame = defaultdict(list)
    for result in instance_results:
        by_frame[result["frame_idx"]].append(result)

    per_frame = {}
    total_correct = 0
    total_instances = 0

    for frame_idx, results in by_frame.items():
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        accuracy = correct / total if total > 0 else 0.0

        per_frame[frame_idx] = accuracy
        total_correct += correct
        total_instances += total

    # Overall accuracy
    overall_accuracy = total_correct / total_instances if total_instances > 0 else 0.0

    return MetricResult(
        name="frame_accuracy",
        value=float(overall_accuracy),
        per_frame=per_frame,
        metadata={
            "total_correct": total_correct,
            "total_instances": total_instances,
            "num_frames": len(per_frame),
        },
    )


def compute_switch_latency(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    track_map: Dict[str, str],
    match_by: str = "spatial",
    distance_threshold: float = 50.0,
) -> MetricResult:
    """Compute frames until wrong identity is corrected.

    For each identity switch event, counts how many frames until:
    - Identity is corrected back to the expected value
    - Track ends
    - Video ends

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.
        track_map: GT to pred track name mapping.
        match_by: Method for matching instances.
        distance_threshold: Maximum distance for spatial matching.

    Returns:
        MetricResult with:
        - value: Mean latency for corrected switches
        - per_instance: List of switch events with latency info
        - metadata: Total switches, corrected switches
    """
    # Get instance accuracy timeline
    instance_results = compute_instance_accuracy(
        gt_labels,
        pred_labels,
        track_map,
        match_by=match_by,
        distance_threshold=distance_threshold,
    )

    # Group by GT track and sort by frame
    by_gt_track = defaultdict(list)
    for result in instance_results:
        by_gt_track[result["gt_track"]].append(result)

    for gt_track in by_gt_track:
        by_gt_track[gt_track].sort(key=lambda x: x["frame_idx"])

    # Find switch events and compute latencies
    latency_events = []

    for gt_track, results in by_gt_track.items():
        expected_pred = track_map.get(gt_track)

        for i in range(len(results)):
            curr = results[i]

            # Check if this is an incorrect assignment
            if not curr["correct"] and curr["pred_track"] is not None:
                # Find when/if it gets corrected
                correction_frame = None
                for j in range(i + 1, len(results)):
                    if results[j]["correct"]:
                        correction_frame = results[j]["frame_idx"]
                        break

                if correction_frame is not None:
                    latency = correction_frame - curr["frame_idx"]
                    corrected = True
                else:
                    # Never corrected - use frames until end of track
                    latency = results[-1]["frame_idx"] - curr["frame_idx"]
                    corrected = False

                latency_events.append(
                    {
                        "frame_idx": curr["frame_idx"],
                        "gt_track": gt_track,
                        "incorrect_pred_track": curr["pred_track"],
                        "expected_pred_track": expected_pred,
                        "latency_frames": latency,
                        "corrected": corrected,
                        "correction_frame": correction_frame,
                    }
                )

    # Calculate statistics
    corrected_latencies = [e["latency_frames"] for e in latency_events if e["corrected"]]
    all_latencies = [e["latency_frames"] for e in latency_events]

    mean_latency = np.mean(corrected_latencies) if corrected_latencies else 0.0
    mean_all_latency = np.mean(all_latencies) if all_latencies else 0.0

    return MetricResult(
        name="switch_latency",
        value=float(mean_latency),
        per_instance=latency_events,
        metadata={
            "total_incorrect_assignments": len(latency_events),
            "corrected_count": len(corrected_latencies),
            "uncorrected_count": len(latency_events) - len(corrected_latencies),
            "mean_latency_all": float(mean_all_latency),
            "description": "Mean frames until incorrect identity is corrected",
        },
    )


def compute_mislabeled_segments(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    track_map: Dict[str, str],
    match_by: str = "spatial",
    distance_threshold: float = 50.0,
) -> MetricResult:
    """Compute contiguous segments of mislabeled frames.

    Identifies runs of consecutive frames where at least one instance
    is incorrectly labeled.

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.
        track_map: GT to pred track name mapping.
        match_by: Method for matching instances.
        distance_threshold: Maximum distance for spatial matching.

    Returns:
        MetricResult with:
        - value: Mean length of mislabeled segments
        - per_frame: Dict mapping frame_idx to is_mislabeled (0 or 1)
        - metadata: Segment info
    """
    instance_results = compute_instance_accuracy(
        gt_labels,
        pred_labels,
        track_map,
        match_by=match_by,
        distance_threshold=distance_threshold,
    )

    # Group by frame
    by_frame = defaultdict(list)
    for result in instance_results:
        by_frame[result["frame_idx"]].append(result)

    # Determine if each frame has any mislabeled instances
    sorted_frames = sorted(by_frame.keys())
    per_frame = {}
    mislabeled_frames = []

    for frame_idx in sorted_frames:
        results = by_frame[frame_idx]
        has_error = any(not r["correct"] for r in results)
        per_frame[frame_idx] = 1 if has_error else 0
        if has_error:
            mislabeled_frames.append(frame_idx)

    # Find contiguous segments
    segments = []
    if mislabeled_frames:
        current_segment = [mislabeled_frames[0]]
        for frame in mislabeled_frames[1:]:
            if frame == current_segment[-1] + 1:
                current_segment.append(frame)
            else:
                segments.append(current_segment)
                current_segment = [frame]
        segments.append(current_segment)

    segment_lengths = [len(s) for s in segments]
    mean_length = np.mean(segment_lengths) if segment_lengths else 0.0

    return MetricResult(
        name="mislabeled_segments",
        value=float(mean_length),
        per_frame=per_frame,
        metadata={
            "num_segments": len(segments),
            "segment_lengths": segment_lengths,
            "total_mislabeled_frames": len(mislabeled_frames),
            "total_frames": len(sorted_frames),
            "segments": [
                {"start": s[0], "end": s[-1], "length": len(s)} for s in segments
            ],
        },
    )


def compute_correct_segments(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    track_map: Dict[str, str],
    match_by: str = "spatial",
    distance_threshold: float = 50.0,
) -> MetricResult:
    """Compute contiguous segments of correctly labeled frames.

    Identifies runs of consecutive frames where all instances are
    correctly labeled.

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.
        track_map: GT to pred track name mapping.
        match_by: Method for matching instances.
        distance_threshold: Maximum distance for spatial matching.

    Returns:
        MetricResult with:
        - value: Mean length of correct segments
        - per_frame: Dict mapping frame_idx to is_correct (0 or 1)
        - metadata: Segment info
    """
    instance_results = compute_instance_accuracy(
        gt_labels,
        pred_labels,
        track_map,
        match_by=match_by,
        distance_threshold=distance_threshold,
    )

    # Group by frame
    by_frame = defaultdict(list)
    for result in instance_results:
        by_frame[result["frame_idx"]].append(result)

    # Determine if each frame is fully correct
    sorted_frames = sorted(by_frame.keys())
    per_frame = {}
    correct_frames = []

    for frame_idx in sorted_frames:
        results = by_frame[frame_idx]
        all_correct = all(r["correct"] for r in results)
        per_frame[frame_idx] = 1 if all_correct else 0
        if all_correct:
            correct_frames.append(frame_idx)

    # Find contiguous segments
    segments = []
    if correct_frames:
        current_segment = [correct_frames[0]]
        for frame in correct_frames[1:]:
            if frame == current_segment[-1] + 1:
                current_segment.append(frame)
            else:
                segments.append(current_segment)
                current_segment = [frame]
        segments.append(current_segment)

    segment_lengths = [len(s) for s in segments]
    mean_length = np.mean(segment_lengths) if segment_lengths else 0.0

    return MetricResult(
        name="correct_segments",
        value=float(mean_length),
        per_frame=per_frame,
        metadata={
            "num_segments": len(segments),
            "segment_lengths": segment_lengths,
            "total_correct_frames": len(correct_frames),
            "total_frames": len(sorted_frames),
            "max_segment_length": max(segment_lengths) if segment_lengths else 0,
        },
    )
