"""Layer attribution analysis for tracking errors."""

from typing import Dict, List, Optional, Set, Union
from pathlib import Path
import sleap_io as sio
import numpy as np
from collections import defaultdict

from sleap_mot.metrics.types import TrackHistoryEntry, MetricResult
from sleap_mot.metrics.per_instance import compute_instance_accuracy


def extract_track_histories(
    labels: sio.Labels,
) -> Dict[str, List[TrackHistoryEntry]]:
    """Extract all track histories from labels.

    Args:
        labels: Labels object with TrackContext instances.

    Returns:
        Dict mapping track names to their history entries.
    """
    from sleap_mot.tracking.base import TrackContext

    histories = defaultdict(list)

    for lf in labels:
        for inst in lf.instances:
            if inst.track is None:
                continue

            # Handle TrackContext
            if isinstance(inst.track, TrackContext):
                track_name = inst.track.name
                for entry_dict in inst.track.track_history:
                    entry = TrackHistoryEntry.from_dict(entry_dict)
                    histories[track_name].append(entry)
            # Handle regular Track with possible track_history attribute
            elif hasattr(inst.track, "track_history"):
                track_name = inst.track.name
                for entry_dict in inst.track.track_history:
                    entry = TrackHistoryEntry.from_dict(entry_dict)
                    histories[track_name].append(entry)

    # Deduplicate and sort by frame_idx
    for track_name in histories:
        seen = set()
        unique = []
        for entry in sorted(histories[track_name], key=lambda e: e.frame_idx):
            # Use tuple of key fields for deduplication
            key = (entry.frame_idx, entry.new_track_name, entry.layer_name)
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        histories[track_name] = unique

    return dict(histories)


def load_track_histories_from_json(
    json_path: Union[str, Path],
) -> Dict[str, List[TrackHistoryEntry]]:
    """Load track histories from a JSON sidecar file.

    This function loads track history data that was saved using
    save_track_history() from sleap_mot.tracking.base. The histories
    are converted to TrackHistoryEntry objects for use with metrics.

    Args:
        json_path: Path to the track history JSON file. Can be:
            - Direct path to *_track_history.json
            - Path to .slp file (will look for matching *_track_history.json)

    Returns:
        Dict mapping track names to their TrackHistoryEntry lists.

    Raises:
        FileNotFoundError: If the JSON file doesn't exist.

    Example:
        >>> histories = load_track_histories_from_json("tracked.slp")
        >>> for track_name, entries in histories.items():
        ...     print(f"{track_name}: {len(entries)} entries")
    """
    from sleap_mot.tracking.base import load_track_history

    # Load raw dict histories from JSON
    raw_histories = load_track_history(json_path)

    # Convert to TrackHistoryEntry objects
    histories = {}
    for track_name, entries in raw_histories.items():
        histories[track_name] = [
            TrackHistoryEntry.from_dict(entry) for entry in entries
        ]

    return histories


def compute_layer_attribution(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    track_map: Dict[str, str],
    match_by: str = "spatial",
    distance_threshold: float = 50.0,
    track_histories: Optional[Dict[str, List[TrackHistoryEntry]]] = None,
) -> MetricResult:
    """Attribute identity switches to specific tracking layers.

    Analyzes track_history to determine which layer caused each switch
    and provides statistics on layer-specific error rates.

    Args:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels with TrackContext containing history.
        track_map: GT to pred track name mapping.
        match_by: Method for matching instances.
        distance_threshold: Maximum distance for spatial matching.
        track_histories: Pre-loaded track histories (from JSON file).
            If provided, these are used instead of extracting from pred_labels.
            Use load_track_histories_from_json() to load from a JSON file.

    Returns:
        MetricResult with:
        - value: Overall error rate across layers
        - per_track: Dict mapping layer names to statistics
        - metadata: Layer-specific error details
    """
    # Use provided histories or extract from predicted labels
    if track_histories is not None:
        histories = track_histories
    else:
        histories = extract_track_histories(pred_labels)

    # Get instance accuracy to identify which predictions are correct/incorrect
    instance_results = compute_instance_accuracy(
        gt_labels,
        pred_labels,
        track_map,
        match_by=match_by,
        distance_threshold=distance_threshold,
    )

    # Build lookup: (frame_idx, pred_track) -> is_correct
    correctness_lookup = {}
    for result in instance_results:
        key = (result["frame_idx"], result["pred_track"])
        correctness_lookup[key] = result["correct"]

    # Invert track_map for expected pred track lookup
    expected_pred_tracks = set(track_map.values())

    # Categorize switches by layer
    incorrect_switches_by_layer = defaultdict(list)
    correct_switches_by_layer = defaultdict(list)
    all_switches_by_layer = defaultdict(list)

    for pred_track_name, entries in histories.items():
        for entry in entries:
            # Determine if this assignment resulted in correct identity
            key = (entry.frame_idx, entry.new_track_name)
            is_correct = correctness_lookup.get(key, None)

            # If we can't determine correctness from instance results,
            # check if new_track_name is in expected pred tracks
            if is_correct is None:
                is_correct = entry.new_track_name in expected_pred_tracks

            switch_info = {
                "frame_idx": entry.frame_idx,
                "pred_track": pred_track_name,
                "old_track": entry.old_track_name,
                "new_track": entry.new_track_name,
                "reason": entry.reason,
                "conflict_resolved": entry.conflict_resolved,
                "propagated": entry.propagated_from_frame is not None,
                "propagated_from_frame": entry.propagated_from_frame,
            }

            all_switches_by_layer[entry.layer_name].append(switch_info)

            if is_correct:
                correct_switches_by_layer[entry.layer_name].append(switch_info)
            else:
                incorrect_switches_by_layer[entry.layer_name].append(switch_info)

    # Compute per-layer statistics
    per_layer_stats = {}
    all_layers = (
        set(incorrect_switches_by_layer.keys())
        | set(correct_switches_by_layer.keys())
        | set(all_switches_by_layer.keys())
    )

    for layer_name in all_layers:
        incorrect = len(incorrect_switches_by_layer.get(layer_name, []))
        correct = len(correct_switches_by_layer.get(layer_name, []))
        total = incorrect + correct

        per_layer_stats[layer_name] = {
            "total_switches": total,
            "incorrect_switches": incorrect,
            "correct_switches": correct,
            "error_rate": incorrect / max(total, 1),
            "incorrect_details": incorrect_switches_by_layer.get(layer_name, []),
            "correct_details": correct_switches_by_layer.get(layer_name, []),
        }

    # Overall statistics
    total_incorrect = sum(s["incorrect_switches"] for s in per_layer_stats.values())
    total_switches = sum(s["total_switches"] for s in per_layer_stats.values())
    overall_error_rate = total_incorrect / max(total_switches, 1)

    return MetricResult(
        name="layer_attribution",
        value=float(overall_error_rate),
        per_track=per_layer_stats,
        metadata={
            "total_switches": total_switches,
            "total_incorrect": total_incorrect,
            "total_correct": total_switches - total_incorrect,
            "layers_analyzed": list(all_layers),
            "description": "Error rate attributed to each tracking layer",
        },
    )


def get_layer_switch_timeline(
    pred_labels: Optional[sio.Labels] = None,
    layer_name: Optional[str] = None,
    track_histories: Optional[Dict[str, List[TrackHistoryEntry]]] = None,
) -> List[Dict]:
    """Get chronological list of all switches, optionally filtered by layer.

    Useful for debugging a specific tracking layer.

    Args:
        pred_labels: Predicted Labels with TrackContext (optional if histories provided).
        layer_name: If provided, filter to only this layer's switches.
        track_histories: Pre-loaded track histories (from JSON file).
            If provided, these are used instead of extracting from pred_labels.

    Returns:
        List of switch events sorted by frame_idx.
    """
    if track_histories is not None:
        histories = track_histories
    elif pred_labels is not None:
        histories = extract_track_histories(pred_labels)
    else:
        raise ValueError("Either pred_labels or track_histories must be provided")
    timeline = []

    for track_name, entries in histories.items():
        for entry in entries:
            if layer_name is not None and entry.layer_name != layer_name:
                continue

            timeline.append(
                {
                    "frame_idx": entry.frame_idx,
                    "track": track_name,
                    "layer_name": entry.layer_name,
                    "old": entry.old_track_name,
                    "new": entry.new_track_name,
                    "reason": entry.reason,
                    "conflict_resolved": entry.conflict_resolved,
                    "propagated_from_frame": entry.propagated_from_frame,
                }
            )

    return sorted(timeline, key=lambda x: x["frame_idx"])


def get_layer_error_summary(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    track_map: Dict[str, str],
    match_by: str = "spatial",
    distance_threshold: float = 50.0,
    track_histories: Optional[Dict[str, List[TrackHistoryEntry]]] = None,
) -> Dict[str, Dict]:
    """Get a summary of errors by tracking layer.

    Provides a high-level view of which layers are contributing most to errors.

    Args:
        gt_labels: Ground truth Labels.
        pred_labels: Predicted Labels with TrackContext.
        track_map: GT to pred track name mapping.
        match_by: Method for matching instances.
        distance_threshold: Maximum distance for spatial matching.
        track_histories: Pre-loaded track histories (from JSON file).

    Returns:
        Dict mapping layer names to summary statistics.
    """
    result = compute_layer_attribution(
        gt_labels,
        pred_labels,
        track_map,
        match_by=match_by,
        distance_threshold=distance_threshold,
        track_histories=track_histories,
    )

    summary = {}
    for layer_name, stats in result.per_track.items():
        summary[layer_name] = {
            "total_switches": stats["total_switches"],
            "error_rate": stats["error_rate"],
            "incorrect_switches": stats["incorrect_switches"],
            "correct_switches": stats["correct_switches"],
        }

    return summary


def analyze_switch_reasons(
    pred_labels: Optional[sio.Labels] = None,
    track_histories: Optional[Dict[str, List[TrackHistoryEntry]]] = None,
) -> Dict[str, int]:
    """Analyze the reasons for identity switches.

    Groups switches by their reason strings to understand patterns.

    Args:
        pred_labels: Predicted Labels with TrackContext (optional if histories provided).
        track_histories: Pre-loaded track histories (from JSON file).

    Returns:
        Dict mapping reason strings to counts.
    """
    if track_histories is not None:
        histories = track_histories
    elif pred_labels is not None:
        histories = extract_track_histories(pred_labels)
    else:
        raise ValueError("Either pred_labels or track_histories must be provided")
    reason_counts = defaultdict(int)

    for track_name, entries in histories.items():
        for entry in entries:
            reason_counts[entry.reason] += 1

    return dict(reason_counts)


def get_propagation_analysis(
    pred_labels: Optional[sio.Labels] = None,
    track_histories: Optional[Dict[str, List[TrackHistoryEntry]]] = None,
) -> Dict[str, any]:
    """Analyze propagation patterns in identity changes.

    Args:
        pred_labels: Predicted Labels with TrackContext (optional if histories provided).
        track_histories: Pre-loaded track histories (from JSON file).

    Returns:
        Dict with propagation statistics.
    """
    if track_histories is not None:
        histories = track_histories
    elif pred_labels is not None:
        histories = extract_track_histories(pred_labels)
    else:
        raise ValueError("Either pred_labels or track_histories must be provided")

    total_switches = 0
    propagated_switches = 0
    propagation_distances = []

    for track_name, entries in histories.items():
        for entry in entries:
            total_switches += 1
            if entry.propagated_from_frame is not None:
                propagated_switches += 1
                distance = abs(entry.frame_idx - entry.propagated_from_frame)
                propagation_distances.append(distance)

    return {
        "total_switches": total_switches,
        "propagated_switches": propagated_switches,
        "non_propagated_switches": total_switches - propagated_switches,
        "propagation_rate": propagated_switches / max(total_switches, 1),
        "mean_propagation_distance": (
            np.mean(propagation_distances) if propagation_distances else 0
        ),
        "max_propagation_distance": (
            max(propagation_distances) if propagation_distances else 0
        ),
    }
