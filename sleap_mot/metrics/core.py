"""Core tracking evaluation class."""

from typing import Dict, List, Optional, Union
from pathlib import Path
import sleap_io as sio

from sleap_mot.metrics.types import MetricResult, TrackHistoryEntry
from sleap_mot.metrics.per_instance import (
    compute_instance_accuracy,
    compute_instance_switch_events,
)
from sleap_mot.metrics.per_track import (
    compute_fragmentation,
    compute_track_purity,
    compute_track_completeness,
    compute_id_switches_per_track,
)
from sleap_mot.metrics.per_frame import (
    compute_frame_accuracy,
    compute_switch_latency,
    compute_mislabeled_segments,
    compute_correct_segments,
)
from sleap_mot.metrics.layer_attribution import (
    compute_layer_attribution,
    extract_track_histories,
    load_track_histories_from_json,
)


class TrackingEvaluator:
    """Main class for evaluating tracking performance.

    Compares ground truth labels against predictions and computes
    various metrics at instance, track, and frame granularity.

    Attributes:
        gt_labels: Ground truth Labels object.
        pred_labels: Predicted Labels object.
        track_map: Mapping of GT track names to predicted track names.
        match_by: Method for matching instances ("spatial" or "track").
        distance_threshold: Maximum distance for spatial matching.
        track_histories: Optional pre-loaded track histories for layer attribution.

    Example:
        >>> gt = sio.load_slp("proofread.slp")
        >>> pred = sio.load_slp("tracked.slp")
        >>> track_map = {"track_1": "A5859", "track_2": "B1234"}
        >>> evaluator = TrackingEvaluator(gt, pred, track_map)
        >>> results = evaluator.compute_all()
        >>> print(results["accuracy"].value)

    Example with track history from JSON:
        >>> gt = sio.load_slp("proofread.slp")
        >>> pred = sio.load_slp("tracked.slp")
        >>> evaluator = TrackingEvaluator(
        ...     gt, pred, track_map,
        ...     track_history_path="tracked.slp"  # Loads tracked_track_history.json
        ... )
        >>> results = evaluator.compute_all()
    """

    def __init__(
        self,
        gt_labels: sio.Labels,
        pred_labels: sio.Labels,
        track_map: Dict[str, str],
        match_by: str = "spatial",
        distance_threshold: float = 50.0,
        track_histories: Optional[Dict[str, List[TrackHistoryEntry]]] = None,
        track_history_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize evaluator.

        Args:
            gt_labels: Ground truth Labels.
            pred_labels: Predicted Labels.
            track_map: GT track name to predicted track name mapping.
            match_by: Method for matching instances - "spatial" (centroid
                distance) or "track" (by mapped track name).
            distance_threshold: Maximum distance for spatial matching (pixels).
            track_histories: Pre-loaded track histories dict. If provided, used
                for layer attribution analysis instead of extracting from labels.
            track_history_path: Path to track history JSON file (or .slp file
                with matching *_track_history.json). If provided, loads histories
                from this file for layer attribution analysis.
        """
        self.gt_labels = gt_labels
        self.pred_labels = pred_labels
        self.track_map = track_map
        self.match_by = match_by
        self.distance_threshold = distance_threshold

        # Load track histories from JSON if path provided
        if track_history_path is not None:
            self.track_histories = load_track_histories_from_json(track_history_path)
        else:
            self.track_histories = track_histories

    def compute_all(self) -> Dict[str, MetricResult]:
        """Compute all available metrics.

        Returns:
            Dict mapping metric names to MetricResult objects.
        """
        return {
            "accuracy": self.compute_accuracy(),
            "frame_accuracy": self.compute_frame_accuracy(),
            "fragmentation": self.compute_fragmentation(),
            "track_purity": self.compute_track_purity(),
            "track_completeness": self.compute_track_completeness(),
            "id_switches": self.compute_id_switches(),
            "switch_latency": self.compute_switch_latency(),
            "mislabeled_segments": self.compute_mislabeled_segments(),
            "correct_segments": self.compute_correct_segments(),
            "layer_attribution": self.compute_layer_attribution(),
        }

    def compute_accuracy(self) -> MetricResult:
        """Compute overall accuracy.

        Returns:
            MetricResult with overall accuracy value.
        """
        instance_results = compute_instance_accuracy(
            self.gt_labels,
            self.pred_labels,
            self.track_map,
            match_by=self.match_by,
            distance_threshold=self.distance_threshold,
        )

        total = len(instance_results)
        correct = sum(1 for r in instance_results if r["correct"])
        accuracy = correct / total if total > 0 else 0.0

        return MetricResult(
            name="accuracy",
            value=float(accuracy),
            per_instance=instance_results,
            metadata={
                "total_instances": total,
                "correct_instances": correct,
                "incorrect_instances": total - correct,
            },
        )

    def compute_frame_accuracy(self) -> MetricResult:
        """Compute per-frame accuracy.

        Returns:
            MetricResult with per-frame accuracy breakdown.
        """
        return compute_frame_accuracy(
            self.gt_labels,
            self.pred_labels,
            self.track_map,
            match_by=self.match_by,
            distance_threshold=self.distance_threshold,
        )

    def compute_fragmentation(self) -> MetricResult:
        """Compute track fragmentation.

        Returns:
            MetricResult with fragmentation per GT track.
        """
        return compute_fragmentation(
            self.gt_labels,
            self.pred_labels,
            self.track_map,
            match_by=self.match_by,
            distance_threshold=self.distance_threshold,
        )

    def compute_track_purity(self) -> MetricResult:
        """Compute track purity.

        Returns:
            MetricResult with purity per predicted track.
        """
        return compute_track_purity(
            self.gt_labels,
            self.pred_labels,
            self.track_map,
            match_by=self.match_by,
            distance_threshold=self.distance_threshold,
        )

    def compute_track_completeness(self) -> MetricResult:
        """Compute track completeness.

        Returns:
            MetricResult with completeness per GT track.
        """
        return compute_track_completeness(
            self.gt_labels,
            self.pred_labels,
            self.track_map,
            match_by=self.match_by,
            distance_threshold=self.distance_threshold,
        )

    def compute_id_switches(self) -> MetricResult:
        """Compute ID switches per track.

        Returns:
            MetricResult with switch counts per GT track.
        """
        return compute_id_switches_per_track(
            self.gt_labels,
            self.pred_labels,
            self.track_map,
            match_by=self.match_by,
            distance_threshold=self.distance_threshold,
        )

    def compute_switch_latency(self) -> MetricResult:
        """Compute switch correction latency.

        Returns:
            MetricResult with latency statistics.
        """
        return compute_switch_latency(
            self.gt_labels,
            self.pred_labels,
            self.track_map,
            match_by=self.match_by,
            distance_threshold=self.distance_threshold,
        )

    def compute_mislabeled_segments(self) -> MetricResult:
        """Compute mislabeled segment statistics.

        Returns:
            MetricResult with segment information.
        """
        return compute_mislabeled_segments(
            self.gt_labels,
            self.pred_labels,
            self.track_map,
            match_by=self.match_by,
            distance_threshold=self.distance_threshold,
        )

    def compute_correct_segments(self) -> MetricResult:
        """Compute correct segment statistics.

        Returns:
            MetricResult with segment information.
        """
        return compute_correct_segments(
            self.gt_labels,
            self.pred_labels,
            self.track_map,
            match_by=self.match_by,
            distance_threshold=self.distance_threshold,
        )

    def compute_layer_attribution(self) -> MetricResult:
        """Compute layer attribution for errors.

        Note: This requires either:
        - pred_labels containing TrackContext objects with track_history populated
        - track_histories provided via constructor (from JSON file)

        If no history is available, returns empty results.

        Returns:
            MetricResult with layer-specific error statistics.
        """
        # Check if histories exist (either from JSON or labels)
        if self.track_histories is not None:
            histories = self.track_histories
        else:
            histories = extract_track_histories(self.pred_labels)

        if not histories:
            return MetricResult(
                name="layer_attribution",
                value=0.0,
                per_track={},
                metadata={
                    "warning": "No track history found",
                    "description": "Layer attribution requires track history from JSON file or TrackContext",
                },
            )

        return compute_layer_attribution(
            self.gt_labels,
            self.pred_labels,
            self.track_map,
            match_by=self.match_by,
            distance_threshold=self.distance_threshold,
            track_histories=histories,
        )

    def get_switch_events(self):
        """Get all identity switch events.

        Returns:
            List of switch event dictionaries.
        """
        return compute_instance_switch_events(
            self.gt_labels,
            self.pred_labels,
            self.track_map,
            match_by=self.match_by,
            distance_threshold=self.distance_threshold,
        )

    def summary(self) -> Dict[str, float]:
        """Get a simple summary of key metrics.

        Returns:
            Dict mapping metric names to scalar values.
        """
        results = self.compute_all()
        return {name: result.value for name, result in results.items()}

    def export_for_viewer(
        self,
        output_path: Union[str, Path],
        indent: int = 2,
    ) -> None:
        """Export metrics in ID Switch Viewer format.

        Creates a JSON file compatible with the ID Switch Viewer web application
        for investigating identity switches in multi-animal tracking.

        Args:
            output_path: Path for the output JSON file.
            indent: JSON indentation level.

        Example:
            >>> evaluator = TrackingEvaluator(
            ...     gt, pred, track_map,
            ...     track_history_path="tracked.slp"
            ... )
            >>> evaluator.export_for_viewer("switch_report.json")
        """
        from sleap_mot.metrics.output import MetricsExporter

        # Compute all metrics
        results = self.compute_all()

        # Get switch events
        switch_events = self.get_switch_events()

        # Get track histories
        track_histories = self.track_histories
        if track_histories is None:
            track_histories = extract_track_histories(self.pred_labels)

        # Export
        exporter = MetricsExporter(results)
        exporter.to_viewer_json(
            path=output_path,
            switch_events=switch_events,
            track_map=self.track_map,
            track_histories=track_histories,
            indent=indent,
        )
