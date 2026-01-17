"""Metrics package for evaluating multi-object tracking performance.

This package provides tools for comparing ground truth labels against
predicted tracking results, with support for:
- Per-instance, per-track, and per-frame metrics
- Multiple output formats (DataFrame, JSON, plots)
- Layer attribution analysis for debugging tracking pipelines
- Export for ID Switch Viewer web application

Example - Basic metrics:
    >>> from sleap_mot.metrics import TrackingEvaluator, MetricsExporter
    >>> import sleap_io as sio
    >>>
    >>> gt = sio.load_slp("proofread.slp")
    >>> pred = sio.load_slp("tracked.slp")
    >>> track_map = {"track_1": "A5859", "track_2": "B1234"}
    >>>
    >>> evaluator = TrackingEvaluator(gt, pred, track_map)
    >>> results = evaluator.compute_all()
    >>>
    >>> exporter = MetricsExporter(results)
    >>> exporter.to_csv("metrics_summary.csv")
    >>> exporter.save_plots("./plots/")

Example - Export for ID Switch Viewer:
    >>> from sleap_mot.metrics import TrackingEvaluator
    >>> import sleap_io as sio
    >>>
    >>> gt = sio.load_slp("proofread.slp")
    >>> pred = sio.load_slp("tracked.slp")
    >>> track_map = {"track_1": "A5859", "track_2": "B1234"}
    >>>
    >>> # Load with track history for layer attribution
    >>> evaluator = TrackingEvaluator(
    ...     gt, pred, track_map,
    ...     track_history_path="tracked.slp"  # Loads tracked_track_history.json
    ... )
    >>>
    >>> # Export for viewer
    >>> evaluator.export_for_viewer("switch_report.json")
"""

from sleap_mot.metrics.types import TrackHistoryEntry, MetricResult
from sleap_mot.metrics.core import TrackingEvaluator
from sleap_mot.metrics.output import MetricsExporter
from sleap_mot.metrics.alignment import (
    apply_track_map,
    build_frame_instance_map,
    infer_track_map_from_overlap,
    get_common_frames,
)
from sleap_mot.metrics.per_instance import (
    compute_instance_accuracy,
    compute_instance_switch_events,
    get_instance_timeline,
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
    extract_track_histories,
    load_track_histories_from_json,
    compute_layer_attribution,
    get_layer_switch_timeline,
    get_layer_error_summary,
    analyze_switch_reasons,
    get_propagation_analysis,
)

__all__ = [
    # Core classes
    "TrackingEvaluator",
    "MetricsExporter",
    "TrackHistoryEntry",
    "MetricResult",
    # Alignment
    "apply_track_map",
    "build_frame_instance_map",
    "infer_track_map_from_overlap",
    "get_common_frames",
    # Per-instance
    "compute_instance_accuracy",
    "compute_instance_switch_events",
    "get_instance_timeline",
    # Per-track
    "compute_fragmentation",
    "compute_track_purity",
    "compute_track_completeness",
    "compute_id_switches_per_track",
    # Per-frame
    "compute_frame_accuracy",
    "compute_switch_latency",
    "compute_mislabeled_segments",
    "compute_correct_segments",
    # Layer attribution
    "extract_track_histories",
    "load_track_histories_from_json",
    "compute_layer_attribution",
    "get_layer_switch_timeline",
    "get_layer_error_summary",
    "analyze_switch_reasons",
    "get_propagation_analysis",
]
