"""Output and export utilities for metrics."""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sleap_mot.metrics.types import MetricResult, TrackHistoryEntry


class MetricsExporter:
    """Export metrics to various formats.

    Supports exporting to:
    - DataFrame/CSV files
    - JSON files
    - Matplotlib plots

    Example:
        >>> results = evaluator.compute_all()
        >>> exporter = MetricsExporter(results)
        >>> exporter.to_csv("metrics.csv")
        >>> exporter.to_json("metrics.json")
        >>> exporter.save_plots("./plots/")
    """

    def __init__(self, results: Dict[str, MetricResult]):
        """Initialize with computed metrics.

        Args:
            results: Dict mapping metric names to MetricResult objects.
        """
        self.results = results

    # --- DataFrame/CSV Export ---

    def to_summary_dataframe(self) -> pd.DataFrame:
        """Create summary DataFrame with one row per metric.

        Returns:
            DataFrame with columns: metric, value, and metadata fields.
        """
        rows = []
        for name, result in self.results.items():
            row = {
                "metric": name,
                "value": result.value,
            }
            # Add select metadata fields
            if result.metadata:
                for key in ["total_instances", "num_frames", "total_gt_tracks"]:
                    if key in result.metadata:
                        row[key] = result.metadata[key]
            rows.append(row)
        return pd.DataFrame(rows)

    def to_per_frame_dataframe(self) -> pd.DataFrame:
        """Create DataFrame with per-frame breakdowns.

        Returns:
            DataFrame with columns: frame_idx, metric, value.
        """
        frames_data = []
        for name, result in self.results.items():
            if result.per_frame:
                for frame_idx, value in result.per_frame.items():
                    frames_data.append(
                        {
                            "frame_idx": frame_idx,
                            "metric": name,
                            "value": value,
                        }
                    )
        return pd.DataFrame(frames_data)

    def to_per_track_dataframe(self) -> pd.DataFrame:
        """Create DataFrame with per-track breakdowns.

        Returns:
            DataFrame with columns: track, metric, and value/stats.
        """
        tracks_data = []
        for name, result in self.results.items():
            if result.per_track:
                for track_name, value in result.per_track.items():
                    if isinstance(value, dict):
                        # Flatten nested dict, exclude detail lists
                        row = {"track": track_name, "metric": name}
                        for k, v in value.items():
                            if not isinstance(v, list):
                                row[k] = v
                        tracks_data.append(row)
                    else:
                        tracks_data.append(
                            {
                                "track": track_name,
                                "metric": name,
                                "value": value,
                            }
                        )
        return pd.DataFrame(tracks_data)

    def to_per_instance_dataframe(self) -> pd.DataFrame:
        """Create DataFrame with per-instance breakdowns.

        Returns:
            DataFrame with instance-level data.
        """
        all_instances = []
        for name, result in self.results.items():
            if result.per_instance:
                for inst in result.per_instance:
                    row = {"metric": name}
                    row.update(inst)
                    all_instances.append(row)
        return pd.DataFrame(all_instances)

    def to_csv(
        self,
        path: Union[str, Path],
        granularity: str = "summary",
    ) -> None:
        """Export to CSV file.

        Args:
            path: Output file path.
            granularity: One of "summary", "per_frame", "per_track", "per_instance".
        """
        if granularity == "summary":
            df = self.to_summary_dataframe()
        elif granularity == "per_frame":
            df = self.to_per_frame_dataframe()
        elif granularity == "per_track":
            df = self.to_per_track_dataframe()
        elif granularity == "per_instance":
            df = self.to_per_instance_dataframe()
        else:
            raise ValueError(f"Unknown granularity: {granularity}")

        df.to_csv(path, index=False)

    # --- JSON/Dict Export ---

    def to_dict(self) -> dict:
        """Convert all results to nested dict.

        Returns:
            Dict representation of all metrics.
        """
        return {name: result.to_dict() for name, result in self.results.items()}

    def to_json(self, path: Union[str, Path], indent: int = 2) -> None:
        """Export to JSON file.

        Args:
            path: Output file path.
            indent: JSON indentation level.
        """
        data = self.to_dict()

        # Custom serializer for numpy types
        def json_serializer(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            return str(obj)

        with open(path, "w") as f:
            json.dump(data, f, indent=indent, default=json_serializer)

    # --- Viewer-compatible JSON Export ---

    def to_viewer_json(
        self,
        path: Union[str, Path],
        switch_events: List[Dict],
        track_map: Dict[str, str],
        track_histories: Optional[Dict[str, List[TrackHistoryEntry]]] = None,
        indent: int = 2,
    ) -> None:
        """Export metrics in ID Switch Viewer format.

        Creates a JSON file compatible with the ID Switch Viewer web application
        for investigating identity switches in multi-animal tracking.

        Args:
            path: Output file path.
            switch_events: List of switch events from get_switch_events().
            track_map: GT track name to predicted track name mapping.
            track_histories: Track histories dict (from TrackHistoryEntry objects).
            indent: JSON indentation level.

        Example:
            >>> evaluator = TrackingEvaluator(gt, pred, track_map, track_history_path="tracked.slp")
            >>> results = evaluator.compute_all()
            >>> switch_events = evaluator.get_switch_events()
            >>> exporter = MetricsExporter(results)
            >>> exporter.to_viewer_json(
            ...     "switch_report.json",
            ...     switch_events=switch_events,
            ...     track_map=track_map,
            ...     track_histories=evaluator.track_histories
            ... )
        """
        data = self._build_viewer_data(
            switch_events=switch_events,
            track_map=track_map,
            track_histories=track_histories,
        )

        # Custom serializer for numpy types
        def json_serializer(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            return str(obj)

        with open(path, "w") as f:
            json.dump(data, f, indent=indent, default=json_serializer)

    def _build_viewer_data(
        self,
        switch_events: List[Dict],
        track_map: Dict[str, str],
        track_histories: Optional[Dict[str, List[TrackHistoryEntry]]] = None,
    ) -> Dict[str, Any]:
        """Build the viewer-compatible data structure.

        Args:
            switch_events: List of switch events.
            track_map: GT to pred track name mapping.
            track_histories: Track histories dict.

        Returns:
            Dict formatted for the ID Switch Viewer.
        """
        # Build frame-to-history lookup for layer attribution
        history_lookup = {}  # (frame_idx, new_track_name) -> TrackHistoryEntry
        if track_histories:
            for track_name, entries in track_histories.items():
                for entry in entries:
                    key = (entry.frame_idx, entry.new_track_name)
                    history_lookup[key] = entry

        # Enhance switch events with layer attribution
        enhanced_events = []
        for event in switch_events:
            enhanced = {
                "frame_idx": event.get("frame_idx"),
                "gt_track": event.get("gt_track"),
                "pred_track": event.get("new_pred_track"),
                "expected_pred": event.get("expected_pred_track"),
                "correct": event.get("switch_type") in (
                    "incorrect_to_correct",
                    "appeared_correct",
                ),
                "switch_type": event.get("switch_type"),
            }

            # Try to find layer attribution from track history
            frame_idx = event.get("frame_idx")
            new_pred_track = event.get("new_pred_track")

            if frame_idx is not None and new_pred_track is not None:
                key = (frame_idx, new_pred_track)
                if key in history_lookup:
                    entry = history_lookup[key]
                    enhanced["layer_attribution"] = {
                        "layer_name": entry.layer_name,
                        "old_track_name": entry.old_track_name,
                        "new_track_name": entry.new_track_name,
                        "reason": entry.reason,
                        "conflict_resolved": entry.conflict_resolved,
                        "propagated_from_frame": entry.propagated_from_frame,
                    }

            enhanced_events.append(enhanced)

        # Build track histories in serializable format
        serialized_histories = {}
        if track_histories:
            for track_name, entries in track_histories.items():
                serialized_histories[track_name] = [
                    entry.to_dict() if hasattr(entry, "to_dict") else {
                        "layer_name": entry.layer_name,
                        "old_track_name": entry.old_track_name,
                        "new_track_name": entry.new_track_name,
                        "frame_idx": entry.frame_idx,
                        "reason": entry.reason,
                        "conflict_resolved": entry.conflict_resolved,
                        "propagated_from_frame": entry.propagated_from_frame,
                    }
                    for entry in entries
                ]

        # Build summary from results
        accuracy_result = self.results.get("accuracy")
        summary = {}
        if accuracy_result:
            summary = {
                "total_instances": accuracy_result.metadata.get("total_instances", 0),
                "correct_instances": accuracy_result.metadata.get("correct_instances", 0),
                "incorrect_instances": accuracy_result.metadata.get("incorrect_instances", 0),
                "accuracy": accuracy_result.value,
            }

        # Add switch counts
        total_switches = len(enhanced_events)
        correct_switches = sum(1 for e in enhanced_events if e.get("correct"))
        incorrect_switches = total_switches - correct_switches
        summary["total_switches"] = total_switches
        summary["correct_switches"] = correct_switches
        summary["incorrect_switches"] = incorrect_switches

        # Build layer attribution summary
        layer_attribution = {}
        layer_result = self.results.get("layer_attribution")
        if layer_result and layer_result.per_track:
            for layer_name, stats in layer_result.per_track.items():
                layer_attribution[layer_name] = {
                    "total": stats.get("total_switches", 0),
                    "incorrect": stats.get("incorrect_switches", 0),
                    "error_rate": stats.get("error_rate", 0),
                }

        return {
            "switch_events": enhanced_events,
            "track_histories": serialized_histories,
            "track_map": track_map,
            "summary": summary,
            "layer_attribution": layer_attribution,
        }

    # --- Matplotlib Plots ---

    def plot_accuracy_over_time(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (12, 4),
        rolling_window: Optional[int] = None,
    ) -> plt.Figure:
        """Plot frame-level accuracy over time.

        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure.
            figsize: Figure size if creating new figure.
            rolling_window: If provided, also plot rolling average.

        Returns:
            Matplotlib Figure object.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        accuracy_result = self.results.get("frame_accuracy") or self.results.get(
            "accuracy"
        )
        if accuracy_result and accuracy_result.per_frame:
            frames = sorted(accuracy_result.per_frame.keys())
            values = [accuracy_result.per_frame[f] for f in frames]

            ax.plot(frames, values, linewidth=0.5, alpha=0.7, label="Per-frame")

            if rolling_window and len(values) >= rolling_window:
                rolling = pd.Series(values).rolling(window=rolling_window).mean()
                ax.plot(
                    frames,
                    rolling,
                    linewidth=2,
                    color="orange",
                    label=f"Rolling avg ({rolling_window})",
                )

            ax.axhline(
                y=accuracy_result.value,
                color="r",
                linestyle="--",
                label=f"Mean: {accuracy_result.value:.3f}",
            )
            ax.set_xlabel("Frame")
            ax.set_ylabel("Accuracy")
            ax.set_title("Tracking Accuracy Over Time")
            ax.legend()
            ax.set_ylim(0, 1.05)

        return fig

    def plot_layer_attribution(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (10, 6),
    ) -> plt.Figure:
        """Plot bar chart of errors by tracking layer.

        Args:
            ax: Matplotlib axes to plot on.
            figsize: Figure size if creating new figure.

        Returns:
            Matplotlib Figure object.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        layer_result = self.results.get("layer_attribution")
        if layer_result and layer_result.per_track:
            layers = list(layer_result.per_track.keys())
            if not layers:
                ax.text(
                    0.5,
                    0.5,
                    "No layer attribution data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                return fig

            incorrect = [
                layer_result.per_track[l].get("incorrect_switches", 0) for l in layers
            ]
            correct = [
                layer_result.per_track[l].get("correct_switches", 0) for l in layers
            ]

            x = range(len(layers))
            width = 0.35

            ax.bar(
                [i - width / 2 for i in x],
                correct,
                width,
                label="Correct",
                color="green",
                alpha=0.7,
            )
            ax.bar(
                [i + width / 2 for i in x],
                incorrect,
                width,
                label="Incorrect",
                color="red",
                alpha=0.7,
            )

            ax.set_xlabel("Tracking Layer")
            ax.set_ylabel("Number of Switches")
            ax.set_title("Identity Switches by Tracking Layer")
            ax.set_xticks(x)
            ax.set_xticklabels(layers, rotation=45, ha="right")
            ax.legend()

        return fig

    def plot_fragmentation_by_track(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (12, 5),
    ) -> plt.Figure:
        """Plot fragmentation per GT track.

        Args:
            ax: Matplotlib axes to plot on.
            figsize: Figure size if creating new figure.

        Returns:
            Matplotlib Figure object.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        frag_result = self.results.get("fragmentation")
        if frag_result and frag_result.per_track:
            tracks = list(frag_result.per_track.keys())
            values = [frag_result.per_track[t] for t in tracks]

            ax.bar(range(len(tracks)), values, color="steelblue", alpha=0.8)
            ax.axhline(
                y=frag_result.value,
                color="r",
                linestyle="--",
                label=f"Mean: {frag_result.value:.2f}",
            )
            ax.axhline(y=1, color="g", linestyle=":", alpha=0.5, label="Ideal: 1")
            ax.set_xlabel("GT Track")
            ax.set_ylabel("Number of Segments")
            ax.set_title("Track Fragmentation (Segments per GT Track)")
            ax.set_xticks(range(len(tracks)))
            ax.set_xticklabels(tracks, rotation=45, ha="right")
            ax.legend()

        return fig

    def plot_switch_latency_histogram(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (10, 5),
        bins: int = 20,
    ) -> plt.Figure:
        """Plot histogram of switch correction latencies.

        Args:
            ax: Matplotlib axes to plot on.
            figsize: Figure size if creating new figure.
            bins: Number of histogram bins.

        Returns:
            Matplotlib Figure object.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        latency_result = self.results.get("switch_latency")
        if latency_result and latency_result.per_instance:
            latencies = [
                l["latency_frames"]
                for l in latency_result.per_instance
                if l.get("corrected", False)
            ]

            if latencies:
                ax.hist(
                    latencies, bins=bins, color="steelblue", alpha=0.8, edgecolor="black"
                )
                ax.axvline(
                    x=latency_result.value,
                    color="r",
                    linestyle="--",
                    label=f"Mean: {latency_result.value:.1f} frames",
                )
                ax.set_xlabel("Latency (frames)")
                ax.set_ylabel("Count")
                ax.set_title("Switch Correction Latency Distribution")
                ax.legend()
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No corrected switches to display",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

        return fig

    def plot_track_purity(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (12, 5),
    ) -> plt.Figure:
        """Plot track purity per predicted track.

        Args:
            ax: Matplotlib axes to plot on.
            figsize: Figure size if creating new figure.

        Returns:
            Matplotlib Figure object.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        purity_result = self.results.get("track_purity")
        if purity_result and purity_result.per_track:
            tracks = list(purity_result.per_track.keys())
            values = [purity_result.per_track[t].get("purity", 0) for t in tracks]

            colors = ["green" if v == 1.0 else "orange" if v >= 0.5 else "red" for v in values]

            ax.bar(range(len(tracks)), values, color=colors, alpha=0.8)
            ax.axhline(
                y=purity_result.value,
                color="b",
                linestyle="--",
                label=f"Mean: {purity_result.value:.2f}",
            )
            ax.axhline(y=1, color="g", linestyle=":", alpha=0.5, label="Ideal: 1")
            ax.set_xlabel("Predicted Track")
            ax.set_ylabel("Purity (1 / num GT identities)")
            ax.set_title("Track Purity (1.0 = pure, only one GT identity)")
            ax.set_xticks(range(len(tracks)))
            ax.set_xticklabels(tracks, rotation=45, ha="right")
            ax.set_ylim(0, 1.1)
            ax.legend()

        return fig

    def plot_id_switches_timeline(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (14, 4),
    ) -> plt.Figure:
        """Plot ID switches as events on a timeline.

        Args:
            ax: Matplotlib axes to plot on.
            figsize: Figure size if creating new figure.

        Returns:
            Matplotlib Figure object.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Get switch latency data which has per-instance details
        latency_result = self.results.get("switch_latency")
        if latency_result and latency_result.per_instance:
            events = latency_result.per_instance

            corrected_frames = [
                e["frame_idx"] for e in events if e.get("corrected", False)
            ]
            uncorrected_frames = [
                e["frame_idx"] for e in events if not e.get("corrected", False)
            ]

            if corrected_frames:
                ax.scatter(
                    corrected_frames,
                    [1] * len(corrected_frames),
                    c="orange",
                    marker="|",
                    s=100,
                    label=f"Corrected ({len(corrected_frames)})",
                )
            if uncorrected_frames:
                ax.scatter(
                    uncorrected_frames,
                    [0.5] * len(uncorrected_frames),
                    c="red",
                    marker="|",
                    s=100,
                    label=f"Uncorrected ({len(uncorrected_frames)})",
                )

            ax.set_xlabel("Frame")
            ax.set_ylabel("")
            ax.set_yticks([0.5, 1])
            ax.set_yticklabels(["Uncorrected", "Corrected"])
            ax.set_title("ID Switch Events Timeline")
            ax.legend()

        return fig

    def plot_summary_dashboard(
        self,
        figsize: tuple = (16, 12),
    ) -> plt.Figure:
        """Create multi-panel summary dashboard.

        Args:
            figsize: Figure size.

        Returns:
            Matplotlib Figure object with 2x2 subplot grid.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        self.plot_accuracy_over_time(ax=axes[0, 0])
        self.plot_layer_attribution(ax=axes[0, 1])
        self.plot_fragmentation_by_track(ax=axes[1, 0])
        self.plot_track_purity(ax=axes[1, 1])

        fig.tight_layout()
        return fig

    def save_plots(
        self,
        output_dir: Union[str, Path],
        format: str = "png",
        dpi: int = 150,
    ) -> List[Path]:
        """Save all plots to directory.

        Args:
            output_dir: Output directory path.
            format: Image format (png, pdf, svg, etc.).
            dpi: Resolution for raster formats.

        Returns:
            List of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = []

        plot_methods = [
            ("accuracy_over_time", self.plot_accuracy_over_time),
            ("layer_attribution", self.plot_layer_attribution),
            ("fragmentation", self.plot_fragmentation_by_track),
            ("track_purity", self.plot_track_purity),
            ("switch_latency_histogram", self.plot_switch_latency_histogram),
            ("id_switches_timeline", self.plot_id_switches_timeline),
            ("dashboard", self.plot_summary_dashboard),
        ]

        for name, method in plot_methods:
            try:
                fig = method()
                path = output_dir / f"{name}.{format}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                saved.append(path)
            except Exception as e:
                print(f"Warning: Could not save {name} plot: {e}")

        return saved
