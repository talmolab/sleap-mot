"""SLPT File Format Implementation.

A .slpt file is a ZIP archive containing:
- labels.slp: Standard SLEAP labels file
- track_contexts.json: TrackContext metadata for all instances
- explanations.json: InstanceExplanationStore records
- pipeline.json: Tracking pipeline metadata

This allows tracking metadata (priorities, history, explanations) to be
preserved across save/load cycles, enabling multi-layer tracking pipelines
with proper conflict resolution.
"""

import json
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import sleap_io as sio

from sleap_mot.tracking.base import TrackContext
from sleap_mot.tracking.instance_explanations import InstanceExplanationStore


@dataclass
class TrackContextData:
    """Serializable representation of TrackContext.

    This dataclass mirrors TrackContext but is designed for JSON serialization.
    It captures all the metadata needed to reconstruct a TrackContext object.

    Attributes:
        frame_idx: Frame index where this context applies.
        instance_idx: Instance index within the frame.
        track_name: Name of the track (e.g., "tracklet_5" or "041B407547").
        priority: Priority level of the layer that made this assignment.
        temporary_track: Whether this is a temporary track (tracklet).
        valid: Whether this track assignment is valid.
        track_history: List of history entry dicts tracking all changes.
    """
    frame_idx: int
    instance_idx: int
    track_name: str
    priority: Optional[int]
    temporary_track: bool
    valid: bool
    track_history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "TrackContextData":
        """Create from dictionary."""
        return cls(**data)

    def to_track_context(self, base_track: sio.Track) -> TrackContext:
        """Convert to a TrackContext object.

        Args:
            base_track: The underlying sio.Track object.

        Returns:
            TrackContext with all metadata restored.
        """
        return TrackContext(
            priority=self.priority,
            track=base_track,
            name=self.track_name,
            temporary_track=self.temporary_track,
            valid=self.valid,
            track_history=self.track_history.copy(),
        )


@dataclass
class PipelineLayerRecord:
    """Record of a tracking layer application.

    Tracks which layers have been applied to the file and in what order,
    along with their configuration.

    Attributes:
        order: Order in which this layer was applied (0-indexed).
        name: Layer name (e.g., "FacingConsistency").
        class_name: Full class name for reconstruction.
        priority: Priority level of this layer.
        config: Configuration dict passed to the layer.
        applied_at: ISO timestamp when layer was applied.
    """
    order: int
    name: str
    class_name: str
    priority: int
    config: Dict
    applied_at: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "PipelineLayerRecord":
        """Create from dictionary."""
        return cls(**data)


class SLPTFile:
    """Reader/writer for .slpt tracking files.

    A .slpt file is a ZIP archive containing:
    - labels.slp: Standard SLEAP labels file
    - track_contexts.json: TrackContext metadata for all instances
    - explanations.json: InstanceExplanationStore records
    - pipeline.json: Tracking pipeline metadata

    This format preserves tracking metadata across save/load cycles,
    enabling proper conflict resolution in multi-layer pipelines.

    Example usage:
        # Create from existing Labels with TrackContext
        slpt = SLPTFile.from_labels(labels, explanation_store)
        slpt.save("output.slpt")

        # Load existing
        slpt = SLPTFile.load("output.slpt")
        labels = slpt.to_labels()  # Labels with TrackContext restored

        # Export to standard SLP (loses metadata)
        slpt.export_slp("output.slp")

    Attributes:
        FORMAT_VERSION: Current format version string.
    """

    FORMAT_VERSION = "1.0"

    def __init__(self):
        """Initialize empty SLPTFile."""
        self._labels: Optional[sio.Labels] = None
        self._track_contexts: Dict[Tuple[int, int], TrackContextData] = {}
        self._explanations: List[Dict] = []
        self._pipeline_layers: List[PipelineLayerRecord] = []
        self._metadata: Dict[str, Any] = {
            "format_version": self.FORMAT_VERSION,
            "created_at": datetime.now().isoformat(),
            "sleap_mot_version": "0.1.0",
            "source_slp_path": None,
        }

    @classmethod
    def from_labels(
        cls,
        labels: sio.Labels,
        explanation_store: Optional[InstanceExplanationStore] = None,
        source_path: Optional[str] = None,
    ) -> "SLPTFile":
        """Create SLPT from Labels object.

        Extracts TrackContext data from instances if present, preserving
        all tracking metadata for later reconstruction.

        Args:
            labels: SLEAP Labels object (may have TrackContext instances).
            explanation_store: Optional InstanceExplanationStore with decisions.
            source_path: Original SLP file path (for metadata).

        Returns:
            New SLPTFile instance with extracted metadata.
        """
        slpt = cls()
        slpt._labels = labels
        slpt._metadata["source_slp_path"] = str(source_path) if source_path else None

        # Extract TrackContext data from instances
        for lf in labels.labeled_frames:
            frame_idx = lf.frame_idx
            for inst_idx, inst in enumerate(lf.instances):
                if inst.track is None:
                    continue

                # Check if it's a TrackContext
                if isinstance(inst.track, TrackContext):
                    ctx_data = TrackContextData(
                        frame_idx=frame_idx,
                        instance_idx=inst_idx,
                        track_name=inst.track.name,
                        priority=inst.track.priority,
                        temporary_track=inst.track.temporary_track,
                        valid=inst.track.valid,
                        track_history=inst.track.track_history.copy(),
                    )
                    slpt._track_contexts[(frame_idx, inst_idx)] = ctx_data
                elif hasattr(inst.track, "name"):
                    # Plain sio.Track - create minimal context
                    ctx_data = TrackContextData(
                        frame_idx=frame_idx,
                        instance_idx=inst_idx,
                        track_name=inst.track.name,
                        priority=None,
                        temporary_track=inst.track.name.startswith("tracklet"),
                        valid=True,
                        track_history=[],
                    )
                    slpt._track_contexts[(frame_idx, inst_idx)] = ctx_data

        # Extract explanations if provided
        if explanation_store is not None:
            # Get all records and convert to dicts
            all_records = explanation_store.get_all_records()
            slpt._explanations = [r.to_dict() for r in all_records]

        return slpt

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SLPTFile":
        """Load .slpt file.

        Args:
            path: Path to .slpt file.

        Returns:
            Loaded SLPTFile instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SLPT file not found: {path}")

        slpt = cls()

        with zipfile.ZipFile(path, "r") as zf:
            # Load manifest/metadata
            if "pipeline.json" in zf.namelist():
                pipeline_data = json.loads(zf.read("pipeline.json").decode("utf-8"))
                slpt._metadata = pipeline_data.get("metadata", slpt._metadata)
                slpt._pipeline_layers = [
                    PipelineLayerRecord.from_dict(layer)
                    for layer in pipeline_data.get("layers", [])
                ]

            # Load track contexts
            if "track_contexts.json" in zf.namelist():
                contexts_data = json.loads(
                    zf.read("track_contexts.json").decode("utf-8")
                )
                for ctx_dict in contexts_data:
                    ctx = TrackContextData.from_dict(ctx_dict)
                    slpt._track_contexts[(ctx.frame_idx, ctx.instance_idx)] = ctx

            # Load explanations
            if "explanations.json" in zf.namelist():
                slpt._explanations = json.loads(
                    zf.read("explanations.json").decode("utf-8")
                )

            # Load embedded SLP
            if "labels.slp" in zf.namelist():
                # Extract to temp file and load
                with tempfile.NamedTemporaryFile(suffix=".slp", delete=False) as tmp:
                    tmp.write(zf.read("labels.slp"))
                    tmp_path = tmp.name

                try:
                    slpt._labels = sio.load_file(tmp_path)
                finally:
                    Path(tmp_path).unlink()

        return slpt

    def save(self, path: Union[str, Path]) -> None:
        """Save to .slpt file.

        Args:
            path: Output path for .slpt file.
        """
        path = Path(path)

        # Ensure .slpt extension
        if path.suffix != ".slpt":
            path = path.with_suffix(".slpt")

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Save labels.slp
            if self._labels is not None:
                with tempfile.NamedTemporaryFile(suffix=".slp", delete=False) as tmp:
                    tmp_path = tmp.name

                try:
                    # Convert TrackContext to plain Track before saving SLP
                    labels_copy = self._prepare_labels_for_save()
                    sio.save_file(labels_copy, tmp_path)
                    zf.write(tmp_path, "labels.slp")
                finally:
                    Path(tmp_path).unlink()

            # Save track contexts
            contexts_list = [ctx.to_dict() for ctx in self._track_contexts.values()]
            zf.writestr(
                "track_contexts.json",
                json.dumps(contexts_list, indent=2, default=self._json_serializer)
            )

            # Merge track_history entries into explanations
            merged_explanations = self._merge_track_history_into_explanations()
            zf.writestr(
                "explanations.json",
                json.dumps(merged_explanations, indent=2, default=self._json_serializer)
            )

            # Save pipeline metadata
            pipeline_data = {
                "metadata": self._metadata,
                "layers": [layer.to_dict() for layer in self._pipeline_layers],
            }
            zf.writestr(
                "pipeline.json",
                json.dumps(pipeline_data, indent=2, default=self._json_serializer)
            )

        print(f"Saved SLPT file: {path}")
        print(f"  - Track contexts: {len(self._track_contexts)}")
        print(f"  - Explanations: {len(self._explanations)}")
        print(f"  - Pipeline layers: {len(self._pipeline_layers)}")

    def _merge_track_history_into_explanations(self) -> list:
        """Merge track_history entries from TrackContexts into explanations.

        Converts track_history entries that aren't already represented in
        the explanation store into explanation-format dicts. This ensures
        that renames from rename_track_globally (e.g., RFID assignment,
        spatial propagation) are visible in the ID Switch Viewer.

        Returns:
            Combined list of explanation dicts.
        """
        # Build a set of existing explanation keys to avoid duplicates
        existing_keys = set()
        for exp in self._explanations:
            key = (
                exp.get("frame_idx"),
                exp.get("instance_idx"),
                exp.get("tracker_name"),
                exp.get("assigned_track_id"),
            )
            existing_keys.add(key)

        # Convert track_history entries to explanation records
        history_records = []
        for (frame_idx, inst_idx), ctx in self._track_contexts.items():
            for entry in ctx.track_history:
                layer_name = entry.get("layer_name", "Unknown")
                new_track = entry.get("new_track_name", ctx.track_name)
                old_track = entry.get("old_track_name")
                entry_frame = entry.get("frame_idx", frame_idx)

                # Skip if already in explanations
                key = (entry_frame, inst_idx, layer_name, new_track)
                if key in existing_keys:
                    continue
                existing_keys.add(key)

                # Determine decision type
                if old_track is None:
                    decision_type = "new_track"
                elif entry.get("propagated_from_frame") is not None:
                    decision_type = "propagated"
                elif old_track != new_track:
                    decision_type = "reassigned"
                else:
                    decision_type = "matched"

                reason = entry.get("reason", "Track assignment")
                conflict = entry.get("conflict_resolved", False)
                propagated_from = entry.get("propagated_from_frame")

                record = {
                    "record_type": "TrackHistoryRecord",
                    "frame_idx": entry_frame,
                    "instance_idx": inst_idx,
                    "tracker_name": layer_name,
                    "tracker_priority": ctx.priority,
                    "decision_type": decision_type,
                    "assigned_track_id": new_track,
                    "previous_track_id": old_track,
                    "summary": reason,
                    "reasons": [reason],
                    "candidate_scores": [],
                    "conflict_resolved": conflict,
                    "propagated_from_frame": propagated_from,
                }

                # Include any nested explanation data
                if "explanation" in entry and entry["explanation"]:
                    exp_data = entry["explanation"]
                    if isinstance(exp_data, dict):
                        record["reasons"] = exp_data.get("reasons", [reason])

                history_records.append(record)

        merged = self._explanations + history_records
        if history_records:
            print(f"  Merged {len(history_records)} track_history entries into explanations")

        return merged

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types and other non-serializable objects."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

    def _prepare_labels_for_save(self) -> sio.Labels:
        """Prepare labels for SLP save by converting TrackContext to Track.

        Creates a copy of labels with plain sio.Track objects that can be
        saved to a standard SLP file.

        Returns:
            Labels with plain sio.Track objects.
        """
        if self._labels is None:
            raise ValueError("No labels to save")

        # Create a mapping of track names to Track objects
        track_cache: Dict[str, sio.Track] = {}

        # First pass: collect all unique track names and create Track objects
        for lf in self._labels.labeled_frames:
            for inst in lf.instances:
                if inst.track is not None:
                    if isinstance(inst.track, TrackContext):
                        track_name = inst.track.name
                        if track_name not in track_cache:
                            track_cache[track_name] = inst.track.track
                    elif hasattr(inst.track, 'name'):
                        track_name = inst.track.name
                        if track_name not in track_cache:
                            track_cache[track_name] = inst.track

        # Second pass: replace TrackContext with plain Track
        for lf in self._labels.labeled_frames:
            for inst in lf.instances:
                if inst.track is not None and isinstance(inst.track, TrackContext):
                    track_name = inst.track.name
                    inst.track = track_cache[track_name]

        # Update tracks list
        self._labels.tracks = list(track_cache.values())

        return self._labels

    def to_labels(self) -> sio.Labels:
        """Get Labels with TrackContext objects restored.

        This method restores TrackContext wrappers around plain Track objects
        using the saved context data, enabling proper conflict resolution.

        Returns:
            Labels with TrackContext instances.
        """
        if self._labels is None:
            raise ValueError("No labels loaded")

        # Create track cache from existing tracks
        track_cache: Dict[str, sio.Track] = {t.name: t for t in self._labels.tracks}

        # Restore TrackContext for each instance
        for lf in self._labels.labeled_frames:
            frame_idx = lf.frame_idx
            for inst_idx, inst in enumerate(lf.instances):
                key = (frame_idx, inst_idx)
                if key in self._track_contexts:
                    ctx_data = self._track_contexts[key]

                    # Get or create base track
                    if ctx_data.track_name not in track_cache:
                        track_cache[ctx_data.track_name] = sio.Track(
                            name=ctx_data.track_name
                        )
                        self._labels.tracks.append(track_cache[ctx_data.track_name])

                    base_track = track_cache[ctx_data.track_name]

                    # Create TrackContext from saved data
                    inst.track = ctx_data.to_track_context(base_track)

        return self._labels

    def export_slp(self, path: Union[str, Path]) -> None:
        """Export to standard .slp file.

        This exports only the Labels data, losing all tracking metadata.
        Use this for compatibility with SLEAP GUI.

        Args:
            path: Output path for .slp file.
        """
        if self._labels is None:
            raise ValueError("No labels to export")

        labels_copy = self._prepare_labels_for_save()
        sio.save_file(labels_copy, str(path))
        print(f"Exported SLP file: {path}")

    def get_track_context(
        self, frame_idx: int, instance_idx: int
    ) -> Optional[TrackContextData]:
        """Get TrackContext data for specific instance.

        Args:
            frame_idx: Frame index.
            instance_idx: Instance index within frame.

        Returns:
            TrackContextData if found, None otherwise.
        """
        return self._track_contexts.get((frame_idx, instance_idx))

    def get_all_track_contexts(self) -> Dict[Tuple[int, int], TrackContextData]:
        """Get all track contexts.

        Returns:
            Dict mapping (frame_idx, instance_idx) to TrackContextData.
        """
        return self._track_contexts.copy()

    def get_explanations_for_frame(self, frame_idx: int) -> List[Dict]:
        """Get all explanation records for a frame.

        Args:
            frame_idx: Frame index.

        Returns:
            List of explanation record dicts.
        """
        return [e for e in self._explanations if e.get("frame_idx") == frame_idx]

    def get_explanations_for_instance(
        self, frame_idx: int, instance_idx: int
    ) -> List[Dict]:
        """Get explanation records for specific instance.

        Args:
            frame_idx: Frame index.
            instance_idx: Instance index.

        Returns:
            List of explanation record dicts.
        """
        return [
            e
            for e in self._explanations
            if e.get("frame_idx") == frame_idx
            and e.get("instance_idx") == instance_idx
        ]

    def get_pipeline_history(self) -> List[PipelineLayerRecord]:
        """Get ordered list of tracking layers applied.

        Returns:
            List of PipelineLayerRecord in application order.
        """
        return sorted(self._pipeline_layers, key=lambda x: x.order)

    def add_layer_to_pipeline(
        self,
        name: str,
        class_name: str,
        priority: int,
        config: Dict,
    ) -> None:
        """Record that a tracking layer was applied.

        Args:
            name: Layer name.
            class_name: Full class name.
            priority: Layer priority.
            config: Layer configuration dict.
        """
        order = len(self._pipeline_layers)
        record = PipelineLayerRecord(
            order=order,
            name=name,
            class_name=class_name,
            priority=priority,
            config=config,
            applied_at=datetime.now().isoformat(),
        )
        self._pipeline_layers.append(record)

    def add_explanation(self, record: Dict) -> None:
        """Add an explanation record.

        Args:
            record: Explanation record dict.
        """
        self._explanations.append(record)

    def set_explanations_from_store(self, store: InstanceExplanationStore) -> None:
        """Set explanations from an InstanceExplanationStore.

        Args:
            store: InstanceExplanationStore with decision records.
        """
        all_records = store.get_all_records()
        self._explanations = [r.to_dict() for r in all_records]

    def merge_explanations(self, store: InstanceExplanationStore) -> None:
        """Merge new explanations with existing ones.

        Unlike set_explanations_from_store which replaces, this method
        adds new explanations while preserving existing ones.

        Args:
            store: InstanceExplanationStore with new decision records.
        """
        new_records = store.get_all_records()
        for record in new_records:
            self._explanations.append(record.to_dict())

    def update_track_context(
        self,
        frame_idx: int,
        instance_idx: int,
        track_name: str,
        priority: Optional[int] = None,
        temporary_track: bool = False,
        valid: bool = True,
        track_history: Optional[List[Dict]] = None,
    ) -> None:
        """Update or create TrackContext data for an instance.

        Args:
            frame_idx: Frame index.
            instance_idx: Instance index.
            track_name: Track name.
            priority: Priority level.
            temporary_track: Whether track is temporary.
            valid: Whether track is valid.
            track_history: History entries.
        """
        key = (frame_idx, instance_idx)
        existing = self._track_contexts.get(key)

        if existing is not None:
            # Update existing
            existing.track_name = track_name
            if priority is not None:
                existing.priority = priority
            existing.temporary_track = temporary_track
            existing.valid = valid
            if track_history is not None:
                existing.track_history = track_history
        else:
            # Create new
            ctx = TrackContextData(
                frame_idx=frame_idx,
                instance_idx=instance_idx,
                track_name=track_name,
                priority=priority,
                temporary_track=temporary_track,
                valid=valid,
                track_history=track_history or [],
            )
            self._track_contexts[key] = ctx

    def update_from_labels(self, labels: sio.Labels) -> None:
        """Update internal state from Labels object.

        Call this after running a tracking layer to update the
        track contexts with new assignments.

        Args:
            labels: Labels object with updated track assignments.
        """
        self._labels = labels

        # Re-extract track contexts
        self._track_contexts.clear()
        for lf in labels.labeled_frames:
            frame_idx = lf.frame_idx
            for inst_idx, inst in enumerate(lf.instances):
                if inst.track is None:
                    continue

                if isinstance(inst.track, TrackContext):
                    ctx_data = TrackContextData(
                        frame_idx=frame_idx,
                        instance_idx=inst_idx,
                        track_name=inst.track.name,
                        priority=inst.track.priority,
                        temporary_track=inst.track.temporary_track,
                        valid=inst.track.valid,
                        track_history=inst.track.track_history.copy(),
                    )
                    self._track_contexts[(frame_idx, inst_idx)] = ctx_data
                elif hasattr(inst.track, "name"):
                    ctx_data = TrackContextData(
                        frame_idx=frame_idx,
                        instance_idx=inst_idx,
                        track_name=inst.track.name,
                        priority=None,
                        temporary_track=inst.track.name.startswith("tracklet"),
                        valid=True,
                        track_history=[],
                    )
                    self._track_contexts[(frame_idx, inst_idx)] = ctx_data

    @property
    def labels(self) -> Optional[sio.Labels]:
        """Get raw Labels object (without TrackContext restoration)."""
        return self._labels

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get file metadata."""
        return self._metadata.copy()

    @property
    def n_track_contexts(self) -> int:
        """Get number of track contexts."""
        return len(self._track_contexts)

    @property
    def n_explanations(self) -> int:
        """Get number of explanation records."""
        return len(self._explanations)

    @property
    def n_pipeline_layers(self) -> int:
        """Get number of pipeline layers."""
        return len(self._pipeline_layers)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the SLPT file.

        Returns:
            Dict with various statistics.
        """
        stats = {
            "n_frames": len(self._labels.labeled_frames) if self._labels else 0,
            "n_instances": sum(len(lf.instances) for lf in self._labels.labeled_frames) if self._labels else 0,
            "n_track_contexts": len(self._track_contexts),
            "n_explanations": len(self._explanations),
            "n_pipeline_layers": len(self._pipeline_layers),
        }

        # Count unique tracks
        if self._track_contexts:
            unique_tracks = set(ctx.track_name for ctx in self._track_contexts.values())
            stats["n_unique_tracks"] = len(unique_tracks)
            stats["track_names"] = sorted(unique_tracks)

            # Count temporary vs permanent
            temp_count = sum(1 for ctx in self._track_contexts.values() if ctx.temporary_track)
            stats["n_temporary_tracks"] = temp_count
            stats["n_permanent_tracks"] = len(unique_tracks) - temp_count

        # Count explanation types
        if self._explanations:
            exp_types = {}
            for exp in self._explanations:
                exp_type = exp.get("decision_type", "unknown")
                exp_types[exp_type] = exp_types.get(exp_type, 0) + 1
            stats["explanation_types"] = exp_types

        return stats

    def __repr__(self) -> str:
        n_frames = len(self._labels.labeled_frames) if self._labels else 0
        n_contexts = len(self._track_contexts)
        n_explanations = len(self._explanations)
        n_layers = len(self._pipeline_layers)
        return (
            f"SLPTFile(frames={n_frames}, contexts={n_contexts}, "
            f"explanations={n_explanations}, pipeline_layers={n_layers})"
        )
