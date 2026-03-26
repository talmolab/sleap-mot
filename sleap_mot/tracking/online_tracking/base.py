"""Base class for online (frame-by-frame) tracking methods.

Online tracking processes frames sequentially, making association decisions
based on local temporal context. This module provides the abstract base class
and common utilities for implementing online tracking algorithms.

Online trackers can operate in two modes:
1. Full Tracking Mode: Assigns identities to all instances across all frames
2. Tracklet Generation Mode: Creates confident short sequences (tracklets)

The mode is determined by threshold parameters specific to each tracking method.
"""

from sleap_mot.tracking.base import TrackingLayer, TrackContext
import sleap_io as sio
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


class OnlineTrackingLayer(TrackingLayer, ABC):
    """Abstract base class for online (frame-by-frame) tracking methods.

    Online trackers process frames sequentially, associating instances
    based on temporal proximity and similarity metrics. They can operate
    in two modes determined by threshold parameters:

    1. Full Tracking Mode: Assigns identities to all instances
       - No confidence filtering
       - Uses optimal assignment (Hungarian) for best matches
       - Generates complete tracks

    2. Tracklet Generation Mode: Creates confident short sequences
       - Applies confidence thresholds to break uncertain tracks
       - Creates temporary track IDs for tracklets
       - Higher priority layers can later assign global identities

    Attributes:
        priority: Priority level for conflict resolution (higher = more authoritative)
        name: Layer name for history tracking
        temporary: Whether tracks from this layer are temporary
        max_gap: Maximum frame gap to search for associations (default 1)
        matching_method: Assignment algorithm ("hungarian" or "greedy")
        clear_tracks: Whether to clear existing tracks before tracking (default True)
    """

    def __init__(
        self,
        priority: int = 5,
        name: str = "OnlineTracker",
        temporary: bool = True,
        max_gap: int = 1,
        matching_method: str = "hungarian",
        clear_tracks: bool = True,
        **kwargs
    ):
        """Initialize the online tracking layer.

        Args:
            priority: Priority level for conflict resolution
            name: Layer name for history tracking
            temporary: Whether tracks from this layer are temporary (overwritten
                to True if in tracklet mode)
            max_gap: Maximum frame gap to search for associations
            matching_method: Assignment algorithm ("hungarian" or "greedy")
            clear_tracks: If True, clear existing track assignments before tracking.
                If False, existing tracks are preserved (but may be overwritten
                during tracking). Default True.
            **kwargs: Additional arguments for subclass configuration
        """
        # Initialize thresholds first so is_tracklet_mode works
        self._configure_thresholds(**kwargs)

        # Determine if temporary based on tracklet mode
        actual_temporary = temporary if not self.is_tracklet_mode else True

        super().__init__(priority=priority, name=name, temporary=actual_temporary)

        self.max_gap = max_gap
        self.matching_method = matching_method
        self.clear_tracks = clear_tracks
        self._track_counter = 0
        self._tracklet_counter = 0

    @abstractmethod
    def _configure_thresholds(self, **kwargs) -> None:
        """Configure threshold parameters that determine tracking mode.

        Subclasses implement this to set up their specific thresholds.
        If any thresholds indicate filtering should occur, the tracker
        operates in tracklet mode.

        Args:
            **kwargs: Threshold parameters specific to the tracking method
        """
        pass

    @property
    @abstractmethod
    def is_tracklet_mode(self) -> bool:
        """Return True if generating tracklets, False if full tracking.

        Tracklet mode is enabled when confidence-filtering thresholds are set.
        In this mode, tracks are broken when confidence drops below thresholds.
        """
        pass

    @abstractmethod
    def compute_association_score(
        self,
        instance1: sio.PredictedInstance,
        instance2: sio.PredictedInstance,
        frame_gap: int = 1,
        track_history: Optional[List[Tuple[int, sio.PredictedInstance]]] = None,
        **kwargs
    ) -> float:
        """Compute association score between two instances.

        Higher scores indicate better matches. The specific metric depends
        on the tracking method (e.g., motion probability, IoU, OKS).

        Args:
            instance1: Instance from earlier frame
            instance2: Instance from later frame
            frame_gap: Number of frames between instances
            track_history: Optional track history for velocity estimation
            **kwargs: Additional arguments for specific tracking methods

        Returns:
            Association score (higher = better match)
        """
        pass

    @abstractmethod
    def should_continue_track(
        self,
        track_instances: List[Tuple[int, sio.PredictedInstance]],
        candidate: sio.PredictedInstance,
        score: float,
        frame_idx: int,
        other_candidates: Optional[List[sio.PredictedInstance]] = None,
        **kwargs
    ) -> bool:
        """Determine if a track should continue with a candidate instance.

        In full tracking mode, typically returns True for the best candidate.
        In tracklet mode, applies confidence thresholds to determine if
        the track should continue or be terminated.

        Args:
            track_instances: Current track as [(frame_idx, instance), ...]
            candidate: Potential next instance
            score: Association score for this candidate
            frame_idx: Current frame index
            other_candidates: Other instances in the same frame (for proximity checks)
            **kwargs: Additional arguments for specific tracking methods

        Returns:
            True if track should continue with this candidate
        """
        pass

    def track(
        self,
        labels: sio.Labels,
        max_instances: Optional[int] = None,
        **kwargs
    ) -> sio.Labels:
        """Main tracking method - processes frames sequentially.

        Iterates through all frames, associating instances to existing tracks
        or creating new tracks as needed. In tracklet mode, tracks are broken
        when confidence thresholds are not met.

        Args:
            labels: SLEAP Labels object containing instances to track
            max_instances: Maximum expected instances per frame (auto-detected if None)
            **kwargs: Additional arguments passed to scoring and continuation methods

        Returns:
            labels: The Labels object with track assignments
        """
        if len(labels.videos) > 1:
            raise NotImplementedError("Multiple videos are not supported.")

        # Clear existing tracks if requested
        if self.clear_tracks:
            for lf in labels.labeled_frames:
                for inst in lf.instances:
                    inst.track = None
            labels.tracks = []

        # Calculate max instances if not provided
        if max_instances is None:
            max_instances = self._calculate_max_instances(labels)

        # Convert existing tracks to context objects if needed
        self.convert_tracks_to_context_objects(labels, self.priority)

        # Initialize active tracks dictionary
        # {track_id: {"instances": [(frame_idx, instance), ...], "last_frame": int}}
        active_tracks: Dict[str, Dict[str, Any]] = {}

        # Process each frame sequentially
        n_frames = len(labels.labeled_frames)
        for frame_idx in range(n_frames):
            self._process_frame(
                labels,
                frame_idx,
                active_tracks,
                max_instances,
                **kwargs
            )

        return labels

    def _process_frame(
        self,
        labels: sio.Labels,
        frame_idx: int,
        active_tracks: Dict[str, Dict[str, Any]],
        max_instances: int,
        **kwargs
    ) -> None:
        """Process a single frame for tracking.

        Associates instances in the current frame with active tracks,
        creates new tracks for unmatched instances, and terminates
        tracks that exceed max_gap without matches.

        Args:
            labels: SLEAP Labels object
            frame_idx: Current frame index
            active_tracks: Dict of active track_id -> track data
            max_instances: Maximum instances per frame
            **kwargs: Additional arguments for scoring methods
        """
        lf = labels.labeled_frames[frame_idx]
        current_instances = list(lf.instances)

        if not current_instances:
            return

        # Get candidates from active tracks
        track_candidates = self._get_track_candidates(active_tracks, frame_idx)

        if not track_candidates:
            # No active tracks - create new tracks for all instances
            for inst in current_instances:
                track_id = self._generate_track_id()
                self._assign_track_to_instance(
                    labels, frame_idx, inst, track_id,
                    reason="New track (no active tracks)"
                )
                active_tracks[track_id] = {
                    "instances": [(frame_idx, inst)],
                    "last_frame": frame_idx
                }
            return

        # Build cost matrix for assignment
        cost_matrix = self._build_cost_matrix(
            track_candidates,
            current_instances,
            active_tracks,
            frame_idx,
            **kwargs
        )

        # Perform assignment
        matched_pairs, unmatched_tracks, unmatched_instances = self._perform_assignment(
            cost_matrix,
            list(track_candidates.keys()),
            current_instances,
            active_tracks,
            frame_idx,
            **kwargs
        )

        # Process matched pairs
        for track_id, inst_idx in matched_pairs:
            inst = current_instances[inst_idx]
            track_data = active_tracks[track_id]

            self._assign_track_to_instance(
                labels, frame_idx, inst, track_id,
                reason=f"Matched to existing track (score-based)"
            )
            track_data["instances"].append((frame_idx, inst))
            track_data["last_frame"] = frame_idx

        # Create new tracks for unmatched instances
        for inst_idx in unmatched_instances:
            inst = current_instances[inst_idx]
            track_id = self._generate_track_id()
            self._assign_track_to_instance(
                labels, frame_idx, inst, track_id,
                reason="New track (unmatched instance)"
            )
            active_tracks[track_id] = {
                "instances": [(frame_idx, inst)],
                "last_frame": frame_idx
            }

        # Clean up inactive tracks (exceeded max_gap)
        tracks_to_remove = []
        for track_id, track_data in active_tracks.items():
            if frame_idx - track_data["last_frame"] > self.max_gap:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del active_tracks[track_id]

    def _get_track_candidates(
        self,
        active_tracks: Dict[str, Dict[str, Any]],
        frame_idx: int
    ) -> Dict[str, Tuple[int, sio.PredictedInstance]]:
        """Get candidate instances from active tracks for matching.

        Returns the most recent instance from each active track that
        is within max_gap frames of the current frame.

        Args:
            active_tracks: Dict of active track data
            frame_idx: Current frame index

        Returns:
            Dict mapping track_id to (last_frame_idx, last_instance)
        """
        candidates = {}
        for track_id, track_data in active_tracks.items():
            if frame_idx - track_data["last_frame"] <= self.max_gap:
                last_frame, last_inst = track_data["instances"][-1]
                candidates[track_id] = (last_frame, last_inst)
        return candidates

    def _build_cost_matrix(
        self,
        track_candidates: Dict[str, Tuple[int, sio.PredictedInstance]],
        current_instances: List[sio.PredictedInstance],
        active_tracks: Dict[str, Dict[str, Any]],
        frame_idx: int,
        **kwargs
    ) -> np.ndarray:
        """Build cost matrix for track-instance assignment.

        Cost = -score (so lower cost = better match for Hungarian algorithm).

        Args:
            track_candidates: Dict of track_id -> (last_frame, last_instance)
            current_instances: Instances in current frame
            active_tracks: Full active tracks dict for history access
            frame_idx: Current frame index
            **kwargs: Additional arguments for scoring

        Returns:
            Cost matrix of shape (num_tracks, num_instances)
        """
        track_ids = list(track_candidates.keys())
        n_tracks = len(track_ids)
        n_instances = len(current_instances)

        # Initialize with large cost (no match)
        cost_matrix = np.full((n_tracks, n_instances), 1e10)

        for t_idx, track_id in enumerate(track_ids):
            last_frame, last_inst = track_candidates[track_id]
            frame_gap = frame_idx - last_frame
            track_history = active_tracks[track_id]["instances"]

            for i_idx, inst in enumerate(current_instances):
                score = self.compute_association_score(
                    last_inst,
                    inst,
                    frame_gap=frame_gap,
                    track_history=track_history,
                    **kwargs
                )
                # Convert score to cost (negate so higher score = lower cost)
                cost_matrix[t_idx, i_idx] = -score

        return cost_matrix

    def _perform_assignment(
        self,
        cost_matrix: np.ndarray,
        track_ids: List[str],
        current_instances: List[sio.PredictedInstance],
        active_tracks: Dict[str, Dict[str, Any]],
        frame_idx: int,
        **kwargs
    ) -> Tuple[List[Tuple[str, int]], List[str], List[int]]:
        """Perform track-instance assignment using specified method.

        Args:
            cost_matrix: Cost matrix (n_tracks x n_instances)
            track_ids: List of track IDs (rows of cost matrix)
            current_instances: Current frame instances (columns)
            active_tracks: Full active tracks dict
            frame_idx: Current frame index
            **kwargs: Additional arguments for continuation checks

        Returns:
            Tuple of:
                - matched_pairs: List of (track_id, instance_idx) pairs
                - unmatched_tracks: List of unmatched track_ids
                - unmatched_instances: List of unmatched instance indices
        """
        n_tracks, n_instances = cost_matrix.shape

        if self.matching_method == "hungarian":
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            # Greedy matching
            row_ind, col_ind = self._greedy_matching(cost_matrix)

        matched_pairs = []
        matched_tracks = set()
        matched_instances = set()

        for row, col in zip(row_ind, col_ind):
            track_id = track_ids[row]
            inst = current_instances[col]
            score = -cost_matrix[row, col]  # Convert cost back to score

            # Check if score is valid (not the large invalid cost)
            if cost_matrix[row, col] >= 1e9:
                continue

            # In tracklet mode, check if track should continue
            if self.is_tracklet_mode:
                track_history = active_tracks[track_id]["instances"]
                other_candidates = [
                    current_instances[i] for i in range(n_instances) if i != col
                ]

                if not self.should_continue_track(
                    track_history,
                    inst,
                    score,
                    frame_idx,
                    other_candidates=other_candidates,
                    **kwargs
                ):
                    continue

            matched_pairs.append((track_id, col))
            matched_tracks.add(track_id)
            matched_instances.add(col)

        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]
        unmatched_instances = [i for i in range(n_instances) if i not in matched_instances]

        return matched_pairs, unmatched_tracks, unmatched_instances

    def _greedy_matching(self, cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform greedy matching on cost matrix.

        Iteratively assigns the lowest cost pair until no valid assignments remain.

        Args:
            cost_matrix: Cost matrix (n_tracks x n_instances)

        Returns:
            Tuple of (row_indices, col_indices) arrays
        """
        n_tracks, n_instances = cost_matrix.shape
        assigned_rows = set()
        assigned_cols = set()
        row_ind = []
        col_ind = []

        # Flatten and sort by cost
        flat_indices = np.argsort(cost_matrix.ravel())

        for flat_idx in flat_indices:
            row = flat_idx // n_instances
            col = flat_idx % n_instances

            if row not in assigned_rows and col not in assigned_cols:
                if cost_matrix[row, col] < 1e9:  # Valid cost
                    row_ind.append(row)
                    col_ind.append(col)
                    assigned_rows.add(row)
                    assigned_cols.add(col)

        return np.array(row_ind), np.array(col_ind)

    def _generate_track_id(self) -> str:
        """Generate a new track ID.

        In tracklet mode, generates temporary tracklet IDs.
        In full tracking mode, generates permanent track IDs.

        Returns:
            New track ID string
        """
        if self.is_tracklet_mode:
            self._tracklet_counter += 1
            return f"tracklet_{self._tracklet_counter}"
        else:
            self._track_counter += 1
            return f"track_{self._track_counter}"

    def _assign_track_to_instance(
        self,
        labels: sio.Labels,
        frame_idx: int,
        instance: sio.PredictedInstance,
        track_id: str,
        reason: str = "Track assignment"
    ) -> None:
        """Assign a track to an instance with TrackContext.

        Creates a TrackContext wrapper with history tracking.

        Args:
            labels: SLEAP Labels object
            frame_idx: Frame index
            instance: Instance to assign track to
            track_id: Track ID to assign
            reason: Reason for assignment (for history)
        """
        # Find or create the sio.Track
        existing_track = next(
            (t for t in labels.tracks if t.name == track_id), None
        )
        base_track = existing_track if existing_track else sio.Track(name=track_id)
        if not existing_track:
            labels.tracks.append(base_track)

        # Get old track name for history
        old_track_name = None
        if instance.track is not None:
            if isinstance(instance.track, TrackContext):
                old_track_name = instance.track.name
            elif hasattr(instance.track, 'name'):
                old_track_name = instance.track.name

        # Create TrackContext with history
        track_context = TrackContext(
            priority=self.priority,
            track=base_track,
            name=track_id,
            temporary_track=self.is_tracklet_mode,
            valid=True,
            track_history=[]
        )

        # Add history entry
        track_context.add_history_entry(
            layer_name=self.name,
            old_track_name=old_track_name,
            new_track_name=track_id,
            frame_idx=frame_idx,
            reason=reason,
            conflict_resolved=False,
            propagated_from_frame=None
        )

        instance.track = track_context

    def _calculate_max_instances(self, labels: sio.Labels) -> int:
        """Calculate the maximum number of instances across all frames.

        Args:
            labels: SLEAP Labels object

        Returns:
            Maximum number of instances in any single frame
        """
        if len(labels) == 0:
            return 0
        return max(len(lf.instances) for lf in labels)

    # Implement abstract methods from TrackingLayer

    def get_track_context(self, labels, frame_idx, track) -> Optional[TrackContext]:
        """Get the TrackContext for a track in a specific frame."""
        if frame_idx >= len(labels):
            return None

        lf = labels.labeled_frames[frame_idx]
        for inst in lf.instances:
            if inst.track is not None:
                if isinstance(inst.track, TrackContext):
                    if inst.track.name == track.name:
                        return inst.track
                elif isinstance(inst.track, sio.Track):
                    if inst.track.name == track.name:
                        return TrackContext(
                            priority=None,
                            track=inst.track,
                            name=inst.track.name,
                            temporary_track=False,
                            valid=True
                        )
        return None

    def has_track_in_frame(self, labels, frame_idx, track) -> bool:
        """Check if a track exists in a specific frame."""
        if frame_idx >= len(labels):
            return False

        lf = labels.labeled_frames[frame_idx]
        for inst in lf.instances:
            if inst.track is not None:
                if hasattr(inst.track, 'name') and inst.track.name == track.name:
                    return True
        return False

    def get_instance_with_track(self, labels, frame_idx, track) -> Optional[int]:
        """Get the instance index that has a specific track in a frame."""
        if frame_idx >= len(labels):
            return None

        lf = labels.labeled_frames[frame_idx]
        for idx, inst in enumerate(lf.instances):
            if inst.track is not None:
                if hasattr(inst.track, 'name') and inst.track.name == track.name:
                    return idx
        return None

    def assign_track(self, labels, frame_idx, instance_idx, track, track_context=None) -> None:
        """Assign a track to an instance."""
        if frame_idx >= len(labels):
            return
        if instance_idx >= len(labels.labeled_frames[frame_idx].instances):
            return

        existing = next((t for t in labels.tracks if t.name == track.name), None)
        if existing:
            labels.labeled_frames[frame_idx].instances[instance_idx].track = existing
        else:
            labels.tracks.append(track)
            labels.labeled_frames[frame_idx].instances[instance_idx].track = track

    def remove_track(self, labels, frame_idx, instance_idx) -> None:
        """Remove track from an instance."""
        if frame_idx >= len(labels):
            return
        if instance_idx >= len(labels.labeled_frames[frame_idx].instances):
            return

        labels.labeled_frames[frame_idx].instances[instance_idx].track = None

    def get_next_frame(self, labels, current_frame, direction) -> Optional[int]:
        """Get the next frame index in a given direction."""
        next_frame = current_frame + direction
        if 0 <= next_frame < len(labels):
            return next_frame
        return None
