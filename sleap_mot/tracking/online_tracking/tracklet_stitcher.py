"""Tracklet Stitcher for merging fragmented tracklets.

This module provides a tracking layer that stitches fragmented tracklets
based on spatial/temporal continuity. It runs after initial tracklet creation
and merges tracklets that are clearly the same animal.

Unlike ReID which assigns tracklets to global tracks, this layer
simply merges tracklets together without changing their identity.
"""

from sleap_mot.tracking.online_tracking.base import OnlineTrackingLayer
from sleap_mot.tracking.online_tracking.motion_tracker import get_facing_direction
from sleap_mot.tracking.base import TrackContext
from sleap_mot.tracking.instance_explanations import (
    InstanceExplanationStore,
    StitchDecisionRecord,
    DecisionType,
)
from sleap_mot.utils import get_centroid
import sleap_io as sio
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Set
from scipy.optimize import linear_sum_assignment
import math


@dataclass
class TrackletInfo:
    """Metadata for a tracklet's boundaries.

    Attributes:
        tracklet_id: The tracklet identifier.
        first_frame: First frame where tracklet appears.
        last_frame: Last frame where tracklet appears.
        first_instance: Instance in first frame.
        last_instance: Instance in last frame.
        first_centroid: Centroid in first frame.
        last_centroid: Centroid in last frame.
        first_facing: Facing direction in first frame (if available).
        last_facing: Facing direction in last frame (if available).
        length: Number of frames in tracklet.
        frame_instances: Dict mapping frame_idx to instance for this tracklet.
    """
    tracklet_id: str
    first_frame: int
    last_frame: int
    first_instance: sio.PredictedInstance
    last_instance: sio.PredictedInstance
    first_centroid: np.ndarray
    last_centroid: np.ndarray
    first_facing: Optional[np.ndarray] = None
    last_facing: Optional[np.ndarray] = None
    length: int = 1
    frame_instances: Dict[int, sio.PredictedInstance] = field(default_factory=dict)


@dataclass
class StitchCandidate:
    """A candidate pair of tracklets for stitching.

    Attributes:
        source: The tracklet being extended (ends earlier).
        target: The tracklet being merged (starts later).
        temporal_gap: Frames between source end and target start.
        spatial_distance: Distance between source last centroid and target first centroid.
        facing_angle_change: Change in facing direction (degrees), if available.
        score: Combined score for this stitch (lower = better).
    """
    source: TrackletInfo
    target: TrackletInfo
    temporal_gap: int
    spatial_distance: float
    facing_angle_change: Optional[float] = None
    score: float = float('inf')


class TrackletStitcher(OnlineTrackingLayer):
    """Stitches fragmented tracklets based on spatial/temporal continuity.

    This layer runs after initial tracklet creation and merges tracklets
    that are clearly the same animal based on:
    - Temporal adjacency (tracklet B starts shortly after A ends)
    - Spatial proximity (A's end position is close to B's start position)
    - Optional: Facing direction consistency

    Unlike ReID which assigns tracklets to global tracks, this layer
    simply merges tracklets together without changing their identity.

    Attributes:
        max_gap_frames: Maximum frames between tracklet end and start.
        max_distance: Maximum centroid distance (pixels) for stitching.
        use_facing_consistency: Whether to check facing direction consistency.
        facing_threshold: Maximum facing direction change (degrees).
        min_tracklet_length: Minimum length of tracklets to consider stitching.
        max_chain_length: Maximum total length of stitched tracklet chain.
    """

    def __init__(
        self,
        priority: int = 6,
        name: str = "TrackletStitcher",
        max_gap_frames: int = 10,
        max_distance: float = 50.0,
        use_facing_consistency: bool = True,
        facing_threshold: float = 60.0,
        min_tracklet_length: int = 1,
        max_chain_length: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the tracklet stitcher.

        Args:
            priority: Priority level for conflict resolution.
            name: Layer name for history tracking.
            max_gap_frames: Max frames between tracklet end and start.
            max_distance: Max centroid distance (pixels) for stitching.
            use_facing_consistency: Check facing direction consistency.
            facing_threshold: Max facing direction change (degrees).
            min_tracklet_length: Min length of tracklets to consider.
            max_chain_length: Max total length of stitched chain (None = no limit).
        """
        self.max_gap_frames = max_gap_frames
        self.max_distance = max_distance
        self.use_facing_consistency = use_facing_consistency
        self.facing_threshold = facing_threshold
        self.min_tracklet_length = min_tracklet_length
        self.max_chain_length = max_chain_length

        # Tracklet index built during tracking
        self._tracklet_index: Dict[str, TrackletInfo] = {}
        self._stitch_history: List[Tuple[str, str, int]] = []  # (source, target, round)

        super().__init__(
            priority=priority,
            name=name,
            temporary=True,  # Tracklets remain temporary
            max_gap=1,
            matching_method="hungarian",
            clear_tracks=False,  # Preserve existing tracks
            **kwargs,
        )

    def _configure_thresholds(self, **kwargs) -> None:
        """Configure thresholds. Required by base class."""
        pass

    @property
    def is_tracklet_mode(self) -> bool:
        """Always returns True - stitcher works with tracklets."""
        return True

    def compute_association_score(
        self,
        instance1: sio.PredictedInstance,
        instance2: sio.PredictedInstance,
        frame_gap: int = 1,
        track_history: Optional[List[Tuple[int, sio.PredictedInstance]]] = None,
        **kwargs
    ) -> float:
        """Not used by stitcher - implemented for interface compliance."""
        return 0.0

    def should_continue_track(
        self,
        track_instances: List[Tuple[int, sio.PredictedInstance]],
        candidate: sio.PredictedInstance,
        score: float,
        frame_idx: int,
        other_candidates: Optional[List[sio.PredictedInstance]] = None,
        **kwargs
    ) -> bool:
        """Not used by stitcher - implemented for interface compliance."""
        return True

    def track(
        self,
        labels: sio.Labels,
        max_instances: Optional[int] = None,
        **kwargs
    ) -> sio.Labels:
        """Main tracking method - stitches fragmented tracklets.

        This method:
        1. Builds a tracklet index from existing track assignments
        2. Iteratively finds and applies stitches until no more are possible
        3. Updates track assignments in the Labels object

        Args:
            labels: SLEAP Labels object with existing tracklet assignments.
            max_instances: Not used by stitcher.
            **kwargs: Additional arguments.

        Returns:
            labels: The Labels object with stitched track assignments.
        """
        # Build tracklet index
        self._build_tracklet_index(labels)

        if not self._tracklet_index:
            return labels

        # Track statistics
        initial_tracklet_count = len(self._tracklet_index)

        # Iterative stitching
        stitch_round = 0
        total_stitches = 0

        while True:
            stitch_round += 1
            stitches = self._find_and_apply_stitches(labels, stitch_round)

            if not stitches:
                break

            total_stitches += len(stitches)

            # Safety check to prevent infinite loops
            if stitch_round > 1000:
                break

        # Log summary
        final_tracklet_count = len(self._tracklet_index)
        if self._explanation_store is not None:
            # Create a summary record at frame 0, instance 0
            summary_record = StitchDecisionRecord(
                frame_idx=0,
                instance_idx=0,
                tracker_name=self.name,
                tracker_priority=self.priority,
                decision_type=DecisionType.MATCHED,
                assigned_track_id=None,
                previous_track_id=None,
                summary=f"Stitching complete: {initial_tracklet_count} -> {final_tracklet_count} tracklets",
                reasons=[
                    f"Total stitches: {total_stitches}",
                    f"Rounds: {stitch_round}",
                    f"Reduction: {initial_tracklet_count - final_tracklet_count} tracklets merged",
                ],
                stitch_round=stitch_round,
            )
            self._explanation_store.add(summary_record)

        return labels

    def _build_tracklet_index(self, labels: sio.Labels) -> None:
        """Build index of tracklet boundaries from Labels.

        Args:
            labels: SLEAP Labels object with track assignments.
        """
        self._tracklet_index.clear()

        # Collect all instances for each tracklet
        tracklet_instances: Dict[str, List[Tuple[int, sio.PredictedInstance]]] = {}

        for lf in labels.labeled_frames:
            frame_idx = lf.frame_idx
            for inst in lf.instances:
                if inst.track is None:
                    continue

                # Get track name (handle TrackContext)
                track_name = inst.track.name if hasattr(inst.track, 'name') else str(inst.track)

                if track_name not in tracklet_instances:
                    tracklet_instances[track_name] = []
                tracklet_instances[track_name].append((frame_idx, inst))

        # Build TrackletInfo for each tracklet
        for tracklet_id, instances in tracklet_instances.items():
            if len(instances) < self.min_tracklet_length:
                continue

            # Sort by frame
            instances.sort(key=lambda x: x[0])

            first_frame, first_inst = instances[0]
            last_frame, last_inst = instances[-1]

            first_centroid = get_centroid(first_inst)
            last_centroid = get_centroid(last_inst)

            if first_centroid is None or last_centroid is None:
                continue

            # Get facing directions
            first_facing = get_facing_direction(first_inst)
            last_facing = get_facing_direction(last_inst)

            # Build frame_instances dict
            frame_instances = {frame_idx: inst for frame_idx, inst in instances}

            self._tracklet_index[tracklet_id] = TrackletInfo(
                tracklet_id=tracklet_id,
                first_frame=first_frame,
                last_frame=last_frame,
                first_instance=first_inst,
                last_instance=last_inst,
                first_centroid=first_centroid,
                last_centroid=last_centroid,
                first_facing=first_facing,
                last_facing=last_facing,
                length=len(instances),
                frame_instances=frame_instances,
            )

    def _find_stitch_candidates(self) -> List[StitchCandidate]:
        """Find all valid stitch candidates between tracklets.

        Returns:
            List of StitchCandidate objects sorted by score.
        """
        candidates = []

        tracklet_ids = list(self._tracklet_index.keys())

        for source_id in tracklet_ids:
            source = self._tracklet_index[source_id]

            for target_id in tracklet_ids:
                if source_id == target_id:
                    continue

                target = self._tracklet_index[target_id]

                # Check temporal adjacency
                # Target must start after source ends
                if target.first_frame <= source.last_frame:
                    continue

                temporal_gap = target.first_frame - source.last_frame - 1

                if temporal_gap > self.max_gap_frames:
                    continue

                # Check spatial proximity
                spatial_distance = np.linalg.norm(
                    target.first_centroid - source.last_centroid
                )

                if spatial_distance > self.max_distance:
                    continue

                # Check facing consistency (optional)
                facing_angle_change = None
                if self.use_facing_consistency:
                    if source.last_facing is not None and target.first_facing is not None:
                        dot = np.clip(
                            np.dot(source.last_facing, target.first_facing),
                            -1.0, 1.0
                        )
                        facing_angle_change = math.degrees(math.acos(dot))

                        if facing_angle_change > self.facing_threshold:
                            continue

                # Check chain length limit
                if self.max_chain_length is not None:
                    combined_length = source.length + target.length
                    if combined_length > self.max_chain_length:
                        continue

                # Compute score (lower = better)
                # Prioritize: smaller gap, smaller distance
                score = spatial_distance + (temporal_gap * 10)

                candidates.append(StitchCandidate(
                    source=source,
                    target=target,
                    temporal_gap=temporal_gap,
                    spatial_distance=spatial_distance,
                    facing_angle_change=facing_angle_change,
                    score=score,
                ))

        # Sort by score
        candidates.sort(key=lambda c: c.score)

        return candidates

    def _find_and_apply_stitches(
        self,
        labels: sio.Labels,
        stitch_round: int
    ) -> List[Tuple[str, str]]:
        """Find optimal stitches and apply them.

        Uses Hungarian algorithm to find optimal one-to-one matching
        between tracklet ends and starts.

        Args:
            labels: SLEAP Labels object.
            stitch_round: Current round number for logging.

        Returns:
            List of (source_id, target_id) tuples that were stitched.
        """
        candidates = self._find_stitch_candidates()

        if not candidates:
            return []

        # Build cost matrix for Hungarian assignment
        # Rows = source tracklets (ending), Cols = target tracklets (starting)
        source_ids = list(set(c.source.tracklet_id for c in candidates))
        target_ids = list(set(c.target.tracklet_id for c in candidates))

        if not source_ids or not target_ids:
            return []

        # Build lookup for candidates
        candidate_lookup: Dict[Tuple[str, str], StitchCandidate] = {}
        for c in candidates:
            key = (c.source.tracklet_id, c.target.tracklet_id)
            if key not in candidate_lookup or c.score < candidate_lookup[key].score:
                candidate_lookup[key] = c

        # Build cost matrix
        n_sources = len(source_ids)
        n_targets = len(target_ids)
        cost_matrix = np.full((n_sources, n_targets), 1e10)

        for i, source_id in enumerate(source_ids):
            for j, target_id in enumerate(target_ids):
                key = (source_id, target_id)
                if key in candidate_lookup:
                    cost_matrix[i, j] = candidate_lookup[key].score

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Collect valid assignments
        stitches = []
        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] < 1e9:
                source_id = source_ids[row]
                target_id = target_ids[col]
                stitches.append((source_id, target_id))

        # Apply stitches (skip any that became invalid due to earlier stitches)
        applied_stitches = []
        for source_id, target_id in stitches:
            # Check if both tracklets still exist (may have been merged earlier this round)
            if source_id not in self._tracklet_index:
                continue
            if target_id not in self._tracklet_index:
                continue

            success = self._apply_stitch(labels, source_id, target_id, stitch_round)
            if success:
                applied_stitches.append((source_id, target_id))

        return applied_stitches

    def _apply_stitch(
        self,
        labels: sio.Labels,
        source_id: str,
        target_id: str,
        stitch_round: int
    ) -> bool:
        """Apply a stitch by renaming target tracklet to source using rename_track_globally.

        This is a RENAME operation - merging the target tracklet into the source
        by propagating the source identity to all frames of the target.
        For temporary tracklets, this is ALWAYS allowed regardless of priority.

        Args:
            labels: SLEAP Labels object.
            source_id: Tracklet ID being extended.
            target_id: Tracklet ID being merged into source.
            stitch_round: Current round number for logging.

        Returns:
            True if stitch was applied, False if blocked.
        """
        source = self._tracklet_index[source_id]
        target = self._tracklet_index[target_id]

        # Calculate details for reason and logging
        spatial_distance = float(np.linalg.norm(
            target.first_centroid - source.last_centroid
        ))
        temporal_gap = target.first_frame - source.last_frame - 1

        facing_angle_change = None
        if source.last_facing is not None and target.first_facing is not None:
            dot = np.clip(
                np.dot(source.last_facing, target.first_facing),
                -1.0, 1.0
            )
            facing_angle_change = float(math.degrees(math.acos(dot)))

        # Build reason for the stitch
        reason = f"Tracklet stitch (gap={temporal_gap}frames, dist={spatial_distance:.1f}px)"

        # Use rename_track_globally for the merge operation
        # This is ALWAYS allowed for temporary tracklets
        success = self.rename_track_globally(
            labels=labels,
            old_track_name=target_id,
            new_track_name=source_id,
            reason=reason,
        )

        if not success:
            return False

        # Log explanation
        if self._explanation_store is not None:
            # Create record at the stitch point (target's first frame)
            record = StitchDecisionRecord(
                frame_idx=target.first_frame,
                instance_idx=0,
                tracker_name=self.name,
                tracker_priority=self.priority,
                decision_type=DecisionType.MATCHED,
                assigned_track_id=source_id,
                previous_track_id=target_id,
                summary=f"Stitched {target_id} -> {source_id}",
                reasons=[
                    f"Temporal gap: {temporal_gap} frames",
                    f"Spatial distance: {spatial_distance:.1f}px",
                    f"Facing change: {facing_angle_change:.1f}deg" if facing_angle_change else "Facing: N/A",
                ],
                source_tracklet_id=source_id,
                target_tracklet_id=target_id,
                temporal_gap=temporal_gap,
                spatial_distance=spatial_distance,
                facing_angle_change=facing_angle_change,
                source_last_centroid=tuple(source.last_centroid.tolist()),
                target_first_centroid=tuple(target.first_centroid.tolist()),
                source_last_frame=source.last_frame,
                target_first_frame=target.first_frame,
                stitch_round=stitch_round,
            )
            self._explanation_store.add(record)

        # Update tracklet index
        # Merge target into source
        combined_frame_instances = {**source.frame_instances, **target.frame_instances}

        updated_source = TrackletInfo(
            tracklet_id=source_id,
            first_frame=source.first_frame,
            last_frame=target.last_frame,
            first_instance=source.first_instance,
            last_instance=target.last_instance,
            first_centroid=source.first_centroid,
            last_centroid=target.last_centroid,
            first_facing=source.first_facing,
            last_facing=target.last_facing,
            length=source.length + target.length,
            frame_instances=combined_frame_instances,
        )

        self._tracklet_index[source_id] = updated_source
        del self._tracklet_index[target_id]

        # Record history
        self._stitch_history.append((source_id, target_id, stitch_round))

        return True

    def get_stitch_statistics(self) -> Dict[str, Any]:
        """Get statistics about stitching performed.

        Returns:
            Dict with statistics including stitch count, rounds, etc.
        """
        if not self._stitch_history:
            return {
                "total_stitches": 0,
                "rounds": 0,
                "final_tracklet_count": len(self._tracklet_index),
            }

        rounds = max(r for _, _, r in self._stitch_history)

        return {
            "total_stitches": len(self._stitch_history),
            "rounds": rounds,
            "final_tracklet_count": len(self._tracklet_index),
            "stitches_by_round": {
                r: sum(1 for _, _, round_num in self._stitch_history if round_num == r)
                for r in range(1, rounds + 1)
            },
        }

    @classmethod
    def with_defaults(
        cls,
        max_gap_frames: int = 10,
        max_distance: float = 50.0,
        use_facing_consistency: bool = True,
        facing_threshold: float = 60.0,
        priority: int = 6,
        name: str = "TrackletStitcher",
        **kwargs
    ) -> "TrackletStitcher":
        """Create TrackletStitcher with sensible defaults.

        Args:
            max_gap_frames: Max frames between tracklet end and start.
            max_distance: Max centroid distance (pixels) for stitching.
            use_facing_consistency: Check facing direction consistency.
            facing_threshold: Max facing direction change (degrees).
            priority: Priority level.
            name: Layer name.
            **kwargs: Additional arguments.

        Returns:
            Configured TrackletStitcher instance.
        """
        return cls(
            priority=priority,
            name=name,
            max_gap_frames=max_gap_frames,
            max_distance=max_distance,
            use_facing_consistency=use_facing_consistency,
            facing_threshold=facing_threshold,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get tracklet stitcher configuration.

        Returns:
            Dict of configuration parameters for this tracklet stitcher.
        """
        config = super().get_config()
        config.update({
            "max_gap_frames": self.max_gap_frames,
            "max_distance": self.max_distance,
            "use_facing_consistency": self.use_facing_consistency,
            "facing_threshold": self.facing_threshold,
            "min_tracklet_length": self.min_tracklet_length,
            "max_chain_length": self.max_chain_length,
        })
        return config
