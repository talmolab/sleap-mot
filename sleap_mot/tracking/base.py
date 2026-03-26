import sleap_io as sio
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path

if TYPE_CHECKING:
    from sleap_mot.tracking.instance_explanations import (
        InstanceExplanationStore,
        PropagationRecord,
        DecisionType,
    )
    from sleap_mot.io.slpt import SLPTFile

@dataclass
class TrackContext:
    """Context wrapper for track assignments with history tracking.

    Attributes:
        priority: Priority level of the layer that made this assignment.
        track: The underlying sleap_io Track object.
        name: Track name/identity.
        temporary_track: Whether this is a temporary track that can be overridden.
        valid: Whether this track assignment is valid.
        track_history: List of history entries recording all assignment changes.
    """

    priority: int or None
    track: sio.Track
    name: str
    temporary_track: bool = False
    valid: bool = True
    track_history: List[Dict] = field(default_factory=list)

    def add_history_entry(
        self,
        layer_name: str,
        old_track_name: Optional[str],
        new_track_name: str,
        frame_idx: int,
        reason: str,
        conflict_resolved: bool = False,
        propagated_from_frame: Optional[int] = None,
        explanation: Optional[Dict] = None,
    ) -> None:
        """Record a track assignment change in history.

        Args:
            layer_name: Name of the tracking layer making the change.
            old_track_name: Previous track name (None if first assignment).
            new_track_name: New track name being assigned.
            frame_idx: Frame index where change occurred.
            reason: Human-readable explanation.
            conflict_resolved: Whether this resolved a priority conflict.
            propagated_from_frame: Source frame if propagated from elsewhere.
            explanation: Optional detailed explanation dict (from SwitchExplanation.to_dict()).
        """
        entry = {
            "layer_name": layer_name,
            "old_track_name": old_track_name,
            "new_track_name": new_track_name,
            "frame_idx": frame_idx,
            "reason": reason,
            "conflict_resolved": conflict_resolved,
            "propagated_from_frame": propagated_from_frame,
        }
        if explanation is not None:
            entry["explanation"] = explanation
        self.track_history.append(entry)

    def get_history_entries(self) -> List:
        """Get typed history entries.

        Returns:
            List of TrackHistoryEntry objects.
        """
        from sleap_mot.metrics.types import TrackHistoryEntry

        return [TrackHistoryEntry.from_dict(d) for d in self.track_history]

    def is_valid(self) -> bool:
        """Check if this track context is valid.

        Returns:
            True if valid, False otherwise.
        """
        return self.valid


def extract_all_track_histories(labels: sio.Labels) -> Dict[str, List[Dict]]:
    """Extract all track histories from labels before converting to sio.Track.

    This function collects track_history data from all TrackContext objects
    in the labels, indexed by track name. Call this BEFORE calling
    convert_context_objects_to_tracks() to preserve history data.

    Args:
        labels: Labels object containing instances with TrackContext tracks.

    Returns:
        Dict mapping track names to lists of history entry dicts.
        Each history entry contains:
            - layer_name: Which tracking layer made the decision
            - old_track_name: Previous identity
            - new_track_name: New identity assigned
            - frame_idx: Frame where change occurred
            - reason: Why the change was made
            - conflict_resolved: Whether a conflict was resolved
            - propagated_from_frame: Source frame if propagated
    """
    histories = {}

    for lf in labels.labeled_frames:
        for inst in lf.instances:
            if inst.track is None:
                continue

            # Handle TrackContext objects
            if isinstance(inst.track, TrackContext):
                track_name = inst.track.name
                if inst.track.track_history:
                    if track_name not in histories:
                        histories[track_name] = []
                    # Add entries we haven't seen yet
                    for entry in inst.track.track_history:
                        # Check for duplicates based on frame_idx and layer_name
                        is_duplicate = any(
                            e["frame_idx"] == entry["frame_idx"]
                            and e["layer_name"] == entry["layer_name"]
                            and e["new_track_name"] == entry["new_track_name"]
                            for e in histories[track_name]
                        )
                        if not is_duplicate:
                            histories[track_name].append(entry)

    # Sort each track's history by frame_idx
    for track_name in histories:
        histories[track_name] = sorted(
            histories[track_name], key=lambda e: e["frame_idx"]
        )

    return histories


def save_track_history(
    histories: Dict[str, List[Dict]],
    output_path: Union[str, Path],
) -> str:
    """Save track history data to a JSON file.

    Creates a sidecar JSON file containing track_history data that would
    otherwise be lost when saving .slp files.

    Args:
        histories: Dict mapping track names to history entry lists.
            Typically obtained from extract_all_track_histories().
        output_path: Path for the JSON file. If it doesn't end with
            '_track_history.json', this suffix will be added.

    Returns:
        The actual path where the file was saved.

    Example:
        >>> histories = extract_all_track_histories(labels)
        >>> save_track_history(histories, "output.slp")
        'output_track_history.json'
    """
    output_path = str(output_path)

    # Generate appropriate filename
    if output_path.endswith("_track_history.json"):
        history_path = output_path
    elif output_path.endswith(".slp"):
        history_path = output_path.replace(".slp", "_track_history.json")
    elif output_path.endswith(".json"):
        history_path = output_path
    else:
        history_path = output_path + "_track_history.json"

    # Add metadata
    data = {
        "version": "1.0",
        "description": "Track history data for sleap-mot layer attribution analysis",
        "track_histories": histories,
    }

    with open(history_path, "w") as f:
        json.dump(data, f, indent=2)

    return history_path


def load_track_history(
    input_path: Union[str, Path],
) -> Dict[str, List[Dict]]:
    """Load track history data from a JSON file.

    Args:
        input_path: Path to the JSON file. Can be either:
            - Direct path to *_track_history.json
            - Path to .slp file (will look for matching *_track_history.json)

    Returns:
        Dict mapping track names to history entry lists.

    Raises:
        FileNotFoundError: If the history file doesn't exist.

    Example:
        >>> histories = load_track_history("output.slp")
        >>> for track_name, entries in histories.items():
        ...     print(f"{track_name}: {len(entries)} history entries")
    """
    input_path = str(input_path)

    # Determine the history file path
    if input_path.endswith("_track_history.json"):
        history_path = input_path
    elif input_path.endswith(".slp"):
        history_path = input_path.replace(".slp", "_track_history.json")
    elif input_path.endswith(".json"):
        history_path = input_path
    else:
        history_path = input_path + "_track_history.json"

    if not os.path.exists(history_path):
        raise FileNotFoundError(
            f"Track history file not found: {history_path}\n"
            f"Track history is only available if save_track_history=True was used "
            f"when saving the .slp file."
        )

    with open(history_path, "r") as f:
        data = json.load(f)

    # Handle both old format (just histories) and new format (with metadata)
    if "track_histories" in data:
        return data["track_histories"]
    else:
        return data


@dataclass
class ConflictResolutionState:
    current_frame: int
    original_frame: int
    target_identity: str  # Track name, not Track object
    old_identity: str     # Track name, not Track object
    instance_idx: int
    direction: int = 0

class TrackingLayer(ABC):
    """Abstract base class for all tracking layers.

    All tracking layers produce explanation dictionaries that document their
    decisions. The dictionaries follow a standardized structure defined by
    `get_explanation_dict_template()`, with method-specific fields added by
    each tracker subclass.

    Standard Explanation Dictionary Fields (all trackers):
        frame_idx: int - Frame index where decision was made
        instance_idx: int - Instance index within the frame
        tracker_name: str - Name of the tracker
        tracker_priority: int - Priority level for conflict resolution
        decision_type: str - Type of decision made:
            - "matched": Instance matched to existing track
            - "new_track": New track created for instance
            - "no_match": No suitable match found
            - "skipped": Instance skipped (e.g., conflict resolution)
            - "propagated": Identity propagated from another frame
        assigned_track_id: Optional[str] - Track ID assigned (None if no assignment)
        previous_track_id: Optional[str] - Previous track ID if any
        summary: str - Human-readable summary of the decision
        reasons: List[str] - List of reasons explaining the decision
        instance_centroid: Optional[Tuple[float, float]] - (x, y) centroid
        candidate_scores: List[Dict] - Candidates considered with scores
            Each candidate dict contains:
            - candidate_id: str - Track/identity ID
            - score: float - Association score
            - passed_thresholds: bool - Whether thresholds passed
            - threshold_results: Dict - Individual threshold check results
            - rejection_reason: Optional[str] - Why rejected if not selected
        association_score: Optional[float] - Score of the winning match
        thresholds: Dict[str, float] - Threshold values used for decisions
        is_propagation: bool - Whether this was propagated from another frame
        propagation_source_frame: Optional[int] - Source frame if propagated

    Method-Specific Fields (added by subclasses):
        Motion Trackers (MotionTracker, DirectionalMotionTracker):
            - kde_model_used: str - Which KDE model was used
            - motion_probability: float - Probability from motion model
            - threshold_checks: Dict - Detailed threshold check results
            - score_type: str - Type of score (e.g., "kde_probability")

        Directional Motion Trackers (DirectionalMotionTracker):
            - facing_direction: Tuple[float, float] - Unit vector of facing
            - alignment_angle: float - Angle between facing and movement
            - is_stagnant: bool - Whether movement was below threshold
            - directional_multiplier: float - Multiplier applied for direction
            - backward_rejected: bool - Whether rejected as backward movement

        RFID Trackers (RFIDFeatureTracker, CoordinateRFIDTracker):
            - rfid_ping_present: bool - Whether RFID ping was present
            - winning_rfid_id: str - RFID ID that won assignment
            - heatmap_probability: float - Probability from heatmap
            - rfid_unit_label: str - Unit label of RFID receiver
            - max_distance_from_rfid: float - Distance threshold used
    """

    # Standard explanation dictionary keys that all trackers should populate
    STANDARD_EXPLANATION_KEYS = [
        "frame_idx",
        "instance_idx",
        "tracker_name",
        "tracker_priority",
        "decision_type",
        "assigned_track_id",
        "previous_track_id",
        "summary",
        "reasons",
        "instance_centroid",
        "candidate_scores",
        "association_score",
        "thresholds",
        "is_propagation",
        "propagation_source_frame",
    ]

    def __init__(self, priority: int, name: str, temporary: bool = False):
        self.priority = priority
        self.name = name
        self.temporary = temporary
        self._explanation_store: Optional["InstanceExplanationStore"] = None

    @property
    def explanation_store(self) -> Optional["InstanceExplanationStore"]:
        """Get the instance explanation store.

        Returns:
            The InstanceExplanationStore, or None if not configured.
        """
        return self._explanation_store

    @explanation_store.setter
    def explanation_store(self, store: Optional["InstanceExplanationStore"]) -> None:
        """Set the instance explanation store.

        Args:
            store: The InstanceExplanationStore to use for logging decisions.
        """
        self._explanation_store = store

    def get_explanation_dict_template(
        self,
        frame_idx: int,
        instance_idx: int,
        decision_type: str,
        assigned_track_id: Optional[str] = None,
        previous_track_id: Optional[str] = None,
        summary: str = "",
        reasons: Optional[List[str]] = None,
        instance_centroid: Optional[tuple] = None,
        candidate_scores: Optional[List[Dict]] = None,
        association_score: Optional[float] = None,
        thresholds: Optional[Dict[str, float]] = None,
        is_propagation: bool = False,
        propagation_source_frame: Optional[int] = None,
    ) -> Dict:
        """Get the standard explanation dictionary template.

        This method provides a base template with all standard fields that
        should be present in every tracker's explanation dictionary. Subclasses
        should call this method and then add their method-specific fields.

        Args:
            frame_idx: Frame index where decision was made.
            instance_idx: Instance index within the frame.
            decision_type: Type of decision ("matched", "new_track", "no_match",
                "skipped", "propagated").
            assigned_track_id: Track ID assigned (None if no assignment).
            previous_track_id: Previous track ID if any.
            summary: Human-readable summary of the decision.
            reasons: List of reasons explaining the decision.
            instance_centroid: (x, y) centroid of the instance.
            candidate_scores: List of candidate dicts with scores. Each dict
                should contain: candidate_id, score, passed_thresholds,
                threshold_results, rejection_reason.
            association_score: Score of the winning match.
            thresholds: Dict of threshold name -> value pairs.
            is_propagation: Whether this was propagated from another frame.
            propagation_source_frame: Source frame if propagated.

        Returns:
            Dict with all standard explanation fields populated.

        Example:
            >>> # In a subclass:
            >>> def _create_explanation_dict(self, frame_idx, instance_idx, ...):
            ...     # Get base template
            ...     explanation = self.get_explanation_dict_template(
            ...         frame_idx=frame_idx,
            ...         instance_idx=instance_idx,
            ...         decision_type="matched",
            ...         ...
            ...     )
            ...     # Add method-specific fields
            ...     explanation["kde_model_used"] = "long_kde"
            ...     explanation["motion_probability"] = 0.85
            ...     return explanation
        """
        return {
            # Core identification fields
            "frame_idx": frame_idx,
            "instance_idx": instance_idx,
            "tracker_name": self.name,
            "tracker_priority": self.priority,
            # Decision outcome fields
            "decision_type": decision_type,
            "assigned_track_id": assigned_track_id,
            "previous_track_id": previous_track_id,
            # Explanation fields
            "summary": summary,
            "reasons": reasons if reasons is not None else [],
            # Instance location
            "instance_centroid": instance_centroid,
            # Candidate evaluation fields
            "candidate_scores": candidate_scores if candidate_scores is not None else [],
            "association_score": association_score,
            "thresholds": thresholds if thresholds is not None else {},
            # Propagation fields
            "is_propagation": is_propagation,
            "propagation_source_frame": propagation_source_frame,
        }

    def create_candidate_score_dict(
        self,
        candidate_id: str,
        score: float,
        passed_thresholds: bool = True,
        threshold_results: Optional[Dict[str, Dict]] = None,
        rejection_reason: Optional[str] = None,
    ) -> Dict:
        """Create a standardized candidate score dictionary.

        This helper method creates a consistent candidate score entry that
        can be added to the candidate_scores list in explanation dictionaries.

        Args:
            candidate_id: The track/identity ID of the candidate.
            score: Association score for this candidate.
            passed_thresholds: Whether all threshold checks passed.
            threshold_results: Dict of threshold check results.
                Format: {threshold_name: {"value": float, "threshold": float, "passed": bool}}
            rejection_reason: Reason for rejection if not selected.

        Returns:
            Dict with standardized candidate score fields.

        Example:
            >>> candidate = self.create_candidate_score_dict(
            ...     candidate_id="track_1",
            ...     score=0.85,
            ...     passed_thresholds=True,
            ...     threshold_results={
            ...         "distance": {"value": 50.0, "threshold": 100.0, "passed": True}
            ...     }
            ... )
        """
        return {
            "candidate_id": candidate_id,
            "score": score,
            "passed_thresholds": passed_thresholds,
            "threshold_results": threshold_results if threshold_results is not None else {},
            "rejection_reason": rejection_reason,
        }

    def convert_tracks_to_context_objects(self, labels: sio.Labels, priority: int = None):
        """Convert sio.Track objects to TrackContext wrappers.

        Only converts tracks that are not already TrackContext objects.

        Args:
            labels: SLEAP Labels object containing instances
            priority: Priority level for the TrackContext (optional)
        """
        for lf in labels.labeled_frames:
            for inst in lf.instances:
                if inst.track is not None and not isinstance(inst.track, TrackContext):
                    inst.track = TrackContext(
                        priority=priority,
                        track=inst.track,
                        name=inst.track.name,
                        temporary_track=False,
                        valid=True,
                        track_history=[]
                    )

    def convert_context_objects_to_tracks(
        self,
        labels: sio.Labels,
        save_track_history_path: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, List[Dict]]]:
        """Convert TrackContext objects back to sio.Track objects.

        This method extracts the underlying sio.Track from each TrackContext
        and removes invalid instances. Optionally saves track_history data
        to a sidecar JSON file before the conversion.

        Args:
            labels: Labels object with TrackContext instances.
            save_track_history_path: If provided, saves track_history data to
                this path before conversion. Can be:
                - Path to .slp file (will create *_track_history.json)
                - Direct path to JSON file
                - True to return histories without saving

        Returns:
            If save_track_history_path is provided, returns the extracted
            track histories dict. Otherwise returns None.
        """
        # Extract histories before conversion if requested
        histories = None
        if save_track_history_path is not None:
            histories = extract_all_track_histories(labels)
            if isinstance(save_track_history_path, (str, Path)):
                save_track_history(histories, save_track_history_path)

        # Convert TrackContext to sio.Track
        for lf in labels.labeled_frames:
            instances = []
            for inst in lf.instances:
                if inst.track.is_valid():
                    inst.track = inst.track.track
                    instances.append(inst)
            lf.instances = instances

        return histories

    def clear_tracks(self, labels: sio.Labels):
        for lf in labels.labeled_frames:
            for inst in lf.instances:
                inst.track = None

    def resolve_identity_conflict(self, labels, frame_idx, instance_idx, new_track, current_track):
        state = ConflictResolutionState(
            current_frame=frame_idx,
            original_frame=frame_idx,
            target_identity=new_track.name,
            old_identity=current_track.name,
            instance_idx=instance_idx,
            direction=0
        )
        self._propogate_identity_change(labels, state)

    def _propogate_identity_change(self, labels, state) -> bool:
        """
        Main propagation loop implementing the flowchart logic.
        
        Returns:
            True if propagation completed successfully
            False if conflict could not be resolved
        """
        while True:
            # STEP 1: Check Priority - Can we override?
            if not self._can_override_identity(labels, state):
                # Can't override in current frame, need to propagate to other frames
                
                # Try to advance direction (NONE → FORWARD → BACKWARD)
                if not self._advance_propagation_direction(state):
                    # Already tried both directions, give up
                    return False
                
                # Move to next frame in current direction
                next_frame = self.get_next_frame(labels, state.current_frame, state.direction)
                
                if next_frame is None:
                    # No more frames in this direction, try other direction
                    if not self._advance_propagation_direction(state):
                        # No more frames in any direction
                        return False
                    continue  # Try other direction
                
                state.current_frame = next_frame
                
                # Check if old identity exists in this new frame
                if not self.has_track_in_frame(labels, state.current_frame, sio.Track(name=state.old_identity)):
                    # Old identity doesn't exist here, try other direction
                    if not self._advance_propagation_direction(state):
                        return False
                    continue
                
                # Find which instance has the old identity in this frame
                state.instance_idx = self.get_instance_with_track(
                    labels, state.current_frame, sio.Track(name=state.old_identity)
                )
                
                if state.instance_idx is None:
                    # Couldn't find instance with old identity
                    if not self._advance_propagation_direction(state):
                        return False
                    continue
                
                # Loop back to check if we can override in this new frame
                continue
            
            # STEP 2: We CAN override! But check for conflicts with target identity
            conflicting_instance_idx = self._check_identity_conflict(labels, state)
            
            if conflicting_instance_idx is not None:
                # Someone else has our target identity
                if not self._resolve_conflicting_assignment(labels, state, conflicting_instance_idx):
                    # Can't resolve the conflict, need to propagate
                    if not self._advance_propagation_direction(state):
                        return False
                    continue
            
            # STEP 3: All clear! Change the identity
            self._change_identity(labels, state)
            
            # STEP 4: Should we continue propagating?
            if not self._should_continue_propagation(state):
                # Direction is NONE, we're done!
                return True
            
            # STEP 5: Continue propagating in current direction
            next_frame = self.get_next_frame(labels, state.current_frame, state.direction)
            
            if next_frame is None:
                # No more frames in this direction, try other direction
                if not self._advance_propagation_direction(state):
                    # Propagation complete
                    return True
                continue
            
            state.current_frame = next_frame
            
            # Check if old identity exists in this new frame
            if not self.has_track_in_frame(labels, state.current_frame, sio.Track(name=state.old_identity)):
                # Old identity doesn't exist here, try other direction
                if not self._advance_propagation_direction(state):
                    # Propagation complete
                    return True
                continue
            
            # Find which instance has the old identity
            state.instance_idx = self.get_instance_with_track(
                labels, state.current_frame, sio.Track(name=state.old_identity)
            )
            
            if state.instance_idx is None:
                # Couldn't find instance
                if not self._advance_propagation_direction(state):
                    return True
                continue
            
            # Loop back to process this new frame

    def get_tracklet_frames(
        self, labels: sio.Labels, track_name: str
    ) -> List[Tuple[int, int]]:
        """Get all (frame_idx, instance_idx) tuples for a tracklet.

        Args:
            labels: SLEAP Labels object.
            track_name: Name of the track to find.

        Returns:
            List of (frame_idx, instance_idx) tuples where this track appears,
            sorted by frame_idx.
        """
        frames = []
        for lf in labels.labeled_frames:
            for inst_idx, inst in enumerate(lf.instances):
                if inst.track is None:
                    continue
                inst_track_name = (
                    inst.track.name if hasattr(inst.track, "name") else str(inst.track)
                )
                if inst_track_name == track_name:
                    frames.append((lf.frame_idx, inst_idx))
        return sorted(frames, key=lambda x: x[0])

    def rename_track_globally(
        self,
        labels: sio.Labels,
        old_track_name: str,
        new_track_name: str,
        reason: str = "Track rename",
    ) -> bool:
        """Rename a track across ALL frames where it appears.

        For temporary tracks, this is ALWAYS allowed regardless of priority.
        For global tracks, this checks priority before proceeding.

        This method propagates the new identity to every frame where the old
        track appears, ensuring consistency across the entire tracklet.

        Args:
            labels: SLEAP Labels object.
            old_track_name: The current track name to rename.
            new_track_name: The new track name to assign.
            reason: Human-readable reason for the rename.

        Returns:
            True if rename was applied, False if blocked by priority.
        """
        # Find all frames with the old track
        tracklet_frames = self.get_tracklet_frames(labels, old_track_name)
        if not tracklet_frames:
            return False

        # Check if the old track is temporary (check first instance)
        first_frame_idx, first_inst_idx = tracklet_frames[0]
        first_lf = next(
            (lf for lf in labels.labeled_frames if lf.frame_idx == first_frame_idx),
            None,
        )
        if first_lf is None:
            return False

        first_inst = first_lf.instances[first_inst_idx]
        old_track = first_inst.track

        is_temporary = False
        old_priority = None
        if isinstance(old_track, TrackContext):
            is_temporary = old_track.temporary_track or old_track_name.startswith(
                "tracklet"
            )
            old_priority = old_track.priority

        # For non-temporary tracks, check priority
        if not is_temporary:
            if old_priority is not None and self.priority <= old_priority:
                # Cannot rename - blocked by priority
                return False

        # Find or create the new track
        new_base_track = next(
            (t for t in labels.tracks if t.name == new_track_name), None
        )
        if new_base_track is None:
            new_base_track = sio.Track(name=new_track_name)
            labels.tracks.append(new_base_track)

        # Apply the rename to all frames
        source_frame = tracklet_frames[0][0]  # First frame as source
        for frame_idx, inst_idx in tracklet_frames:
            lf = next(
                (lf for lf in labels.labeled_frames if lf.frame_idx == frame_idx), None
            )
            if lf is None or inst_idx >= len(lf.instances):
                continue

            inst = lf.instances[inst_idx]

            # Preserve existing history if available
            existing_history = []
            if isinstance(inst.track, TrackContext):
                existing_history = inst.track.track_history.copy()

            # Create new TrackContext
            track_context = TrackContext(
                priority=self.priority,
                track=new_base_track,
                name=new_track_name,
                temporary_track=False,  # Renamed track is no longer temporary
                valid=True,
                track_history=existing_history,
            )

            # Add history entry
            is_source = frame_idx == source_frame
            track_context.add_history_entry(
                layer_name=self.name,
                old_track_name=old_track_name,
                new_track_name=new_track_name,
                frame_idx=frame_idx,
                reason=reason if is_source else f"Propagated from frame {source_frame}: {reason}",
                conflict_resolved=False,
                propagated_from_frame=None if is_source else source_frame,
            )

            inst.track = track_context

        # Remove old track from labels.tracks if it exists and is no longer used
        old_base_track = next(
            (t for t in labels.tracks if t.name == old_track_name), None
        )
        if old_base_track is not None:
            # Check if any instance still uses it
            still_used = any(
                inst.track is not None
                and hasattr(inst.track, "name")
                and inst.track.name == old_track_name
                for lf in labels.labeled_frames
                for inst in lf.instances
            )
            if not still_used:
                labels.tracks.remove(old_base_track)

        return True

    def assign_with_priority_resolution(
        self,
        labels: sio.Labels,
        frame_idx: int,
        instance_idx: int,
        new_track_name: str,
        reason: str = "Track assignment",
        explanation: Optional[Dict] = None,
        propagate_to_tracklet: bool = True,
    ) -> bool:
        """Assign a track with priority-based conflict resolution.

        This is the main method for assigning tracks with proper inter-layer
        conflict resolution. It handles two types of operations:

        1. RENAME (propagate_to_tracklet=True): If the instance belongs to a
           temporary tracklet, propagate the new identity to ALL frames of
           that tracklet. This is ALWAYS allowed for temporary tracklets.

        2. SLICE/OVERRIDE (propagate_to_tracklet=False): Assign identity only
           to this specific instance. This is subject to priority rules -
           lower priority cannot override higher priority assignments.

        Args:
            labels: SLEAP Labels object.
            frame_idx: Frame index of the instance to assign.
            instance_idx: Instance index within the frame.
            new_track_name: Track name to assign.
            reason: Human-readable reason for the assignment.
            explanation: Optional detailed explanation dict.
            propagate_to_tracklet: If True and instance belongs to a temporary
                tracklet, propagate the assignment to ALL frames of that tracklet.
                If False, this becomes a "slice" operation subject to priority.

        Returns:
            True if assignment was made, False if blocked by higher priority.
        """
        # Get the labeled frame
        lf = next(
            (lf for lf in labels.labeled_frames if lf.frame_idx == frame_idx), None
        )
        if lf is None or instance_idx >= len(lf.instances):
            return False

        inst = lf.instances[instance_idx]
        existing_track = inst.track

        # Case 1: No existing track - simple assignment
        if existing_track is None:
            return self._do_simple_assignment(
                labels, lf, inst, frame_idx, instance_idx, new_track_name, reason, explanation
            )

        # Get existing track info
        existing_name = (
            existing_track.name if hasattr(existing_track, "name") else str(existing_track)
        )
        is_temporary = False
        existing_priority = None

        if isinstance(existing_track, TrackContext):
            is_temporary = existing_track.temporary_track or existing_name.startswith(
                "tracklet"
            )
            existing_priority = existing_track.priority
        else:
            is_temporary = existing_name.startswith("tracklet")

        # Case 2: Same track - no change needed
        if existing_name == new_track_name:
            return True

        # Case 3: RENAME operation - propagate to entire tracklet
        if propagate_to_tracklet and is_temporary:
            return self.rename_track_globally(
                labels,
                old_track_name=existing_name,
                new_track_name=new_track_name,
                reason=reason,
            )

        # Case 4: SLICE/OVERRIDE operation - check priority
        if existing_priority is not None and self.priority <= existing_priority:
            # Cannot override - blocked by priority
            return False

        # Handle conflicts with target identity in this frame
        if self._has_track_name_in_frame(labels, frame_idx, new_track_name):
            # Find the instance with the conflicting track
            conflict_inst_idx = self._get_instance_with_track_name(
                labels, frame_idx, new_track_name
            )
            if conflict_inst_idx is not None and conflict_inst_idx != instance_idx:
                # Check if we can override the conflict
                conflict_inst = lf.instances[conflict_inst_idx]
                if isinstance(conflict_inst.track, TrackContext):
                    conflict_priority = conflict_inst.track.priority
                    if conflict_priority is not None and self.priority <= conflict_priority:
                        # Cannot resolve conflict
                        return False
                    # Remove conflicting assignment
                    conflict_inst.track = None

        # Proceed with assignment
        return self._do_simple_assignment(
            labels, lf, inst, frame_idx, instance_idx, new_track_name, reason, explanation,
            old_track_name=existing_name,
        )

    def _do_simple_assignment(
        self,
        labels: sio.Labels,
        lf,
        inst,
        frame_idx: int,
        instance_idx: int,
        new_track_name: str,
        reason: str,
        explanation: Optional[Dict],
        old_track_name: Optional[str] = None,
    ) -> bool:
        """Perform a simple track assignment to a single instance.

        Args:
            labels: SLEAP Labels object.
            lf: LabeledFrame containing the instance.
            inst: Instance to assign track to.
            frame_idx: Frame index.
            instance_idx: Instance index.
            new_track_name: Track name to assign.
            reason: Reason for assignment.
            explanation: Optional explanation dict.
            old_track_name: Previous track name if any.

        Returns:
            True (assignment always succeeds at this point).
        """
        # Find or create the track
        new_base_track = next(
            (t for t in labels.tracks if t.name == new_track_name), None
        )
        if new_base_track is None:
            new_base_track = sio.Track(name=new_track_name)
            labels.tracks.append(new_base_track)

        # Get old track name if not provided
        if old_track_name is None and inst.track is not None:
            old_track_name = (
                inst.track.name if hasattr(inst.track, "name") else str(inst.track)
            )

        # Preserve existing history
        existing_history = []
        if isinstance(inst.track, TrackContext):
            existing_history = inst.track.track_history.copy()

        # Create new TrackContext
        track_context = TrackContext(
            priority=self.priority,
            track=new_base_track,
            name=new_track_name,
            temporary_track=self.temporary,
            valid=True,
            track_history=existing_history,
        )

        # Add history entry
        track_context.add_history_entry(
            layer_name=self.name,
            old_track_name=old_track_name,
            new_track_name=new_track_name,
            frame_idx=frame_idx,
            reason=reason,
            conflict_resolved=old_track_name is not None,
            propagated_from_frame=None,
            explanation=explanation,
        )

        inst.track = track_context
        return True

    def _has_track_name_in_frame(
        self, labels: sio.Labels, frame_idx: int, track_name: str
    ) -> bool:
        """Check if a track name exists in a specific frame.

        Args:
            labels: SLEAP Labels object.
            frame_idx: Frame index.
            track_name: Track name to look for.

        Returns:
            True if track exists in frame.
        """
        lf = next(
            (lf for lf in labels.labeled_frames if lf.frame_idx == frame_idx), None
        )
        if lf is None:
            return False
        for inst in lf.instances:
            if inst.track is not None:
                inst_name = (
                    inst.track.name if hasattr(inst.track, "name") else str(inst.track)
                )
                if inst_name == track_name:
                    return True
        return False

    def _get_instance_with_track_name(
        self, labels: sio.Labels, frame_idx: int, track_name: str
    ) -> Optional[int]:
        """Get the instance index that has a specific track name in a frame.

        Args:
            labels: SLEAP Labels object.
            frame_idx: Frame index.
            track_name: Track name to look for.

        Returns:
            Instance index if found, None otherwise.
        """
        lf = next(
            (lf for lf in labels.labeled_frames if lf.frame_idx == frame_idx), None
        )
        if lf is None:
            return None
        for idx, inst in enumerate(lf.instances):
            if inst.track is not None:
                inst_name = (
                    inst.track.name if hasattr(inst.track, "name") else str(inst.track)
                )
                if inst_name == track_name:
                    return idx
        return None

    def _can_override_identity(self, labels, state) -> bool:
        """Check if current layer can override the existing identity.

        This method is used for SLICE operations (breaking a tracklet into
        separate identities). For RENAME operations (propagating to entire
        tracklet), use `rename_track_globally()` instead.

        Implements: "Priority of current layer > Priority of Identity Y OR Identity Y is temporary?"

        Returns:
            True if we can override (proceed with change)
            False if we cannot override (must propagate to other frames)
        """
        # Get the TrackContext for the instance with old identity
        track_context = self.get_track_context(
            labels, 
            state.current_frame, 
            sio.Track(name=state.old_identity)
        )
        
        if track_context is None:
            # No track assigned, we can proceed
            return True
        
        # Check if it's a temporary track
        if track_context.temporary_track:
            # Temporary tracks can always be overridden
            return True
        
        # Check if they have no priority set
        if track_context.priority is None:
            # No priority means we can override
            return True
        
        # Compare priorities
        return self.priority > track_context.priority

    def _check_identity_conflict(self, labels, state) -> Optional[int]:
        """
        Check if target identity is already assigned to a different instance.
        
        Implements: "Different instance has Identity X in this frame?"
        
        Returns:
            The conflicting instance index if conflict exists, None otherwise
        """
        # Check if target identity exists in this frame
        if not self.has_track_in_frame(
            labels, 
            state.current_frame, 
            sio.Track(name=state.target_identity)
        ):
            # Target identity doesn't exist in this frame, no conflict
            return None
        
        # Find which instance has the target identity
        conflicting_instance_idx = self.get_instance_with_track(
            labels,
            state.current_frame,
            sio.Track(name=state.target_identity)
        )
        
        if conflicting_instance_idx is None:
            # Shouldn't happen, but handle gracefully
            return None
        
        if conflicting_instance_idx == state.instance_idx:
            # Same instance, no conflict
            return None
        
        # Different instance has our target identity - that's a conflict!
        return conflicting_instance_idx

    def _resolve_conflicting_assignment(self, labels, state, conflicting_instance_idx) -> bool:
        """
        Attempt to resolve conflicting identity assignment.
        
        Implements: "Conflicting X Priority < Current Layer Priority?"
        
        Args:
            conflicting_instance_idx: The instance that has our target identity
        
        Returns:
            True if conflict was resolved (identity removed from conflicting instance)
            False if conflict could not be resolved (must propagate)
        """
        # Get the TrackContext of the conflicting assignment
        conflicting_track_context = self.get_track_context(
            labels,
            state.current_frame,
            sio.Track(name=state.target_identity)
        )
        
        if conflicting_track_context is None:
            # Shouldn't happen, but if no track context, we can proceed
            return True
        
        # Check if we can override the conflicting assignment
        can_override = False
        
        # Can override if it's temporary
        if conflicting_track_context.temporary_track:
            can_override = True
        
        # Can override if they have no priority
        elif conflicting_track_context.priority is None:
            can_override = True
        
        # Can override if our priority is higher
        elif self.priority > conflicting_track_context.priority:
            can_override = True
        
        if can_override:
            # Remove the identity from the conflicting instance
            self.remove_track(labels, state.current_frame, conflicting_instance_idx)
            return True
        
        # Cannot override the conflicting assignment
        return False

    def _change_identity(self, labels, state) -> None:
        """
        Change identity from Y to X for the current instance and frame.

        Implements: "Change Identity Y → X for this frame"

        This is where the actual assignment happens.
        """
        # Get the current track context to preserve history
        current_track_context = labels[state.current_frame].instances[state.instance_idx].track

        # Determine if this is propagated
        is_propagated = state.direction != 0
        propagated_from = state.original_frame if is_propagated else None

        # Create a new TrackContext with our layer's priority
        new_track_context = TrackContext(
            priority=self.priority,
            track=sio.Track(name=state.target_identity),
            temporary_track=False,
            valid=True,
            track_history=current_track_context.track_history.copy() if current_track_context else []
        )

        # Add history entry
        new_track_context.add_history_entry(
            layer_name=self.__class__.__name__,
            old_track_name=state.old_identity,
            new_track_name=state.target_identity,
            frame_idx=state.current_frame,
            reason=self._get_change_reason(state),
            conflict_resolved=True,
            propagated_from_frame=propagated_from
        )

        # Log propagation to instance explanation store
        if self._explanation_store is not None and is_propagated:
            from sleap_mot.tracking.instance_explanations import (
                PropagationRecord,
                DecisionType,
            )

            # Determine conflict resolution method
            conflict_method = None
            if state.direction == 1:
                conflict_method = "forward_propagation"
            elif state.direction == -1:
                conflict_method = "backward_propagation"

            record = PropagationRecord(
                frame_idx=state.current_frame,
                instance_idx=state.instance_idx,
                tracker_name=self.name,
                tracker_priority=self.priority,
                decision_type=DecisionType.PROPAGATED,
                assigned_track_id=state.target_identity,
                previous_track_id=state.old_identity,
                summary=f"Identity propagated from frame {state.original_frame}",
                reasons=[self._get_change_reason(state)],
                source_frame_idx=state.original_frame,
                propagation_direction=state.direction,
                conflict_existed=True,
                conflict_resolution_method=conflict_method,
                propagated_track_id=state.target_identity,
            )
            self._explanation_store.add(record)

        # Assign it to the instance
        self.assign_track(
            labels,
            state.current_frame,
            state.instance_idx,
            sio.Track(name=state.target_identity),
            new_track_context
        )

    def _get_change_reason(self, state: ConflictResolutionState) -> str:
        """
        Generate human-readable reason for identity change.
        
        Args:
            state: Current conflict resolution state
        
        Returns:
            Reason string explaining the change
        """
        if state.direction == 0:
            return f"Override priority conflict: {self.priority} > previous priority"
        elif state.direction == 1:
            return f"Forward propagation from frame {state.original_frame}"
        elif state.direction == -1:
            return f"Backward propagation from frame {state.original_frame}"
        else:
            return "Identity assignment"

    def _advance_propagation_direction(self, state) -> bool:
        """
        Advance the propagation direction through state machine.
        
        State transitions: NONE → FORWARD → BACKWARD → DONE
        
        Implements: "Direction = none?" → "Set Forward" → "Set Backward"
        
        Args:
            state: Modified in place to advance direction
        
        Returns:
            True if direction was advanced (more directions to try)
            False if already tried all directions (propagation complete/failed)
        """
        if state.direction == 0:
            # First time, start with FORWARD
            state.direction = 1
            return True
        
        elif state.direction == 1:
            # Already tried forward, now try BACKWARD
            state.direction = -1
            return True
        
        else:  # state.direction == PropagationDirection.BACKWARD
            # Already tried both directions, we're done
            return False

    def _should_continue_propagation(self, state) -> bool:
        """
        Check if propagation should continue in current direction.

        Implements: "Direction = none?" check after changing identity

        Returns:
            True if we should continue propagating (FORWARD or BACKWARD)
            False if propagation is complete (NONE)
        """
        # If direction is still set (FORWARD or BACKWARD), keep propagating
        # If direction is NONE, we're done
        return state.direction != 0

    def get_config(self) -> Dict[str, Any]:
        """Get tracker configuration for pipeline metadata.

        Subclasses should override to include their specific parameters.
        This enables reconstruction of tracking pipelines from saved SLPT files.

        Returns:
            Dict of configuration parameters that can be used to recreate
            this tracker with the same settings.
        """
        return {
            "priority": self.priority,
            "name": self.name,
            "temporary": self.temporary,
        }

    def export_explanations_json(
        self,
        output_path: Union[str, Path],
    ) -> Optional[str]:
        """Export tracking explanations to JSON for debugging.

        Exports the explanation store to a JSON file with the following format:
        {
            "version": "2.0",
            "statistics": {
                "total_records": int,
                "trackers": {tracker_name: count, ...},
                "decision_types": {type: count, ...},
                "frames_with_records": int
            },
            "records": {
                "frame_idx": {
                    "instance_idx": [
                        {
                            "record_type": "DirectionalMotionDecisionRecord",
                            "frame_idx": int,
                            "instance_idx": int,
                            "tracker_name": str,
                            "tracker_priority": int,
                            "decision_type": "matched" | "new_track" | ...,
                            "assigned_track_id": str,
                            "summary": str,
                            "reasons": [str, ...],
                            "candidate_scores": [...],
                            ...
                        }
                    ]
                }
            }
        }

        Args:
            output_path: Path for the output JSON file. Can be:
                - Direct path to .json file
                - Path to .slp file (will create *_explanations.json)

        Returns:
            The actual path where the file was saved, or None if no
            explanation store is configured.

        Example:
            >>> tracker = DirectionalMotionTracker.for_tracklets(...)
            >>> store = InstanceExplanationStore()
            >>> tracker.explanation_store = store
            >>> labels = tracker.track(labels)
            >>> tracker.export_explanations_json("debug_explanations.json")
            'debug_explanations.json'

            >>> # Or with .slp path:
            >>> tracker.export_explanations_json("tracked_output.slp")
            'tracked_output_explanations.json'
        """
        if self._explanation_store is None:
            return None

        output_path = str(output_path)

        # Generate appropriate filename
        if output_path.endswith("_explanations.json"):
            json_path = output_path
        elif output_path.endswith(".slp"):
            json_path = output_path.replace(".slp", "_explanations.json")
        elif output_path.endswith(".json"):
            json_path = output_path
        else:
            json_path = output_path + "_explanations.json"

        self._explanation_store.save_json(json_path)
        return json_path

    def track_slpt(
        self,
        slpt: "SLPTFile",
        **kwargs
    ) -> "SLPTFile":
        """Track using SLPT file, preserving all metadata.

        This is the preferred method for multi-layer pipelines.
        It:
        1. Loads Labels with TrackContext objects restored
        2. Runs tracking with conflict resolution enabled
        3. Updates SLPT with new track contexts and explanations
        4. Records this layer in the pipeline metadata

        Args:
            slpt: SLPTFile object to track.
            **kwargs: Additional arguments passed to the track() method.

        Returns:
            Updated SLPTFile with tracking results.

        Example:
            >>> from sleap_mot.io.slpt import SLPTFile
            >>> slpt = SLPTFile.from_labels(labels)
            >>> slpt = tracker.track_slpt(slpt)
            >>> slpt.save("output.slpt")
        """
        from sleap_mot.io.slpt import SLPTFile

        # Get labels with TrackContext restored
        labels = slpt.to_labels()

        # Run tracking (existing method)
        labels = self.track(labels, **kwargs)

        # Update SLPT with new state
        slpt.update_from_labels(labels)

        # Add explanations if we have an explanation store
        if hasattr(self, '_explanation_store') and self._explanation_store is not None:
            slpt.merge_explanations(self._explanation_store)

        # Record this layer in pipeline
        slpt.add_layer_to_pipeline(
            name=self.name,
            class_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
            priority=self.priority,
            config=self.get_config(),
        )

        return slpt

    @abstractmethod
    def get_track_context(self, labels, frame_idx, track) -> Optional[TrackContext]:
        pass

    @abstractmethod
    def has_track_in_frame(self, labels, frame_idx, track) -> bool:
        pass

    @abstractmethod
    def get_instance_with_track(self, labels, frame_idx, track) -> Optional[int]:
        pass

    @abstractmethod
    def assign_track(self, labels, frame_idx, instance_idx, track, track_context) -> None:
        pass

    @abstractmethod
    def remove_track(self, labels, frame_idx, instance_idx) -> None:
        pass

    @abstractmethod
    def get_next_frame(self, labels, current_frame, direction) -> Optional[int]:
        pass

    @abstractmethod
    def track(self, labels: sio.Labels, priority: int, max_instances: int):
        pass