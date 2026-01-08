import sleap_io as sio
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from dataclasses import dataclass, field

@dataclass
class TrackContext:
    priority: int or None
    track: sio.Track
    name: str
    temporary_track: bool = False
    valid: bool = True
    track_history: List[Dict] = field(default_factory=list)

@dataclass
class ConflictResolutionState:
    current_frame: int
    original_frame: int
    target_identity: str  # Track name, not Track object
    old_identity: str     # Track name, not Track object
    instance_idx: int
    direction: int = 0

class TrackingLayer(ABC):
    def __init__(self, priority: int, name: str, temporary: bool = False):
        self.priority = priority
        self.name = name
        self.temporary = temporary

    def convert_tracks_to_context_objects(self, labels: sio.Labels, priority: int = None):
        for lf in labels.labeled_frames:
            for inst in lf.instances:
                inst.track = TrackContext(priority, inst.track)

    def convert_context_objects_to_tracks(self, labels: sio.Labels):
        for lf in labels.labeled_frames:
            instances = []
            for inst in lf.instances:
                if inst.track.is_valid():
                    inst.track = inst.track.track
                    instances.append(inst)
            lf.instances = instances

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

    def _can_override_identity(self, labels, state) -> bool:
        """
        Check if current layer can override the existing identity.
        
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
    def track(self, labels: sio.Labels, priority: int):
        pass