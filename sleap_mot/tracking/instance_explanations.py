"""Per-instance tracking explanation system.

This module provides a new instance-centric explanation system that stores
explanations indexed by (frame_idx, instance_idx) for maximum flexibility
and comprehensive logging of ALL tracker interactions.

The system uses tracker-specific dataclasses for maximum flexibility and
supports propagation event logging.

Standardized Explanation Structure
==================================

All tracker explanations follow a standardized structure with common fields
defined in BaseDecisionRecord, plus method-specific fields in subclasses.

Standard Fields (present in ALL explanation records):
    - frame_idx: Frame index where decision was made
    - instance_idx: Instance index within the frame
    - tracker_name: Name of the tracker that made this decision
    - tracker_priority: Priority level of the tracker
    - decision_type: Type of decision (DecisionType enum)
    - assigned_track_id: Track ID assigned (None if no assignment)
    - previous_track_id: Previous track ID if any
    - summary: Human-readable summary of the decision
    - reasons: List of reasons explaining the decision

Method-Specific Fields:
    MotionDecisionRecord (motion-based trackers):
        - thresholds: Dict of threshold name -> value pairs
        - instance_centroid: (x, y) centroid of the instance
        - candidate_scores: List of CandidateScore for all candidates
        - winning_track_id: Track ID that won the assignment
        - kde_model_used: Which KDE model was used
        - motion_probability: Motion probability from KDE
        - threshold_checks: Dict of threshold check results
        - association_score: The score used for track assignment
        - score_type: Type of score used

    DirectionalMotionDecisionRecord (extends MotionDecisionRecord):
        - facing_direction: (x, y) unit vector of facing direction
        - alignment_angle: Angle between facing and movement (degrees)
        - is_stagnant: Whether movement was below stagnant threshold
        - directional_multiplier: Multiplier applied based on direction
        - backward_rejected: Whether movement was rejected as backward

    RFIDDecisionRecord (RFID-based trackers):
        - rfid_ping_present: Whether an RFID ping was present
        - candidate_rfids: List of candidate RFID IDs with scores
        - winning_rfid_id: RFID ID that won the assignment
        - winning_probability: Probability of the winning RFID
        - is_propagation: Whether this was propagated from another frame
        - propagation_source_frame: Source frame if propagated
        - heatmap_probability: Probability from RFID heatmap
        - instance_centroid: (x, y) centroid of the instance

    RFIDNoMatchBroadcast (unmatched RFID pings):
        - rfid_id: The RFID ID that had no match
        - no_match_reason: Reason why no match was found
        - probability_at_this_instance: Probability at this instance's location
        - max_probability_found: Maximum probability found across all instances
        - rfid_unit_label: The unit label of the RFID receiver

    PropagationRecord (identity propagation events):
        - source_frame_idx: Original frame where assignment was made
        - propagation_direction: Direction of propagation (1=forward, -1=backward)
        - conflict_existed: Whether a conflict existed before propagation
        - conflict_resolution_method: How the conflict was resolved
        - propagated_track_id: Track ID that was propagated

Classes:
    DecisionType: Enum of possible decision types
    CandidateScore: Score for a single candidate track
    BaseDecisionRecord: Abstract base for all tracker decision records
    MotionDecisionRecord: For motion-based trackers
    DirectionalMotionDecisionRecord: For directional motion trackers
    RFIDDecisionRecord: For RFID-based trackers
    RFIDNoMatchBroadcast: For unmatched RFID pings
    PropagationRecord: For identity propagation events
    InstanceExplanationStore: Central store for all decision records
"""

from abc import ABC
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from pathlib import Path
import numpy as np


class DecisionType(Enum):
    """Type of tracking decision made for an instance."""
    MATCHED = "matched"
    NEW_TRACK = "new_track"
    NO_MATCH = "no_match"
    SKIPPED = "skipped"
    PROPAGATED = "propagated"


@dataclass
class CandidateScore:
    """Score for a single candidate track.

    Attributes:
        candidate_id: The track ID of the candidate.
        score: Association score for this candidate.
        passed_thresholds: Whether all threshold checks passed.
        threshold_results: Dict of threshold check results.
            Format: {name: {value, threshold, passed}}
        rejection_reason: Reason for rejection if not selected.
    """
    candidate_id: str
    score: float
    passed_thresholds: bool
    threshold_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "passed_thresholds": self.passed_thresholds,
            "threshold_results": self.threshold_results,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateScore":
        """Create from dictionary."""
        return cls(
            candidate_id=data["candidate_id"],
            score=data["score"],
            passed_thresholds=data["passed_thresholds"],
            threshold_results=data.get("threshold_results", {}),
            rejection_reason=data.get("rejection_reason"),
        )


@dataclass
class BaseDecisionRecord(ABC):
    """Base for all tracker decision records.

    This is the abstract base class for all tracker-specific decision records.
    It contains the common fields that all records must have.

    Standard Fields (present in all subclasses):
        frame_idx: Frame index where this decision was made.
        instance_idx: Instance index within the frame.
        tracker_name: Name of the tracker that made this decision.
        tracker_priority: Priority level of the tracker.
        decision_type: Type of decision made (matched, new_track, etc.).
        assigned_track_id: Track ID assigned (None if no assignment).
        previous_track_id: Previous track ID if any.
        summary: Human-readable summary of the decision.
        reasons: List of reasons explaining the decision.

    Subclasses should add method-specific fields while maintaining these
    standard fields for consistency across all tracker types.
    """

    # Standard keys that are present in all decision records
    STANDARD_KEYS = [
        "frame_idx",
        "instance_idx",
        "tracker_name",
        "tracker_priority",
        "decision_type",
        "assigned_track_id",
        "previous_track_id",
        "summary",
        "reasons",
    ]

    frame_idx: int
    instance_idx: int
    tracker_name: str
    tracker_priority: int
    decision_type: DecisionType
    assigned_track_id: Optional[str]
    previous_track_id: Optional[str]
    summary: str
    reasons: List[str] = field(default_factory=list)

    def get_record_type(self) -> str:
        """Return the type name for JSON serialization."""
        return self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "record_type": self.get_record_type(),
            "frame_idx": self.frame_idx,
            "instance_idx": self.instance_idx,
            "tracker_name": self.tracker_name,
            "tracker_priority": self.tracker_priority,
            "decision_type": self.decision_type.value,
            "assigned_track_id": self.assigned_track_id,
            "previous_track_id": self.previous_track_id,
            "summary": self.summary,
            "reasons": self.reasons,
        }
        return data


@dataclass
class MotionDecisionRecord(BaseDecisionRecord):
    """Decision record for motion-based trackers.

    Extends BaseDecisionRecord with motion-specific fields including
    thresholds, candidate scores, and KDE model information.

    Standard Fields (inherited from BaseDecisionRecord):
        frame_idx, instance_idx, tracker_name, tracker_priority,
        decision_type, assigned_track_id, previous_track_id, summary, reasons

    Extended Standard Fields (common across motion trackers):
        thresholds: Dict of threshold name -> value pairs.
        instance_centroid: (x, y) centroid of the instance.
        candidate_scores: List of CandidateScore for all candidates.
        winning_track_id: Track ID that won the assignment.
        association_score: The score used for track assignment.

    Motion-Specific Fields:
        kde_model_used: Which KDE model was used (long/short/distance).
        motion_probability: Motion probability from KDE.
        threshold_checks: Dict of threshold check results.
        score_type: Type of score (e.g., "euclidean_distance", "kde_probability").
    """

    # Extended standard keys added by this record type
    EXTENDED_KEYS = [
        "thresholds",
        "instance_centroid",
        "candidate_scores",
        "winning_track_id",
        "association_score",
    ]

    # Motion-specific keys
    MOTION_SPECIFIC_KEYS = [
        "kde_model_used",
        "motion_probability",
        "threshold_checks",
        "score_type",
    ]

    thresholds: Dict[str, float] = field(default_factory=dict)
    instance_centroid: Optional[Tuple[float, float]] = None
    candidate_scores: List[CandidateScore] = field(default_factory=list)
    winning_track_id: Optional[str] = None
    kde_model_used: Optional[str] = None
    motion_probability: Optional[float] = None
    threshold_checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    association_score: Optional[float] = None
    score_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = super().to_dict()
        data.update({
            "thresholds": self.thresholds,
            "instance_centroid": list(self.instance_centroid) if self.instance_centroid else None,
            "candidate_scores": [cs.to_dict() for cs in self.candidate_scores],
            "winning_track_id": self.winning_track_id,
            "kde_model_used": self.kde_model_used,
            "motion_probability": self.motion_probability,
            "threshold_checks": self.threshold_checks,
            "association_score": self.association_score,
            "score_type": self.score_type,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MotionDecisionRecord":
        """Create from dictionary."""
        candidate_scores = [
            CandidateScore.from_dict(cs) for cs in data.get("candidate_scores", [])
        ]
        centroid = data.get("instance_centroid")
        if centroid is not None:
            centroid = tuple(centroid)

        return cls(
            frame_idx=data["frame_idx"],
            instance_idx=data["instance_idx"],
            tracker_name=data["tracker_name"],
            tracker_priority=data["tracker_priority"],
            decision_type=DecisionType(data["decision_type"]),
            assigned_track_id=data.get("assigned_track_id"),
            previous_track_id=data.get("previous_track_id"),
            summary=data["summary"],
            reasons=data.get("reasons", []),
            thresholds=data.get("thresholds", {}),
            instance_centroid=centroid,
            candidate_scores=candidate_scores,
            winning_track_id=data.get("winning_track_id"),
            kde_model_used=data.get("kde_model_used"),
            motion_probability=data.get("motion_probability"),
            threshold_checks=data.get("threshold_checks", {}),
            association_score=data.get("association_score"),
            score_type=data.get("score_type"),
        )


@dataclass
class DirectionalMotionDecisionRecord(MotionDecisionRecord):
    """Decision record for directional motion trackers.

    Extends MotionDecisionRecord with directional-specific fields including
    facing direction, alignment angle, and directional penalties.

    Inherited Fields:
        From BaseDecisionRecord: frame_idx, instance_idx, tracker_name,
            tracker_priority, decision_type, assigned_track_id,
            previous_track_id, summary, reasons
        From MotionDecisionRecord: thresholds, instance_centroid,
            candidate_scores, winning_track_id, kde_model_used,
            motion_probability, threshold_checks, association_score, score_type

    Directional-Specific Fields:
        facing_direction: (x, y) unit vector of facing direction.
        alignment_angle: Angle between facing and movement (degrees).
        is_stagnant: Whether movement was below stagnant threshold.
        directional_multiplier: Multiplier applied based on direction.
        backward_rejected: Whether movement was rejected as backward.
    """

    # Directional-specific keys
    DIRECTIONAL_SPECIFIC_KEYS = [
        "facing_direction",
        "alignment_angle",
        "is_stagnant",
        "directional_multiplier",
        "backward_rejected",
    ]

    facing_direction: Optional[Tuple[float, float]] = None
    alignment_angle: Optional[float] = None
    is_stagnant: bool = False
    directional_multiplier: float = 1.0
    backward_rejected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = super().to_dict()
        data.update({
            "facing_direction": list(self.facing_direction) if self.facing_direction else None,
            "alignment_angle": self.alignment_angle,
            "is_stagnant": self.is_stagnant,
            "directional_multiplier": self.directional_multiplier,
            "backward_rejected": self.backward_rejected,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DirectionalMotionDecisionRecord":
        """Create from dictionary."""
        candidate_scores = [
            CandidateScore.from_dict(cs) for cs in data.get("candidate_scores", [])
        ]
        centroid = data.get("instance_centroid")
        if centroid is not None:
            centroid = tuple(centroid)
        facing = data.get("facing_direction")
        if facing is not None:
            facing = tuple(facing)

        return cls(
            frame_idx=data["frame_idx"],
            instance_idx=data["instance_idx"],
            tracker_name=data["tracker_name"],
            tracker_priority=data["tracker_priority"],
            decision_type=DecisionType(data["decision_type"]),
            assigned_track_id=data.get("assigned_track_id"),
            previous_track_id=data.get("previous_track_id"),
            summary=data["summary"],
            reasons=data.get("reasons", []),
            thresholds=data.get("thresholds", {}),
            instance_centroid=centroid,
            candidate_scores=candidate_scores,
            winning_track_id=data.get("winning_track_id"),
            kde_model_used=data.get("kde_model_used"),
            motion_probability=data.get("motion_probability"),
            threshold_checks=data.get("threshold_checks", {}),
            facing_direction=facing,
            alignment_angle=data.get("alignment_angle"),
            is_stagnant=data.get("is_stagnant", False),
            directional_multiplier=data.get("directional_multiplier", 1.0),
            backward_rejected=data.get("backward_rejected", False),
        )


@dataclass
class RFIDDecisionRecord(BaseDecisionRecord):
    """Decision record for RFID-based trackers.

    Standard Fields (inherited from BaseDecisionRecord):
        frame_idx, instance_idx, tracker_name, tracker_priority,
        decision_type, assigned_track_id, previous_track_id, summary, reasons

    Extended Standard Fields (common candidate/location info):
        instance_centroid: (x, y) centroid of the instance.
        candidate_rfids: List of CandidateScore for candidate RFIDs.
        winning_probability: Probability/score of the winning match.
        is_propagation: Whether this was propagated from another frame.
        propagation_source_frame: Source frame if propagated.

    RFID-Specific Fields:
        rfid_ping_present: Whether an RFID ping was present.
        winning_rfid_id: RFID ID that won the assignment.
        heatmap_probability: Probability from RFID heatmap.
    """

    # Extended standard keys (common across feature trackers)
    EXTENDED_KEYS = [
        "instance_centroid",
        "candidate_rfids",
        "winning_probability",
        "is_propagation",
        "propagation_source_frame",
    ]

    # RFID-specific keys
    RFID_SPECIFIC_KEYS = [
        "rfid_ping_present",
        "winning_rfid_id",
        "heatmap_probability",
    ]

    rfid_ping_present: bool = False
    candidate_rfids: List[CandidateScore] = field(default_factory=list)
    winning_rfid_id: Optional[str] = None
    winning_probability: Optional[float] = None
    is_propagation: bool = False
    propagation_source_frame: Optional[int] = None
    heatmap_probability: Optional[float] = None
    instance_centroid: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = super().to_dict()
        data.update({
            "rfid_ping_present": self.rfid_ping_present,
            "candidate_rfids": [cs.to_dict() for cs in self.candidate_rfids],
            "winning_rfid_id": self.winning_rfid_id,
            "winning_probability": self.winning_probability,
            "is_propagation": self.is_propagation,
            "propagation_source_frame": self.propagation_source_frame,
            "heatmap_probability": self.heatmap_probability,
            "instance_centroid": list(self.instance_centroid) if self.instance_centroid else None,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RFIDDecisionRecord":
        """Create from dictionary."""
        candidate_rfids = [
            CandidateScore.from_dict(cs) for cs in data.get("candidate_rfids", [])
        ]
        centroid = data.get("instance_centroid")
        if centroid is not None:
            centroid = tuple(centroid)

        return cls(
            frame_idx=data["frame_idx"],
            instance_idx=data["instance_idx"],
            tracker_name=data["tracker_name"],
            tracker_priority=data["tracker_priority"],
            decision_type=DecisionType(data["decision_type"]),
            assigned_track_id=data.get("assigned_track_id"),
            previous_track_id=data.get("previous_track_id"),
            summary=data["summary"],
            reasons=data.get("reasons", []),
            rfid_ping_present=data.get("rfid_ping_present", False),
            candidate_rfids=candidate_rfids,
            winning_rfid_id=data.get("winning_rfid_id"),
            winning_probability=data.get("winning_probability"),
            is_propagation=data.get("is_propagation", False),
            propagation_source_frame=data.get("propagation_source_frame"),
            heatmap_probability=data.get("heatmap_probability"),
            instance_centroid=centroid,
        )


@dataclass
class RFIDNoMatchBroadcast(BaseDecisionRecord):
    """Record for RFID pings that couldn't find a match.

    When an RFID ping has no match, this record is created for ALL instances
    in the frame to document why the ping didn't match any of them.

    Standard Fields (inherited from BaseDecisionRecord):
        frame_idx, instance_idx, tracker_name, tracker_priority,
        decision_type, assigned_track_id, previous_track_id, summary, reasons

    No-Match-Specific Fields:
        rfid_id: The RFID ID that had no match.
        no_match_reason: Reason why no match was found.
        probability_at_this_instance: Probability at this instance's location.
        max_probability_found: Maximum probability found across all instances.
        rfid_unit_label: The unit label of the RFID receiver.
    """

    # No-match-specific keys
    NO_MATCH_SPECIFIC_KEYS = [
        "rfid_id",
        "no_match_reason",
        "probability_at_this_instance",
        "max_probability_found",
        "rfid_unit_label",
    ]

    rfid_id: str = ""
    no_match_reason: str = ""
    probability_at_this_instance: float = 0.0
    max_probability_found: float = 0.0
    rfid_unit_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = super().to_dict()
        data.update({
            "rfid_id": self.rfid_id,
            "no_match_reason": self.no_match_reason,
            "probability_at_this_instance": self.probability_at_this_instance,
            "max_probability_found": self.max_probability_found,
            "rfid_unit_label": self.rfid_unit_label,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RFIDNoMatchBroadcast":
        """Create from dictionary."""
        return cls(
            frame_idx=data["frame_idx"],
            instance_idx=data["instance_idx"],
            tracker_name=data["tracker_name"],
            tracker_priority=data["tracker_priority"],
            decision_type=DecisionType(data["decision_type"]),
            assigned_track_id=data.get("assigned_track_id"),
            previous_track_id=data.get("previous_track_id"),
            summary=data["summary"],
            reasons=data.get("reasons", []),
            rfid_id=data.get("rfid_id", ""),
            no_match_reason=data.get("no_match_reason", ""),
            probability_at_this_instance=data.get("probability_at_this_instance", 0.0),
            max_probability_found=data.get("max_probability_found", 0.0),
            rfid_unit_label=data.get("rfid_unit_label"),
        )


@dataclass
class PropagationRecord(BaseDecisionRecord):
    """Record for identity propagation events.

    Created when an identity is propagated to another frame during
    conflict resolution.

    Standard Fields (inherited from BaseDecisionRecord):
        frame_idx, instance_idx, tracker_name, tracker_priority,
        decision_type, assigned_track_id, previous_track_id, summary, reasons

    Propagation-Specific Fields:
        source_frame_idx: Original frame where assignment was made.
        propagation_direction: Direction of propagation (1=forward, -1=backward).
        conflict_existed: Whether a conflict existed before propagation.
        conflict_resolution_method: How the conflict was resolved.
        propagated_track_id: Track ID that was propagated.
    """

    # Propagation-specific keys
    PROPAGATION_SPECIFIC_KEYS = [
        "source_frame_idx",
        "propagation_direction",
        "conflict_existed",
        "conflict_resolution_method",
        "propagated_track_id",
    ]

    source_frame_idx: int = 0
    propagation_direction: int = 0
    conflict_existed: bool = False
    conflict_resolution_method: Optional[str] = None
    propagated_track_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = super().to_dict()
        data.update({
            "source_frame_idx": self.source_frame_idx,
            "propagation_direction": self.propagation_direction,
            "conflict_existed": self.conflict_existed,
            "conflict_resolution_method": self.conflict_resolution_method,
            "propagated_track_id": self.propagated_track_id,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PropagationRecord":
        """Create from dictionary."""
        return cls(
            frame_idx=data["frame_idx"],
            instance_idx=data["instance_idx"],
            tracker_name=data["tracker_name"],
            tracker_priority=data["tracker_priority"],
            decision_type=DecisionType(data["decision_type"]),
            assigned_track_id=data.get("assigned_track_id"),
            previous_track_id=data.get("previous_track_id"),
            summary=data["summary"],
            reasons=data.get("reasons", []),
            source_frame_idx=data.get("source_frame_idx", 0),
            propagation_direction=data.get("propagation_direction", 0),
            conflict_existed=data.get("conflict_existed", False),
            conflict_resolution_method=data.get("conflict_resolution_method"),
            propagated_track_id=data.get("propagated_track_id"),
        )


@dataclass
class IdentityVote:
    """A single identity vote from a feature tracker.

    Used by feature trackers (RFID, fur color, etc.) to cast votes for
    the identity of instances/tracklets. Multiple votes are collected
    and analyzed for temporal consistency.

    Attributes:
        frame_idx: Frame where the vote was cast.
        instance_idx: Instance index in that frame.
        identity: The identity being voted for (e.g., RFID ID, color signature).
        confidence: Confidence/probability (0-1).
        source_info: Tracker-specific info (e.g., distance for RFID, color histogram for fur).
    """
    frame_idx: int
    instance_idx: int
    identity: str
    confidence: float
    source_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "frame_idx": self.frame_idx,
            "instance_idx": self.instance_idx,
            "identity": self.identity,
            "confidence": self.confidence,
            "source_info": self.source_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdentityVote":
        """Create from dictionary."""
        return cls(
            frame_idx=data["frame_idx"],
            instance_idx=data["instance_idx"],
            identity=data["identity"],
            confidence=data["confidence"],
            source_info=data.get("source_info", {}),
        )


@dataclass
class StitchDecisionRecord(BaseDecisionRecord):
    """Decision record for tracklet stitching decisions.

    Records why two tracklets were or weren't stitched together based on
    spatial/temporal continuity.

    Standard Fields (inherited from BaseDecisionRecord):
        frame_idx, instance_idx, tracker_name, tracker_priority,
        decision_type, assigned_track_id, previous_track_id, summary, reasons

    Stitch-Specific Fields:
        source_tracklet_id: The tracklet being extended (A).
        target_tracklet_id: The tracklet being merged into source (B).
        temporal_gap: Number of frames between tracklet end and start.
        spatial_distance: Distance between A's last centroid and B's first centroid.
        facing_angle_change: Change in facing direction (degrees) if available.
        source_last_centroid: Last centroid of source tracklet.
        target_first_centroid: First centroid of target tracklet.
        source_last_frame: Last frame of source tracklet.
        target_first_frame: First frame of target tracklet.
        stitch_round: Which round of iterative stitching this occurred in.
        competing_candidates: Other tracklets that could have been stitched.
    """

    # Stitch-specific keys
    STITCH_SPECIFIC_KEYS = [
        "source_tracklet_id",
        "target_tracklet_id",
        "temporal_gap",
        "spatial_distance",
        "facing_angle_change",
        "source_last_centroid",
        "target_first_centroid",
        "source_last_frame",
        "target_first_frame",
        "stitch_round",
        "competing_candidates",
    ]

    source_tracklet_id: str = ""
    target_tracklet_id: str = ""
    temporal_gap: int = 0
    spatial_distance: float = 0.0
    facing_angle_change: Optional[float] = None
    source_last_centroid: Optional[Tuple[float, float]] = None
    target_first_centroid: Optional[Tuple[float, float]] = None
    source_last_frame: int = 0
    target_first_frame: int = 0
    stitch_round: int = 1
    competing_candidates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = super().to_dict()
        data.update({
            "source_tracklet_id": self.source_tracklet_id,
            "target_tracklet_id": self.target_tracklet_id,
            "temporal_gap": self.temporal_gap,
            "spatial_distance": self.spatial_distance,
            "facing_angle_change": self.facing_angle_change,
            "source_last_centroid": list(self.source_last_centroid) if self.source_last_centroid else None,
            "target_first_centroid": list(self.target_first_centroid) if self.target_first_centroid else None,
            "source_last_frame": self.source_last_frame,
            "target_first_frame": self.target_first_frame,
            "stitch_round": self.stitch_round,
            "competing_candidates": self.competing_candidates,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StitchDecisionRecord":
        """Create from dictionary."""
        source_centroid = data.get("source_last_centroid")
        if source_centroid is not None:
            source_centroid = tuple(source_centroid)
        target_centroid = data.get("target_first_centroid")
        if target_centroid is not None:
            target_centroid = tuple(target_centroid)

        return cls(
            frame_idx=data["frame_idx"],
            instance_idx=data["instance_idx"],
            tracker_name=data["tracker_name"],
            tracker_priority=data["tracker_priority"],
            decision_type=DecisionType(data["decision_type"]),
            assigned_track_id=data.get("assigned_track_id"),
            previous_track_id=data.get("previous_track_id"),
            summary=data["summary"],
            reasons=data.get("reasons", []),
            source_tracklet_id=data.get("source_tracklet_id", ""),
            target_tracklet_id=data.get("target_tracklet_id", ""),
            temporal_gap=data.get("temporal_gap", 0),
            spatial_distance=data.get("spatial_distance", 0.0),
            facing_angle_change=data.get("facing_angle_change"),
            source_last_centroid=source_centroid,
            target_first_centroid=target_centroid,
            source_last_frame=data.get("source_last_frame", 0),
            target_first_frame=data.get("target_first_frame", 0),
            stitch_round=data.get("stitch_round", 1),
            competing_candidates=data.get("competing_candidates", []),
        )


@dataclass
class SwitchRepairDecisionRecord(BaseDecisionRecord):
    """Record for identity switch detection and repair decisions.

    Created when a feature tracker detects and repairs an identity switch
    within a tracklet. Documents the switch location, voting evidence,
    and repair action taken.

    Standard Fields (inherited from BaseDecisionRecord):
        frame_idx, instance_idx, tracker_name, tracker_priority,
        decision_type, assigned_track_id, previous_track_id, summary, reasons

    Switch-Specific Fields:
        switch_frame: Frame where the switch was detected.
        identity_before: Identity before the switch.
        identity_after: Identity after the switch.
        votes_before: Number of votes supporting identity_before.
        votes_after: Number of votes supporting identity_after.
        switch_confidence: Confidence in the switch detection.
        partner_tracklet: Partner tracklet if cross-validated.
        partner_validated: Whether switch was cross-validated with a partner.
        repair_action: Action taken ("split", "swap", "majority_vote", "none").
        original_tracklet: Original tracklet name before repair.
        resulting_tracks: List of track names after repair.
    """

    # Switch-specific keys
    SWITCH_SPECIFIC_KEYS = [
        "switch_frame",
        "identity_before",
        "identity_after",
        "votes_before",
        "votes_after",
        "switch_confidence",
        "partner_tracklet",
        "partner_validated",
        "repair_action",
        "original_tracklet",
        "resulting_tracks",
    ]

    switch_frame: int = 0
    identity_before: str = ""
    identity_after: str = ""
    votes_before: int = 0
    votes_after: int = 0
    switch_confidence: float = 0.0
    partner_tracklet: Optional[str] = None
    partner_validated: bool = False
    repair_action: str = "none"
    original_tracklet: str = ""
    resulting_tracks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = super().to_dict()
        data.update({
            "switch_frame": self.switch_frame,
            "identity_before": self.identity_before,
            "identity_after": self.identity_after,
            "votes_before": self.votes_before,
            "votes_after": self.votes_after,
            "switch_confidence": self.switch_confidence,
            "partner_tracklet": self.partner_tracklet,
            "partner_validated": self.partner_validated,
            "repair_action": self.repair_action,
            "original_tracklet": self.original_tracklet,
            "resulting_tracks": self.resulting_tracks,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwitchRepairDecisionRecord":
        """Create from dictionary."""
        return cls(
            frame_idx=data["frame_idx"],
            instance_idx=data["instance_idx"],
            tracker_name=data["tracker_name"],
            tracker_priority=data["tracker_priority"],
            decision_type=DecisionType(data["decision_type"]),
            assigned_track_id=data.get("assigned_track_id"),
            previous_track_id=data.get("previous_track_id"),
            summary=data["summary"],
            reasons=data.get("reasons", []),
            switch_frame=data.get("switch_frame", 0),
            identity_before=data.get("identity_before", ""),
            identity_after=data.get("identity_after", ""),
            votes_before=data.get("votes_before", 0),
            votes_after=data.get("votes_after", 0),
            switch_confidence=data.get("switch_confidence", 0.0),
            partner_tracklet=data.get("partner_tracklet"),
            partner_validated=data.get("partner_validated", False),
            repair_action=data.get("repair_action", "none"),
            original_tracklet=data.get("original_tracklet", ""),
            resulting_tracks=data.get("resulting_tracks", []),
        )


@dataclass
class TailDecisionRecord(BaseDecisionRecord):
    """Record for tail feature tracker decisions.

    Created when the TailFeatureTracker assigns or propagates an identity
    based on tail segment pattern clustering.

    Standard Fields (inherited from BaseDecisionRecord):
        frame_idx, instance_idx, tracker_name, tracker_priority,
        decision_type, assigned_track_id, previous_track_id, summary, reasons

    Tail-Specific Fields:
        cluster_id: K-means cluster assignment for this instance.
        mapping_method: Mapping method used ("trust_first", "backfill", "majority_vote").
        is_checkpoint_frame: Whether this frame was used as a checkpoint.
        barcode_valid: Whether the tail barcode extraction was valid.
        propagation_source_frame: Source frame if identity was propagated.
        proximity_frame: Frame of proximity event if propagation was used.
        instance_centroid: (x, y) centroid of the instance.
        cluster_confidence: Confidence in cluster assignment (purity).
    """

    # Tail-specific keys
    TAIL_SPECIFIC_KEYS = [
        "cluster_id",
        "mapping_method",
        "is_checkpoint_frame",
        "barcode_valid",
        "propagation_source_frame",
        "proximity_frame",
        "instance_centroid",
        "cluster_confidence",
    ]

    cluster_id: Optional[int] = None
    mapping_method: str = "trust_first"
    is_checkpoint_frame: bool = False
    barcode_valid: bool = True
    propagation_source_frame: Optional[int] = None
    proximity_frame: Optional[int] = None
    instance_centroid: Optional[Tuple[float, float]] = None
    cluster_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = super().to_dict()
        data.update({
            "cluster_id": self.cluster_id,
            "mapping_method": self.mapping_method,
            "is_checkpoint_frame": self.is_checkpoint_frame,
            "barcode_valid": self.barcode_valid,
            "propagation_source_frame": self.propagation_source_frame,
            "proximity_frame": self.proximity_frame,
            "instance_centroid": list(self.instance_centroid) if self.instance_centroid else None,
            "cluster_confidence": self.cluster_confidence,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TailDecisionRecord":
        """Create from dictionary."""
        centroid = data.get("instance_centroid")
        if centroid is not None:
            centroid = tuple(centroid)
        return cls(
            frame_idx=data["frame_idx"],
            instance_idx=data["instance_idx"],
            tracker_name=data["tracker_name"],
            tracker_priority=data["tracker_priority"],
            decision_type=DecisionType(data["decision_type"]),
            assigned_track_id=data.get("assigned_track_id"),
            previous_track_id=data.get("previous_track_id"),
            summary=data["summary"],
            reasons=data.get("reasons", []),
            cluster_id=data.get("cluster_id"),
            mapping_method=data.get("mapping_method", "trust_first"),
            is_checkpoint_frame=data.get("is_checkpoint_frame", False),
            barcode_valid=data.get("barcode_valid", True),
            propagation_source_frame=data.get("propagation_source_frame"),
            proximity_frame=data.get("proximity_frame"),
            instance_centroid=centroid,
            cluster_confidence=data.get("cluster_confidence"),
        )


# Registry for deserialization
_RECORD_TYPES: Dict[str, type] = {
    "BaseDecisionRecord": BaseDecisionRecord,
    "MotionDecisionRecord": MotionDecisionRecord,
    "DirectionalMotionDecisionRecord": DirectionalMotionDecisionRecord,
    "RFIDDecisionRecord": RFIDDecisionRecord,
    "RFIDNoMatchBroadcast": RFIDNoMatchBroadcast,
    "PropagationRecord": PropagationRecord,
    "StitchDecisionRecord": StitchDecisionRecord,
    "SwitchRepairDecisionRecord": SwitchRepairDecisionRecord,
    "TailDecisionRecord": TailDecisionRecord,
}


def _record_from_dict(data: Dict[str, Any]) -> BaseDecisionRecord:
    """Create appropriate record type from dictionary."""
    record_type = data.get("record_type", "BaseDecisionRecord")
    cls = _RECORD_TYPES.get(record_type)
    if cls is None:
        raise ValueError(f"Unknown record type: {record_type}")
    return cls.from_dict(data)


class InstanceExplanationStore:
    """Central store for per-instance tracking explanations.

    Indexed by (frame_idx, instance_idx) for efficient lookup.
    Supports adding records, querying by frame/instance/tracker,
    and JSON serialization.

    Attributes:
        _records: Dict mapping (frame_idx, instance_idx) to list of records.
        _tracker_index: Dict mapping tracker_name to list of records.

    Example:
        >>> store = InstanceExplanationStore()
        >>> record = MotionDecisionRecord(
        ...     frame_idx=0, instance_idx=0, tracker_name="MotionTracker",
        ...     tracker_priority=5, decision_type=DecisionType.MATCHED,
        ...     assigned_track_id="track_1", previous_track_id=None,
        ...     summary="Matched with score 0.85", reasons=["Best match"]
        ... )
        >>> store.add(record)
        >>> records = store.get_records(0, 0)
        >>> len(records)
        1
    """

    def __init__(self):
        """Initialize an empty explanation store."""
        # Main storage: (frame_idx, instance_idx) -> List[BaseDecisionRecord]
        self._records: Dict[Tuple[int, int], List[BaseDecisionRecord]] = {}
        # Secondary index: tracker_name -> List[BaseDecisionRecord]
        self._tracker_index: Dict[str, List[BaseDecisionRecord]] = {}

    def add(self, record: BaseDecisionRecord) -> None:
        """Add a decision record to the store.

        Args:
            record: The decision record to add.
        """
        key = (record.frame_idx, record.instance_idx)
        if key not in self._records:
            self._records[key] = []
        self._records[key].append(record)

        # Update tracker index
        tracker_name = record.tracker_name
        if tracker_name not in self._tracker_index:
            self._tracker_index[tracker_name] = []
        self._tracker_index[tracker_name].append(record)

    def get_records(
        self,
        frame_idx: int,
        instance_idx: int
    ) -> List[BaseDecisionRecord]:
        """Get all records for a specific frame and instance.

        Args:
            frame_idx: Frame index.
            instance_idx: Instance index within the frame.

        Returns:
            List of decision records for this instance, or empty list.
        """
        key = (frame_idx, instance_idx)
        return self._records.get(key, [])

    def get_records_for_frame(
        self,
        frame_idx: int
    ) -> Dict[int, List[BaseDecisionRecord]]:
        """Get all records for a specific frame.

        Args:
            frame_idx: Frame index.

        Returns:
            Dict mapping instance_idx to list of records.
        """
        result: Dict[int, List[BaseDecisionRecord]] = {}
        for (f_idx, i_idx), records in self._records.items():
            if f_idx == frame_idx:
                result[i_idx] = records
        return result

    def get_records_by_tracker(
        self,
        tracker_name: str
    ) -> List[BaseDecisionRecord]:
        """Get all records from a specific tracker.

        Args:
            tracker_name: Name of the tracker.

        Returns:
            List of all records from this tracker.
        """
        return self._tracker_index.get(tracker_name, [])

    def get_all_records(self) -> List[BaseDecisionRecord]:
        """Get all records in the store.

        Returns:
            Flat list of all records.
        """
        all_records = []
        for records in self._records.values():
            all_records.extend(records)
        return all_records

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the store.

        Returns:
            Dict with statistics including:
                - total_records: Total number of records
                - trackers: Dict of tracker_name -> count
                - decision_types: Dict of decision_type -> count
                - frames_with_records: Number of unique frames
        """
        all_records = self.get_all_records()

        # Count by tracker
        tracker_counts: Dict[str, int] = {}
        for record in all_records:
            tracker_counts[record.tracker_name] = (
                tracker_counts.get(record.tracker_name, 0) + 1
            )

        # Count by decision type
        decision_counts: Dict[str, int] = {}
        for record in all_records:
            dt = record.decision_type.value
            decision_counts[dt] = decision_counts.get(dt, 0) + 1

        # Count unique frames
        frames = set(key[0] for key in self._records.keys())

        return {
            "total_records": len(all_records),
            "trackers": tracker_counts,
            "decision_types": decision_counts,
            "frames_with_records": len(frames),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert store to dictionary for JSON serialization.

        Returns:
            Dict with version, statistics, and records.
        """
        # Organize records by frame_idx -> instance_idx
        records_by_frame: Dict[str, Dict[str, List[Dict]]] = {}
        for (frame_idx, instance_idx), records in self._records.items():
            frame_key = str(frame_idx)
            instance_key = str(instance_idx)
            if frame_key not in records_by_frame:
                records_by_frame[frame_key] = {}
            records_by_frame[frame_key][instance_key] = [
                r.to_dict() for r in records
            ]

        return {
            "version": "2.0",
            "statistics": self.get_statistics(),
            "records": records_by_frame,
        }

    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization.

        Args:
            obj: Object to convert (can be dict, list, or primitive).

        Returns:
            Object with numpy types converted to Python native types.
        """
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def save_json(self, path: Union[str, Path]) -> None:
        """Save the store to a JSON file.

        Args:
            path: Path to save the JSON file.
        """
        path = Path(path)
        data = self._convert_numpy_types(self.to_dict())
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstanceExplanationStore":
        """Create store from dictionary.

        Args:
            data: Dict with records organized by frame_idx -> instance_idx.

        Returns:
            New InstanceExplanationStore with loaded records.
        """
        store = cls()

        records_data = data.get("records", {})
        for frame_key, instances in records_data.items():
            for instance_key, records_list in instances.items():
                for record_dict in records_list:
                    record = _record_from_dict(record_dict)
                    store.add(record)

        return store

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "InstanceExplanationStore":
        """Load store from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            New InstanceExplanationStore with loaded records.
        """
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def clear(self) -> None:
        """Clear all records from the store."""
        self._records.clear()
        self._tracker_index.clear()

    def __len__(self) -> int:
        """Return total number of records."""
        return sum(len(records) for records in self._records.values())

    def __repr__(self) -> str:
        """Return string representation."""
        stats = self.get_statistics()
        return (
            f"InstanceExplanationStore("
            f"records={stats['total_records']}, "
            f"frames={stats['frames_with_records']}, "
            f"trackers={list(stats['trackers'].keys())})"
        )
