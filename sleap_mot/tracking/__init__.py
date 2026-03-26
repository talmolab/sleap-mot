"""Tracking layer implementations for SLEAP-MOT.

This module provides a priority-based tracking system where different tracking
methods can be applied at any point. Tracks are only reassigned if the new
tracker has higher priority than the existing assignment.

Classes
-------
TrackContext
    Wrapper for tracks with priority information.
TrackingLayer
    Abstract base class for all tracking layers.
ConflictResolutionState
    State object for tracking conflict resolution.
InstanceExplanationStore
    Central store for per-instance tracking explanations.
DecisionType
    Enum of possible decision types.
CandidateScore
    Score for a single candidate track.
BaseDecisionRecord
    Abstract base for all tracker decision records.
MotionDecisionRecord
    Decision record for motion-based trackers.
DirectionalMotionDecisionRecord
    Decision record for directional motion trackers.
RFIDDecisionRecord
    Decision record for RFID-based trackers.
RFIDNoMatchBroadcast
    Record for RFID pings that couldn't find a match.
PropagationRecord
    Record for identity propagation events.
StitchDecisionRecord
    Decision record for tracklet stitching decisions.
IdentityVote
    A single identity vote from a feature tracker.
SwitchRepairDecisionRecord
    Record for identity switch detection and repair decisions.
TrackletSegment
    A temporal segment of a tracklet with consistent identity.
DetectedSwitch
    A detected identity switch within a tracklet.
TrackletIdentityAnalysis
    Analysis of identity votes for a single tracklet.
"""

# Base classes
from .base import TrackContext, TrackingLayer, ConflictResolutionState

# Instance explanation system
from .instance_explanations import (
    InstanceExplanationStore,
    DecisionType,
    CandidateScore,
    BaseDecisionRecord,
    MotionDecisionRecord,
    DirectionalMotionDecisionRecord,
    RFIDDecisionRecord,
    RFIDNoMatchBroadcast,
    PropagationRecord,
    StitchDecisionRecord,
    IdentityVote,
    SwitchRepairDecisionRecord,
)

# Identity switch detection data structures
from .feature_tracking.base import (
    TrackletSegment,
    DetectedSwitch,
    TrackletIdentityAnalysis,
)

# Concrete implementations
# (Uncomment as you create each module)
# from .deduplicate import DeduplicatePose
# from .rfid_tracker import RFIDTracker
# from .fur_color_tracker import FurColorTracker

# Feature tracking implementations
from .feature_tracking.RFID_tracking import CoordinateRFIDTracker

__all__ = [
    # Base classes
    "TrackContext",
    "TrackingLayer",
    "ConflictResolutionState",
    # Instance explanation system
    "InstanceExplanationStore",
    "DecisionType",
    "CandidateScore",
    "BaseDecisionRecord",
    "MotionDecisionRecord",
    "DirectionalMotionDecisionRecord",
    "RFIDDecisionRecord",
    "RFIDNoMatchBroadcast",
    "PropagationRecord",
    "StitchDecisionRecord",
    "IdentityVote",
    "SwitchRepairDecisionRecord",
    # Identity switch detection
    "TrackletSegment",
    "DetectedSwitch",
    "TrackletIdentityAnalysis",
    # Feature tracking
    "CoordinateRFIDTracker",
    # Implementations
    # "DeduplicatePose",
    # "RFIDTracker",
    # "FurColorTracker",
]