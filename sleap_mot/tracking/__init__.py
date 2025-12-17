"""Tracking layer implementations for SLEAP-MOT.

This module provides a priority-based tracking system where different tracking
methods can be applied at any point. Tracks are only reassigned if the new
tracker has higher priority than the existing assignment.

Classes
-------
IdTrackObject
    Wrapper for tracks with priority information.
IdTrackLayer
    Abstract base class for all tracking layers.
DeduplicatePose
    Remove duplicate poses within frames.
RFIDTracker
    Track using RFID feature data.
FurColorTracker
    Track using fur color features.
"""

# Base classes
from .base import IdTrackObject, IdTrackLayer

# Concrete implementations
# (Uncomment as you create each module)
# from .deduplicate import DeduplicatePose
# from .rfid_tracker import RFIDTracker
# from .fur_color_tracker import FurColorTracker

__all__ = [
    # Base classes
    "IdTrackObject",
    "IdTrackLayer",
    # Implementations
    # "DeduplicatePose",
    # "RFIDTracker",
    # "FurColorTracker",
]