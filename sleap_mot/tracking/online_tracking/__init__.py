"""Online tracking module for frame-by-frame tracking methods.

This module provides the base class and implementations for online tracking
algorithms that process frames sequentially.

Classes:
    OnlineTrackingLayer: Abstract base class for online trackers
    MotionTracker: Motion-based tracker using KDE velocity prediction
"""

from sleap_mot.tracking.online_tracking.base import OnlineTrackingLayer
from sleap_mot.tracking.online_tracking.motion_tracker import MotionTracker

__all__ = [
    "OnlineTrackingLayer",
    "MotionTracker",
]
