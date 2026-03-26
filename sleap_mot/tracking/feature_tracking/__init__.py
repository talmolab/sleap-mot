"""Feature tracking implementations for SLEAP-MOT.

This module provides feature-based tracking methods that use visual or
sensor features to assign identities across frames.

Classes:
    FeatureTracker: Abstract base class for feature-based trackers
    TailFeatureTracker: Tail segment pattern clustering
    RFIDFeatureTracker: RFID heatmap-based tracking
    CoordinateRFIDTracker: Coordinate-based RFID tracking
    VisualPatchTracker: Visual patch re-identification
"""

from sleap_mot.tracking.feature_tracking.base import FeatureTracker
from sleap_mot.tracking.feature_tracking.tail_tracking import TailFeatureTracker
from sleap_mot.tracking.feature_tracking.RFID_tracking import (
    RFIDFeatureTracker,
    CoordinateRFIDTracker,
)
from sleap_mot.tracking.feature_tracking.visual_patch_tracking import (
    VisualPatchTracker,
)

__all__ = [
    "FeatureTracker",
    "TailFeatureTracker",
    "RFIDFeatureTracker",
    "CoordinateRFIDTracker",
    "VisualPatchTracker",
]
