"""Processing layers for SLEAP-MOT.

This module provides post-processing tracking layers such as
deduplication and cleanup.

Classes:
    DeduplicatePose: Remove duplicate pose detections.
"""

from sleap_mot.tracking.processing.deduplicate_pose import DeduplicatePose

__all__ = [
    "DeduplicatePose",
]
