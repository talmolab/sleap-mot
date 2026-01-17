"""Online tracking module for frame-by-frame tracking methods.

This module provides the base class and implementations for online tracking
algorithms that process frames sequentially.

Classes:
    OnlineTrackingLayer: Abstract base class for online trackers
    MotionTracker: Motion-based tracker using KDE velocity prediction
    DirectionalMotionTracker: Motion tracker with pose-based directional penalties
    DirectionalKDEModel: KDE model with out-of-bounds fallback
    KDEModel: Basic KDE model wrapper

Functions:
    get_facing_direction: Get animal facing direction from pose keypoints
    directional_motion_model: Transform positions to facing-aligned coordinates
    compute_directional_alignment: Compute alignment between facing and movement
    compute_directional_multiplier: Compute score multiplier based on direction
"""

from sleap_mot.tracking.online_tracking.base import OnlineTrackingLayer
from sleap_mot.tracking.online_tracking.motion_tracker import (
    MotionTracker,
    DirectionalMotionTracker,
    KDEModel,
    DirectionalKDEModel,
    get_facing_direction,
    directional_motion_model,
    compute_directional_alignment,
    compute_directional_multiplier,
    rotate_points,
    tri_point_motion_model,
    compute_iou,
)

__all__ = [
    # Base classes
    "OnlineTrackingLayer",
    # Trackers
    "MotionTracker",
    "DirectionalMotionTracker",
    # KDE models
    "KDEModel",
    "DirectionalKDEModel",
    # Helper functions
    "get_facing_direction",
    "directional_motion_model",
    "compute_directional_alignment",
    "compute_directional_multiplier",
    "rotate_points",
    "tri_point_motion_model",
    "compute_iou",
]
