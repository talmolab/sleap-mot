"""Online tracking module for frame-by-frame tracking methods.

This module provides the base class and implementations for online tracking
algorithms that process frames sequentially.

Classes:
    OnlineTrackingLayer: Abstract base class for online trackers
    MotionTracker: Motion-based tracker using KDE velocity prediction
    DirectionalMotionTracker: Motion tracker with pose-based directional penalties
    FacingConsistencyTracker: Motion tracker with facing direction consistency
    GeneralOnlineTracker: Configurable tracker with multiple feature/scoring options
    TrackletStitcher: Stitches fragmented tracklets based on spatial/temporal continuity
    DirectionalKDEModel: KDE model with out-of-bounds fallback
    KDEModel: Basic KDE model wrapper

Feature Extractors (for GeneralOnlineTracker):
    FeatureExtractor: Abstract base class for feature extraction
    KeypointFeatureExtractor: Extract keypoint coordinates as features
    CentroidFeatureExtractor: Extract centroid position as feature
    BBoxFeatureExtractor: Extract bounding box as feature

Scorers (for GeneralOnlineTracker):
    Scorer: Abstract base class for similarity scoring
    OKSScorer: Object Keypoint Similarity scorer
    IoUScorer: Intersection over Union scorer
    EuclideanDistanceScorer: Euclidean distance-based scorer
    MahalanobisScorer: Mahalanobis distance-based scorer

Functions:
    get_facing_direction: Get animal facing direction from pose keypoints
    directional_motion_model: Transform positions to facing-aligned coordinates
    compute_directional_alignment: Compute alignment between facing and movement
    compute_directional_multiplier: Compute score multiplier based on direction
    compute_facing_consistency: Compute facing direction consistency score
"""

from sleap_mot.tracking.online_tracking.base import OnlineTrackingLayer
from sleap_mot.tracking.online_tracking.motion_tracker import (
    MotionTracker,
    DirectionalMotionTracker,
    FacingConsistencyTracker,
    KDEModel,
    DirectionalKDEModel,
    get_facing_direction,
    directional_motion_model,
    compute_directional_alignment,
    compute_directional_multiplier,
    compute_facing_consistency,
    rotate_points,
    tri_point_motion_model,
    compute_iou,
)
from sleap_mot.tracking.online_tracking.general_tracker import (
    GeneralOnlineTracker,
    # Feature extractors
    FeatureExtractor,
    KeypointFeatureExtractor,
    CentroidFeatureExtractor,
    BBoxFeatureExtractor,
    # Scorers
    Scorer,
    OKSScorer,
    IoUScorer,
    EuclideanDistanceScorer,
    MahalanobisScorer,
)
from sleap_mot.tracking.online_tracking.tracklet_stitcher import TrackletStitcher

__all__ = [
    # Base classes
    "OnlineTrackingLayer",
    # Trackers
    "MotionTracker",
    "DirectionalMotionTracker",
    "FacingConsistencyTracker",
    "GeneralOnlineTracker",
    "TrackletStitcher",
    # KDE models
    "KDEModel",
    "DirectionalKDEModel",
    # Feature extractors
    "FeatureExtractor",
    "KeypointFeatureExtractor",
    "CentroidFeatureExtractor",
    "BBoxFeatureExtractor",
    # Scorers
    "Scorer",
    "OKSScorer",
    "IoUScorer",
    "EuclideanDistanceScorer",
    "MahalanobisScorer",
    # Helper functions
    "get_facing_direction",
    "directional_motion_model",
    "compute_directional_alignment",
    "compute_directional_multiplier",
    "compute_facing_consistency",
    "rotate_points",
    "tri_point_motion_model",
    "compute_iou",
]
