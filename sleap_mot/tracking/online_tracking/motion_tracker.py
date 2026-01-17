"""Motion-based online tracker using velocity prediction.

This module provides a motion-based tracking implementation that uses
KDE (Kernel Density Estimation) motion models to predict instance positions
and score association candidates.

The tracker supports both full tracking and tracklet generation modes,
controlled by threshold parameters.
"""

from sleap_mot.tracking.online_tracking.base import OnlineTrackingLayer
from sleap_mot.tracking.base import TrackContext
from sleap_mot.utils import get_centroid, get_bbox
import sleap_io as sio
import numpy as np
import joblib
import math
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path


def rotate_points(point: Tuple[float, float], theta: float) -> Tuple[float, float]:
    """Rotate a point by angle theta around the origin.

    Args:
        point: (x, y) coordinates
        theta: Rotation angle in radians

    Returns:
        Rotated (x, y) coordinates
    """
    x, y = point
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t
    return (x_rot, y_rot)


def tri_point_motion_model(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray
) -> Tuple[float, Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Transform three points to a normalized coordinate system.

    The tri-point motion model transforms three consecutive positions
    to a normalized coordinate system where:
    - p2 (middle point) is at the origin
    - p1 (first point) is aligned with the negative y-axis
    - p3 (third point) is in the transformed space

    This allows motion patterns to be compared regardless of absolute position.

    Args:
        p1: First point (x, y) - from frame t-2
        p2: Second point (x, y) - from frame t-1
        p3: Third point (x, y) - from frame t (candidate)

    Returns:
        Tuple of (theta, p1_hat, p2_hat, p3_hat) where:
            - theta: Rotation angle applied
            - p1_hat: Transformed first point
            - p2_hat: Transformed second point (origin)
            - p3_hat: Transformed third point
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Translate so p2 is at origin
    p2_hat = (0, 0)
    p1_hat = (x1 - x2, y1 - y2)
    p3_hat = (x3 - x2, y3 - y2)

    # Find angle to rotate p1_hat to align with negative y-axis
    theta = math.atan2(-p1_hat[0], p1_hat[1])

    # Rotate all points
    p1_hat = rotate_points(p1_hat, theta)
    p3_hat = rotate_points(p3_hat, theta)

    return theta, p1_hat, p2_hat, p3_hat


def compute_iou(bbox1: Tuple, bbox2: Tuple) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        bbox1: ((x_min, y_min), (x_max, y_max))
        bbox2: ((x_min, y_min), (x_max, y_max))

    Returns:
        IoU score between 0 and 1
    """
    (x1_min, y1_min), (x1_max, y1_max) = bbox1
    (x2_min, y2_min), (x2_max, y2_max) = bbox2

    # Compute intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0

    intersection = (xi_max - xi_min) * (yi_max - yi_min)

    # Compute union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


class KDEModel:
    """Wrapper for Kernel Density Estimation motion model.

    Provides probability lookup for motion predictions based on
    pre-trained KDE models.

    Attributes:
        kde: The sklearn KernelDensity model
        x_min, x_max, y_min, y_max: Bounds for valid predictions
        max_density: Maximum density value for normalization
    """

    def __init__(self, kde, bounds: Tuple[float, float, float, float]):
        """Initialize the KDE model wrapper.

        Args:
            kde: Pre-trained sklearn KernelDensity model
            bounds: (x_min, x_max, y_min, y_max) bounds for predictions
        """
        x_min, x_max, y_min, y_max = bounds
        self.kde = kde
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        # Create grid of points for finding max density
        x = np.linspace(self.x_min, self.x_max, 100)
        y = np.linspace(self.y_min, self.y_max, 100)
        X_grid, Y_grid = np.meshgrid(x, y)
        xy_grid = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

        # Get density values and find maximum
        density = np.exp(kde.score_samples(xy_grid))
        self.max_density = density.max()

    def get_probability(self, point: Tuple[float, float]) -> float:
        """Get probability density at a point, normalized by maximum.

        Args:
            point: (x, y) coordinates in normalized motion space

        Returns:
            Probability between 0 and 1
        """
        x, y = point

        # Check bounds
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return 0.0

        # Reshape point for KDE
        point_array = np.array(point).reshape(1, -1)

        # Get density at point
        density = np.exp(self.kde.score_samples(point_array))[0]

        # Normalize by maximum density
        prob = density / self.max_density

        return prob


class MotionTracker(OnlineTrackingLayer):
    """Motion-based online tracker using KDE velocity prediction.

    Uses KDE (Kernel Density Estimation) motion models to predict
    instance positions based on velocity and scores association
    candidates based on motion probability.

    The tracker uses a tri-point motion model:
    - Requires at least 2 previous frames for velocity estimation
    - Uses short-distance KDE for slow movements
    - Uses long-distance KDE for fast movements

    Supports two modes:
    1. Full Tracking Mode: All thresholds are None
       - Assigns all instances using optimal (Hungarian) matching
       - Generates complete tracks

    2. Tracklet Generation Mode: One or more thresholds are set
       - min_probability_threshold: Break track if motion probability too low
       - proximity_threshold: Break track if other instances too close
       - iou_threshold: Break track if IoU with other instances too high
       - max_match_distance: Break track if distance to candidate too large

    Attributes:
        long_kde: KDEModel for long-distance (fast) movements
        short_kde: KDEModel for short-distance (slow) movements
        short_distance_threshold: Distance threshold for KDE selection
        min_probability_threshold: Minimum motion probability for tracklets
        proximity_threshold: Minimum distance to other instances
        iou_threshold: Maximum IoU with other instances
        max_match_distance: Maximum distance to matched candidate instance
    """

    def __init__(
        self,
        priority: int = 5,
        name: str = "MotionTracker",
        # Motion model parameters
        long_kde_path: Optional[str] = None,
        short_kde_path: Optional[str] = None,
        short_distance_threshold: float = 80.0,
        kde_percentile: int = 95,
        kde_samples: int = 100000,
        # Tracklet mode thresholds (None = full tracking mode)
        min_probability_threshold: Optional[float] = None,
        proximity_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_match_distance: Optional[float] = None,
        # General parameters
        max_gap: int = 1,
        matching_method: str = "hungarian",
        **kwargs
    ):
        """Initialize the motion tracker.

        Args:
            priority: Priority level for conflict resolution
            name: Layer name for history tracking
            long_kde_path: Path to long-distance KDE model (joblib)
            short_kde_path: Path to short-distance KDE model (joblib)
            short_distance_threshold: Distance threshold to select KDE model (px)
            kde_percentile: Percentile for KDE bounds computation
            kde_samples: Number of samples for KDE bounds estimation
            min_probability_threshold: Minimum probability to continue track (tracklet mode)
            proximity_threshold: Minimum distance to other instances (tracklet mode)
            iou_threshold: Maximum IoU with other instances (tracklet mode)
            max_match_distance: Maximum distance (px) between consecutive matched instances (tracklet mode)
            max_gap: Maximum frame gap for track continuation
            matching_method: "hungarian" or "greedy"
            **kwargs: Additional arguments
        """
        # Store KDE parameters before calling parent __init__
        self.long_kde_path = long_kde_path
        self.short_kde_path = short_kde_path
        self.short_distance_threshold = short_distance_threshold
        self.kde_percentile = kde_percentile
        self.kde_samples = kde_samples

        # Store tracklet thresholds before calling parent __init__
        self.min_probability_threshold = min_probability_threshold
        self.proximity_threshold = proximity_threshold
        self.iou_threshold = iou_threshold
        self.max_match_distance = max_match_distance

        # Initialize KDE models
        self.long_kde: Optional[KDEModel] = None
        self.short_kde: Optional[KDEModel] = None
        self._load_kde_models()

        # Call parent init (which calls _configure_thresholds)
        super().__init__(
            priority=priority,
            name=name,
            temporary=self._determine_temporary(),
            max_gap=max_gap,
            matching_method=matching_method,
            **kwargs
        )

    def _determine_temporary(self) -> bool:
        """Determine if tracks should be temporary (tracklet mode)."""
        return self.is_tracklet_mode

    def _configure_thresholds(self, **kwargs) -> None:
        """Configure motion-specific thresholds.

        Thresholds are already set in __init__ before this is called.
        """
        pass

    @property
    def is_tracklet_mode(self) -> bool:
        """Return True if any tracklet-mode threshold is set."""
        return any([
            self.min_probability_threshold is not None,
            self.proximity_threshold is not None,
            self.iou_threshold is not None,
            self.max_match_distance is not None
        ])

    def _load_kde_models(self) -> None:
        """Load KDE models from joblib files.

        Creates KDEModel wrappers with computed bounds.
        """
        if self.long_kde_path is None or self.short_kde_path is None:
            return

        # Load raw KDE models
        long_kde_raw = joblib.load(self.long_kde_path)
        short_kde_raw = joblib.load(self.short_kde_path)

        # Compute bounds from samples
        min_percentile = 100 - self.kde_percentile
        max_percentile = self.kde_percentile

        # Long KDE bounds
        long_samples = long_kde_raw.sample(self.kde_samples)
        x_vals = long_samples[:, 0]
        y_vals = long_samples[:, 1]
        x_min, x_max = np.percentile(x_vals, [min_percentile, max_percentile])
        y_min, y_max = np.percentile(y_vals, [min_percentile, max_percentile])
        self.long_kde = KDEModel(long_kde_raw, (x_min, x_max, y_min, y_max))

        # Short KDE bounds
        short_samples = short_kde_raw.sample(self.kde_samples)
        x_vals = short_samples[:, 0]
        y_vals = short_samples[:, 1]
        x_min, x_max = np.percentile(x_vals, [min_percentile, max_percentile])
        y_min, y_max = np.percentile(y_vals, [min_percentile, max_percentile])
        self.short_kde = KDEModel(short_kde_raw, (x_min, x_max, y_min, y_max))

    def compute_association_score(
        self,
        instance1: sio.PredictedInstance,
        instance2: sio.PredictedInstance,
        frame_gap: int = 1,
        track_history: Optional[List[Tuple[int, sio.PredictedInstance]]] = None,
        **kwargs
    ) -> float:
        """Compute motion-based association score.

        Uses the tri-point motion model to predict instance position
        and scores candidates based on KDE probability.

        If track history has fewer than 2 instances (no velocity available),
        falls back to centroid distance scoring.

        Args:
            instance1: Instance from previous frame
            instance2: Candidate instance from current frame
            frame_gap: Number of frames between instances
            track_history: Full track history for velocity computation
            **kwargs: Additional arguments

        Returns:
            Association score (higher = better match)
        """
        # Get centroids
        p2 = get_centroid(instance1)  # Previous frame (t-1)
        p3 = get_centroid(instance2)  # Current frame (t)

        if p2 is None or p3 is None:
            return 0.0

        # Check if we have enough history for motion model
        if track_history is None or len(track_history) < 2:
            # Fall back to distance-based scoring (closer = higher score)
            distance = np.linalg.norm(p3 - p2)
            # Convert distance to score (inverse relationship)
            # Use a sigmoid-like function for smooth scoring
            max_dist = self.short_distance_threshold * 2
            score = max(0, 1 - (distance / max_dist))
            return score

        # Get p1 from track history (t-2)
        _, prev_instance = track_history[-2]
        p1 = get_centroid(prev_instance)

        if p1 is None:
            # Fall back to distance-based scoring
            distance = np.linalg.norm(p3 - p2)
            max_dist = self.short_distance_threshold * 2
            score = max(0, 1 - (distance / max_dist))
            return score

        # Apply tri-point motion model
        theta, p1_hat, p2_hat, p3_hat = tri_point_motion_model(p1, p2, p3)

        # Compute distance between p1 and p2 to select KDE
        dist_p1_p2 = np.linalg.norm(np.array(p2) - np.array(p1))

        # Select appropriate KDE based on movement speed
        if self.long_kde is None or self.short_kde is None:
            # No KDE models - use distance-based scoring
            distance = np.linalg.norm(p3 - p2)
            max_dist = self.short_distance_threshold * 2
            score = max(0, 1 - (distance / max_dist))
            return score

        if dist_p1_p2 >= self.short_distance_threshold:
            # Fast movement - use long-distance KDE
            prob = self.long_kde.get_probability(p3_hat)
        else:
            # Slow movement - use short-distance KDE
            prob = self.short_kde.get_probability(p3_hat)

        return prob

    def should_continue_track(
        self,
        track_instances: List[Tuple[int, sio.PredictedInstance]],
        candidate: sio.PredictedInstance,
        score: float,
        frame_idx: int,
        other_candidates: Optional[List[sio.PredictedInstance]] = None,
        **kwargs
    ) -> bool:
        """Determine if track should continue with candidate.

        In full tracking mode, always returns True.
        In tracklet mode, checks threshold conditions.

        Args:
            track_instances: Current track history
            candidate: Potential next instance
            score: Association score (motion probability)
            frame_idx: Current frame index
            other_candidates: Other instances in the frame
            **kwargs: Additional arguments

        Returns:
            True if track should continue
        """
        if not self.is_tracklet_mode:
            return True

        # Check probability threshold
        if self.min_probability_threshold is not None:
            if score < self.min_probability_threshold:
                return False

        # Check proximity to other instances
        if self.proximity_threshold is not None and other_candidates:
            candidate_centroid = get_centroid(candidate)
            if candidate_centroid is not None:
                for other in other_candidates:
                    other_centroid = get_centroid(other)
                    if other_centroid is not None:
                        dist = np.linalg.norm(candidate_centroid - other_centroid)
                        if dist < self.proximity_threshold:
                            return False

        # Check IoU with other instances
        if self.iou_threshold is not None and other_candidates:
            candidate_bbox = get_bbox(candidate)
            if candidate_bbox is not None:
                for other in other_candidates:
                    other_bbox = get_bbox(other)
                    if other_bbox is not None:
                        iou = compute_iou(candidate_bbox, other_bbox)
                        if iou > self.iou_threshold:
                            return False

        # Check distance between previous instance and candidate
        if self.max_match_distance is not None and track_instances:
            _, prev_instance = track_instances[-1]
            prev_centroid = get_centroid(prev_instance)
            candidate_centroid = get_centroid(candidate)
            if prev_centroid is not None and candidate_centroid is not None:
                dist = np.linalg.norm(candidate_centroid - prev_centroid)
                if dist >= self.max_match_distance:
                    return False

        return True

    @classmethod
    def from_kde_paths(
        cls,
        long_kde_path: str,
        short_kde_path: str,
        priority: int = 5,
        name: str = "MotionTracker",
        short_distance_threshold: float = 80.0,
        min_probability_threshold: Optional[float] = None,
        proximity_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_match_distance: Optional[float] = None,
        max_gap: int = 1,
        matching_method: str = "hungarian",
        **kwargs
    ) -> "MotionTracker":
        """Create MotionTracker from KDE model paths.

        Convenience constructor for loading pre-trained KDE models.

        Args:
            long_kde_path: Path to long-distance KDE model
            short_kde_path: Path to short-distance KDE model
            priority: Priority level
            name: Layer name
            short_distance_threshold: Distance threshold for KDE selection
            min_probability_threshold: Tracklet mode threshold
            proximity_threshold: Tracklet mode threshold
            iou_threshold: Tracklet mode threshold
            max_match_distance: Maximum distance to matched candidate (tracklet mode)
            max_gap: Maximum frame gap
            matching_method: Matching algorithm
            **kwargs: Additional arguments

        Returns:
            Configured MotionTracker instance
        """
        return cls(
            priority=priority,
            name=name,
            long_kde_path=long_kde_path,
            short_kde_path=short_kde_path,
            short_distance_threshold=short_distance_threshold,
            min_probability_threshold=min_probability_threshold,
            proximity_threshold=proximity_threshold,
            iou_threshold=iou_threshold,
            max_match_distance=max_match_distance,
            max_gap=max_gap,
            matching_method=matching_method,
            **kwargs
        )

    @classmethod
    def for_tracklets(
        cls,
        long_kde_path: str,
        short_kde_path: str,
        min_probability_threshold: float = 0.1,
        proximity_threshold: float = 50.0,
        iou_threshold: float = 0.3,
        max_match_distance: Optional[float] = None,
        priority: int = 5,
        name: str = "MotionTrackletGenerator",
        **kwargs
    ) -> "MotionTracker":
        """Create MotionTracker configured for tracklet generation.

        Convenience constructor for tracklet generation mode with
        sensible default thresholds.

        Args:
            long_kde_path: Path to long-distance KDE model
            short_kde_path: Path to short-distance KDE model
            min_probability_threshold: Minimum motion probability
            proximity_threshold: Minimum distance to other instances
            iou_threshold: Maximum IoU with other instances
            max_match_distance: Maximum distance (px) between consecutive matched instances
            priority: Priority level
            name: Layer name
            **kwargs: Additional arguments

        Returns:
            MotionTracker configured for tracklet generation
        """
        return cls(
            priority=priority,
            name=name,
            long_kde_path=long_kde_path,
            short_kde_path=short_kde_path,
            min_probability_threshold=min_probability_threshold,
            proximity_threshold=proximity_threshold,
            iou_threshold=iou_threshold,
            max_match_distance=max_match_distance,
            **kwargs
        )

    @classmethod
    def for_full_tracking(
        cls,
        long_kde_path: str,
        short_kde_path: str,
        priority: int = 5,
        name: str = "MotionTracker",
        **kwargs
    ) -> "MotionTracker":
        """Create MotionTracker configured for full tracking.

        Convenience constructor for full tracking mode (no thresholds).

        Args:
            long_kde_path: Path to long-distance KDE model
            short_kde_path: Path to short-distance KDE model
            priority: Priority level
            name: Layer name
            **kwargs: Additional arguments

        Returns:
            MotionTracker configured for full tracking
        """
        return cls(
            priority=priority,
            name=name,
            long_kde_path=long_kde_path,
            short_kde_path=short_kde_path,
            min_probability_threshold=None,
            proximity_threshold=None,
            iou_threshold=None,
            max_match_distance=None,
            **kwargs
        )
