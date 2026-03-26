"""General-purpose online tracker with configurable features and scoring.

This module provides a flexible online tracking implementation that supports
multiple feature types (keypoints, centroids, bounding boxes) and scoring
methods (OKS, IoU, Euclidean distance, Mahalanobis distance).

The tracker focuses on geometric frame-by-frame tracking and integrates with
the layer-based tracking architecture for priority-based conflict resolution.
"""

from sleap_mot.tracking.online_tracking.base import OnlineTrackingLayer
from sleap_mot.tracking.base import TrackContext
from sleap_mot.tracking.instance_explanations import (
    MotionDecisionRecord,
    CandidateScore,
    DecisionType,
)
from sleap_mot.utils import get_centroid, get_bbox
import sleap_io as sio
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, Any, Union
from scipy.optimize import linear_sum_assignment


# =============================================================================
# Feature Extractors
# =============================================================================


class FeatureExtractor(ABC):
    """Abstract base class for feature extraction from instances."""

    @abstractmethod
    def extract(self, instance: sio.PredictedInstance) -> Optional[np.ndarray]:
        """Extract features from an instance.

        Args:
            instance: PredictedInstance to extract features from.

        Returns:
            Feature array, or None if extraction fails.
        """
        pass

    @property
    @abstractmethod
    def feature_shape(self) -> Tuple[int, ...]:
        """Return the expected shape of extracted features."""
        pass


class KeypointFeatureExtractor(FeatureExtractor):
    """Extract keypoint coordinates as features.

    Returns an (N, 2) array of (x, y) coordinates where N is the number
    of keypoints (or subset if keypoint_indices is specified).
    """

    def __init__(self, keypoint_indices: Optional[List[int]] = None):
        """Initialize the keypoint feature extractor.

        Args:
            keypoint_indices: Optional list of keypoint indices to use.
                If None, all keypoints are used.
        """
        self.keypoint_indices = keypoint_indices

    def extract(self, instance: sio.PredictedInstance) -> Optional[np.ndarray]:
        """Extract keypoint coordinates from an instance.

        Args:
            instance: PredictedInstance with keypoints.

        Returns:
            (N, 2) array of keypoint coordinates, or None if no valid keypoints.
        """
        pts = instance.numpy()

        if self.keypoint_indices is not None:
            # Filter to specified keypoints
            valid_indices = [i for i in self.keypoint_indices if i < pts.shape[0]]
            if not valid_indices:
                return None
            pts = pts[valid_indices]

        # Check if we have any valid (non-NaN) keypoints
        valid_mask = ~np.any(np.isnan(pts), axis=1)
        if not np.any(valid_mask):
            return None

        return pts

    @property
    def feature_shape(self) -> Tuple[int, ...]:
        """Return shape hint (actual shape depends on skeleton)."""
        return (-1, 2)  # Variable number of keypoints, 2 coordinates each


class CentroidFeatureExtractor(FeatureExtractor):
    """Extract centroid (center of mass) as feature.

    Returns a (2,) array containing the (x, y) centroid coordinates.
    """

    def extract(self, instance: sio.PredictedInstance) -> Optional[np.ndarray]:
        """Extract centroid from an instance.

        Args:
            instance: PredictedInstance with keypoints.

        Returns:
            (2,) array of centroid coordinates, or None if extraction fails.
        """
        centroid = get_centroid(instance)
        if centroid is None or np.any(np.isnan(centroid)):
            return None
        return centroid

    @property
    def feature_shape(self) -> Tuple[int, ...]:
        """Return the shape of centroid features."""
        return (2,)


class BBoxFeatureExtractor(FeatureExtractor):
    """Extract bounding box as feature.

    Returns a (4,) array containing [x_min, y_min, width, height].
    """

    def extract(self, instance: sio.PredictedInstance) -> Optional[np.ndarray]:
        """Extract bounding box from an instance.

        Args:
            instance: PredictedInstance with keypoints.

        Returns:
            (4,) array of [x_min, y_min, width, height], or None if extraction fails.
        """
        try:
            bbox = get_bbox(instance)
            if bbox is None:
                return None
            (x_min, y_min), (x_max, y_max) = bbox
            width = x_max - x_min
            height = y_max - y_min
            if width <= 0 or height <= 0:
                return None
            return np.array([x_min, y_min, width, height])
        except (ValueError, IndexError):
            return None

    @property
    def feature_shape(self) -> Tuple[int, ...]:
        """Return the shape of bounding box features."""
        return (4,)


# =============================================================================
# Scorers
# =============================================================================


class Scorer(ABC):
    """Abstract base class for computing similarity scores between features."""

    @abstractmethod
    def score(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        instance_a: Optional[sio.PredictedInstance] = None,
        instance_b: Optional[sio.PredictedInstance] = None,
    ) -> float:
        """Compute similarity score between two feature sets.

        Args:
            features_a: Features from first instance.
            features_b: Features from second instance.
            instance_a: Optional original instance for additional context.
            instance_b: Optional original instance for additional context.

        Returns:
            Similarity score in range [0, 1] where 1 = identical.
        """
        pass


class OKSScorer(Scorer):
    """Object Keypoint Similarity (OKS) scorer for pose comparison.

    OKS is the standard COCO metric for comparing pose similarity.
    It computes per-keypoint gaussian distance normalized by instance scale.
    """

    def __init__(self, sigma: float = 0.1):
        """Initialize the OKS scorer.

        Args:
            sigma: Sigma parameter controlling matching tolerance.
                Higher values are more forgiving of keypoint drift.
        """
        self.sigma = sigma

    def score(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        instance_a: Optional[sio.PredictedInstance] = None,
        instance_b: Optional[sio.PredictedInstance] = None,
    ) -> float:
        """Compute OKS between two keypoint sets.

        Args:
            features_a: (N, 2) keypoint coordinates from first instance.
            features_b: (N, 2) keypoint coordinates from second instance.
            instance_a: Optional first instance for scale computation.
            instance_b: Optional second instance for scale computation.

        Returns:
            OKS score in range [0, 1].
        """
        if features_a is None or features_b is None:
            return 0.0

        # Handle shape mismatches by using minimum common keypoints
        n_kpts = min(features_a.shape[0], features_b.shape[0])
        pts_a = features_a[:n_kpts]
        pts_b = features_b[:n_kpts]

        # Find valid (non-NaN) keypoints in both
        valid_a = ~np.any(np.isnan(pts_a), axis=1)
        valid_b = ~np.any(np.isnan(pts_b), axis=1)
        valid_both = valid_a & valid_b

        if not np.any(valid_both):
            return 0.0

        # Compute instance scale (area of bounding box)
        scale = self._compute_scale(pts_a, pts_b, valid_both)
        if scale <= 0:
            scale = 1.0  # Fallback

        # Compute per-keypoint distances
        valid_pts_a = pts_a[valid_both]
        valid_pts_b = pts_b[valid_both]
        distances = np.linalg.norm(valid_pts_a - valid_pts_b, axis=1)

        # Compute OKS: exp(-d^2 / (2 * sigma^2 * scale^2))
        oks_per_kpt = np.exp(-(distances ** 2) / (2 * (self.sigma ** 2) * (scale ** 2)))

        # Return mean OKS across valid keypoints
        return float(np.mean(oks_per_kpt))

    def _compute_scale(
        self,
        pts_a: np.ndarray,
        pts_b: np.ndarray,
        valid_mask: np.ndarray
    ) -> float:
        """Compute scale factor from keypoint bounding boxes.

        Uses the geometric mean of the two instances' bounding box areas.
        """
        def bbox_area(pts: np.ndarray, mask: np.ndarray) -> float:
            valid_pts = pts[mask]
            if len(valid_pts) < 2:
                return 0.0
            x_range = np.ptp(valid_pts[:, 0])
            y_range = np.ptp(valid_pts[:, 1])
            return max(x_range * y_range, 1.0)

        area_a = bbox_area(pts_a, valid_mask)
        area_b = bbox_area(pts_b, valid_mask)

        if area_a <= 0 or area_b <= 0:
            return max(area_a, area_b, 1.0)

        return np.sqrt(area_a * area_b)


class IoUScorer(Scorer):
    """Intersection over Union (IoU) scorer for bounding boxes."""

    def score(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        instance_a: Optional[sio.PredictedInstance] = None,
        instance_b: Optional[sio.PredictedInstance] = None,
    ) -> float:
        """Compute IoU between two bounding boxes.

        Args:
            features_a: (4,) array [x_min, y_min, width, height].
            features_b: (4,) array [x_min, y_min, width, height].
            instance_a: Not used.
            instance_b: Not used.

        Returns:
            IoU score in range [0, 1].
        """
        if features_a is None or features_b is None:
            return 0.0

        # Extract coordinates
        x1_min, y1_min, w1, h1 = features_a
        x2_min, y2_min, w2, h2 = features_b

        x1_max = x1_min + w1
        y1_max = y1_min + h1
        x2_max = x2_min + w2
        y2_max = y2_min + h2

        # Compute intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0

        intersection = (xi_max - xi_min) * (yi_max - yi_min)

        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return float(intersection / union)


class EuclideanDistanceScorer(Scorer):
    """Euclidean distance scorer with configurable scale.

    Converts distance to similarity using: score = 1 / (1 + dist/scale)
    """

    def __init__(self, distance_scale: float = 100.0):
        """Initialize the Euclidean distance scorer.

        Args:
            distance_scale: Scale factor for distance-to-similarity conversion.
                Larger values are more tolerant of larger distances.
        """
        self.distance_scale = distance_scale

    def score(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        instance_a: Optional[sio.PredictedInstance] = None,
        instance_b: Optional[sio.PredictedInstance] = None,
    ) -> float:
        """Compute Euclidean distance-based similarity.

        Args:
            features_a: Feature vector (centroid or keypoint mean).
            features_b: Feature vector (centroid or keypoint mean).
            instance_a: Not used.
            instance_b: Not used.

        Returns:
            Similarity score in range (0, 1].
        """
        if features_a is None or features_b is None:
            return 0.0

        # For keypoints, use centroid (mean of valid keypoints)
        if features_a.ndim == 2:
            valid_a = ~np.any(np.isnan(features_a), axis=1)
            features_a = np.mean(features_a[valid_a], axis=0) if np.any(valid_a) else None
        if features_b.ndim == 2:
            valid_b = ~np.any(np.isnan(features_b), axis=1)
            features_b = np.mean(features_b[valid_b], axis=0) if np.any(valid_b) else None

        if features_a is None or features_b is None:
            return 0.0

        # For bounding boxes, use center
        if len(features_a) == 4:
            features_a = np.array([
                features_a[0] + features_a[2] / 2,
                features_a[1] + features_a[3] / 2
            ])
        if len(features_b) == 4:
            features_b = np.array([
                features_b[0] + features_b[2] / 2,
                features_b[1] + features_b[3] / 2
            ])

        distance = np.linalg.norm(features_a - features_b)
        return float(1.0 / (1.0 + distance / self.distance_scale))


class MahalanobisScorer(Scorer):
    """Mahalanobis distance scorer with covariance normalization.

    Accounts for typical movement patterns by normalizing by covariance.
    """

    def __init__(
        self,
        distance_scale: float = 100.0,
        covariance_matrix: Optional[np.ndarray] = None
    ):
        """Initialize the Mahalanobis distance scorer.

        Args:
            distance_scale: Scale factor for distance-to-similarity conversion.
            covariance_matrix: 2x2 covariance matrix for normalization.
                If None, uses identity matrix (equivalent to Euclidean).
        """
        self.distance_scale = distance_scale
        if covariance_matrix is not None:
            self.inv_cov = np.linalg.inv(covariance_matrix)
        else:
            self.inv_cov = np.eye(2)

    def score(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        instance_a: Optional[sio.PredictedInstance] = None,
        instance_b: Optional[sio.PredictedInstance] = None,
    ) -> float:
        """Compute Mahalanobis distance-based similarity.

        Args:
            features_a: Feature vector (centroid or keypoint mean).
            features_b: Feature vector (centroid or keypoint mean).
            instance_a: Not used.
            instance_b: Not used.

        Returns:
            Similarity score in range (0, 1].
        """
        if features_a is None or features_b is None:
            return 0.0

        # For keypoints, use centroid (mean of valid keypoints)
        if features_a.ndim == 2:
            valid_a = ~np.any(np.isnan(features_a), axis=1)
            features_a = np.mean(features_a[valid_a], axis=0) if np.any(valid_a) else None
        if features_b.ndim == 2:
            valid_b = ~np.any(np.isnan(features_b), axis=1)
            features_b = np.mean(features_b[valid_b], axis=0) if np.any(valid_b) else None

        if features_a is None or features_b is None:
            return 0.0

        # For bounding boxes, use center
        if len(features_a) == 4:
            features_a = np.array([
                features_a[0] + features_a[2] / 2,
                features_a[1] + features_a[3] / 2
            ])
        if len(features_b) == 4:
            features_b = np.array([
                features_b[0] + features_b[2] / 2,
                features_b[1] + features_b[3] / 2
            ])

        diff = features_a - features_b
        mahal_dist = np.sqrt(diff @ self.inv_cov @ diff)
        return float(1.0 / (1.0 + mahal_dist / self.distance_scale))


# =============================================================================
# General Online Tracker
# =============================================================================


class GeneralOnlineTracker(OnlineTrackingLayer):
    """General-purpose online tracker with configurable features and scoring.

    This tracker provides flexible geometric tracking using configurable
    feature extraction and scoring methods. It supports three feature types
    (keypoints, centroids, bounding boxes) and four scoring methods
    (OKS, IoU, Euclidean distance, Mahalanobis distance).

    The tracker integrates with the layer-based tracking architecture,
    supporting priority-based conflict resolution and tracklet generation.

    Attributes:
        feature_type: Type of features to extract ("keypoints", "centroids", "bboxes").
        scoring_method: Scoring method to use ("oks", "iou", "euclidean_dist", "mahalanobis").
        feature_extractor: FeatureExtractor instance for the selected feature type.
        scorer: Scorer instance for the selected scoring method.
        max_cost: Maximum cost threshold for valid assignment.
    """

    def __init__(
        self,
        # Feature configuration
        feature_type: str = "keypoints",
        keypoint_indices: Optional[List[int]] = None,
        # Scoring configuration
        scoring_method: str = "oks",
        oks_sigma: float = 0.1,
        distance_scale: float = 100.0,
        covariance_matrix: Optional[np.ndarray] = None,
        # Cost matrix configuration
        max_cost: Optional[float] = None,
        cost_transform: str = "one_minus",
        # Fixed track mode
        n_tracks: Optional[int] = None,
        init_frame: Optional[int] = None,
        # Tracklet mode thresholds
        min_probability_threshold: Optional[float] = None,
        proximity_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_match_distance: Optional[float] = None,
        # Tracklet propagation
        propagate_to_tracklet: bool = False,
        # Base parameters
        priority: int = 5,
        name: str = "GeneralTracker",
        max_gap: Optional[int] = None,
        matching_method: str = "hungarian",
        clear_tracks: bool = True,
        **kwargs
    ):
        """Initialize the general online tracker.

        Args:
            feature_type: Feature representation. Options:
                - "keypoints": Raw (x, y) keypoint coordinates
                - "centroids": Single centroid point
                - "bboxes": Bounding box [x, y, w, h]
            keypoint_indices: Subset of keypoints to use (only for feature_type="keypoints").
            scoring_method: Similarity metric. Options:
                - "oks": Object Keypoint Similarity (best for keypoints)
                - "iou": Intersection over Union (best for bboxes)
                - "euclidean_dist": Euclidean distance (best for centroids)
                - "mahalanobis": Mahalanobis distance (for directional movement)
            oks_sigma: Sigma parameter for OKS scoring.
            distance_scale: Scale factor for distance-based scoring.
            covariance_matrix: Covariance matrix for Mahalanobis distance.
            max_cost: Maximum cost threshold for valid assignment. None = no threshold.
            cost_transform: How to convert similarity to cost. Options:
                - "one_minus": cost = 1 - similarity
                - "negative": cost = -similarity
                - "inverse": cost = 1 / similarity
            n_tracks: Fixed number of tracks. When set, every instance is force-matched
                to one of n_tracks tracks. No new tracks are created after initialization.
            init_frame: Frame index to use for track initialization in fixed track mode.
                If None, uses first frame with >= n_tracks instances. If no such frame
                exists, uses the first frame with the most instances.
            min_probability_threshold: Minimum score to continue track (tracklet mode).
            proximity_threshold: Minimum distance to other instances (tracklet mode).
            iou_threshold: Maximum IoU with other instances (tracklet mode).
            max_match_distance: Maximum distance for valid match (tracklet mode).
            propagate_to_tracklet: If True, when an instance belonging to a
                temporary tracklet is assigned a new identity, the entire tracklet
                is renamed. Useful as a final layer to assign remaining tracklets
                to known global identities. Default False.
            priority: Priority level for conflict resolution.
            name: Layer name for history tracking.
            max_gap: Maximum frame gap for track continuation. None = infinite (tracks never expire).
            matching_method: Assignment algorithm ("hungarian" or "greedy").
            clear_tracks: Whether to clear existing tracks before tracking.
            **kwargs: Additional arguments.
        """
        # Store configuration
        self.feature_type = feature_type
        self.scoring_method = scoring_method
        self.max_cost = max_cost
        self.cost_transform = cost_transform
        self.keypoint_indices = keypoint_indices
        self.oks_sigma = oks_sigma
        self.distance_scale = distance_scale
        self.covariance_matrix = covariance_matrix
        self.n_tracks = n_tracks
        self.init_frame = init_frame
        self.propagate_to_tracklet = propagate_to_tracklet

        # Store tracklet thresholds
        self.min_probability_threshold = min_probability_threshold
        self.proximity_threshold = proximity_threshold
        self.iou_threshold = iou_threshold
        self.max_match_distance = max_match_distance

        # Initialize feature extractor
        self.feature_extractor = self._create_feature_extractor()

        # Initialize scorer
        self.scorer = self._create_scorer()

        # Handle max_gap: None means infinite (never expire tracks)
        effective_max_gap = max_gap if max_gap is not None else 999999

        # Call parent init
        super().__init__(
            priority=priority,
            name=name,
            temporary=self._determine_temporary(),
            max_gap=effective_max_gap,
            matching_method=matching_method,
            clear_tracks=clear_tracks,
            **kwargs
        )

    def _create_feature_extractor(self) -> FeatureExtractor:
        """Create the appropriate feature extractor."""
        if self.feature_type == "keypoints":
            return KeypointFeatureExtractor(keypoint_indices=self.keypoint_indices)
        elif self.feature_type == "centroids":
            return CentroidFeatureExtractor()
        elif self.feature_type == "bboxes":
            return BBoxFeatureExtractor()
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

    def _create_scorer(self) -> Scorer:
        """Create the appropriate scorer."""
        if self.scoring_method == "oks":
            return OKSScorer(sigma=self.oks_sigma)
        elif self.scoring_method == "iou":
            return IoUScorer()
        elif self.scoring_method == "euclidean_dist":
            return EuclideanDistanceScorer(distance_scale=self.distance_scale)
        elif self.scoring_method == "mahalanobis":
            return MahalanobisScorer(
                distance_scale=self.distance_scale,
                covariance_matrix=self.covariance_matrix
            )
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")

    def _determine_temporary(self) -> bool:
        """Determine if tracks should be temporary (tracklet mode)."""
        return self.is_tracklet_mode

    def _configure_thresholds(self, **kwargs) -> None:
        """Configure thresholds (already set in __init__)."""
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

    def compute_association_score(
        self,
        instance1: sio.PredictedInstance,
        instance2: sio.PredictedInstance,
        frame_gap: int = 1,
        track_history: Optional[List[Tuple[int, sio.PredictedInstance]]] = None,
        **kwargs
    ) -> float:
        """Compute association score between two instances.

        Extracts features from both instances and computes similarity
        using the configured scorer.

        Args:
            instance1: Instance from previous frame.
            instance2: Candidate instance from current frame.
            frame_gap: Number of frames between instances.
            track_history: Track history (not used in basic scoring).
            **kwargs: Additional arguments.

        Returns:
            Association score (higher = better match).
        """
        # Extract features
        features_a = self.feature_extractor.extract(instance1)
        features_b = self.feature_extractor.extract(instance2)

        if features_a is None or features_b is None:
            return 0.0

        # Compute and return score
        return self.scorer.score(
            features_a, features_b,
            instance_a=instance1, instance_b=instance2
        )

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
            track_instances: Current track history.
            candidate: Potential next instance.
            score: Association score.
            frame_idx: Current frame index.
            other_candidates: Other instances in the frame.
            **kwargs: Additional arguments.

        Returns:
            True if track should continue.
        """
        if not self.is_tracklet_mode:
            return True

        # Check score threshold
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
            candidate_bbox = self._get_bbox_for_iou(candidate)
            if candidate_bbox is not None:
                for other in other_candidates:
                    other_bbox = self._get_bbox_for_iou(other)
                    if other_bbox is not None:
                        iou = self._compute_iou(candidate_bbox, other_bbox)
                        if iou > self.iou_threshold:
                            return False

        # Check distance between previous instance and candidate
        if self.max_match_distance is not None and track_instances:
            _, prev_instance = track_instances[-1]
            prev_centroid = get_centroid(prev_instance)
            candidate_centroid = get_centroid(candidate)
            if prev_centroid is not None and candidate_centroid is not None:
                dist = np.linalg.norm(candidate_centroid - prev_centroid)
                if dist > self.max_match_distance:
                    return False

        return True

    def _get_bbox_for_iou(
        self,
        instance: sio.PredictedInstance
    ) -> Optional[Tuple[float, float, float, float]]:
        """Get bounding box coordinates for IoU computation."""
        try:
            bbox = get_bbox(instance)
            if bbox is None:
                return None
            (x_min, y_min), (x_max, y_max) = bbox
            return (x_min, y_min, x_max, y_max)
        except (ValueError, IndexError):
            return None

    def _compute_iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """Compute IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

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

    def _build_cost_matrix(
        self,
        track_candidates: Dict[str, Tuple[int, sio.PredictedInstance]],
        current_instances: List[sio.PredictedInstance],
        active_tracks: Dict[str, Dict[str, Any]],
        frame_idx: int,
        **kwargs
    ) -> np.ndarray:
        """Build cost matrix for track-instance assignment.

        Overrides parent to apply configurable cost transform.

        Args:
            track_candidates: Dict of track_id -> (last_frame, last_instance).
            current_instances: Instances in current frame.
            active_tracks: Full active tracks dict for history access.
            frame_idx: Current frame index.
            **kwargs: Additional arguments for scoring.

        Returns:
            Cost matrix of shape (num_tracks, num_instances).
        """
        track_ids = list(track_candidates.keys())
        n_tracks = len(track_ids)
        n_instances = len(current_instances)

        # Initialize with large cost (no match)
        cost_matrix = np.full((n_tracks, n_instances), 1e10)

        for t_idx, track_id in enumerate(track_ids):
            last_frame, last_inst = track_candidates[track_id]
            frame_gap = frame_idx - last_frame
            track_history = active_tracks[track_id]["instances"]

            for i_idx, inst in enumerate(current_instances):
                score = self.compute_association_score(
                    last_inst,
                    inst,
                    frame_gap=frame_gap,
                    track_history=track_history,
                    **kwargs
                )

                # Apply cost transform
                cost = self._transform_score_to_cost(score)

                # Apply max_cost threshold only if set
                if self.max_cost is None or cost <= self.max_cost:
                    cost_matrix[t_idx, i_idx] = cost

        return cost_matrix

    def _transform_score_to_cost(self, score: float) -> float:
        """Transform similarity score to cost for assignment.

        Args:
            score: Similarity score in [0, 1].

        Returns:
            Cost value (lower = better match).
        """
        if self.cost_transform == "one_minus":
            return 1.0 - score
        elif self.cost_transform == "negative":
            return -score
        elif self.cost_transform == "inverse":
            return 1.0 / max(score, 1e-10)
        else:
            return 1.0 - score  # Default to one_minus

    def _create_decision_record(
        self,
        frame_idx: int,
        instance_idx: int,
        decision_type: DecisionType,
        assigned_track_id: Optional[str],
        previous_track_id: Optional[str],
        summary: str,
        reasons: List[str],
        candidate_scores: List[CandidateScore],
        instance_centroid: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> MotionDecisionRecord:
        """Create a decision record for this tracker.

        Args:
            frame_idx: Frame index.
            instance_idx: Instance index within the frame.
            decision_type: Type of decision made.
            assigned_track_id: Track ID assigned (None if no assignment).
            previous_track_id: Previous track ID if any.
            summary: Human-readable summary.
            reasons: List of reasons explaining the decision.
            candidate_scores: List of CandidateScore for all candidates.
            instance_centroid: (x, y) centroid of the instance.
            **kwargs: Additional context.

        Returns:
            MotionDecisionRecord for this decision.
        """
        # Build threshold dict
        thresholds = {
            'feature_type': self.feature_type,
            'scoring_method': self.scoring_method,
            'max_cost': self.max_cost,
        }
        if self.max_match_distance is not None:
            thresholds['max_match_distance'] = self.max_match_distance
        if self.min_probability_threshold is not None:
            thresholds['min_probability_threshold'] = self.min_probability_threshold
        if self.proximity_threshold is not None:
            thresholds['proximity_threshold'] = self.proximity_threshold
        if self.iou_threshold is not None:
            thresholds['iou_threshold'] = self.iou_threshold

        # Get association score - use explicit value or fall back to motion_probability
        association_score = kwargs.get('association_score')
        if association_score is None:
            association_score = kwargs.get('motion_probability')
        score_type = self._get_score_type_name()

        return MotionDecisionRecord(
            frame_idx=frame_idx,
            instance_idx=instance_idx,
            tracker_name=self.name,
            tracker_priority=self.priority,
            decision_type=decision_type,
            assigned_track_id=assigned_track_id,
            previous_track_id=previous_track_id,
            summary=summary,
            reasons=reasons,
            thresholds=thresholds,
            instance_centroid=instance_centroid,
            candidate_scores=candidate_scores,
            winning_track_id=assigned_track_id,
            kde_model_used=None,
            motion_probability=kwargs.get('motion_probability'),
            threshold_checks=kwargs.get('threshold_checks', {}),
            association_score=association_score,
            score_type=score_type,
        )

    def _get_score_type_name(self) -> str:
        """Get a descriptive name for the score type based on configuration.

        Returns:
            Human-readable score type name.
        """
        score_type_map = {
            ("centroids", "euclidean_dist"): "centroid_euclidean_distance",
            ("centroids", "mahalanobis"): "centroid_mahalanobis_distance",
            ("keypoints", "oks"): "keypoint_oks",
            ("keypoints", "euclidean_dist"): "keypoint_euclidean_distance",
            ("keypoints", "mahalanobis"): "keypoint_mahalanobis_distance",
            ("bboxes", "iou"): "bbox_iou",
            ("bboxes", "euclidean_dist"): "bbox_center_euclidean_distance",
        }
        key = (self.feature_type, self.scoring_method)
        return score_type_map.get(key, f"{self.feature_type}_{self.scoring_method}")

    # =========================================================================
    # Fixed Track Mode
    # =========================================================================

    def track(
        self,
        labels: sio.Labels,
        max_instances: Optional[int] = None,
        **kwargs
    ) -> sio.Labels:
        """Main tracking method with optional fixed track mode.

        When n_tracks is set, operates in fixed track mode:
        - Initializes exactly n_tracks tracks from the first frame
        - Every instance is force-matched to one of the existing tracks
        - No new tracks are created after initialization

        Args:
            labels: SLEAP Labels object containing instances to track.
            max_instances: Maximum expected instances per frame (ignored in fixed track mode).
            **kwargs: Additional arguments passed to scoring methods.

        Returns:
            labels: The Labels object with track assignments.
        """
        if self.n_tracks is None:
            # Standard tracking mode - use parent implementation
            return super().track(labels, max_instances=max_instances, **kwargs)

        # Fixed track mode
        return self._track_fixed_n_tracks(labels, **kwargs)

    def _track_fixed_n_tracks(
        self,
        labels: sio.Labels,
        **kwargs
    ) -> sio.Labels:
        """Track with a fixed number of tracks (no new track creation).

        Every instance is force-matched to one of n_tracks tracks using
        Hungarian assignment. Tracks never expire and are never created
        after initialization.

        Args:
            labels: SLEAP Labels object containing instances to track.
            **kwargs: Additional arguments passed to scoring methods.

        Returns:
            labels: The Labels object with track assignments.
        """
        if len(labels.videos) > 1:
            raise NotImplementedError("Multiple videos are not supported.")

        # Clear existing tracks if requested
        if self.clear_tracks:
            for lf in labels.labeled_frames:
                for inst in lf.instances:
                    inst.track = None
            labels.tracks = []

        # Convert existing tracks to context objects if needed
        self.convert_tracks_to_context_objects(labels, self.priority)

        # Build frame_idx -> LabeledFrame mapping
        self._frame_idx_to_lf = {lf.frame_idx: lf for lf in labels.labeled_frames}

        # Find the initialization frame
        init_frame_idx = self._find_init_frame(labels)

        # Initialize active tracks dictionary
        # {track_id: {"instances": [(frame_idx, instance), ...], "last_frame": int}}
        active_tracks: Dict[str, Dict[str, Any]] = {}

        # Track initialization flag
        tracks_initialized = False

        # Process each labeled frame sequentially
        for lf in labels.labeled_frames:
            frame_idx = lf.frame_idx
            current_instances = list(lf.instances)

            if not current_instances:
                continue

            if not tracks_initialized:
                # Wait for the initialization frame
                if frame_idx < init_frame_idx:
                    continue

                # Initialize tracks from this frame
                self._initialize_fixed_tracks(
                    labels, frame_idx, current_instances, active_tracks
                )
                tracks_initialized = True
                continue

            # Force-match all instances to existing tracks
            self._process_frame_fixed_tracks(
                labels, lf, current_instances, active_tracks, **kwargs
            )

        # Go back and process frames before init_frame (backwards tracking)
        if init_frame_idx > 0:
            frames_before_init = [
                lf for lf in labels.labeled_frames
                if lf.frame_idx < init_frame_idx and len(lf.instances) > 0
            ]
            # Create a separate active_tracks for backwards tracking
            # Start from the init frame instances
            backward_tracks: Dict[str, Dict[str, Any]] = {}
            for track_id, track_data in active_tracks.items():
                if track_data["instances"]:
                    # Use the first instance (from init frame) as reference
                    backward_tracks[track_id] = {
                        "instances": [track_data["instances"][0]],
                        "last_frame": track_data["instances"][0][0]
                    }
                else:
                    backward_tracks[track_id] = {
                        "instances": [],
                        "last_frame": init_frame_idx
                    }

            for lf in reversed(frames_before_init):
                current_instances = list(lf.instances)
                self._process_frame_fixed_tracks(
                    labels, lf, current_instances, backward_tracks, **kwargs
                )
                # Update backward_tracks with new assignments for next iteration
                for track_id in backward_tracks:
                    # Keep only the most recent (earliest) instance for backwards matching
                    if backward_tracks[track_id]["instances"]:
                        earliest = min(backward_tracks[track_id]["instances"], key=lambda x: x[0])
                        backward_tracks[track_id]["instances"] = [earliest]
                        backward_tracks[track_id]["last_frame"] = earliest[0]

        return labels

    def _find_init_frame(self, labels: sio.Labels) -> int:
        """Find the best frame for track initialization.

        Priority:
        1. Use init_frame if explicitly specified
        2. Find first frame with >= n_tracks instances
        3. Fall back to frame with most instances

        Args:
            labels: SLEAP Labels object.

        Returns:
            Frame index to use for initialization.
        """
        if self.init_frame is not None:
            return self.init_frame

        # Find first frame with >= n_tracks instances
        best_frame = None
        best_count = 0

        for lf in labels.labeled_frames:
            n_instances = len(lf.instances)

            if n_instances >= self.n_tracks:
                # Found a frame with enough instances
                return lf.frame_idx

            if n_instances > best_count:
                best_count = n_instances
                best_frame = lf.frame_idx

        # Fall back to frame with most instances
        return best_frame if best_frame is not None else 0

    def _initialize_fixed_tracks(
        self,
        labels: sio.Labels,
        frame_idx: int,
        instances: List[sio.PredictedInstance],
        active_tracks: Dict[str, Dict[str, Any]]
    ) -> None:
        """Initialize exactly n_tracks tracks from the first frame.

        If there are fewer instances than n_tracks, creates placeholder tracks.
        If there are more instances, assigns the first n_tracks instances.

        Args:
            labels: SLEAP Labels object.
            frame_idx: Frame index.
            instances: Instances in the frame.
            active_tracks: Dict to populate with initial track data.
        """
        for i in range(self.n_tracks):
            track_id = f"track_{i + 1}"

            if i < len(instances):
                inst = instances[i]
                self._assign_track_to_instance(
                    labels, frame_idx, inst, track_id,
                    reason="Track initialization"
                )
                active_tracks[track_id] = {
                    "instances": [(frame_idx, inst)],
                    "last_frame": frame_idx
                }

                # Generate explanation record
                if self.explanation_store is not None:
                    inst_centroid = get_centroid(inst)
                    centroid_tuple = tuple(inst_centroid.tolist()) if inst_centroid is not None else None

                    record = self._create_decision_record(
                        frame_idx=frame_idx,
                        instance_idx=i,
                        decision_type=DecisionType.NEW_TRACK,
                        assigned_track_id=track_id,
                        previous_track_id=None,
                        summary=f"Initialized {track_id} from instance {i}",
                        reasons=["Track initialization", f"Init frame: {frame_idx}"],
                        candidate_scores=[CandidateScore(
                            candidate_id=track_id,
                            score=1.0,
                            passed_thresholds=True,
                            rejection_reason=None
                        )],
                        instance_centroid=centroid_tuple,
                        association_score=1.0,
                    )
                    self.explanation_store.add(record)
            else:
                # Create placeholder track (no instance yet)
                active_tracks[track_id] = {
                    "instances": [],
                    "last_frame": frame_idx
                }

    def _process_frame_fixed_tracks(
        self,
        labels: sio.Labels,
        lf: sio.LabeledFrame,
        current_instances: List[sio.PredictedInstance],
        active_tracks: Dict[str, Dict[str, Any]],
        **kwargs
    ) -> None:
        """Process a frame in fixed track mode (force-match all instances).

        Uses Hungarian assignment to optimally match instances to tracks.
        All instances are assigned to the best available track.

        Args:
            labels: SLEAP Labels object.
            lf: Current LabeledFrame.
            current_instances: Instances in the current frame.
            active_tracks: Dict of active track data.
            **kwargs: Additional arguments for scoring.
        """
        frame_idx = lf.frame_idx
        track_ids = list(active_tracks.keys())
        n_tracks = len(track_ids)
        n_instances = len(current_instances)

        # Build cost matrix and score matrix (tracks x instances)
        cost_matrix = np.full((n_tracks, n_instances), 1e10)
        score_matrix = np.full((n_tracks, n_instances), 0.0)

        for t_idx, track_id in enumerate(track_ids):
            track_data = active_tracks[track_id]

            # Get the most recent instance for this track
            if track_data["instances"]:
                last_frame, last_inst = track_data["instances"][-1]
                frame_gap = abs(frame_idx - last_frame)
            else:
                # Track has no instances yet - use centroid distance as fallback
                last_inst = None
                frame_gap = 1

            for i_idx, inst in enumerate(current_instances):
                if last_inst is not None:
                    score = self.compute_association_score(
                        last_inst,
                        inst,
                        frame_gap=frame_gap,
                        track_history=track_data["instances"],
                        **kwargs
                    )
                else:
                    # No previous instance - assign small cost to allow matching
                    score = 0.5

                cost = self._transform_score_to_cost(score)
                cost_matrix[t_idx, i_idx] = cost
                score_matrix[t_idx, i_idx] = score

        # Perform Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Build assignment map: instance_idx -> (track_id, score, is_overflow)
        assignments: Dict[int, Tuple[str, float, bool]] = {}
        for row, col in zip(row_ind, col_ind):
            track_id = track_ids[row]
            score = score_matrix[row, col]
            assignments[col] = (track_id, score, False)

        # Handle any unassigned instances (more instances than tracks)
        assigned_instances = set(assignments.keys())
        for i_idx, inst in enumerate(current_instances):
            if i_idx not in assigned_instances:
                # Find best track for this instance
                best_track = None
                best_cost = float('inf')
                best_score = 0.0
                for t_idx, track_id in enumerate(track_ids):
                    if cost_matrix[t_idx, i_idx] < best_cost:
                        best_cost = cost_matrix[t_idx, i_idx]
                        best_track = track_id
                        best_score = score_matrix[t_idx, i_idx]

                if best_track is not None:
                    assignments[i_idx] = (best_track, best_score, True)

        # Apply assignments and generate explanations
        for i_idx, inst in enumerate(current_instances):
            if i_idx not in assignments:
                continue

            track_id, score, is_overflow = assignments[i_idx]
            reason = "Force-matched to best track (overflow)" if is_overflow else "Force-matched to track"

            self._assign_track_to_instance(
                labels, frame_idx, inst, track_id,
                reason=reason,
                propagate_to_tracklet=self.propagate_to_tracklet,
            )

            active_tracks[track_id]["instances"].append((frame_idx, inst))
            active_tracks[track_id]["last_frame"] = frame_idx

            # Generate explanation record
            if self.explanation_store is not None:
                # Build candidate scores for all tracks
                candidate_scores = []
                for t_idx, tid in enumerate(track_ids):
                    candidate_scores.append(CandidateScore(
                        candidate_id=tid,
                        score=float(score_matrix[t_idx, i_idx]),
                        passed_thresholds=True,
                        rejection_reason=None if tid == track_id else "Not selected by Hungarian"
                    ))

                # Get instance centroid
                inst_centroid = get_centroid(inst)
                centroid_tuple = tuple(inst_centroid.tolist()) if inst_centroid is not None else None

                # Create decision record
                record = self._create_decision_record(
                    frame_idx=frame_idx,
                    instance_idx=i_idx,
                    decision_type=DecisionType.MATCHED,
                    assigned_track_id=track_id,
                    previous_track_id=None,
                    summary=f"Matched to {track_id} with {self._get_score_type_name()} score {score:.3f}",
                    reasons=[reason, f"{self._get_score_type_name()}: {score:.3f}", "Hungarian assignment"],
                    candidate_scores=candidate_scores,
                    instance_centroid=centroid_tuple,
                    association_score=score,
                )
                self.explanation_store.add(record)

    # =========================================================================
    # Factory Methods (Presets)
    # =========================================================================

    @classmethod
    def pose_tracker(
        cls,
        oks_sigma: float = 0.1,
        keypoint_indices: Optional[List[int]] = None,
        max_cost: float = 0.5,
        priority: int = 5,
        name: str = "PoseTracker",
        **kwargs
    ) -> "GeneralOnlineTracker":
        """Create a pose-based tracker using OKS scoring.

        Best for multi-animal pose tracking with similar appearances.

        Args:
            oks_sigma: Sigma parameter for OKS scoring.
            keypoint_indices: Subset of keypoints to use.
            max_cost: Maximum cost threshold.
            priority: Priority level.
            name: Layer name.
            **kwargs: Additional arguments.

        Returns:
            GeneralOnlineTracker configured for pose tracking.
        """
        return cls(
            feature_type="keypoints",
            scoring_method="oks",
            oks_sigma=oks_sigma,
            keypoint_indices=keypoint_indices,
            max_cost=max_cost,
            priority=priority,
            name=name,
            **kwargs
        )

    @classmethod
    def centroid_tracker(
        cls,
        distance_scale: float = 100.0,
        max_cost: float = 0.5,
        priority: int = 5,
        name: str = "CentroidTracker",
        **kwargs
    ) -> "GeneralOnlineTracker":
        """Create a centroid-based tracker using Euclidean distance.

        Best for well-separated instances where position alone suffices.

        Args:
            distance_scale: Scale factor for distance-to-similarity conversion.
            max_cost: Maximum cost threshold.
            priority: Priority level.
            name: Layer name.
            **kwargs: Additional arguments.

        Returns:
            GeneralOnlineTracker configured for centroid tracking.
        """
        return cls(
            feature_type="centroids",
            scoring_method="euclidean_dist",
            distance_scale=distance_scale,
            max_cost=max_cost,
            priority=priority,
            name=name,
            **kwargs
        )

    @classmethod
    def bbox_tracker(
        cls,
        max_cost: float = 0.5,
        priority: int = 5,
        name: str = "BBoxTracker",
        **kwargs
    ) -> "GeneralOnlineTracker":
        """Create a bounding box tracker using IoU scoring.

        Best for detection-based tracking without pose requirements.

        Args:
            max_cost: Maximum cost threshold (IoU < 0.5 = no match).
            priority: Priority level.
            name: Layer name.
            **kwargs: Additional arguments.

        Returns:
            GeneralOnlineTracker configured for bbox tracking.
        """
        return cls(
            feature_type="bboxes",
            scoring_method="iou",
            max_cost=max_cost,
            priority=priority,
            name=name,
            **kwargs
        )

    @classmethod
    def for_tracklets(
        cls,
        feature_type: str = "keypoints",
        scoring_method: str = "oks",
        min_probability_threshold: float = 0.3,
        proximity_threshold: float = 50.0,
        iou_threshold: float = 0.3,
        max_match_distance: Optional[float] = 100.0,
        priority: int = 5,
        name: str = "GeneralTrackletGenerator",
        **kwargs
    ) -> "GeneralOnlineTracker":
        """Create a tracker configured for tracklet generation.

        Tracklet mode creates short, confident track segments that can
        be refined by higher-priority layers.

        Args:
            feature_type: Feature type to use.
            scoring_method: Scoring method to use.
            min_probability_threshold: Minimum score to continue track.
            proximity_threshold: Minimum distance to other instances.
            iou_threshold: Maximum IoU with other instances.
            max_match_distance: Maximum distance for valid match.
            priority: Priority level.
            name: Layer name.
            **kwargs: Additional arguments.

        Returns:
            GeneralOnlineTracker configured for tracklet generation.
        """
        return cls(
            feature_type=feature_type,
            scoring_method=scoring_method,
            min_probability_threshold=min_probability_threshold,
            proximity_threshold=proximity_threshold,
            iou_threshold=iou_threshold,
            max_match_distance=max_match_distance,
            priority=priority,
            name=name,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get general tracker configuration.

        Returns:
            Dict of configuration parameters for this general tracker.
        """
        config = super().get_config()
        config.update({
            "feature_type": self.feature_type,
            "keypoint_indices": self.keypoint_indices,
            "scoring_method": self.scoring_method,
            "oks_sigma": self.oks_sigma,
            "distance_scale": self.distance_scale,
            "covariance_matrix": self.covariance_matrix.tolist() if self.covariance_matrix is not None else None,
            "max_cost": self.max_cost,
            "cost_transform": self.cost_transform,
            "n_tracks": self.n_tracks,
            "init_frame": self.init_frame,
            "min_probability_threshold": self.min_probability_threshold,
            "proximity_threshold": self.proximity_threshold,
            "iou_threshold": self.iou_threshold,
            "max_match_distance": self.max_match_distance,
        })
        return config
