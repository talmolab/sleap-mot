"""Tail Feature Tracker for sleap-mot.

This module implements a feature tracker that uses tail segment patterns for
identity matching. It extracts features from tail nodes, clusters them using
k-means, and maps clusters to track identities.

Date: 2026-01-30
"""

from sleap_mot.tracking.feature_tracking.base import FeatureTracker
from sleap_mot.tracking.base import TrackContext
from sleap_mot.tracking.instance_explanations import (
    TailDecisionRecord,
    DecisionType,
)
from sleap_mot.utils import get_centroid

import bisect
import numpy as np
import cv2
import sleap_io as sio
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import warnings
from typing import Optional, Dict, List, Any, Tuple

warnings.filterwarnings("ignore")


class TailFeatureTracker(FeatureTracker):
    """Feature tracker using tail segment patterns for identity matching.

    This tracker extracts visual features from tail segments, clusters them
    using PCA + k-means, and maps clusters to track identities using one of
    three mapping methods: trust_first, backfill, or majority_vote.

    The tracker supports optional forward propagation based on proximity
    detection, which can correct identity switches by swapping tracks from
    the last proximity event forward to the end of the video.

    Example:
        >>> from sleap_mot.tracking.feature_tracking.tail_tracking import TailFeatureTracker
        >>> tracker = TailFeatureTracker(
        ...     tail_nodes=["Tail_0", "Tail_1", "Tail_2", "TailTip"],
        ...     mapping_method="trust_first",
        ... )
        >>> result = tracker.track(labels=labels, video=video)

    Attributes:
        tail_nodes: List of tail node names for segment extraction.
        n_clusters: Number of k-means clusters (auto-detected if None).
        min_node_distance: Minimum pixel distance between consecutive nodes.
        max_segment_angle: Maximum angle between segments in degrees (optional).
        segment_image_size: (width, height) for segment image extraction.
        n_pca_components: Number of PCA components before k-means.
        require_all_tails: If True, require all instances have valid tails.
        mapping_method: One of "trust_first", "backfill", "majority_vote".
        propagate_forward: If True, swap tracks forward from proximity event to end.
        proximity_method: Method for proximity detection.
        proximity_threshold: Distance/IOU threshold for proximity detection.
    """

    def __init__(
        self,
        tail_nodes: List[str],
        n_clusters: Optional[int] = None,
        min_node_distance: Optional[float] = None,
        max_segment_angle: Optional[float] = None,
        segment_image_size: Optional[Tuple[int, int]] = None,
        n_pca_components: int = 30,
        require_all_tails: bool = False,
        mapping_method: str = "trust_first",
        propagate_forward: bool = False,
        proximity_method: str = "pose_centroid",
        proximity_threshold: float = 150.0,
        priority: int = 10,
        name: str = "TailFeatureTracker",
        # Image processing parameters
        segment_width: int = 25,
        segment_length: int = 70,
        contrast: float = 4.5,
        brightness: float = 40.0,
        # Post-processing smoothing parameters
        smoothing_window: int = 0,
        spatial_max_jump: Optional[float] = None,
        # Classification method
        classification_method: str = "kmeans",
        knn_neighbors: int = 1,
    ):
        """Initialize the TailFeatureTracker.

        Args:
            tail_nodes: Node names for tail segments (e.g., ["Tail_0", "Tail_1", "Tail_2", "TailTip"]).
            n_clusters: Number of k-means clusters. If None, auto-detected from tracks.
            min_node_distance: Min distance between consecutive tail nodes (px).
                If None, auto-detected from pose scale.
            max_segment_angle: Max angle between segments (degrees). None = disabled.
            segment_image_size: (width, height) for segment extraction.
                If None, auto-detected from pose scale.
            n_pca_components: PCA components before k-means.
            require_all_tails: Require all instances have valid tails to process frame.
            mapping_method: One of: "trust_first", "backfill", "majority_vote".
            propagate_forward: If True, swap tracks forward from proximity event to end of video.
            proximity_method: "pose_centroid", "bbox_centroid", or "bbox_iou".
            proximity_threshold: Distance (px) or IOU threshold for switch point detection.
            priority: Priority level for conflict resolution (higher = more authoritative).
            name: Name of this tracking layer.
            segment_width: Width of extracted segment images in pixels.
            segment_length: Length of extracted segment images in pixels.
            contrast: Contrast adjustment factor for image enhancement.
            brightness: Brightness adjustment for image enhancement.
            smoothing_window: Size of the temporal majority-vote smoothing window
                (in frames). Must be odd or 0. If 0, smoothing is disabled. A window
                of 11 means 5 frames on each side vote on the identity.
            spatial_max_jump: Maximum allowed centroid displacement (in pixels)
                between consecutive frames for a track assignment to be accepted.
                Assignments that would require an instance to "teleport" farther
                than this are rejected and fall back to the previous identity.
                If None, spatial gating is disabled.
            classification_method: Classification method for assigning track
                identities. "kmeans" uses raw k-means cluster assignments.
                "knn" trains a KNN classifier on checkpoint frames and
                reclassifies all samples.
            knn_neighbors: Number of neighbors for KNN classification.
        """
        super().__init__(priority=priority, name=name)

        # Validate mapping method
        valid_methods = {"trust_first", "backfill", "majority_vote"}
        if mapping_method not in valid_methods:
            raise ValueError(
                f"Invalid mapping_method '{mapping_method}'. "
                f"Must be one of: {valid_methods}"
            )

        # Validate proximity method
        valid_proximity = {"pose_centroid", "bbox_centroid", "bbox_iou"}
        if proximity_method not in valid_proximity:
            raise ValueError(
                f"Invalid proximity_method '{proximity_method}'. "
                f"Must be one of: {valid_proximity}"
            )

        # Validate classification method
        valid_classification = {"kmeans", "knn"}
        if classification_method not in valid_classification:
            raise ValueError(
                f"Invalid classification_method '{classification_method}'. "
                f"Must be one of: {valid_classification}"
            )

        self.tail_nodes = tail_nodes
        self.n_clusters = n_clusters
        self.min_node_distance = min_node_distance
        self.max_segment_angle = max_segment_angle
        self.segment_image_size = segment_image_size
        self.n_pca_components = n_pca_components
        self.require_all_tails = require_all_tails
        self.mapping_method = mapping_method
        self.propagate_forward = propagate_forward
        self.proximity_method = proximity_method
        self.proximity_threshold = proximity_threshold

        # Image processing parameters
        self.segment_width = segment_width
        self.segment_length = segment_length
        self.contrast = contrast
        self.brightness = brightness

        # Post-processing smoothing parameters
        self.smoothing_window = smoothing_window
        self.spatial_max_jump = spatial_max_jump

        # Classification method
        self.classification_method = classification_method
        self.knn_neighbors = knn_neighbors

        # Internal state (populated during tracking)
        self._tail_node_indices: Dict[str, int] = {}
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._kmeans: Optional[KMeans] = None
        self._proximity_events: Dict[Tuple[str, str], List[int]] = {}

    # =========================================================================
    # Phase 1: Core Feature Extraction
    # =========================================================================

    def _get_tail_node_indices(self, skeleton: sio.Skeleton) -> Dict[str, int]:
        """Get skeleton indices for tail nodes.

        Args:
            skeleton: SLEAP skeleton object.

        Returns:
            Dict mapping node name to skeleton index.

        Raises:
            ValueError: If any tail node is not found in skeleton.
        """
        indices = {}
        for node_name in self.tail_nodes:
            try:
                idx = skeleton.index(node_name)
                indices[node_name] = idx
            except ValueError:
                raise ValueError(
                    f"Tail node '{node_name}' not found in skeleton. "
                    f"Available nodes: {skeleton.node_names}"
                )
        return indices

    def _adjust_contrast(self, img: np.ndarray) -> np.ndarray:
        """Apply contrast and brightness adjustment with CLAHE.

        Args:
            img: Grayscale image.

        Returns:
            Enhanced grayscale image.
        """
        img_float = img.astype(np.float32)
        mean = np.mean(img_float)
        adjusted = np.clip(
            (img_float - mean) * self.contrast + mean + self.brightness, 0, 255
        ).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(adjusted)

    def _validate_tail(
        self, instance: sio.Instance, tail_node_indices: Dict[str, int]
    ) -> Tuple[bool, Optional[List[Tuple[float, float]]]]:
        """Check if tail passes quality filters.

        Validates that:
        1. All specified tail nodes are present and have valid (non-NaN) coordinates
        2. Distance between each consecutive pair of nodes >= min_node_distance
        3. If max_segment_angle is set: angle between consecutive segments <= threshold

        Args:
            instance: SLEAP instance to validate.
            tail_node_indices: Dict mapping node names to skeleton indices.

        Returns:
            Tuple of (is_valid, keypoints) where keypoints is a list of (x, y)
            tuples if valid, or None if invalid.
        """
        angle_threshold = None
        if self.max_segment_angle is not None:
            angle_threshold = np.deg2rad(self.max_segment_angle)

        min_dist = self.min_node_distance if self.min_node_distance is not None else 5.0

        # Get all tail keypoints
        keypoints = []
        pts = instance.numpy()

        for node_name in self.tail_nodes:
            idx = tail_node_indices[node_name]
            pt = pts[idx]

            if np.isnan(pt[0]) or np.isnan(pt[1]):
                return False, None

            keypoints.append((float(pt[0]), float(pt[1])))

        # Check minimum distance between consecutive nodes
        for i in range(len(keypoints) - 1):
            p1 = np.array(keypoints[i])
            p2 = np.array(keypoints[i + 1])
            dist = np.linalg.norm(p2 - p1)
            if dist < min_dist:
                return False, None

        # Check angles between consecutive segments
        if angle_threshold is not None:
            for i in range(len(keypoints) - 2):
                p0 = np.array(keypoints[i])
                p1 = np.array(keypoints[i + 1])
                p2 = np.array(keypoints[i + 2])

                v1 = p1 - p0
                v2 = p2 - p1

                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                if norm1 == 0 or norm2 == 0:
                    return False, None

                v1 = v1 / norm1
                v2 = v2 / norm2

                dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(dot)

                if angle > angle_threshold:
                    return False, None

        return True, keypoints

    def _get_segment_image(
        self,
        img_enhanced: np.ndarray,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> np.ndarray:
        """Extract a single segment image between two keypoints.

        Args:
            img_enhanced: Contrast-enhanced grayscale image.
            p1: Start point (x, y).
            p2: End point (x, y).

        Returns:
            Rectified segment image of shape (width, length).
        """
        x1, y1 = p1
        x2, y2 = p2
        angle = np.arctan2(y2 - y1, x2 - x1)

        dx = self.segment_width / 2 * np.sin(angle)
        dy = self.segment_width / 2 * -np.cos(angle)

        src_points = np.float32(
            [
                [x1 - dx, y1 - dy],
                [x1 + dx, y1 + dy],
                [x2 + dx, y2 + dy],
                [x2 - dx, y2 - dy],
            ]
        )

        dst_points = np.float32(
            [
                [0, 0],
                [0, self.segment_width],
                [self.segment_length, self.segment_width],
                [self.segment_length, 0],
            ]
        )

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(
            img_enhanced, M, (self.segment_length, self.segment_width)
        )

    def _extract_tail_segments(
        self,
        frame_image: np.ndarray,
        keypoints: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Extract and concatenate tail segment images.

        Args:
            frame_image: Video frame (RGB or grayscale).
            keypoints: List of (x, y) tail keypoints.

        Returns:
            Concatenated segment image.
        """
        # Convert to grayscale if needed
        if len(frame_image.shape) >= 3 and frame_image.shape[-1] >= 3:
            img_gray = cv2.cvtColor(frame_image, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = frame_image

        # Enhance contrast
        img_enhanced = self._adjust_contrast(img_gray)

        # Extract segments (n_nodes - 1 segments)
        segments = []
        for i in range(len(keypoints) - 1):
            seg_img = self._get_segment_image(
                img_enhanced, keypoints[i], keypoints[i + 1]
            )
            segments.append(seg_img)

        # Concatenate horizontally
        return np.hstack(segments)

    def _compute_barcode(
        self, concatenated_img: np.ndarray, weight_power: float = 2.0
    ) -> np.ndarray:
        """Convert concatenated segment image to 1D barcode.

        Uses weighted column averaging where darker pixels have higher weight.

        Args:
            concatenated_img: Concatenated segment image.
            weight_power: Power for inverse intensity weighting.

        Returns:
            1D normalized barcode array.
        """
        smoothed = np.zeros(concatenated_img.shape[1], dtype=np.float32)

        for col in range(concatenated_img.shape[1]):
            column = concatenated_img[:, col].astype(np.float32)
            inv = np.max(column) - column + 1e-3
            weights = inv**weight_power
            weights = weights / np.sum(weights)
            smoothed[col] = np.sum(column * weights)

        # Normalize to [0, 1]
        min_val, max_val = np.min(smoothed), np.max(smoothed)
        if max_val > min_val:
            smoothed = (smoothed - min_val) / (max_val - min_val)
        else:
            smoothed = np.zeros_like(smoothed)

        return smoothed

    # =========================================================================
    # Phase 2: Clustering Pipeline
    # =========================================================================

    def _collect_checkpoint_features(
        self,
        labels: sio.Labels,
        video: sio.Video,
        tail_node_indices: Dict[str, int],
    ) -> Tuple[
        np.ndarray,  # features
        np.ndarray,  # frame_indices
        np.ndarray,  # instance_indices
        np.ndarray,  # validity_mask
        List[Dict],  # metadata for each sample
    ]:
        """Gather features from all valid frames.

        Args:
            labels: SLEAP Labels object.
            video: Video object for frame extraction.
            tail_node_indices: Dict mapping node names to skeleton indices.

        Returns:
            Tuple of:
                - features: (N, feature_dim) array of barcodes
                - frame_indices: (N,) array of frame indices
                - instance_indices: (N,) array of instance indices within frames
                - validity_mask: (N,) boolean array of valid samples
                - metadata: List of dicts with additional info per sample
        """
        features = []
        frame_indices = []
        instance_indices = []
        validity_mask = []
        metadata = []

        print(f"Collecting tail features from {len(labels.labeled_frames)} frames...")

        for lf in labels.labeled_frames:
            frame_idx = lf.frame_idx

            # Load frame image
            try:
                frame_image = video[frame_idx]
            except Exception:
                continue

            frame_valid_count = 0
            frame_data = []

            for inst_idx, inst in enumerate(lf.instances):
                is_valid, keypoints = self._validate_tail(inst, tail_node_indices)

                if is_valid and keypoints is not None:
                    # Extract features
                    concat_img = self._extract_tail_segments(frame_image, keypoints)
                    barcode = self._compute_barcode(concat_img)

                    frame_data.append(
                        {
                            "frame_idx": frame_idx,
                            "inst_idx": inst_idx,
                            "barcode": barcode,
                            "valid": True,
                            "track_name": inst.track.name
                            if inst.track is not None
                            else f"inst_{inst_idx}",
                        }
                    )
                    frame_valid_count += 1
                else:
                    frame_data.append(
                        {
                            "frame_idx": frame_idx,
                            "inst_idx": inst_idx,
                            "barcode": None,
                            "valid": False,
                            "track_name": inst.track.name
                            if inst.track is not None
                            else f"inst_{inst_idx}",
                        }
                    )

            # Apply require_all_tails filter
            n_instances = len(lf.instances)
            if self.require_all_tails and frame_valid_count < n_instances:
                # Mark all as invalid for this frame
                for data in frame_data:
                    data["valid"] = False
                    frame_valid_count = 0

            # Add to results
            for data in frame_data:
                if data["valid"]:
                    features.append(data["barcode"])
                    frame_indices.append(data["frame_idx"])
                    instance_indices.append(data["inst_idx"])
                    validity_mask.append(True)
                    metadata.append(data)

        if len(features) == 0:
            return (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                [],
            )

        return (
            np.array(features),
            np.array(frame_indices),
            np.array(instance_indices),
            np.array(validity_mask),
            metadata,
        )

    def _run_clustering(
        self, features: np.ndarray, n_clusters: int
    ) -> Tuple[np.ndarray, StandardScaler, PCA, KMeans]:
        """Run PCA + k-means clustering on features.

        Args:
            features: (N, feature_dim) array of barcodes.
            n_clusters: Number of clusters for k-means.

        Returns:
            Tuple of (cluster_labels, scaler, pca, kmeans).
        """
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # PCA
        n_comp = min(self.n_pca_components, features.shape[0], features.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)

        return cluster_labels, scaler, pca, kmeans

    def _filter_same_cluster_duplicates(
        self,
        cluster_labels: np.ndarray,
        frame_indices: np.ndarray,
        instance_indices: np.ndarray,
        metadata: List[Dict],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Filter out frames where multiple instances share the same cluster.

        Behavior depends on require_all_tails:
        - False: Remove only the duplicate instances from that frame
        - True: Discard the entire frame if any instances share a cluster

        Args:
            cluster_labels: Array of cluster assignments.
            frame_indices: Array of frame indices.
            instance_indices: Array of instance indices.
            metadata: List of metadata dicts.

        Returns:
            Filtered (cluster_labels, frame_indices, instance_indices, validity_mask, metadata).
        """
        # Group by frame
        frame_to_samples = defaultdict(list)
        for i, (frame_idx, cluster) in enumerate(zip(frame_indices, cluster_labels)):
            frame_to_samples[frame_idx].append((i, cluster))

        valid_indices = []

        for frame_idx, samples in frame_to_samples.items():
            clusters_in_frame = [s[1] for s in samples]
            unique_clusters = set(clusters_in_frame)

            if len(unique_clusters) < len(samples):
                # Some instances share clusters
                if self.require_all_tails:
                    # Discard entire frame
                    continue
                else:
                    # Keep only instances with unique clusters
                    seen_clusters = set()
                    for idx, cluster in samples:
                        if cluster not in seen_clusters:
                            valid_indices.append(idx)
                            seen_clusters.add(cluster)
            else:
                # All different clusters - keep all
                for idx, _ in samples:
                    valid_indices.append(idx)

        valid_indices = np.array(valid_indices)

        if len(valid_indices) == 0:
            return (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                [],
            )

        return (
            cluster_labels[valid_indices],
            frame_indices[valid_indices],
            instance_indices[valid_indices],
            np.ones(len(valid_indices), dtype=bool),
            [metadata[i] for i in valid_indices],
        )

    # =========================================================================
    # Phase 3: Cluster-to-Track Mapping
    # =========================================================================

    def _get_track_name_for_instance(
        self, labels: sio.Labels, frame_idx: int, instance_idx: int
    ) -> str:
        """Get the track name for a specific instance.

        Args:
            labels: SLEAP Labels object.
            frame_idx: Frame index.
            instance_idx: Instance index within frame.

        Returns:
            Track name or instance index as string if no track.
        """
        lf = self._get_labeled_frame(labels, frame_idx)
        if lf is None or instance_idx >= len(lf.instances):
            return f"inst_{instance_idx}"

        inst = lf.instances[instance_idx]
        if inst.track is None:
            return f"inst_{instance_idx}"

        if isinstance(inst.track, TrackContext):
            return inst.track.name
        return inst.track.name

    def _build_mapping_trust_first(
        self,
        cluster_labels: np.ndarray,
        frame_indices: np.ndarray,
        instance_indices: np.ndarray,
        labels: sio.Labels,
        metadata: List[Dict],
    ) -> Dict[int, str]:
        """Build cluster-to-track mapping using first valid checkpoint.

        At the first valid checkpoint frame, use the incoming track assignments
        to build the cluster-to-track mapping.

        Args:
            cluster_labels: Array of cluster assignments.
            frame_indices: Array of frame indices.
            instance_indices: Array of instance indices.
            labels: SLEAP Labels object.
            metadata: List of metadata dicts.

        Returns:
            Dict mapping cluster_id -> track_name.
        """
        # Find first complete frame (all clusters represented)
        n_clusters = len(set(cluster_labels))
        frame_to_samples = defaultdict(list)

        for i, frame_idx in enumerate(frame_indices):
            frame_to_samples[frame_idx].append(i)

        mapping = {}
        first_checkpoint_frame = None

        for frame_idx in sorted(frame_to_samples.keys()):
            sample_indices = frame_to_samples[frame_idx]
            clusters_in_frame = set(cluster_labels[sample_indices])

            if len(clusters_in_frame) == n_clusters:
                # This is a complete frame - use it for mapping
                first_checkpoint_frame = frame_idx
                for idx in sample_indices:
                    cluster = cluster_labels[idx]
                    track_name = metadata[idx]["track_name"]
                    mapping[cluster] = track_name
                break

        if first_checkpoint_frame is not None:
            print(
                f"  Trust First: Using frame {first_checkpoint_frame} for mapping"
            )
        else:
            # Fallback: use most common track for each cluster
            print("  Trust First: No complete frame found, using majority vote fallback")
            return self._build_mapping_majority_vote(
                cluster_labels, frame_indices, instance_indices, labels, metadata
            )

        return mapping

    def _build_mapping_backfill(
        self,
        cluster_labels: np.ndarray,
        frame_indices: np.ndarray,
        instance_indices: np.ndarray,
        labels: sio.Labels,
        metadata: List[Dict],
    ) -> Tuple[Dict[int, str], List[Tuple[int, int, str, str]]]:
        """Build mapping with backfill correction.

        Same as trust_first, but additionally identifies frames before the
        first checkpoint that need correction.

        Args:
            cluster_labels: Array of cluster assignments.
            frame_indices: Array of frame indices.
            instance_indices: Array of instance indices.
            labels: SLEAP Labels object.
            metadata: List of metadata dicts.

        Returns:
            Tuple of (mapping, corrections) where corrections is a list of
            (frame_idx, instance_idx, old_track, new_track) tuples.
        """
        mapping = self._build_mapping_trust_first(
            cluster_labels, frame_indices, instance_indices, labels, metadata
        )

        # Find first checkpoint frame
        n_clusters = len(set(cluster_labels))
        frame_to_samples = defaultdict(list)

        for i, frame_idx in enumerate(frame_indices):
            frame_to_samples[frame_idx].append(i)

        first_checkpoint_frame = None
        for frame_idx in sorted(frame_to_samples.keys()):
            sample_indices = frame_to_samples[frame_idx]
            clusters_in_frame = set(cluster_labels[sample_indices])
            if len(clusters_in_frame) == n_clusters:
                first_checkpoint_frame = frame_idx
                break

        corrections = []

        if first_checkpoint_frame is not None:
            # Check frames before the checkpoint
            for frame_idx in sorted(frame_to_samples.keys()):
                if frame_idx >= first_checkpoint_frame:
                    break

                sample_indices = frame_to_samples[frame_idx]
                for idx in sample_indices:
                    cluster = cluster_labels[idx]
                    current_track = metadata[idx]["track_name"]
                    expected_track = mapping.get(cluster)

                    if expected_track and current_track != expected_track:
                        inst_idx = instance_indices[idx]
                        corrections.append(
                            (frame_idx, inst_idx, current_track, expected_track)
                        )

        print(f"  Backfill: {len(corrections)} corrections identified")
        return mapping, corrections

    def _build_mapping_majority_vote(
        self,
        cluster_labels: np.ndarray,
        frame_indices: np.ndarray,
        instance_indices: np.ndarray,
        labels: sio.Labels,
        metadata: List[Dict],
    ) -> Dict[int, str]:
        """Build mapping using majority vote across all checkpoints.

        For each cluster, tally votes from all checkpoints and assign the
        track with the most votes.

        Args:
            cluster_labels: Array of cluster assignments.
            frame_indices: Array of frame indices.
            instance_indices: Array of instance indices.
            labels: SLEAP Labels object.
            metadata: List of metadata dicts.

        Returns:
            Dict mapping cluster_id -> track_name.
        """
        # Count track votes per cluster
        cluster_to_track_counts: Dict[int, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for i, cluster in enumerate(cluster_labels):
            track_name = metadata[i]["track_name"]
            cluster_to_track_counts[cluster][track_name] += 1

        # Build mapping with majority vote
        mapping = {}
        for cluster, track_counts in cluster_to_track_counts.items():
            best_track = max(track_counts, key=track_counts.get)
            mapping[cluster] = best_track
            total = sum(track_counts.values())
            confidence = track_counts[best_track] / total
            print(
                f"  Cluster {cluster} -> {best_track} "
                f"({track_counts[best_track]}/{total} = {confidence:.1%})"
            )

        return mapping

    # =========================================================================
    # Phase 3b: KNN Reclassification
    # =========================================================================

    def _reclassify_with_knn(
        self,
        cluster_labels: np.ndarray,
        frame_indices: np.ndarray,
        metadata: List[Dict],
        mapping: Dict[int, str],
        n_clusters: int,
    ) -> np.ndarray:
        """Reclassify samples using KNN trained on checkpoint frames.

        Checkpoint frames (where all n_clusters are represented) have high
        accuracy in their cluster-to-track assignments. This method trains
        a KNN classifier on checkpoint PCA features with mapped track names
        as labels, then reclassifies all samples.

        Samples whose k nearest neighbors do not unanimously agree on the
        same track are rejected (set to None) and will not be used for
        tracking. This acts as a confidence filter.

        Args:
            cluster_labels: Array of k-means cluster assignments.
            frame_indices: Array of frame indices per sample.
            metadata: List of metadata dicts per sample.
            mapping: Dict mapping cluster_id -> track_name.
            n_clusters: Number of clusters.

        Returns:
            Array of predicted track name strings (one per sample),
            or None for rejected samples.
        """
        import math

        # Reconstruct PCA features from metadata barcodes
        features = np.array([m["barcode"] for m in metadata])
        X_scaled = self._scaler.transform(features)
        X_pca = self._pca.transform(X_scaled)

        # Group by frame, find checkpoint frames
        frame_to_samples = defaultdict(list)
        for i, frame_idx in enumerate(frame_indices):
            frame_to_samples[frame_idx].append(i)

        checkpoint_indices = []
        for frame_idx, sample_indices in frame_to_samples.items():
            clusters_in_frame = set(cluster_labels[sample_indices])
            if len(clusters_in_frame) == n_clusters:
                checkpoint_indices.extend(sample_indices)

        if len(checkpoint_indices) == 0:
            print("  KNN: No checkpoint frames found, falling back to k-means")
            return np.array(
                [mapping.get(int(c), metadata[i]["track_name"])
                 for i, c in enumerate(cluster_labels)],
                dtype=object,
            )

        # Build training data from checkpoint frames
        X_train = X_pca[checkpoint_indices]
        y_train = np.array(
            [mapping[int(cluster_labels[i])] for i in checkpoint_indices]
        )

        # Determine k: float < 1.0 means fraction of checkpoint samples
        if isinstance(self.knn_neighbors, float) and self.knn_neighbors < 1.0:
            k = math.ceil(len(checkpoint_indices) * self.knn_neighbors)
        else:
            k = int(self.knn_neighbors)
        k = max(1, min(k, len(checkpoint_indices)))

        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Get neighbor labels for all samples to check unanimity
        neighbor_indices = knn.kneighbors(X_pca, return_distance=False)
        neighbor_labels = y_train[neighbor_indices]  # (n_samples, k)

        # Build result: only accept samples where all k neighbors agree
        y_pred = np.empty(len(X_pca), dtype=object)
        n_accepted = 0
        n_rejected = 0
        for i in range(len(X_pca)):
            unique_labels = set(neighbor_labels[i])
            if len(unique_labels) == 1:
                y_pred[i] = neighbor_labels[i, 0]
                n_accepted += 1
            else:
                y_pred[i] = None
                n_rejected += 1

        print(
            f"  KNN (k={k}): trained on {len(checkpoint_indices)} checkpoint samples, "
            f"accepted {n_accepted}/{len(y_pred)}, "
            f"rejected {n_rejected} (non-unanimous neighbors)"
        )

        return y_pred

    # =========================================================================
    # Phase 4: Proximity-Based Propagation
    # =========================================================================

    def _compute_proximity(
        self,
        inst_a: sio.Instance,
        inst_b: sio.Instance,
        method: str,
    ) -> float:
        """Compute proximity between two instances.

        Args:
            inst_a: First instance.
            inst_b: Second instance.
            method: "pose_centroid", "bbox_centroid", or "bbox_iou".

        Returns:
            Distance in pixels (for centroid methods) or IOU (for bbox_iou).
            For IOU, returns 1 - IOU so smaller values indicate closer proximity.
        """
        if method == "pose_centroid":
            centroid_a = get_centroid(inst_a)
            centroid_b = get_centroid(inst_b)

            if centroid_a is None or centroid_b is None:
                return float("inf")
            if np.isnan(centroid_a).any() or np.isnan(centroid_b).any():
                return float("inf")

            return float(np.linalg.norm(centroid_a - centroid_b))

        elif method == "bbox_centroid":
            pts_a = inst_a.numpy()
            pts_b = inst_b.numpy()

            valid_a = pts_a[~np.isnan(pts_a).any(axis=1)]
            valid_b = pts_b[~np.isnan(pts_b).any(axis=1)]

            if len(valid_a) == 0 or len(valid_b) == 0:
                return float("inf")

            center_a = (valid_a.min(axis=0) + valid_a.max(axis=0)) / 2
            center_b = (valid_b.min(axis=0) + valid_b.max(axis=0)) / 2

            return float(np.linalg.norm(center_a - center_b))

        elif method == "bbox_iou":
            pts_a = inst_a.numpy()
            pts_b = inst_b.numpy()

            valid_a = pts_a[~np.isnan(pts_a).any(axis=1)]
            valid_b = pts_b[~np.isnan(pts_b).any(axis=1)]

            if len(valid_a) == 0 or len(valid_b) == 0:
                return float("inf")

            # Compute bounding boxes
            min_a, max_a = valid_a.min(axis=0), valid_a.max(axis=0)
            min_b, max_b = valid_b.min(axis=0), valid_b.max(axis=0)

            # Compute intersection
            inter_min = np.maximum(min_a, min_b)
            inter_max = np.minimum(max_a, max_b)

            if (inter_max <= inter_min).any():
                return float("inf")  # No intersection

            inter_area = np.prod(inter_max - inter_min)

            # Compute union
            area_a = np.prod(max_a - min_a)
            area_b = np.prod(max_b - min_b)
            union_area = area_a + area_b - inter_area

            iou = inter_area / union_area if union_area > 0 else 0

            # Return 1 - IOU so smaller values indicate closer proximity
            return 1.0 - iou

        else:
            raise ValueError(f"Unknown proximity method: {method}")

    def _build_proximity_index(
        self,
        labels: sio.Labels,
        method: str,
        threshold: float,
    ) -> None:
        """Pre-compute proximity events for all track pairs.

        Scans all frames once and builds a lookup from (track_a, track_b) to a
        sorted list of frame indices where the pair was within the threshold.
        Subsequent calls to ``_find_last_proximity_event`` use binary search on
        this index instead of re-scanning frames each time.

        Args:
            labels: SLEAP Labels object.
            method: Proximity method.
            threshold: Proximity threshold.
        """
        self._build_frame_mapping(labels)

        # _proximity_events[(ta, tb)] = sorted list of frame indices
        self._proximity_events: Dict[Tuple[str, str], List[int]] = defaultdict(list)

        for lf in labels.labeled_frames:
            # Collect track-named instances in this frame
            track_instances: Dict[str, Any] = {}
            for inst in lf.instances:
                if inst.track is None:
                    continue
                track_name = (
                    inst.track.name
                    if isinstance(inst.track, TrackContext)
                    else inst.track.name
                )
                track_instances[track_name] = inst

            # Check all pairs
            track_names = sorted(track_instances.keys())
            for i in range(len(track_names)):
                for j in range(i + 1, len(track_names)):
                    ta, tb = track_names[i], track_names[j]
                    prox = self._compute_proximity(
                        track_instances[ta], track_instances[tb], method
                    )
                    if prox < threshold:
                        # Store under both orderings for easy lookup
                        self._proximity_events[(ta, tb)].append(lf.frame_idx)
                        self._proximity_events[(tb, ta)].append(lf.frame_idx)

        # Sort each list (frames may not be in order in labels)
        for key in self._proximity_events:
            self._proximity_events[key].sort()

        total = sum(len(v) for v in self._proximity_events.values()) // 2
        print(f"  Proximity index built: {total} events across {len(self._proximity_events)//2} track pairs")

    def _find_last_proximity_event(
        self,
        labels: sio.Labels,
        track_a: str,
        track_b: str,
        before_frame: int,
        method: str,
        threshold: float,
    ) -> Optional[int]:
        """Search backwards for the last frame where tracks were close.

        Uses the pre-computed proximity index for O(log n) lookup.

        Args:
            labels: SLEAP Labels object.
            track_a: First track name.
            track_b: Second track name.
            before_frame: Frame to search backwards from.
            method: Proximity method.
            threshold: Proximity threshold.

        Returns:
            Frame index of last proximity event, or None if not found.
        """
        frames = self._proximity_events.get((track_a, track_b))
        if not frames:
            return None

        # Binary search: find rightmost frame < before_frame
        idx = bisect.bisect_left(frames, before_frame) - 1
        if idx >= 0:
            return frames[idx]
        return None

    def _propagate_correction_to_proximity_event(
        self,
        labels: sio.Labels,
        track_from: str,
        track_to: str,
        checkpoint_frame: int,
    ) -> int:
        """Apply identity correction from proximity event to checkpoint.

        Finds the last proximity event between track_from and track_to before
        checkpoint_frame, then propagates the correction from that point.

        Args:
            labels: SLEAP Labels object.
            track_from: Current (incorrect) track name.
            track_to: Target (correct) track name.
            checkpoint_frame: Frame where mismatch was detected.

        Returns:
            Number of frames corrected.
        """
        # Find last proximity event
        proximity_frame = self._find_last_proximity_event(
            labels=labels,
            track_a=track_from,
            track_b=track_to,
            before_frame=checkpoint_frame,
            method=self.proximity_method,
            threshold=self.proximity_threshold,
        )

        if proximity_frame is None:
            print(
                f"  No proximity event found between {track_from} and {track_to} "
                f"before frame {checkpoint_frame}"
            )
            return 0

        print(
            f"  Found proximity event at frame {proximity_frame}, "
            f"propagating correction to frame {checkpoint_frame}"
        )

        # Apply correction from proximity_frame to checkpoint_frame
        corrected = 0
        for frame_idx in range(proximity_frame, checkpoint_frame + 1):
            lf = self._get_labeled_frame(labels, frame_idx)
            if lf is None:
                continue

            for inst_idx, inst in enumerate(lf.instances):
                if inst.track is None:
                    continue

                track_name = (
                    inst.track.name
                    if isinstance(inst.track, TrackContext)
                    else inst.track.name
                )

                if track_name == track_from:
                    # Apply correction using priority-based assignment
                    success = self.assign_with_priority_resolution(
                        labels=labels,
                        frame_idx=frame_idx,
                        instance_idx=inst_idx,
                        new_track_name=track_to,
                        reason=f"Tail tracker propagation from proximity frame {proximity_frame}",
                        propagate_to_tracklet=False,  # Don't propagate further
                    )
                    if success:
                        corrected += 1

        return corrected

    def _swap_tracks_in_range(
        self,
        labels: sio.Labels,
        track_a: str,
        track_b: str,
        start_frame: int,
        end_frame: int,
    ) -> int:
        """Bidirectionally swap track_a <-> track_b from start_frame to end_frame.

        For every frame in [start_frame, end_frame], instances on track_a are
        moved to track_b and vice versa.  Swaps are performed simultaneously
        within each frame so no intermediate state is visible.

        Args:
            labels: SLEAP Labels object (modified in place).
            track_a: First track name.
            track_b: Second track name.
            start_frame: First frame index (inclusive).
            end_frame: Last frame index (inclusive).

        Returns:
            Number of instances whose track was changed.
        """
        swapped = 0
        for frame_idx in range(start_frame, end_frame + 1):
            lf = self._get_labeled_frame(labels, frame_idx)
            if lf is None:
                continue

            # Collect indices that need swapping (don't mutate while iterating)
            a_to_b = []  # instance indices currently on track_a
            b_to_a = []  # instance indices currently on track_b
            for inst_idx, inst in enumerate(lf.instances):
                if inst.track is None:
                    continue
                track_name = (
                    inst.track.name
                    if isinstance(inst.track, TrackContext)
                    else inst.track.name
                )
                if track_name == track_a:
                    a_to_b.append(inst_idx)
                elif track_name == track_b:
                    b_to_a.append(inst_idx)

            # Apply swaps: A → B
            for inst_idx in a_to_b:
                success = self.assign_with_priority_resolution(
                    labels=labels,
                    frame_idx=frame_idx,
                    instance_idx=inst_idx,
                    new_track_name=track_b,
                    reason=f"Forward propagation swap {track_a}->{track_b}",
                    propagate_to_tracklet=False,
                )
                if success:
                    swapped += 1

            # Apply swaps: B → A
            for inst_idx in b_to_a:
                success = self.assign_with_priority_resolution(
                    labels=labels,
                    frame_idx=frame_idx,
                    instance_idx=inst_idx,
                    new_track_name=track_a,
                    reason=f"Forward propagation swap {track_b}->{track_a}",
                    propagate_to_tracklet=False,
                )
                if success:
                    swapped += 1

        return swapped

    # =========================================================================
    # Phase 4b: Post-Clustering Smoothing
    # =========================================================================

    def _smooth_assignments(
        self,
        labels: sio.Labels,
        cluster_labels: np.ndarray,
        frame_indices: np.ndarray,
        instance_indices: np.ndarray,
        metadata: List[Dict],
        mapping: Dict[int, str],
        pre_proposed: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Smooth cluster assignments using spatial gating and majority vote.

        Two-stage post-processing applied after cluster-to-track mapping:

        1. Spatial gating (if spatial_max_jump is set): For each instance, check
           whether accepting the cluster-based track would require the instance
           to "teleport" farther than spatial_max_jump pixels from where that
           track was in the previous frame. If so, revert to the instance's
           original (pre-clustering) track identity.

        2. Temporal majority vote (if smoothing_window > 0): For each
           (instance position), apply a sliding window majority vote over the
           assigned track names, eliminating short-lived identity flickers.

        Args:
            labels: SLEAP Labels object.
            cluster_labels: Array of cluster assignments per sample.
            frame_indices: Array of frame indices per sample.
            instance_indices: Array of instance indices per sample.
            metadata: List of metadata dicts per sample.
            mapping: Dict mapping cluster_id -> track_name.
            pre_proposed: Optional pre-computed track name predictions (e.g.
                from KNN). If provided, used instead of building from mapping.

        Returns:
            Array of smoothed track names (one per sample), parallel to inputs.
        """
        n_samples = len(cluster_labels)

        # Build proposed track assignment for each sample
        if pre_proposed is not None:
            proposed = np.array(pre_proposed, dtype=object)
        else:
            proposed = np.array(
                [mapping.get(int(c), metadata[i]["track_name"]) for i, c in enumerate(cluster_labels)],
                dtype=object,
            )

        # --- Stage 1: Spatial gating ---
        if self.spatial_max_jump is not None and self.spatial_max_jump > 0:
            proposed = self._apply_spatial_gating(
                labels, proposed, frame_indices, instance_indices, metadata
            )

        # --- Stage 2: Temporal majority vote ---
        if self.smoothing_window > 1:
            proposed = self._apply_majority_vote(
                proposed, frame_indices, instance_indices, metadata
            )

        return proposed

    def _apply_spatial_gating(
        self,
        labels: sio.Labels,
        proposed: np.ndarray,
        frame_indices: np.ndarray,
        instance_indices: np.ndarray,
        metadata: List[Dict],
    ) -> np.ndarray:
        """Reject track assignments that require impossible spatial jumps.

        For each sample, if the proposed track differs from the original track,
        check whether the instance centroid is within spatial_max_jump of the
        last known position of the proposed track. If not, keep the original.

        Args:
            labels: SLEAP Labels object.
            proposed: Array of proposed track names per sample.
            frame_indices: Array of frame indices per sample.
            instance_indices: Array of instance indices per sample.
            metadata: List of metadata dicts per sample.

        Returns:
            Updated proposed array with gated assignments.
        """
        result = proposed.copy()

        # Build centroid lookup: (frame_idx, inst_idx) -> (x, y)
        centroid_cache = {}
        for lf in labels.labeled_frames:
            for inst_idx, inst in enumerate(lf.instances):
                c = get_centroid(inst)
                if c is not None and not np.isnan(c).any():
                    centroid_cache[(lf.frame_idx, inst_idx)] = c

        # Build last-known position per track, scanning in frame order
        # Group samples by frame for ordered processing
        order = np.argsort(frame_indices)
        last_pos = {}  # track_name -> (x, y)
        gated_count = 0

        for idx in order:
            frame_idx = int(frame_indices[idx])
            inst_idx = int(instance_indices[idx])
            original_track = metadata[idx]["track_name"]
            new_track = result[idx]

            # Skip samples rejected by KNN unanimity filter
            if new_track is None:
                continue

            centroid = centroid_cache.get((frame_idx, inst_idx))
            if centroid is None:
                continue

            if new_track != original_track and new_track in last_pos:
                dist = float(np.linalg.norm(centroid - last_pos[new_track]))
                if dist > self.spatial_max_jump:
                    # Reject: would require teleportation
                    result[idx] = original_track
                    gated_count += 1

            # Update last known position for the (possibly reverted) track
            final_track = result[idx]
            last_pos[final_track] = centroid

        if gated_count > 0:
            print(f"  Spatial gating: rejected {gated_count} assignments (max_jump={self.spatial_max_jump}px)")

        return result

    def _apply_majority_vote(
        self,
        proposed: np.ndarray,
        frame_indices: np.ndarray,
        instance_indices: np.ndarray,
        metadata: List[Dict],
    ) -> np.ndarray:
        """Smooth track assignments with a sliding-window majority vote.

        Groups samples by the original track identity (instance position in
        the input labels), sorts by frame, and applies a centered sliding
        window. Within each window the most common proposed track wins.

        Args:
            proposed: Array of proposed track names per sample.
            frame_indices: Array of frame indices per sample.
            instance_indices: Array of instance indices per sample.
            metadata: List of metadata dicts per sample.

        Returns:
            Smoothed proposed array.
        """
        result = proposed.copy()
        half_w = self.smoothing_window // 2

        # Group sample indices by original track (so we smooth each animal's
        # timeline independently)
        track_groups = defaultdict(list)
        for i in range(len(proposed)):
            track_groups[metadata[i]["track_name"]].append(i)

        changed = 0
        for original_track, indices in track_groups.items():
            # Sort by frame
            indices_sorted = sorted(indices, key=lambda i: frame_indices[i])

            for pos, idx in enumerate(indices_sorted):
                # Skip samples already rejected (None)
                if result[idx] is None:
                    continue

                lo = max(0, pos - half_w)
                hi = min(len(indices_sorted), pos + half_w + 1)
                votes = Counter()
                for j in range(lo, hi):
                    v = result[indices_sorted[j]]
                    if v is not None:
                        votes[v] += 1
                if not votes:
                    continue
                winner = votes.most_common(1)[0][0]
                if winner != result[idx]:
                    changed += 1
                result[idx] = winner

        if changed > 0:
            print(f"  Majority vote (window={self.smoothing_window}): changed {changed} assignments")

        return result

    # =========================================================================
    # Phase 5: Track Assignment and Explanation Logging
    # =========================================================================

    def _log_tail_decision(
        self,
        frame_idx: int,
        instance_idx: int,
        decision_type: DecisionType,
        assigned_track_id: Optional[str],
        previous_track_id: Optional[str],
        summary: str,
        reasons: List[str],
        cluster_id: Optional[int] = None,
        is_checkpoint_frame: bool = False,
        barcode_valid: bool = True,
        propagation_source_frame: Optional[int] = None,
        proximity_frame: Optional[int] = None,
        instance_centroid: Optional[Tuple[float, float]] = None,
        cluster_confidence: Optional[float] = None,
    ) -> None:
        """Log a tail tracking decision to the explanation store.

        Args:
            frame_idx: Frame index.
            instance_idx: Instance index.
            decision_type: Type of decision made.
            assigned_track_id: Track ID assigned.
            previous_track_id: Previous track ID.
            summary: Human-readable summary.
            reasons: List of reasons.
            cluster_id: K-means cluster assignment.
            is_checkpoint_frame: Whether this was a checkpoint frame.
            barcode_valid: Whether barcode extraction was valid.
            propagation_source_frame: Source frame if propagated.
            proximity_frame: Frame of proximity event.
            instance_centroid: (x, y) centroid.
            cluster_confidence: Confidence in cluster assignment.
        """
        if self._explanation_store is None:
            return

        record = TailDecisionRecord(
            frame_idx=frame_idx,
            instance_idx=instance_idx,
            tracker_name=self.name,
            tracker_priority=self.priority,
            decision_type=decision_type,
            assigned_track_id=assigned_track_id,
            previous_track_id=previous_track_id,
            summary=summary,
            reasons=reasons,
            cluster_id=cluster_id,
            mapping_method=self.mapping_method,
            is_checkpoint_frame=is_checkpoint_frame,
            barcode_valid=barcode_valid,
            propagation_source_frame=propagation_source_frame,
            proximity_frame=proximity_frame,
            instance_centroid=instance_centroid,
            cluster_confidence=cluster_confidence,
        )
        self._explanation_store.add(record)

    def _apply_assignment_with_logging(
        self,
        labels: sio.Labels,
        frame_idx: int,
        instance_idx: int,
        new_track: str,
        current_track: str,
        cluster_id: int,
        is_checkpoint: bool,
        reason: str,
        propagation_source_frame: Optional[int] = None,
        proximity_frame: Optional[int] = None,
    ) -> bool:
        """Apply a track assignment and log the decision.

        Args:
            labels: SLEAP Labels object.
            frame_idx: Frame index.
            instance_idx: Instance index.
            new_track: New track name to assign.
            current_track: Current track name.
            cluster_id: Cluster ID for this instance.
            is_checkpoint: Whether this is a checkpoint frame.
            reason: Reason for assignment.
            propagation_source_frame: Source frame if propagated.
            proximity_frame: Frame of proximity event.

        Returns:
            True if assignment was successful.
        """
        # Get instance centroid for logging
        lf = self._get_labeled_frame(labels, frame_idx)
        centroid_tuple = None
        if lf is not None and instance_idx < len(lf.instances):
            centroid = get_centroid(lf.instances[instance_idx])
            if centroid is not None and not np.isnan(centroid).any():
                centroid_tuple = (float(centroid[0]), float(centroid[1]))

        # Determine decision type
        if current_track == new_track:
            decision_type = DecisionType.MATCHED
            summary = f"Confirmed cluster {cluster_id} -> {new_track}"
        elif propagation_source_frame is not None:
            decision_type = DecisionType.PROPAGATED
            summary = f"Propagated from frame {propagation_source_frame}: {current_track} -> {new_track}"
        else:
            decision_type = DecisionType.MATCHED
            summary = f"Assigned cluster {cluster_id} -> {new_track} (was {current_track})"

        # Build explanation dict for track_history (used by ID Switch Viewer)
        explanation_dict = {
            "tracker_type": "TailFeatureTracker",
            "cluster_id": cluster_id,
            "mapping_method": self.mapping_method,
            "is_checkpoint_frame": is_checkpoint,
            "propagation_source_frame": propagation_source_frame,
            "proximity_frame": proximity_frame,
            "instance_centroid": list(centroid_tuple) if centroid_tuple else None,
            "decision_type": decision_type.value,
            "summary": summary,
        }

        # Apply assignment
        success = True
        if current_track != new_track:
            success = self.assign_with_priority_resolution(
                labels=labels,
                frame_idx=frame_idx,
                instance_idx=instance_idx,
                new_track_name=new_track,
                reason=reason,
                explanation=explanation_dict,  # Pass to track_history
                propagate_to_tracklet=True,
            )

        # Log the decision to InstanceExplanationStore
        reasons = [reason]
        if propagation_source_frame is not None:
            reasons.append(f"Propagated from frame {propagation_source_frame}")
        if proximity_frame is not None:
            reasons.append(f"Proximity event at frame {proximity_frame}")

        self._log_tail_decision(
            frame_idx=frame_idx,
            instance_idx=instance_idx,
            decision_type=decision_type,
            assigned_track_id=new_track if success else current_track,
            previous_track_id=current_track,
            summary=summary,
            reasons=reasons,
            cluster_id=cluster_id,
            is_checkpoint_frame=is_checkpoint,
            barcode_valid=True,
            propagation_source_frame=propagation_source_frame,
            proximity_frame=proximity_frame,
            instance_centroid=centroid_tuple,
        )

        return success

    def track(
        self,
        labels: sio.Labels,
        video: Optional[sio.Video] = None,
        video_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> sio.Labels:
        """Track instances using tail segment features.

        Args:
            labels: SLEAP Labels object containing instances to track.
            video: Video object. If None, uses labels.video.
            video_path: Path to video (alternative to video parameter).
            output_path: Optional path to save tracking results.

        Returns:
            Labels object with track assignments.
        """
        print(f"\n{'='*70}")
        print(f"Tail Feature Tracker: {self.name}")
        print(f"{'='*70}")

        # Get video
        if video is None:
            if video_path is not None:
                video = sio.load_video(video_path)
            elif labels.videos:
                video = labels.videos[0]
            else:
                raise ValueError("No video provided or found in labels")

        # Get skeleton
        if not labels.skeletons:
            raise ValueError("No skeleton found in labels")
        skeleton = labels.skeletons[0]

        # Build frame mapping
        self._build_frame_mapping(labels)

        # Wrap existing tracks in TrackContext with low priority so that
        # conflict resolution can properly displace them when the tracker
        # (higher priority) assigns a different identity.
        self.convert_tracks_to_context_objects(labels, priority=0)

        # Get tail node indices
        print(f"\nTail nodes: {self.tail_nodes}")
        self._tail_node_indices = self._get_tail_node_indices(skeleton)
        print(f"Node indices: {self._tail_node_indices}")

        # Determine n_clusters
        n_clusters = self.n_clusters
        if n_clusters is None:
            # Auto-detect from number of unique tracks or max instances per frame
            unique_tracks = set()
            max_instances = 0
            for lf in labels.labeled_frames:
                max_instances = max(max_instances, len(lf.instances))
                for inst in lf.instances:
                    if inst.track is not None:
                        track_name = (
                            inst.track.name
                            if isinstance(inst.track, TrackContext)
                            else inst.track.name
                        )
                        unique_tracks.add(track_name)

            n_clusters = len(unique_tracks) if unique_tracks else max_instances
            print(f"Auto-detected n_clusters: {n_clusters}")

        if n_clusters < 2:
            print("Warning: n_clusters < 2, skipping tracking")
            return labels

        # Phase 1 & 2: Collect features
        print(f"\nPhase 1-2: Feature extraction and clustering...")
        features, frame_indices, instance_indices, validity_mask, metadata = (
            self._collect_checkpoint_features(
                labels, video, self._tail_node_indices
            )
        )

        if len(features) == 0:
            print("Warning: No valid tail features found")
            return labels

        print(f"  Collected {len(features)} valid samples")

        # Run clustering
        cluster_labels, self._scaler, self._pca, self._kmeans = self._run_clustering(
            features, n_clusters
        )
        print(f"  Clustering complete: {n_clusters} clusters")

        # Filter duplicates
        (
            cluster_labels,
            frame_indices,
            instance_indices,
            validity_mask,
            metadata,
        ) = self._filter_same_cluster_duplicates(
            cluster_labels, frame_indices, instance_indices, metadata
        )
        print(f"  After filtering: {len(cluster_labels)} samples")

        if len(cluster_labels) == 0:
            print("Warning: No valid checkpoints after filtering")
            return labels

        # Phase 3: Build cluster-to-track mapping
        print(f"\nPhase 3: Building cluster-to-track mapping ({self.mapping_method})...")

        corrections = []
        if self.mapping_method == "trust_first":
            mapping = self._build_mapping_trust_first(
                cluster_labels, frame_indices, instance_indices, labels, metadata
            )
        elif self.mapping_method == "backfill":
            mapping, corrections = self._build_mapping_backfill(
                cluster_labels, frame_indices, instance_indices, labels, metadata
            )
        else:  # majority_vote
            mapping = self._build_mapping_majority_vote(
                cluster_labels, frame_indices, instance_indices, labels, metadata
            )

        print(f"  Mapping: {mapping}")

        # Phase 3b: KNN reclassification (if enabled)
        knn_predictions = None
        if self.classification_method == "knn":
            print(f"\nPhase 3b: KNN reclassification...")
            knn_predictions = self._reclassify_with_knn(
                cluster_labels, frame_indices, metadata, mapping, n_clusters
            )

        # Identify checkpoint frames (frames where all clusters are represented)
        frame_to_samples = defaultdict(list)
        for i, frame_idx in enumerate(frame_indices):
            frame_to_samples[frame_idx].append(i)

        checkpoint_frames = set()
        for frame_idx, sample_indices in frame_to_samples.items():
            clusters_in_frame = set(cluster_labels[sample_indices])
            if len(clusters_in_frame) == n_clusters:
                checkpoint_frames.add(frame_idx)

        # Phase 4 & 5: Apply assignments
        print(f"\nPhase 4-5: Applying track assignments...")

        # Pre-compute proximity index for fast lookups
        if self.propagate_forward:
            print(f"  Building proximity index ({self.proximity_method}, {self.proximity_threshold}px)...")
            self._build_proximity_index(
                labels, self.proximity_method, self.proximity_threshold
            )

        # Apply corrections for backfill (only used when propagate_forward=False)
        if corrections and not self.propagate_forward:
            print(f"  Applying {len(corrections)} backfill corrections...")
            for frame_idx, inst_idx, old_track, new_track in corrections:
                cluster_id = None
                for i, (fi, ii) in enumerate(zip(frame_indices, instance_indices)):
                    if fi == frame_idx and ii == inst_idx:
                        cluster_id = cluster_labels[i]
                        break

                self._apply_assignment_with_logging(
                    labels=labels,
                    frame_idx=frame_idx,
                    instance_idx=inst_idx,
                    new_track=new_track,
                    current_track=old_track,
                    cluster_id=cluster_id if cluster_id is not None else -1,
                    is_checkpoint=frame_idx in checkpoint_frames,
                    reason="Tail tracker backfill correction",
                )

        # Phase 4b: Smooth assignments (spatial gating + majority vote)
        if self.smoothing_window > 1 or self.spatial_max_jump is not None:
            print(f"\nPhase 4b: Smoothing assignments...")
            smoothed_tracks = self._smooth_assignments(
                labels, cluster_labels, frame_indices, instance_indices,
                metadata, mapping, pre_proposed=knn_predictions,
            )
        else:
            if knn_predictions is not None:
                smoothed_tracks = np.array(knn_predictions, dtype=object)
            else:
                smoothed_tracks = np.array(
                    [mapping.get(int(c), metadata[i]["track_name"]) for i, c in enumerate(cluster_labels)],
                    dtype=object,
                )

        # ---------------------------------------------------------------
        # Forward propagation mode
        # ---------------------------------------------------------------
        if self.propagate_forward:
            last_frame = max(lf.frame_idx for lf in labels.labeled_frames)

            # Sort decisions by frame_idx for correct temporal ordering
            order = np.argsort(frame_indices)
            assigned_count = 0

            for idx in order:
                frame_idx = int(frame_indices[idx])
                inst_idx = int(instance_indices[idx])
                cluster = int(cluster_labels[idx])
                new_track = smoothed_tracks[idx]
                if new_track is None:
                    continue
                is_checkpoint = frame_idx in checkpoint_frames

                # Read current_track from LABELS so prior propagations
                # are visible (not from metadata which is stale)
                lf = self._get_labeled_frame(labels, frame_idx)
                if lf is None or inst_idx >= len(lf.instances):
                    continue
                inst = lf.instances[inst_idx]
                if inst.track is None:
                    continue
                current_track = (
                    inst.track.name
                    if isinstance(inst.track, TrackContext)
                    else inst.track.name
                )

                if current_track == new_track:
                    # Confirmation — log and continue
                    self._log_tail_decision(
                        frame_idx=frame_idx,
                        instance_idx=inst_idx,
                        decision_type=DecisionType.MATCHED,
                        assigned_track_id=new_track,
                        previous_track_id=current_track,
                        summary=f"Cluster {cluster} confirms track {new_track}",
                        reasons=[f"Cluster {cluster} maps to {new_track} (already assigned)"],
                        cluster_id=cluster,
                        is_checkpoint_frame=is_checkpoint,
                    )
                    assigned_count += 1
                    continue

                # Swap needed — find last proximity event
                proximity_frame = self._find_last_proximity_event(
                    labels, current_track, new_track, frame_idx,
                    self.proximity_method, self.proximity_threshold,
                )
                swap_start = proximity_frame if proximity_frame is not None else frame_idx

                swapped = self._swap_tracks_in_range(
                    labels, current_track, new_track, swap_start, last_frame
                )
                assigned_count += swapped

                # Log the decision
                self._log_tail_decision(
                    frame_idx=frame_idx,
                    instance_idx=inst_idx,
                    decision_type=DecisionType.PROPAGATED,
                    assigned_track_id=new_track,
                    previous_track_id=current_track,
                    summary=(
                        f"Cluster {cluster}: swapped {current_track}<->{new_track} "
                        f"from frame {swap_start} to {last_frame} ({swapped} instances)"
                    ),
                    reasons=[
                        f"Cluster {cluster} maps to {new_track} but found {current_track}",
                        f"Proximity event at frame {proximity_frame}" if proximity_frame is not None else "No proximity found, swap from current frame",
                        f"Propagated swap range: [{swap_start}, {last_frame}]",
                    ],
                    cluster_id=cluster,
                    is_checkpoint_frame=is_checkpoint,
                    proximity_frame=proximity_frame,
                )

            print(f"  Forward propagation: {assigned_count} instances affected")

        # ---------------------------------------------------------------
        # Standard (non-propagating) mode
        # ---------------------------------------------------------------
        else:
            assigned_count = 0
            for i, (frame_idx, inst_idx, cluster) in enumerate(
                zip(frame_indices, instance_indices, cluster_labels)
            ):
                new_track = smoothed_tracks[i]
                if new_track is None:
                    continue

                current_track = metadata[i]["track_name"]
                is_checkpoint = frame_idx in checkpoint_frames

                if current_track != new_track:
                    success = self._apply_assignment_with_logging(
                        labels=labels,
                        frame_idx=frame_idx,
                        instance_idx=inst_idx,
                        new_track=new_track,
                        current_track=current_track,
                        cluster_id=int(cluster),
                        is_checkpoint=is_checkpoint,
                        reason=f"Tail tracker cluster {cluster} assignment",
                    )
                    if success:
                        assigned_count += 1
                else:
                    # Track already matches - log confirmation
                    self._log_tail_decision(
                        frame_idx=frame_idx,
                        instance_idx=inst_idx,
                        decision_type=DecisionType.MATCHED,
                        assigned_track_id=new_track,
                        previous_track_id=current_track,
                        summary=f"Cluster {cluster} confirms track {new_track}",
                        reasons=[f"Cluster {cluster} maps to {new_track} (already assigned)"],
                        cluster_id=int(cluster),
                        is_checkpoint_frame=is_checkpoint,
                    )
                    assigned_count += 1

            print(f"  Assigned {assigned_count} instances")

        # Save if output path provided
        if output_path is not None:
            print(f"\nSaving results to {output_path}...")
            sio.save_file(labels, output_path)

        print(f"\nTracking complete!")
        return labels

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization.

        Returns:
            Dict of configuration parameters.
        """
        config = super().get_config()
        config.update(
            {
                "tail_nodes": self.tail_nodes,
                "n_clusters": self.n_clusters,
                "min_node_distance": self.min_node_distance,
                "max_segment_angle": self.max_segment_angle,
                "segment_image_size": self.segment_image_size,
                "n_pca_components": self.n_pca_components,
                "require_all_tails": self.require_all_tails,
                "mapping_method": self.mapping_method,
                "propagate_forward": self.propagate_forward,
                "proximity_method": self.proximity_method,
                "proximity_threshold": self.proximity_threshold,
                "segment_width": self.segment_width,
                "segment_length": self.segment_length,
                "contrast": self.contrast,
                "brightness": self.brightness,
                "smoothing_window": self.smoothing_window,
                "spatial_max_jump": self.spatial_max_jump,
                "classification_method": self.classification_method,
                "knn_neighbors": self.knn_neighbors,
            }
        )
        return config
