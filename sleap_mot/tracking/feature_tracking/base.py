from sleap_mot.tracking.base import TrackingLayer
import numpy as np
import sleap_io as sio
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sleap_mot.tracking.base import TrackContext
from collections import Counter

from typing import Optional, Dict, List


class FeatureTracker(TrackingLayer, ABC):
    """Abstract base class for feature-based tracking algorithms in SLEAP-MOT.

    This class defines the interface and common utilities for implementing
    feature-based trackers. Subclasses should implement the required abstract
    methods to provide specific tracking logic.

    Methods
    -------
    load_and_preprocess_labels(labels: sio.Labels, video_path: str)
        Load and preprocess SLEAP labels for tracking.

    Attributes
    ----------
    motion_kde_paths : Any
        Stores motion kernel density estimation paths, if used by the tracker.
    """

    def __init__(self, priority: int = 10, name: str = "FeatureTracker"):
        """Initialize the FeatureTracker.

        Args:
            priority: Priority level for conflict resolution (higher = more authoritative)
            name: Name of this tracking layer
        """
        super().__init__(priority=priority, name=name)
        self.only_apply_to_tracklets = True

    def _calculate_max_instances(self, labels: sio.Labels) -> int:
        """Calculate the maximum number of instances across all frames.

        Args:
            labels: SLEAP Labels object

        Returns:
            Maximum number of instances in any single frame
        """
        if len(labels) == 0:
            return 0
        return max(len(lf.instances) for lf in labels)

    def run_pca(self, freqs, confidence_vector, max_instances, n_tracks):
        """Run PCA on features and confidence vector."""
        freqs = np.array(freqs)
        freqs_reshaped = freqs.reshape(freqs.shape[0] * n_tracks, freqs.shape[2])

        # Flatten confidence vector to align with reshaped features
        conf_flat = confidence_vector.flatten()

        # For each array in freqs_reshaped, if every value is -1, set conf_flat to False at this position
        for i in range(freqs_reshaped.shape[0]):
            if np.all(freqs_reshaped[i] == -1):
                conf_flat[i] = False

        # Select only valid poses using boolean indexing
        X = freqs_reshaped[conf_flat]

        # Fit PCA and transform data
        pcs = PCA(n_components=6)
        pcs = pcs.fit(X)
        Z = pcs.transform(X)

        # Cluster the transformed data
        kmeans = KMeans(n_clusters=max_instances).fit(Z)
        G = kmeans.labels_

        return G, X, Z

    def run_knn(
        self,
        X,
        G,
        Z,
        confidence_vector,
        max_instances,
        n_neighbors,
        n_components,
    ):
        """Run KNN on features and confidence vector."""
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(X)
        distances, indices = knn.kneighbors(X)

        nn_G = G[indices]
        is_unambiguous = (nn_G == G.reshape(-1, 1)).all(axis=1)

        kmeans = KMeans(n_clusters=max_instances).fit(Z[:, :n_components])
        G = kmeans.labels_

        # Set ambiguous points to -1 in G
        if G is None:
            return None

        G[~is_unambiguous] = -1

        # Create array same size as confidence_vector filled with -1
        G_mapped = np.full(confidence_vector.shape, -1)

        true_positions = np.where(confidence_vector)
        G_mapped[true_positions] = G

        for frame in G_mapped:
            # Find duplicates
            seen = set()
            for i, cluster_id in enumerate(frame):
                if cluster_id != -1:  # Only check non-unassigned poses
                    if cluster_id in seen:
                        # This is a duplicate, set to -1
                        frame[i] = -1
                    else:
                        # First time seeing this cluster, keep it
                        seen.add(cluster_id)

        return G_mapped

    def get_current_tracklets(self, labels):
        tracklets = {}
        for lf in labels:
            for instance_idx, inst in enumerate(lf.instances):
                if inst.track is not None:
                    if inst.track.name not in tracklets:
                        tracklets[inst.track.name] = []
                    tracklets[inst.track.name].append((lf.frame_idx, instance_idx))
        if len(tracklets) == 0:
            return None
        return tracklets

    def assign_track_ids_to_tracklets(self, probabilities_df, tracklets):
        """Assign RFID IDs to tracklets based on probabilities DataFrame.

        For each tracklet, collects all RFID probabilities and uses majority
        voting to determine the best track ID.

        Args:
            probabilities_df: DataFrame of shape (num_frames, max_instances) where
                each cell contains {rfid_id: probability} dict or None
            tracklets: Dict of {track_name: [(frame_idx, instance_idx), ...]}

        Returns:
            List of (tracklet, [track_ids]) pairs
        """
        tracklet_id_pairs = []

        for track_name, frame_instance_list in tracklets.items():
            track_ids = []
            for frame_idx, pose_idx in frame_instance_list:
                if pose_idx >= probabilities_df.shape[1]:
                    continue
                probs_dict = probabilities_df.loc[frame_idx, pose_idx]
                if probs_dict is None:
                    continue
                # Find best RFID for this instance
                best_prob = 0
                best_rfid_name = None
                for rfid_name, prob in probs_dict.items():
                    if prob > best_prob:
                        best_prob = prob
                        best_rfid_name = rfid_name
                if best_rfid_name is not None:
                    track_ids.append(best_rfid_name)

            tracklet_id_pairs.append((frame_instance_list, track_ids))

        return tracklet_id_pairs

    def assign_track_ids(self, probabilities_df, labels):
        """Assign track IDs to instances based on probabilities DataFrame.

        This method iterates through the probabilities DataFrame and assigns
        the best RFID identity to each instance. It handles conflicts when
        multiple instances in the same frame want the same identity.

        Args:
            probabilities_df: DataFrame of shape (num_frames, max_instances) where
                each cell contains {rfid_id: probability} dict or None
            labels: SLEAP Labels object to assign tracks to
        """
        # First, reset all tracks
        for lf in labels:
            for inst in lf.instances:
                inst.track = None

        # Get existing tracklets if any
        tracklets = self.get_current_tracklets(labels)

        if tracklets is not None and self.only_apply_to_tracklets:
            # Use tracklet-based assignment with majority voting
            tracklet_id_pairs = self.assign_track_ids_to_tracklets(probabilities_df, tracklets)
            self._assign_tracks_from_tracklet_pairs(tracklet_id_pairs, labels)
        else:
            # Direct frame-by-frame assignment
            self._assign_tracks_from_probabilities(probabilities_df, labels)

    def _assign_tracks_from_tracklet_pairs(self, tracklet_id_pairs, labels):
        """Assign tracks to instances based on tracklet ID pairs.

        Uses majority voting for tracklets with multiple ID assignments and
        handles conflicts between overlapping tracklets.

        Args:
            tracklet_id_pairs: List of (tracklet, [track_ids]) where tracklet is
                a list of (frame_idx, instance_idx) tuples
            labels: SLEAP Labels object
        """
        def set_track_with_context(tracklet, track_name, labels, reason="Tracklet assignment"):
            """Set track with TrackContext for all poses in a tracklet."""
            if track_name is None:
                # Clear tracks
                for frame_idx, pose_idx in tracklet:
                    if frame_idx < len(labels) and pose_idx < len(labels[frame_idx].instances):
                        labels[frame_idx].instances[pose_idx].track = None
                return

            # Find or create the sio.Track
            existing_track = next(
                (t for t in labels.tracks if t.name == track_name), None
            )
            base_track = existing_track if existing_track else sio.Track(name=track_name)
            if not existing_track:
                labels.tracks.append(base_track)

            # Assign TrackContext with history to each instance
            for frame_idx, pose_idx in tracklet:
                if frame_idx < len(labels) and pose_idx < len(labels[frame_idx].instances):
                    inst = labels[frame_idx].instances[pose_idx]

                    # Get old track name for history
                    old_track_name = None
                    if inst.track is not None:
                        if isinstance(inst.track, TrackContext):
                            old_track_name = inst.track.name
                        elif hasattr(inst.track, 'name'):
                            old_track_name = inst.track.name

                    # Create TrackContext with history
                    track_context = TrackContext(
                        priority=self.priority,
                        track=base_track,
                        name=track_name,
                        temporary_track=False,
                        valid=True,
                        track_history=[]
                    )

                    # Add history entry
                    track_context.add_history_entry(
                        layer_name=self.name,
                        old_track_name=old_track_name,
                        new_track_name=track_name,
                        frame_idx=frame_idx,
                        reason=reason,
                        conflict_resolved=False,
                        propagated_from_frame=None
                    )

                    inst.track = track_context

        # Process each tracklet with majority voting
        for index, (tracklet, track_id_list) in enumerate(tracklet_id_pairs):
            if len(set(track_id_list)) > 1:
                # Multiple IDs - use majority voting
                id_counts = Counter(track_id_list)
                filtered_counts = {k: v for k, v in id_counts.items() if k is not None}

                if filtered_counts:
                    max_count = max(filtered_counts.values())
                    most_common = [k for k, v in filtered_counts.items() if v == max_count]

                    if len(most_common) == 1:
                        track_id_list = [most_common[0]] * len(track_id_list)
                    else:
                        # Tie - clear the list
                        track_id_list = []
                else:
                    track_id_list = []

                tracklet_id_pairs[index] = (tracklet, track_id_list)

            if len(track_id_list) > 0:
                track_id = track_id_list[0]
                current_frames = set(frame for frame, _ in tracklet)

                # Check for conflicts with other tracklets
                conflict = False
                for other_tracklet, other_track_ids in tracklet_id_pairs:
                    if (len(other_track_ids) > 0 and
                        other_track_ids[0] == track_id and
                        other_tracklet != tracklet):

                        other_frames = set(frame for frame, _ in other_tracklet)
                        if current_frames & other_frames:
                            # Overlap exists - resolve by track_id_list length
                            if len(track_id_list) > len(other_track_ids):
                                set_track_with_context(
                                    other_tracklet, None, labels,
                                    reason=f"Conflict resolution: shorter tracklet cleared"
                                )
                            elif len(track_id_list) < len(other_track_ids):
                                set_track_with_context(
                                    tracklet, None, labels,
                                    reason=f"Conflict resolution: shorter tracklet cleared"
                                )
                                conflict = True
                                break
                            else:
                                # Equal - clear both
                                set_track_with_context(
                                    other_tracklet, None, labels,
                                    reason=f"Conflict resolution: equal length tie"
                                )
                                set_track_with_context(
                                    tracklet, None, labels,
                                    reason=f"Conflict resolution: equal length tie"
                                )
                                conflict = True
                                break

                if not conflict:
                    set_track_with_context(
                        tracklet, track_id, labels,
                        reason=f"RFID majority vote assignment"
                    )

    def _assign_tracks_from_probabilities(self, probabilities_df, labels):
        """Assign tracks directly from probabilities without tracklets.

        For each frame, assigns the best RFID identity to each instance,
        resolving conflicts by giving priority to higher probabilities.

        Args:
            probabilities_df: DataFrame of probabilities
            labels: SLEAP Labels object
        """
        num_frames = len(labels)
        track_cache = {}  # Cache of track_name -> sio.Track object

        for frame_idx in range(num_frames):
            if frame_idx >= len(labels):
                continue

            lf = labels[frame_idx]
            num_instances = len(lf.instances)

            # Collect best identities for each instance in this frame
            instance_assignments = []  # [(instance_idx, rfid_name, probability), ...]

            for pose_idx in range(num_instances):
                if pose_idx >= probabilities_df.shape[1]:
                    continue

                probs_dict = probabilities_df.loc[frame_idx, pose_idx]
                if probs_dict is None:
                    continue

                # Find best RFID for this instance
                best_prob = 0
                best_rfid = None
                for rfid_name, prob in probs_dict.items():
                    if prob > best_prob:
                        best_prob = prob
                        best_rfid = rfid_name

                if best_rfid is not None and best_prob > 0:
                    instance_assignments.append((pose_idx, best_rfid, best_prob))

            # Resolve conflicts - if multiple instances want same ID, highest prob wins
            used_rfids = set()
            # Sort by probability descending so highest prob gets first pick
            instance_assignments.sort(key=lambda x: x[2], reverse=True)

            for pose_idx, rfid_name, prob in instance_assignments:
                if rfid_name in used_rfids:
                    continue  # Already assigned to another instance

                # Get or create the base sio.Track
                if rfid_name not in track_cache:
                    existing = next((t for t in labels.tracks if t.name == rfid_name), None)
                    if existing:
                        track_cache[rfid_name] = existing
                    else:
                        new_track = sio.Track(name=rfid_name)
                        labels.tracks.append(new_track)
                        track_cache[rfid_name] = new_track

                base_track = track_cache[rfid_name]
                inst = lf.instances[pose_idx]

                # Get old track name for history
                old_track_name = None
                if inst.track is not None:
                    if isinstance(inst.track, TrackContext):
                        old_track_name = inst.track.name
                    elif hasattr(inst.track, 'name'):
                        old_track_name = inst.track.name

                # Create TrackContext with history
                track_context = TrackContext(
                    priority=self.priority,
                    track=base_track,
                    name=rfid_name,
                    temporary_track=False,
                    valid=True,
                    track_history=[]
                )

                # Add history entry
                track_context.add_history_entry(
                    layer_name=self.name,
                    old_track_name=old_track_name,
                    new_track_name=rfid_name,
                    frame_idx=frame_idx,
                    reason=f"RFID probability assignment (prob={prob:.3f})",
                    conflict_resolved=False,
                    propagated_from_frame=None
                )

                inst.track = track_context
                used_rfids.add(rfid_name)

    def get_track_context(self, labels, frame_idx, track) -> Optional[TrackContext]:
        """Get the TrackContext for a track in a specific frame.

        Args:
            labels: SLEAP Labels object
            frame_idx: Frame index
            track: Track to look for

        Returns:
            TrackContext if found, None otherwise
        """
        if frame_idx >= len(labels):
            return None

        lf = labels[frame_idx]
        for inst in lf.instances:
            if inst.track is not None:
                if isinstance(inst.track, TrackContext):
                    if inst.track.name == track.name:
                        return inst.track
                elif isinstance(inst.track, sio.Track):
                    if inst.track.name == track.name:
                        # Convert to TrackContext for compatibility
                        return TrackContext(
                            priority=None,
                            track=inst.track,
                            name=inst.track.name,
                            temporary_track=False,
                            valid=True
                        )
        return None

    def has_track_in_frame(self, labels, frame_idx, track) -> bool:
        """Check if a track exists in a specific frame.

        Args:
            labels: SLEAP Labels object
            frame_idx: Frame index
            track: Track to look for

        Returns:
            True if track exists in frame, False otherwise
        """
        if frame_idx >= len(labels):
            return False

        lf = labels[frame_idx]
        for inst in lf.instances:
            if inst.track is not None:
                track_name = inst.track.name if isinstance(inst.track, sio.Track) else inst.track
                if hasattr(inst.track, 'name') and inst.track.name == track.name:
                    return True
        return False

    def get_instance_with_track(self, labels, frame_idx, track) -> Optional[int]:
        """Get the instance index that has a specific track in a frame.

        Args:
            labels: SLEAP Labels object
            frame_idx: Frame index
            track: Track to look for

        Returns:
            Instance index if found, None otherwise
        """
        if frame_idx >= len(labels):
            return None

        lf = labels[frame_idx]
        for idx, inst in enumerate(lf.instances):
            if inst.track is not None:
                if hasattr(inst.track, 'name') and inst.track.name == track.name:
                    return idx
        return None

    def assign_track(self, labels, frame_idx, instance_idx, track, track_context=None) -> None:
        """Assign a track to an instance.

        Args:
            labels: SLEAP Labels object
            frame_idx: Frame index
            instance_idx: Instance index
            track: Track to assign
            track_context: Optional TrackContext (not used in simple assignment)
        """
        if frame_idx >= len(labels):
            return
        if instance_idx >= len(labels[frame_idx].instances):
            return

        # Find or create the track in labels.tracks
        existing = next((t for t in labels.tracks if t.name == track.name), None)
        if existing:
            labels[frame_idx].instances[instance_idx].track = existing
        else:
            labels.tracks.append(track)
            labels[frame_idx].instances[instance_idx].track = track

    def remove_track(self, labels, frame_idx, instance_idx) -> None:
        """Remove track from an instance.

        Args:
            labels: SLEAP Labels object
            frame_idx: Frame index
            instance_idx: Instance index
        """
        if frame_idx >= len(labels):
            return
        if instance_idx >= len(labels[frame_idx].instances):
            return

        labels[frame_idx].instances[instance_idx].track = None

    def get_next_frame(self, labels, current_frame, direction) -> Optional[int]:
        """Get the next frame index in a given direction.

        Args:
            labels: SLEAP Labels object
            current_frame: Current frame index
            direction: 1 for forward, -1 for backward

        Returns:
            Next frame index if valid, None otherwise
        """
        next_frame = current_frame + direction
        if 0 <= next_frame < len(labels):
            return next_frame
        return None

    @abstractmethod
    def track(
        self,
        labels: sio.Labels,
        max_instances: Optional[int] = None,
        **kwargs
    ):
        """Track instances across frames using feature-based tracking.

        Args:
            labels: SLEAP Labels object containing instances to track
            max_instances: Maximum number of instances per frame. If None,
                calculated automatically as the max instances across all frames.
            **kwargs: Additional arguments for specific tracker implementations

        Returns:
            labels: The Labels object with track assignments
        """
        pass