from sleap_mot.tracking.base import IdTrackLayer
import numpy as np
import sleap_io as sio
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class FeatureTracker(IdTrackLayer, ABC):
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

    def __init__(self):
        """Initialize the FeatureTracker."""
        self.only_apply_to_tracklets = True

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
        tracklets = []
        for lf in labels:
            for instance_idx, inst in enumerate(lf.instances):
                if inst.track is not None:
                    if inst.track.name not in tracklets:
                        tracklets[inst.track.name] = []
                    tracklets[inst.track.name].append((lf.frame_idx, instance_idx))
        if len(tracklets) == 0:
            return None
        return tracklets

    def assign_track_ids_to_tracklets(self, probabilities_df, labels, tracklets):
        
        tracklet_id_pairs = []
        for index, (tracklet, track_id_list) in enumerate(tracklets):
            for frame_idx, pose_idx in tracklet:
                probs_dict = probabilities_df.loc[frame_idx, pose_idx]
                if probs_dict is None:
                    continue
                best_prob = 0
                best_rfid_name = None
                for rfid_name, prob in probs_dict.items():
                    if prob > best_prob:
                        best_prob = prob
                        best_rfid_name = rfid_name
                if best_rfid_name is not None:
                    track_id_list.append(best_rfid_name)



    def assign_track_ids(self, probabilities_df, labels, tracklets):
        if self.only_apply_to_tracklets:

        

    @abstractmethod
    def track(self, *args, **kwargs):
        """Abstract method to be implemented by subclasses."""
        pass