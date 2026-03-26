from sleap_mot.tracking.base import TrackingLayer
from sleap_mot.tracking.explanations import ExplanationGenerator, FeatureExplanationGenerator
from sleap_mot.tracking.instance_explanations import (
    InstanceExplanationStore,
    RFIDDecisionRecord,
    RFIDNoMatchBroadcast,
    CandidateScore,
    DecisionType,
    IdentityVote,
    SwitchRepairDecisionRecord,
)
from sleap_mot.utils import get_centroid
import numpy as np
import sleap_io as sio
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sleap_mot.tracking.base import TrackContext
from collections import Counter
from dataclasses import dataclass, field

from typing import Optional, Dict, List, Any, Tuple


@dataclass
class TrackletSegment:
    """A temporal segment of a tracklet with consistent identity.

    Represents a contiguous portion of a tracklet where a single identity
    dominates the votes. Used for detecting identity switches.

    Attributes:
        start_frame: First frame of this segment.
        end_frame: Last frame of this segment (inclusive).
        dominant_identity: The identity with most votes in this segment.
        vote_count: Number of votes for the dominant identity.
        total_votes_in_segment: Total votes cast in this segment.
        confidence: vote_count / total_votes_in_segment.
    """
    start_frame: int
    end_frame: int
    dominant_identity: str
    vote_count: int
    total_votes_in_segment: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "dominant_identity": self.dominant_identity,
            "vote_count": self.vote_count,
            "total_votes_in_segment": self.total_votes_in_segment,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackletSegment":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DetectedSwitch:
    """A detected identity switch within a tracklet.

    Represents a point in time where the tracklet appears to switch
    from one identity to another, based on voting evidence.

    Attributes:
        switch_frame: Frame where switch occurs (first frame of new identity).
        identity_before: Dominant identity before switch.
        identity_after: Dominant identity after switch.
        votes_before: Number of votes for identity_before.
        votes_after: Number of votes for identity_after.
        confidence: Based on vote counts on each side.
        partner_tracklet: If cross-validated, the partner tracklet name.
        partner_switch_frame: If cross-validated, the partner's switch frame.
    """
    switch_frame: int
    identity_before: str
    identity_after: str
    votes_before: int
    votes_after: int
    confidence: float
    partner_tracklet: Optional[str] = None
    partner_switch_frame: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "switch_frame": self.switch_frame,
            "identity_before": self.identity_before,
            "identity_after": self.identity_after,
            "votes_before": self.votes_before,
            "votes_after": self.votes_after,
            "confidence": self.confidence,
            "partner_tracklet": self.partner_tracklet,
            "partner_switch_frame": self.partner_switch_frame,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectedSwitch":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TrackletIdentityAnalysis:
    """Analysis of identity votes for a single tracklet.

    Contains the full analysis results for a tracklet, including
    segments, detected switches, and overall consistency assessment.

    Attributes:
        tracklet_name: Name of the tracklet.
        frame_range: (first_frame, last_frame) tuple.
        votes_by_identity: Dict mapping identity -> list of IdentityVote.
        segments: Time-ordered list of TrackletSegment.
        detected_switches: List of DetectedSwitch found.
        is_consistent: True if single identity dominates entire tracklet.
        dominant_identity: If consistent, the winning identity.
        confidence: Overall confidence in the analysis.
    """
    tracklet_name: str
    frame_range: Tuple[int, int]
    votes_by_identity: Dict[str, List[IdentityVote]]
    segments: List[TrackletSegment]
    detected_switches: List[DetectedSwitch]
    is_consistent: bool
    dominant_identity: Optional[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tracklet_name": self.tracklet_name,
            "frame_range": list(self.frame_range),
            "votes_by_identity": {
                identity: [v.to_dict() for v in votes]
                for identity, votes in self.votes_by_identity.items()
            },
            "segments": [s.to_dict() for s in self.segments],
            "detected_switches": [s.to_dict() for s in self.detected_switches],
            "is_consistent": self.is_consistent,
            "dominant_identity": self.dominant_identity,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackletIdentityAnalysis":
        """Create from dictionary."""
        return cls(
            tracklet_name=data["tracklet_name"],
            frame_range=tuple(data["frame_range"]),
            votes_by_identity={
                identity: [IdentityVote.from_dict(v) for v in votes]
                for identity, votes in data["votes_by_identity"].items()
            },
            segments=[TrackletSegment.from_dict(s) for s in data["segments"]],
            detected_switches=[DetectedSwitch.from_dict(s) for s in data["detected_switches"]],
            is_consistent=data["is_consistent"],
            dominant_identity=data.get("dominant_identity"),
            confidence=data["confidence"],
        )


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

    def __init__(
        self,
        priority: int = 10,
        name: str = "FeatureTracker",
        min_votes_for_identity: int = 2,
        switch_confidence_threshold: float = 0.8,
        split_on_switch: bool = True,
        require_partner_validation: bool = False,
    ):
        """Initialize the FeatureTracker.

        Args:
            priority: Priority level for conflict resolution (higher = more authoritative)
            name: Name of this tracking layer
            min_votes_for_identity: Minimum votes required on each side of a switch
                to consider it valid. Default: 2.
            switch_confidence_threshold: Confidence required to confirm a switch
                (fraction of votes agreeing on each side). Default: 0.8.
            split_on_switch: Whether to split tracklets at detected switch points.
                Default: True.
            require_partner_validation: Only split/swap if partner tracklet confirms
                the switch (cross-validation). Default: False.
        """
        super().__init__(priority=priority, name=name)
        self.only_apply_to_tracklets = True
        self._explanation_generator: Optional[ExplanationGenerator] = FeatureExplanationGenerator()
        self._explanation_store: Optional[InstanceExplanationStore] = None
        # Frame index mapping - built by _build_frame_mapping()
        # IMPORTANT: labels[x] returns the x-th labeled frame, NOT the frame at index x
        # Use this mapping to access frames by their actual frame index
        self._frame_idx_to_lf: Dict[int, Any] = {}

        # Identity switch detection parameters
        self.min_votes_for_identity = min_votes_for_identity
        self.switch_confidence_threshold = switch_confidence_threshold
        self.split_on_switch = split_on_switch
        self.require_partner_validation = require_partner_validation

    def _build_frame_mapping(self, labels: sio.Labels) -> None:
        """Build mapping from frame index to LabeledFrame.

        IMPORTANT: labels[x] returns the x-th labeled frame, NOT the frame at index x.
        This mapping allows correct lookup by actual video frame index.

        Args:
            labels: SLEAP Labels object
        """
        self._frame_idx_to_lf = {lf.frame_idx: lf for lf in labels.labeled_frames}

    def _get_labeled_frame(self, labels: sio.Labels, frame_idx: int) -> Optional[Any]:
        """Get LabeledFrame by actual frame index.

        Args:
            labels: SLEAP Labels object
            frame_idx: The actual video frame index (not list index)

        Returns:
            LabeledFrame if found, None otherwise
        """
        # Use cached mapping if available
        if self._frame_idx_to_lf:
            return self._frame_idx_to_lf.get(frame_idx)
        # Fallback: build mapping on demand
        return {lf.frame_idx: lf for lf in labels.labeled_frames}.get(frame_idx)

    @property
    def explanation_generator(self) -> Optional[ExplanationGenerator]:
        """Get the explanation generator for this tracker.

        Returns:
            The explanation generator instance, or None if not configured.
        """
        return self._explanation_generator

    @explanation_generator.setter
    def explanation_generator(self, generator: Optional[ExplanationGenerator]) -> None:
        """Set the explanation generator for this tracker.

        Args:
            generator: The explanation generator to use.
        """
        self._explanation_generator = generator

    @property
    def explanation_store(self) -> Optional[InstanceExplanationStore]:
        """Get the instance explanation store.

        Returns:
            The InstanceExplanationStore, or None if not configured.
        """
        return self._explanation_store

    @explanation_store.setter
    def explanation_store(self, store: Optional[InstanceExplanationStore]) -> None:
        """Set the instance explanation store.

        Args:
            store: The InstanceExplanationStore to use for logging decisions.
        """
        self._explanation_store = store

    def _create_rfid_decision_record(
        self,
        frame_idx: int,
        instance_idx: int,
        decision_type: DecisionType,
        assigned_track_id: Optional[str],
        previous_track_id: Optional[str],
        summary: str,
        reasons: List[str],
        rfid_ping_present: bool = False,
        candidate_rfids: Optional[List[CandidateScore]] = None,
        winning_rfid_id: Optional[str] = None,
        winning_probability: Optional[float] = None,
        is_propagation: bool = False,
        propagation_source_frame: Optional[int] = None,
        heatmap_probability: Optional[float] = None,
        instance_centroid: Optional[Tuple[float, float]] = None,
    ) -> RFIDDecisionRecord:
        """Create an RFIDDecisionRecord for this tracker.

        Args:
            frame_idx: Frame index.
            instance_idx: Instance index within the frame.
            decision_type: Type of decision made.
            assigned_track_id: Track ID assigned (None if no assignment).
            previous_track_id: Previous track ID if any.
            summary: Human-readable summary.
            reasons: List of reasons explaining the decision.
            rfid_ping_present: Whether an RFID ping was present.
            candidate_rfids: List of candidate RFID IDs with scores.
            winning_rfid_id: RFID ID that won the assignment.
            winning_probability: Probability of the winning RFID.
            is_propagation: Whether this was propagated from another frame.
            propagation_source_frame: Source frame if propagated.
            heatmap_probability: Probability from RFID heatmap.
            instance_centroid: (x, y) centroid of the instance.

        Returns:
            RFIDDecisionRecord for this decision.
        """
        return RFIDDecisionRecord(
            frame_idx=frame_idx,
            instance_idx=instance_idx,
            tracker_name=self.name,
            tracker_priority=self.priority,
            decision_type=decision_type,
            assigned_track_id=assigned_track_id,
            previous_track_id=previous_track_id,
            summary=summary,
            reasons=reasons,
            rfid_ping_present=rfid_ping_present,
            candidate_rfids=candidate_rfids or [],
            winning_rfid_id=winning_rfid_id,
            winning_probability=winning_probability,
            is_propagation=is_propagation,
            propagation_source_frame=propagation_source_frame,
            heatmap_probability=heatmap_probability,
            instance_centroid=instance_centroid,
        )

    def _broadcast_rfid_no_match(
        self,
        labels: sio.Labels,
        frame_idx: int,
        rfid_id: str,
        no_match_reason: str,
        instance_probabilities: Dict[int, float],
        max_probability_found: float,
        rfid_unit_label: Optional[str] = None,
    ) -> None:
        """Broadcast an RFID no-match event to all instances in a frame.

        When an RFID ping has no match, this creates RFIDNoMatchBroadcast
        records for ALL instances in the frame to document why the ping
        didn't match any of them.

        Args:
            labels: SLEAP Labels object.
            frame_idx: Frame index.
            rfid_id: The RFID ID that had no match.
            no_match_reason: Reason why no match was found.
            instance_probabilities: Dict mapping instance_idx to probability at that location.
            max_probability_found: Maximum probability found across all instances.
            rfid_unit_label: The unit label of the RFID receiver.
        """
        if self._explanation_store is None:
            return

        lf = self._get_labeled_frame(labels, frame_idx)
        if lf is None:
            return

        for inst_idx, inst in enumerate(lf.instances):
            prob_at_inst = instance_probabilities.get(inst_idx, 0.0)
            centroid = get_centroid(inst)
            centroid_tuple = tuple(centroid.tolist()) if centroid is not None else None

            record = RFIDNoMatchBroadcast(
                frame_idx=frame_idx,
                instance_idx=inst_idx,
                tracker_name=self.name,
                tracker_priority=self.priority,
                decision_type=DecisionType.NO_MATCH,
                assigned_track_id=None,
                previous_track_id=None,
                summary=f"RFID {rfid_id} had no match in this frame",
                reasons=[no_match_reason],
                rfid_id=rfid_id,
                no_match_reason=no_match_reason,
                probability_at_this_instance=prob_at_inst,
                max_probability_found=max_probability_found,
                rfid_unit_label=rfid_unit_label,
            )
            self._explanation_store.add(record)

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
                # Check bounds for both frame and instance indices
                if frame_idx >= probabilities_df.shape[0]:
                    continue
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

    def _is_temporary_track(self, track) -> bool:
        """Check if a track is temporary.

        A track is considered temporary if:
        - It's a TrackContext with temporary_track=True
        - Its name starts with 'tracklet' (naming convention)

        Args:
            track: Track object (sio.Track or TrackContext)

        Returns:
            True if the track is temporary, False otherwise.
        """
        if track is None:
            return False
        if isinstance(track, TrackContext):
            return track.temporary_track or track.name.startswith("tracklet")
        if hasattr(track, 'name'):
            return track.name.startswith("tracklet")
        return False

    def assign_track_ids(self, probabilities_df, labels, clear_existing_tracks: bool = True):
        """Assign track IDs to instances based on probabilities DataFrame.

        When temporary tracklets exist, this method propagates RFID assignments
        to the entire tracklet: if instance X on frame Y matches RFID ID "041B...",
        all instances in that temporary tracklet get assigned to "041B...".

        Temporary tracklets that don't get an RFID match remain untouched.

        Args:
            probabilities_df: DataFrame of shape (num_frames, max_instances) where
                each cell contains {rfid_id: probability} dict or None
            labels: SLEAP Labels object to assign tracks to
            clear_existing_tracks: If True, clear all existing track assignments before
                assigning new ones. If False, preserve existing temporary tracks and
                propagate RFID assignments to entire tracklets. Defaults to True.
        """
        # Get existing tracklets BEFORE any modifications
        tracklets = self.get_current_tracklets(labels)

        if clear_existing_tracks:
            # Old behavior: clear all tracks and assign directly
            for lf in labels:
                for inst in lf.instances:
                    inst.track = None
            tracklets = None
            # Direct frame-by-frame assignment
            self._assign_tracks_from_probabilities(probabilities_df, labels)
        elif tracklets is not None:
            # New behavior: propagate RFID to entire temporary tracklets
            self._assign_tracks_propagating_to_tracklets(probabilities_df, labels, tracklets)
        else:
            # No tracklets and not clearing - direct assignment
            self._assign_tracks_from_probabilities(probabilities_df, labels)

    def _assign_tracks_propagating_to_tracklets(self, probabilities_df, labels, tracklets):
        """Assign RFID IDs by propagating to entire temporary tracklets.

        For each temporary tracklet:
        1. Scan through all frames to find ANY RFID probability match
        2. When a match is found, assign that RFID ID to the ENTIRE tracklet
        3. Tracklets with no matches remain unchanged
        4. Conflict resolution: if multiple tracklets want the same RFID and overlap
           in frames, only the one with highest probability wins

        Args:
            probabilities_df: DataFrame of probabilities
            labels: SLEAP Labels object
            tracklets: Dict of {track_name: [(frame_idx, instance_idx), ...]}
        """
        track_cache = {}  # Cache of track_name -> sio.Track object

        # First pass: find best RFID match for each temporary tracklet
        tracklet_matches = {}  # {track_name: (rfid, prob, frame_idx, frame_instance_list)}

        for track_name, frame_instance_list in tracklets.items():
            # Check if this is a temporary track (by naming convention)
            if not track_name.startswith("tracklet"):
                continue

            # Find the best RFID match across all frames in this tracklet
            best_rfid = None
            best_prob = 0
            best_frame_idx = None

            for frame_idx, pose_idx in frame_instance_list:
                # Check bounds
                if frame_idx >= probabilities_df.shape[0]:
                    continue
                if pose_idx >= probabilities_df.shape[1]:
                    continue

                probs_dict = probabilities_df.loc[frame_idx, pose_idx]
                if probs_dict is None:
                    continue

                # Find best RFID for this instance
                for rfid_name, prob in probs_dict.items():
                    if prob > best_prob:
                        best_prob = prob
                        best_rfid = rfid_name
                        best_frame_idx = frame_idx

            if best_rfid is not None and best_prob > 0:
                tracklet_matches[track_name] = (best_rfid, best_prob, best_frame_idx, frame_instance_list)

        # Second pass: resolve conflicts - group by RFID
        rfid_to_tracklets = {}  # {rfid: [(track_name, prob, frame_idx, frame_instance_list), ...]}
        for track_name, (rfid, prob, frame_idx, frame_instance_list) in tracklet_matches.items():
            if rfid not in rfid_to_tracklets:
                rfid_to_tracklets[rfid] = []
            rfid_to_tracklets[rfid].append((track_name, prob, frame_idx, frame_instance_list))

        # Resolve conflicts: for each RFID, find overlapping tracklets and pick highest prob
        assigned_tracklets = set()
        conflict_losers = set()

        for rfid, candidates in rfid_to_tracklets.items():
            if len(candidates) == 1:
                # No conflict - assign directly
                track_name, prob, frame_idx, frame_instance_list = candidates[0]
                self._assign_rfid_to_tracklet(
                    labels, track_name, frame_instance_list,
                    rfid, prob, frame_idx, track_cache
                )
                assigned_tracklets.add(track_name)
            else:
                # Multiple tracklets want this RFID - check for overlaps
                # Build frame sets for each tracklet
                tracklet_frames = {}
                for track_name, prob, frame_idx, frame_instance_list in candidates:
                    tracklet_frames[track_name] = set(f for f, _ in frame_instance_list)

                # Find overlapping groups
                # For simplicity, we'll resolve globally: highest prob wins for this RFID
                # Sort by probability descending
                candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)

                # Track which frames have been assigned
                assigned_frames = set()

                for track_name, prob, frame_idx, frame_instance_list in candidates_sorted:
                    frames = tracklet_frames[track_name]
                    # Check if any frames overlap with already assigned
                    if frames & assigned_frames:
                        # Conflict - this tracklet loses, preserve its original track
                        conflict_losers.add(track_name)
                    else:
                        # No overlap - assign
                        self._assign_rfid_to_tracklet(
                            labels, track_name, frame_instance_list,
                            rfid, prob, frame_idx, track_cache
                        )
                        assigned_tracklets.add(track_name)
                        assigned_frames.update(frames)

        # Log results
        total_tracklets = len([t for t in tracklets if t.startswith("tracklet")])
        preserved = total_tracklets - len(assigned_tracklets)
        print(f"  Assigned RFID IDs to {len(assigned_tracklets)}/{total_tracklets} temporary tracklets")
        print(f"  {len(conflict_losers)} tracklets lost conflict resolution (preserved)")
        print(f"  {preserved - len(conflict_losers)} tracklets had no RFID match (preserved)")

    def _assign_rfid_to_tracklet(self, labels, old_track_name, frame_instance_list,
                                  rfid_name, prob, source_frame_idx, track_cache):
        """Assign an RFID ID to all instances in a tracklet using rename_track_globally.

        This is a RENAME operation - it propagates the RFID identity to ALL
        frames of the tracklet. For temporary tracklets, this is ALWAYS allowed
        regardless of priority.

        Args:
            labels: SLEAP Labels object
            old_track_name: Original temporary track name
            frame_instance_list: List of (frame_idx, instance_idx) tuples
            rfid_name: RFID ID to assign
            prob: Probability of the match
            source_frame_idx: Frame where the match was found
            track_cache: Cache of track_name -> sio.Track
        """
        # Use rename_track_globally for the rename operation
        # This is ALWAYS allowed for temporary tracklets (which is what we expect here)
        reason = f"RFID match (prob={prob:.6f})"
        success = self.rename_track_globally(
            labels=labels,
            old_track_name=old_track_name,
            new_track_name=rfid_name,
            reason=reason,
        )

        if success:
            # Update track cache
            new_track = next((t for t in labels.tracks if t.name == rfid_name), None)
            if new_track is not None:
                track_cache[rfid_name] = new_track

            # Log to explanation store if available
            if self._explanation_store is not None:
                for frame_idx, pose_idx in frame_instance_list:
                    lf = self._get_labeled_frame(labels, frame_idx)
                    if lf is None or pose_idx >= len(lf.instances):
                        continue

                    inst = lf.instances[pose_idx]
                    centroid = get_centroid(inst)
                    centroid_tuple = tuple(centroid.tolist()) if centroid is not None else None

                    is_source_frame = (frame_idx == source_frame_idx)
                    decision_type = DecisionType.MATCHED if is_source_frame else DecisionType.PROPAGATED

                    record = self._create_rfid_decision_record(
                        frame_idx=frame_idx,
                        instance_idx=pose_idx,
                        decision_type=decision_type,
                        assigned_track_id=rfid_name,
                        previous_track_id=old_track_name,
                        summary=f"RFID {rfid_name} assigned via tracklet rename" if is_source_frame
                                else f"Propagated from frame {source_frame_idx}",
                        reasons=[reason if is_source_frame else f"Propagated from frame {source_frame_idx}"],
                        rfid_ping_present=is_source_frame,
                        winning_rfid_id=rfid_name,
                        winning_probability=prob,
                        is_propagation=not is_source_frame,
                        propagation_source_frame=None if is_source_frame else source_frame_idx,
                        heatmap_probability=prob,
                        instance_centroid=centroid_tuple,
                    )
                    self._explanation_store.add(record)

    def _assign_tracks_from_tracklet_pairs(self, tracklet_id_pairs, labels):
        """Assign tracks to instances based on tracklet ID pairs using priority resolution.

        Uses majority voting for tracklets with multiple ID assignments and
        handles conflicts between overlapping tracklets. Uses `rename_track_globally()`
        for track assignments to ensure proper priority-based conflict resolution.

        Args:
            tracklet_id_pairs: List of (tracklet, [track_ids]) where tracklet is
                a list of (frame_idx, instance_idx) tuples
            labels: SLEAP Labels object
        """
        def clear_tracklet_tracks(tracklet, labels):
            """Clear tracks from all poses in a tracklet."""
            for frame_idx, pose_idx in tracklet:
                lf = self._get_labeled_frame(labels, frame_idx)
                if lf is not None and pose_idx < len(lf.instances):
                    lf.instances[pose_idx].track = None

        def get_tracklet_old_name(tracklet, labels) -> Optional[str]:
            """Get the current track name of a tracklet's first instance."""
            if not tracklet:
                return None
            frame_idx, pose_idx = tracklet[0]
            lf = self._get_labeled_frame(labels, frame_idx)
            if lf is None or pose_idx >= len(lf.instances):
                return None
            inst = lf.instances[pose_idx]
            if inst.track is None:
                return None
            return inst.track.name if hasattr(inst.track, 'name') else str(inst.track)

        def assign_tracklet_with_priority(
            tracklet, track_name, labels, reason="Tracklet assignment"
        ) -> bool:
            """Assign track to all poses in a tracklet using rename_track_globally.

            Returns True if assignment succeeded, False if blocked.
            """
            if track_name is None:
                clear_tracklet_tracks(tracklet, labels)
                return True

            # Get current track name of the tracklet
            old_name = get_tracklet_old_name(tracklet, labels)
            if old_name is None:
                # No existing track - use assign_with_priority_resolution on first instance
                if tracklet:
                    frame_idx, pose_idx = tracklet[0]
                    success = self.assign_with_priority_resolution(
                        labels=labels,
                        frame_idx=frame_idx,
                        instance_idx=pose_idx,
                        new_track_name=track_name,
                        reason=reason,
                        propagate_to_tracklet=True,
                    )
                    return success
                return False

            # Use rename_track_globally for the rename operation
            return self.rename_track_globally(
                labels=labels,
                old_track_name=old_name,
                new_track_name=track_name,
                reason=reason,
            )

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
                                clear_tracklet_tracks(other_tracklet, labels)
                            elif len(track_id_list) < len(other_track_ids):
                                clear_tracklet_tracks(tracklet, labels)
                                conflict = True
                                break
                            else:
                                # Equal - clear both
                                clear_tracklet_tracks(other_tracklet, labels)
                                clear_tracklet_tracks(tracklet, labels)
                                conflict = True
                                break

                if not conflict:
                    # Build reason with voting results
                    id_counts = Counter(track_id_list)
                    filtered_counts = {k: v for k, v in id_counts.items() if k is not None}
                    total_votes = sum(filtered_counts.values())
                    winning_votes = filtered_counts.get(track_id, 0)
                    vote_ratio = winning_votes / total_votes if total_votes > 0 else 0.0

                    reason = f"RFID majority vote assignment (votes={winning_votes}/{total_votes}, ratio={vote_ratio:.2f})"

                    # Use priority-based assignment
                    assign_tracklet_with_priority(
                        tracklet, track_id, labels,
                        reason=reason,
                    )

    def _assign_tracks_from_probabilities(self, probabilities_df, labels):
        """Assign tracks directly from probabilities using priority-based resolution.

        For each frame, assigns the best RFID identity to each instance,
        resolving conflicts by giving priority to higher probabilities.

        Uses `assign_with_priority_resolution()` with `propagate_to_tracklet=True`
        so that if an instance belongs to a temporary tracklet, the RFID identity
        is propagated to ALL frames of that tracklet.

        Args:
            probabilities_df: DataFrame of probabilities (indexed by actual frame indices)
            labels: SLEAP Labels object
        """
        # Track which RFIDs have been assigned in each frame to avoid conflicts
        # {frame_idx: set of assigned rfid_names}
        assigned_rfids_by_frame = {}

        # Track which tracklets have been assigned to avoid duplicate propagation
        assigned_tracklets = set()

        # Build frame mapping
        self._build_frame_mapping(labels)

        # Iterate over labeled frames directly
        for lf in labels.labeled_frames:
            frame_idx = lf.frame_idx
            # Skip if frame not in probabilities DataFrame
            if frame_idx not in probabilities_df.index:
                continue

            num_instances = len(lf.instances)

            # Initialize frame's assigned RFIDs
            if frame_idx not in assigned_rfids_by_frame:
                assigned_rfids_by_frame[frame_idx] = set()

            # Collect best identities for each instance in this frame
            instance_assignments = []  # [(instance_idx, rfid_name, probability), ...]

            for pose_idx in range(num_instances):
                if pose_idx >= probabilities_df.shape[1]:
                    continue

                inst = lf.instances[pose_idx]

                # Check if this instance's tracklet was already assigned
                if inst.track is not None:
                    track_name = inst.track.name if hasattr(inst.track, 'name') else str(inst.track)
                    if track_name in assigned_tracklets:
                        continue  # Skip - already assigned via propagation
                    # If track doesn't start with 'tracklet', it's already permanent
                    if not track_name.startswith('tracklet'):
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

            # Sort by probability descending so highest prob gets first pick
            instance_assignments.sort(key=lambda x: x[2], reverse=True)

            for pose_idx, rfid_name, prob in instance_assignments:
                inst = lf.instances[pose_idx]

                # Check if this RFID is already assigned in this frame
                if rfid_name in assigned_rfids_by_frame[frame_idx]:
                    # Log skipped assignment due to conflict
                    if self._explanation_store is not None:
                        centroid = get_centroid(inst)
                        centroid_tuple = tuple(centroid.tolist()) if centroid is not None else None

                        probs_dict = probabilities_df.loc[frame_idx, pose_idx] or {}
                        candidate_rfids = [
                            CandidateScore(
                                candidate_id=name,
                                score=p,
                                passed_thresholds=(name not in assigned_rfids_by_frame[frame_idx]),
                                threshold_results={},
                                rejection_reason="Already assigned to another instance" if name in assigned_rfids_by_frame[frame_idx] else None,
                            )
                            for name, p in probs_dict.items()
                        ]

                        record = self._create_rfid_decision_record(
                            frame_idx=frame_idx,
                            instance_idx=pose_idx,
                            decision_type=DecisionType.SKIPPED,
                            assigned_track_id=None,
                            previous_track_id=None,
                            summary=f"RFID {rfid_name} already assigned to another instance",
                            reasons=[f"Best match {rfid_name} (prob={prob:.4f}) already assigned"],
                            rfid_ping_present=True,
                            candidate_rfids=candidate_rfids,
                            winning_rfid_id=None,
                            winning_probability=prob,
                            heatmap_probability=prob,
                            instance_centroid=centroid_tuple,
                        )
                        self._explanation_store.add(record)
                    continue

                # Get old track name
                old_track_name = None
                if inst.track is not None:
                    old_track_name = inst.track.name if hasattr(inst.track, 'name') else str(inst.track)

                # Generate explanation dict
                explanation_dict = None
                if self._explanation_generator is not None:
                    all_scores = {
                        str(idx): p for idx, name, p in instance_assignments if name == rfid_name
                    }
                    explanation = self._explanation_generator.explain_match(
                        track_id=rfid_name,
                        instance_idx=pose_idx,
                        score=prob,
                        all_scores=all_scores,
                        feature_type="feature",
                        feature_context={"probability": prob},
                    )
                    explanation_dict = explanation.to_dict() if hasattr(explanation, 'to_dict') else explanation

                # Use assign_with_priority_resolution with propagate_to_tracklet=True
                # This will propagate RFID to entire tracklet if instance belongs to one
                success = self.assign_with_priority_resolution(
                    labels=labels,
                    frame_idx=frame_idx,
                    instance_idx=pose_idx,
                    new_track_name=rfid_name,
                    reason=f"RFID probability assignment (prob={prob:.3f})",
                    explanation=explanation_dict,
                    propagate_to_tracklet=True,  # Propagate to entire tracklet
                )

                if success:
                    # Mark RFID as used in this frame
                    assigned_rfids_by_frame[frame_idx].add(rfid_name)

                    # If there was a tracklet, mark it as assigned
                    if old_track_name is not None and old_track_name.startswith('tracklet'):
                        assigned_tracklets.add(old_track_name)
                        # Also mark RFID as used in all frames of the tracklet
                        tracklet_frames = self.get_tracklet_frames(labels, rfid_name)
                        for tf_idx, _ in tracklet_frames:
                            if tf_idx not in assigned_rfids_by_frame:
                                assigned_rfids_by_frame[tf_idx] = set()
                            assigned_rfids_by_frame[tf_idx].add(rfid_name)

                    # Log to explanation store
                    if self._explanation_store is not None:
                        centroid = get_centroid(inst)
                        centroid_tuple = tuple(centroid.tolist()) if centroid is not None else None

                        probs_dict = probabilities_df.loc[frame_idx, pose_idx] or {}
                        candidate_rfids = [
                            CandidateScore(
                                candidate_id=name,
                                score=p,
                                passed_thresholds=(name == rfid_name),
                                threshold_results={},
                                rejection_reason=None if name == rfid_name else "Lower probability",
                            )
                            for name, p in probs_dict.items()
                        ]

                        record = self._create_rfid_decision_record(
                            frame_idx=frame_idx,
                            instance_idx=pose_idx,
                            decision_type=DecisionType.MATCHED,
                            assigned_track_id=rfid_name,
                            previous_track_id=old_track_name,
                            summary=f"Matched to RFID {rfid_name} with probability {prob:.4f}",
                            reasons=[f"Best RFID match: {rfid_name} (prob={prob:.4f})"],
                            rfid_ping_present=True,
                            candidate_rfids=candidate_rfids,
                            winning_rfid_id=rfid_name,
                            winning_probability=prob,
                            heatmap_probability=prob,
                            instance_centroid=centroid_tuple,
                        )
                        self._explanation_store.add(record)

    def get_track_context(self, labels, frame_idx, track) -> Optional[TrackContext]:
        """Get the TrackContext for a track in a specific frame.

        Args:
            labels: SLEAP Labels object
            frame_idx: Frame index
            track: Track to look for

        Returns:
            TrackContext if found, None otherwise
        """
        lf = self._get_labeled_frame(labels, frame_idx)
        if lf is None:
            return None

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
            frame_idx: Frame index (actual video frame index)
            track: Track to look for

        Returns:
            True if track exists in frame, False otherwise
        """
        lf = self._get_labeled_frame(labels, frame_idx)
        if lf is None:
            return False

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
            frame_idx: Frame index (actual video frame index)
            track: Track to look for

        Returns:
            Instance index if found, None otherwise
        """
        lf = self._get_labeled_frame(labels, frame_idx)
        if lf is None:
            return None

        for idx, inst in enumerate(lf.instances):
            if inst.track is not None:
                if hasattr(inst.track, 'name') and inst.track.name == track.name:
                    return idx
        return None

    def assign_track(self, labels, frame_idx, instance_idx, track, track_context=None) -> None:
        """Assign a track to an instance.

        Args:
            labels: SLEAP Labels object
            frame_idx: Frame index (actual video frame index)
            instance_idx: Instance index
            track: Track to assign
            track_context: Optional TrackContext (not used in simple assignment)
        """
        lf = self._get_labeled_frame(labels, frame_idx)
        if lf is None:
            return
        if instance_idx >= len(lf.instances):
            return

        # Find or create the track in labels.tracks
        existing = next((t for t in labels.tracks if t.name == track.name), None)
        if existing:
            lf.instances[instance_idx].track = existing
        else:
            labels.tracks.append(track)
            lf.instances[instance_idx].track = track

    def remove_track(self, labels, frame_idx, instance_idx) -> None:
        """Remove track from an instance.

        Args:
            labels: SLEAP Labels object
            frame_idx: Frame index (actual video frame index)
            instance_idx: Instance index
        """
        lf = self._get_labeled_frame(labels, frame_idx)
        if lf is None:
            return
        if instance_idx >= len(lf.instances):
            return

        lf.instances[instance_idx].track = None

    def get_next_frame(self, labels, current_frame, direction) -> Optional[int]:
        """Get the next labeled frame index in a given direction.

        Args:
            labels: SLEAP Labels object
            current_frame: Current frame index (actual video frame index)
            direction: 1 for forward, -1 for backward

        Returns:
            Next labeled frame index if valid, None otherwise
        """
        # Build sorted list of labeled frame indices if not cached
        if not self._frame_idx_to_lf:
            self._build_frame_mapping(labels)

        labeled_indices = sorted(self._frame_idx_to_lf.keys())
        if not labeled_indices:
            return None

        if direction > 0:
            # Find next labeled frame after current_frame
            for idx in labeled_indices:
                if idx > current_frame:
                    return idx
        else:
            # Find previous labeled frame before current_frame
            for idx in reversed(labeled_indices):
                if idx < current_frame:
                    return idx
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

    def get_config(self) -> Dict[str, Any]:
        """Get feature tracker configuration.

        Subclasses should override to include their specific parameters.

        Returns:
            Dict of configuration parameters for this feature tracker.
        """
        config = super().get_config()
        config.update({
            "only_apply_to_tracklets": self.only_apply_to_tracklets,
            "min_votes_for_identity": self.min_votes_for_identity,
            "switch_confidence_threshold": self.switch_confidence_threshold,
            "split_on_switch": self.split_on_switch,
            "require_partner_validation": self.require_partner_validation,
        })
        return config

    # =========================================================================
    # Identity Switch Detection Methods
    # =========================================================================

    def analyze_tracklet_consistency(
        self,
        tracklet_name: str,
        votes: List[IdentityVote],
        tracklet_frames: List[Tuple[int, int]],
    ) -> TrackletIdentityAnalysis:
        """Analyze identity votes for a tracklet to detect switches.

        This method examines all identity votes for a tracklet and determines
        whether the tracklet has a consistent identity or contains switches.

        Algorithm:
        1. Group votes by identity
        2. Sort votes temporally
        3. Find dominant identity for each temporal region
        4. Detect changes in dominant identity (potential switch points)
        5. Validate switches have sufficient votes on each side
        6. Calculate confidence scores

        Args:
            tracklet_name: Name of the tracklet being analyzed.
            votes: All identity votes for this tracklet.
            tracklet_frames: All (frame_idx, instance_idx) in the tracklet.

        Returns:
            TrackletIdentityAnalysis with segments, switches, and recommendations.
        """
        if not votes:
            # No votes - cannot determine identity
            frame_indices = [f for f, _ in tracklet_frames]
            return TrackletIdentityAnalysis(
                tracklet_name=tracklet_name,
                frame_range=(min(frame_indices), max(frame_indices)) if frame_indices else (0, 0),
                votes_by_identity={},
                segments=[],
                detected_switches=[],
                is_consistent=True,
                dominant_identity=None,
                confidence=0.0,
            )

        # Group votes by identity
        votes_by_identity: Dict[str, List[IdentityVote]] = {}
        for vote in votes:
            if vote.identity not in votes_by_identity:
                votes_by_identity[vote.identity] = []
            votes_by_identity[vote.identity].append(vote)

        # Sort votes within each identity by frame
        for identity in votes_by_identity:
            votes_by_identity[identity].sort(key=lambda v: v.frame_idx)

        # Get tracklet frame range
        frame_indices = [f for f, _ in tracklet_frames]
        first_frame = min(frame_indices)
        last_frame = max(frame_indices)

        # Find all unique identities
        unique_identities = list(votes_by_identity.keys())

        # If only one identity, tracklet is consistent
        if len(unique_identities) == 1:
            identity = unique_identities[0]
            total_votes = len(votes_by_identity[identity])
            segment = TrackletSegment(
                start_frame=first_frame,
                end_frame=last_frame,
                dominant_identity=identity,
                vote_count=total_votes,
                total_votes_in_segment=total_votes,
                confidence=1.0,
            )
            return TrackletIdentityAnalysis(
                tracklet_name=tracklet_name,
                frame_range=(first_frame, last_frame),
                votes_by_identity=votes_by_identity,
                segments=[segment],
                detected_switches=[],
                is_consistent=True,
                dominant_identity=identity,
                confidence=1.0,
            )

        # Multiple identities - need to detect switches
        # For each pair of identities, find potential switch points
        detected_switches: List[DetectedSwitch] = []
        segments: List[TrackletSegment] = []

        # Get the two most common identities (most likely to be the real switch)
        identity_counts = {id: len(v) for id, v in votes_by_identity.items()}
        sorted_identities = sorted(identity_counts.keys(), key=lambda x: identity_counts[x], reverse=True)

        # For two-identity case, find optimal switch point
        if len(sorted_identities) >= 2:
            identity_a = sorted_identities[0]
            identity_b = sorted_identities[1]

            # Find switch point between the two main identities
            switch_result = self.detect_switch_point(
                votes_identity_a=votes_by_identity[identity_a],
                votes_identity_b=votes_by_identity[identity_b],
                min_votes=self.min_votes_for_identity,
            )

            if switch_result is not None:
                switch_frame, confidence = switch_result

                # Determine which identity is before and which is after
                votes_a_before = sum(1 for v in votes_by_identity[identity_a] if v.frame_idx < switch_frame)
                votes_a_after = sum(1 for v in votes_by_identity[identity_a] if v.frame_idx >= switch_frame)
                votes_b_before = sum(1 for v in votes_by_identity[identity_b] if v.frame_idx < switch_frame)
                votes_b_after = sum(1 for v in votes_by_identity[identity_b] if v.frame_idx >= switch_frame)

                # Determine the direction of the switch
                if votes_a_before > votes_a_after and votes_b_after > votes_b_before:
                    identity_before = identity_a
                    identity_after = identity_b
                    v_before = votes_a_before
                    v_after = votes_b_after
                else:
                    identity_before = identity_b
                    identity_after = identity_a
                    v_before = votes_b_before
                    v_after = votes_a_after

                detected_switch = DetectedSwitch(
                    switch_frame=switch_frame,
                    identity_before=identity_before,
                    identity_after=identity_after,
                    votes_before=v_before,
                    votes_after=v_after,
                    confidence=confidence,
                )
                detected_switches.append(detected_switch)

                # Create two segments
                segment_before = TrackletSegment(
                    start_frame=first_frame,
                    end_frame=switch_frame - 1,
                    dominant_identity=identity_before,
                    vote_count=v_before,
                    total_votes_in_segment=votes_a_before + votes_b_before,
                    confidence=v_before / (votes_a_before + votes_b_before) if (votes_a_before + votes_b_before) > 0 else 0.0,
                )
                segment_after = TrackletSegment(
                    start_frame=switch_frame,
                    end_frame=last_frame,
                    dominant_identity=identity_after,
                    vote_count=v_after,
                    total_votes_in_segment=votes_a_after + votes_b_after,
                    confidence=v_after / (votes_a_after + votes_b_after) if (votes_a_after + votes_b_after) > 0 else 0.0,
                )
                segments = [segment_before, segment_after]

        # If no valid switch detected, use majority voting
        if not detected_switches:
            dominant_identity = sorted_identities[0]
            total_votes = sum(len(v) for v in votes_by_identity.values())
            dominant_votes = len(votes_by_identity[dominant_identity])
            segment = TrackletSegment(
                start_frame=first_frame,
                end_frame=last_frame,
                dominant_identity=dominant_identity,
                vote_count=dominant_votes,
                total_votes_in_segment=total_votes,
                confidence=dominant_votes / total_votes if total_votes > 0 else 0.0,
            )
            segments = [segment]

            return TrackletIdentityAnalysis(
                tracklet_name=tracklet_name,
                frame_range=(first_frame, last_frame),
                votes_by_identity=votes_by_identity,
                segments=segments,
                detected_switches=[],
                is_consistent=True,
                dominant_identity=dominant_identity,
                confidence=dominant_votes / total_votes if total_votes > 0 else 0.0,
            )

        # Return analysis with detected switch
        return TrackletIdentityAnalysis(
            tracklet_name=tracklet_name,
            frame_range=(first_frame, last_frame),
            votes_by_identity=votes_by_identity,
            segments=segments,
            detected_switches=detected_switches,
            is_consistent=False,
            dominant_identity=None,  # Multiple identities
            confidence=detected_switches[0].confidence if detected_switches else 0.0,
        )

    def detect_switch_point(
        self,
        votes_identity_a: List[IdentityVote],
        votes_identity_b: List[IdentityVote],
        min_votes: int,
    ) -> Optional[Tuple[int, float]]:
        """Find the optimal frame to split between two identities.

        Searches for the frame F that maximizes:
            score = (votes_A_before_F / total_A) * (votes_B_after_F / total_B)

        This finds the frame that best separates the two identity clusters.

        Args:
            votes_identity_a: Votes for identity A.
            votes_identity_b: Votes for identity B.
            min_votes: Minimum votes required on each side.

        Returns:
            (switch_frame, confidence) if valid switch found, None otherwise.
        """
        if not votes_identity_a or not votes_identity_b:
            return None

        # Get all unique frames from both vote lists
        all_frames = sorted(set(
            [v.frame_idx for v in votes_identity_a] +
            [v.frame_idx for v in votes_identity_b]
        ))

        if len(all_frames) < 2:
            return None

        # Extract frame indices
        frames_a = [v.frame_idx for v in votes_identity_a]
        frames_b = [v.frame_idx for v in votes_identity_b]
        total_a = len(frames_a)
        total_b = len(frames_b)

        best_frame = None
        best_score = -1.0

        # Test each possible switch point (gaps between frames)
        for i in range(len(all_frames) - 1):
            switch_frame = all_frames[i] + 1  # First frame of "after" segment

            # Count votes on each side
            a_before = sum(1 for f in frames_a if f < switch_frame)
            a_after = total_a - a_before
            b_before = sum(1 for f in frames_b if f < switch_frame)
            b_after = total_b - b_before

            # Check minimum vote requirements
            # Try both orderings: A before / B after, or B before / A after
            score_ab = 0.0
            score_ba = 0.0

            if a_before >= min_votes and b_after >= min_votes:
                score_ab = (a_before / total_a) * (b_after / total_b)

            if b_before >= min_votes and a_after >= min_votes:
                score_ba = (b_before / total_b) * (a_after / total_a)

            max_score = max(score_ab, score_ba)

            if max_score > best_score:
                best_score = max_score
                best_frame = switch_frame

        # Check if the best score meets the confidence threshold
        if best_frame is not None and best_score >= self.switch_confidence_threshold:
            return (best_frame, best_score)

        return None

    def cross_validate_switches(
        self,
        analyses: Dict[str, TrackletIdentityAnalysis],
        frame_tolerance: int = 10,
    ) -> List[Tuple[str, str, DetectedSwitch, DetectedSwitch]]:
        """Find pairs of tracklets with complementary switches.

        A complementary switch occurs when:
        - Tracklet A switches from identity X -> Y at frame F
        - Tracklet B switches from identity Y -> X at approximately frame F

        This pattern strongly indicates a motion tracker ID switch that
        should be repaired by swapping the post-switch segments.

        Args:
            analyses: Dict mapping tracklet_name -> TrackletIdentityAnalysis.
            frame_tolerance: Maximum frame difference to consider switches related.

        Returns:
            List of (tracklet_a, tracklet_b, switch_a, switch_b) tuples
            representing cross-validated switch pairs.
        """
        cross_validated: List[Tuple[str, str, DetectedSwitch, DetectedSwitch]] = []

        # Build list of tracklets with detected switches
        tracklets_with_switches = [
            (name, analysis)
            for name, analysis in analyses.items()
            if analysis.detected_switches
        ]

        # Check each pair for complementary switches
        for i, (name_a, analysis_a) in enumerate(tracklets_with_switches):
            for name_b, analysis_b in tracklets_with_switches[i + 1:]:
                for switch_a in analysis_a.detected_switches:
                    for switch_b in analysis_b.detected_switches:
                        # Check if switches are complementary (identities swap)
                        if (switch_a.identity_before == switch_b.identity_after and
                            switch_a.identity_after == switch_b.identity_before):
                            # Check temporal proximity
                            frame_diff = abs(switch_a.switch_frame - switch_b.switch_frame)
                            if frame_diff <= frame_tolerance:
                                # Mark as cross-validated
                                switch_a.partner_tracklet = name_b
                                switch_a.partner_switch_frame = switch_b.switch_frame
                                switch_b.partner_tracklet = name_a
                                switch_b.partner_switch_frame = switch_a.switch_frame

                                cross_validated.append((name_a, name_b, switch_a, switch_b))

        return cross_validated

    def split_tracklet_at_frame(
        self,
        labels: sio.Labels,
        tracklet_name: str,
        tracklet_frames: List[Tuple[int, int]],
        split_frame: int,
        identity_before: str,
        identity_after: str,
        reason: str,
    ) -> Tuple[str, str]:
        """Split a tracklet at a specific frame into two separate tracks.

        Creates two new tracks:
        - Frames before split_frame get identity_before
        - Frames >= split_frame get identity_after

        Args:
            labels: SLEAP Labels object.
            tracklet_name: Original tracklet name to split.
            tracklet_frames: All (frame_idx, instance_idx) in the tracklet.
            split_frame: Frame at which to split (first frame of "after" segment).
            identity_before: Identity to assign to frames < split_frame.
            identity_after: Identity to assign to frames >= split_frame.
            reason: Explanation for the split (for track history).

        Returns:
            (track_name_before, track_name_after) tuple.
        """
        # Ensure frame mapping is built
        if not self._frame_idx_to_lf:
            self._build_frame_mapping(labels)

        # Get or create tracks for both identities
        track_before = next((t for t in labels.tracks if t.name == identity_before), None)
        if track_before is None:
            track_before = sio.Track(name=identity_before)
            labels.tracks.append(track_before)

        track_after = next((t for t in labels.tracks if t.name == identity_after), None)
        if track_after is None:
            track_after = sio.Track(name=identity_after)
            labels.tracks.append(track_after)

        # Assign instances to appropriate tracks
        for frame_idx, inst_idx in tracklet_frames:
            lf = self._frame_idx_to_lf.get(frame_idx)
            if lf is None or inst_idx >= len(lf.instances):
                continue

            inst = lf.instances[inst_idx]

            if frame_idx < split_frame:
                # Assign to identity_before
                track_context = TrackContext(
                    priority=self.priority,
                    track=track_before,
                    name=identity_before,
                    temporary_track=False,
                    valid=True,
                    track_history=[],
                )
                track_context.add_history_entry(
                    layer_name=self.name,
                    old_track_name=tracklet_name,
                    new_track_name=identity_before,
                    frame_idx=frame_idx,
                    reason=f"{reason} (before split at frame {split_frame})",
                    conflict_resolved=False,
                    propagated_from_frame=None,
                )
                inst.track = track_context
            else:
                # Assign to identity_after
                track_context = TrackContext(
                    priority=self.priority,
                    track=track_after,
                    name=identity_after,
                    temporary_track=False,
                    valid=True,
                    track_history=[],
                )
                track_context.add_history_entry(
                    layer_name=self.name,
                    old_track_name=tracklet_name,
                    new_track_name=identity_after,
                    frame_idx=frame_idx,
                    reason=f"{reason} (after split at frame {split_frame})",
                    conflict_resolved=False,
                    propagated_from_frame=None,
                )
                inst.track = track_context

            # Log to explanation store
            if self._explanation_store is not None:
                is_before = frame_idx < split_frame
                record = SwitchRepairDecisionRecord(
                    frame_idx=frame_idx,
                    instance_idx=inst_idx,
                    tracker_name=self.name,
                    tracker_priority=self.priority,
                    decision_type=DecisionType.MATCHED,
                    assigned_track_id=identity_before if is_before else identity_after,
                    previous_track_id=tracklet_name,
                    summary=f"Split tracklet at frame {split_frame}",
                    reasons=[reason],
                    switch_frame=split_frame,
                    identity_before=identity_before,
                    identity_after=identity_after,
                    repair_action="split",
                    original_tracklet=tracklet_name,
                    resulting_tracks=[identity_before, identity_after],
                )
                self._explanation_store.add(record)

        return (identity_before, identity_after)

    def swap_tracklet_segments(
        self,
        labels: sio.Labels,
        tracklet_a: str,
        tracklet_b: str,
        frames_a: List[Tuple[int, int]],
        frames_b: List[Tuple[int, int]],
        switch_frame: int,
        identity_a: str,
        identity_b: str,
        reason: str,
    ) -> None:
        """Swap post-switch segments between two tracklets.

        When a cross-validated switch is detected, this swaps the segments
        to repair the identity switch:

        Before:
            tracklet_a: [frames 0-855 = mouse_a] [frames 856-1000 = mouse_b (WRONG)]
            tracklet_b: [frames 0-855 = mouse_b] [frames 856-1000 = mouse_a (WRONG)]

        After:
            identity_a: [tracklet_a: 0-855] + [tracklet_b: 856-1000]
            identity_b: [tracklet_b: 0-855] + [tracklet_a: 856-1000]

        Args:
            labels: SLEAP Labels object.
            tracklet_a: First tracklet name.
            tracklet_b: Second tracklet name.
            frames_a: All (frame_idx, instance_idx) in tracklet_a.
            frames_b: All (frame_idx, instance_idx) in tracklet_b.
            switch_frame: Frame where the swap should occur.
            identity_a: Final identity for tracklet_a[:switch] + tracklet_b[switch:].
            identity_b: Final identity for tracklet_b[:switch] + tracklet_a[switch:].
            reason: Explanation for the swap (for track history).
        """
        # Ensure frame mapping is built
        if not self._frame_idx_to_lf:
            self._build_frame_mapping(labels)

        # Get or create tracks for both identities
        track_a = next((t for t in labels.tracks if t.name == identity_a), None)
        if track_a is None:
            track_a = sio.Track(name=identity_a)
            labels.tracks.append(track_a)

        track_b = next((t for t in labels.tracks if t.name == identity_b), None)
        if track_b is None:
            track_b = sio.Track(name=identity_b)
            labels.tracks.append(track_b)

        # Assign instances from tracklet_a
        for frame_idx, inst_idx in frames_a:
            lf = self._frame_idx_to_lf.get(frame_idx)
            if lf is None or inst_idx >= len(lf.instances):
                continue

            inst = lf.instances[inst_idx]

            if frame_idx < switch_frame:
                # Before switch: tracklet_a stays as identity_a
                assigned_identity = identity_a
                assigned_track = track_a
            else:
                # After switch: tracklet_a becomes identity_b
                assigned_identity = identity_b
                assigned_track = track_b

            track_context = TrackContext(
                priority=self.priority,
                track=assigned_track,
                name=assigned_identity,
                temporary_track=False,
                valid=True,
                track_history=[],
            )
            track_context.add_history_entry(
                layer_name=self.name,
                old_track_name=tracklet_a,
                new_track_name=assigned_identity,
                frame_idx=frame_idx,
                reason=f"{reason} (swap at frame {switch_frame})",
                conflict_resolved=False,
                propagated_from_frame=None,
            )
            inst.track = track_context

            # Log to explanation store
            if self._explanation_store is not None:
                record = SwitchRepairDecisionRecord(
                    frame_idx=frame_idx,
                    instance_idx=inst_idx,
                    tracker_name=self.name,
                    tracker_priority=self.priority,
                    decision_type=DecisionType.MATCHED,
                    assigned_track_id=assigned_identity,
                    previous_track_id=tracklet_a,
                    summary=f"Swapped segments between {tracklet_a} and {tracklet_b}",
                    reasons=[reason],
                    switch_frame=switch_frame,
                    identity_before=identity_a,
                    identity_after=identity_b,
                    partner_tracklet=tracklet_b,
                    partner_validated=True,
                    repair_action="swap",
                    original_tracklet=tracklet_a,
                    resulting_tracks=[identity_a, identity_b],
                )
                self._explanation_store.add(record)

        # Assign instances from tracklet_b
        for frame_idx, inst_idx in frames_b:
            lf = self._frame_idx_to_lf.get(frame_idx)
            if lf is None or inst_idx >= len(lf.instances):
                continue

            inst = lf.instances[inst_idx]

            if frame_idx < switch_frame:
                # Before switch: tracklet_b stays as identity_b
                assigned_identity = identity_b
                assigned_track = track_b
            else:
                # After switch: tracklet_b becomes identity_a
                assigned_identity = identity_a
                assigned_track = track_a

            track_context = TrackContext(
                priority=self.priority,
                track=assigned_track,
                name=assigned_identity,
                temporary_track=False,
                valid=True,
                track_history=[],
            )
            track_context.add_history_entry(
                layer_name=self.name,
                old_track_name=tracklet_b,
                new_track_name=assigned_identity,
                frame_idx=frame_idx,
                reason=f"{reason} (swap at frame {switch_frame})",
                conflict_resolved=False,
                propagated_from_frame=None,
            )
            inst.track = track_context

            # Log to explanation store
            if self._explanation_store is not None:
                record = SwitchRepairDecisionRecord(
                    frame_idx=frame_idx,
                    instance_idx=inst_idx,
                    tracker_name=self.name,
                    tracker_priority=self.priority,
                    decision_type=DecisionType.MATCHED,
                    assigned_track_id=assigned_identity,
                    previous_track_id=tracklet_b,
                    summary=f"Swapped segments between {tracklet_a} and {tracklet_b}",
                    reasons=[reason],
                    switch_frame=switch_frame,
                    identity_before=identity_a,
                    identity_after=identity_b,
                    partner_tracklet=tracklet_a,
                    partner_validated=True,
                    repair_action="swap",
                    original_tracklet=tracklet_b,
                    resulting_tracks=[identity_a, identity_b],
                )
                self._explanation_store.add(record)

    def collect_identity_votes(
        self,
        labels: sio.Labels,
        tracklets: Dict[str, List[Tuple[int, int]]],
        **kwargs
    ) -> Dict[str, List[IdentityVote]]:
        """Collect identity votes for each tracklet.

        Each subclass implements this based on its feature type:
        - RFID: votes from coordinate matches to RFID pings
        - Fur color: votes from color signature matches
        - Tail tattoo: votes from tattoo pattern recognition

        This is an optional method. If not implemented, subclasses should
        continue to use their existing assignment logic.

        Args:
            labels: SLEAP Labels object.
            tracklets: Dict mapping tracklet_name -> [(frame_idx, instance_idx), ...].
            **kwargs: Tracker-specific args (e.g., rfid_pings for RFID).

        Returns:
            Dict mapping tracklet_name -> list of IdentityVote objects.
            Each vote represents one observation of identity evidence.

        Raises:
            NotImplementedError: If subclass does not implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement collect_identity_votes(). "
            "Subclasses should implement this method to enable switch detection."
        )

    def _get_identity_candidates(
        self,
        analysis: TrackletIdentityAnalysis,
        votes: List[IdentityVote],
    ) -> List[Tuple[str, int, float]]:
        """Get ranked identity candidates for a tracklet.

        Returns list of (identity, vote_count, avg_distance) sorted by preference:
        - More votes = higher preference
        - If tied on votes, lower avg_distance = higher preference

        Args:
            analysis: Analysis results for the tracklet.
            votes: All votes for this tracklet.

        Returns:
            List of (identity, vote_count, avg_distance) tuples, sorted best-first.
        """
        if not votes:
            return []

        # Group votes by identity
        identity_stats: Dict[str, List[float]] = {}
        for vote in votes:
            if vote.identity not in identity_stats:
                identity_stats[vote.identity] = []
            # Get distance from source_info if available
            dist = vote.source_info.get('distance', 50.0)  # Default if not available
            identity_stats[vote.identity].append(dist)

        # Calculate (identity, vote_count, avg_distance)
        candidates = []
        for identity, distances in identity_stats.items():
            vote_count = len(distances)
            avg_distance = sum(distances) / len(distances)
            candidates.append((identity, vote_count, avg_distance))

        # Sort by vote_count desc, then avg_distance asc
        candidates.sort(key=lambda x: (-x[1], x[2]))
        return candidates

    def _check_identity_conflict(
        self,
        tracklet_frames: List[Tuple[int, int]],
        identity: str,
        assigned_frames: Dict[str, Dict[int, str]],
    ) -> Optional[str]:
        """Check if assigning identity to tracklet would conflict with existing assignments.

        Args:
            tracklet_frames: Frames in the tracklet [(frame_idx, inst_idx), ...].
            identity: Identity to check.
            assigned_frames: Dict mapping identity -> {frame_idx -> tracklet_name}.

        Returns:
            Conflicting tracklet name if conflict exists, None otherwise.
        """
        if identity not in assigned_frames:
            return None

        identity_frames = assigned_frames[identity]
        for frame_idx, _ in tracklet_frames:
            if frame_idx in identity_frames:
                return identity_frames[frame_idx]
        return None

    def _resolve_conflict(
        self,
        tracklet_a: str,
        tracklet_b: str,
        identity: str,
        votes_a: List[IdentityVote],
        votes_b: List[IdentityVote],
    ) -> str:
        """Resolve conflict between two tracklets wanting the same identity.

        Winner is determined by:
        1. More votes for the identity
        2. If tied, lower average distance

        Args:
            tracklet_a: First tracklet name.
            tracklet_b: Second tracklet name.
            identity: The contested identity.
            votes_a: Votes for tracklet_a.
            votes_b: Votes for tracklet_b.

        Returns:
            Name of winning tracklet.
        """
        # Get votes for the specific identity
        votes_a_for_id = [v for v in votes_a if v.identity == identity]
        votes_b_for_id = [v for v in votes_b if v.identity == identity]

        count_a = len(votes_a_for_id)
        count_b = len(votes_b_for_id)

        if count_a != count_b:
            return tracklet_a if count_a > count_b else tracklet_b

        # Tied on count, compare average distance
        avg_dist_a = sum(v.source_info.get('distance', 50.0) for v in votes_a_for_id) / max(count_a, 1)
        avg_dist_b = sum(v.source_info.get('distance', 50.0) for v in votes_b_for_id) / max(count_b, 1)

        return tracklet_a if avg_dist_a <= avg_dist_b else tracklet_b

    def _apply_switch_splits(
        self,
        labels: sio.Labels,
        tracklets: Dict[str, List[Tuple[int, int]]],
        votes: Dict[str, List[IdentityVote]],
        analyses: Dict[str, "TrackletIdentityAnalysis"],
    ) -> int:
        """Split tracklets at detected switch points.

        When a tracklet has a detected identity switch (e.g., identity A dominates
        frames 0-352, identity B dominates frames 353+), this method splits the
        tracklet into two separate tracklets at the switch point.

        This is critical for correct identity assignment when a motion tracker
        has swapped two animals' identities mid-tracklet.

        Args:
            labels: SLEAP Labels object.
            tracklets: Dict mapping tracklet_name -> [(frame_idx, instance_idx), ...].
                       Modified in place.
            votes: Dict mapping tracklet_name -> list of IdentityVote.
                   Modified in place.
            analyses: Dict mapping tracklet_name -> TrackletIdentityAnalysis.
                      Modified in place.

        Returns:
            Number of splits applied.
        """
        split_count = 0
        tracklets_to_split = []

        # Find tracklets with detected switches that meet confidence threshold
        for tracklet_name, analysis in analyses.items():
            if not analysis.detected_switches:
                continue

            # Check each detected switch
            for switch in analysis.detected_switches:
                # Validate switch has enough votes on each side
                if (switch.votes_before >= self.min_votes_for_identity and
                    switch.votes_after >= self.min_votes_for_identity and
                    switch.confidence >= self.switch_confidence_threshold):
                    tracklets_to_split.append((tracklet_name, switch))
                    break  # Only split at first valid switch per tracklet

        # Apply splits
        for tracklet_name, switch in tracklets_to_split:
            tracklet_frames = tracklets[tracklet_name]
            tracklet_votes = votes.get(tracklet_name, [])

            # Split frames into before and after switch
            frames_before = [(f, i) for f, i in tracklet_frames if f < switch.switch_frame]
            frames_after = [(f, i) for f, i in tracklet_frames if f >= switch.switch_frame]

            # Skip if either segment is empty
            if not frames_before or not frames_after:
                continue

            # Split votes into before and after switch
            votes_before = [v for v in tracklet_votes if v.frame_idx < switch.switch_frame]
            votes_after = [v for v in tracklet_votes if v.frame_idx >= switch.switch_frame]

            # Create new tracklet names
            name_before = f"{tracklet_name}_pre{switch.switch_frame}"
            name_after = f"{tracklet_name}_post{switch.switch_frame}"

            # Log the split
            print(f"    Splitting {tracklet_name} at frame {switch.switch_frame}: "
                  f"{switch.identity_before} ({switch.votes_before} votes) -> "
                  f"{switch.identity_after} ({switch.votes_after} votes)")

            # Build frame mapping if needed
            if not self._frame_idx_to_lf:
                self._build_frame_mapping(labels)

            # CRITICAL: Directly assign the RFID identities to the split segments
            # We know exactly what identity each segment should get, so we skip conflict
            # resolution and directly assign. This ensures the switch is properly repaired.
            #
            # However, we must check for conflicts with OTHER tracklets to avoid creating
            # duplicate identities in the same frame.

            # Get or create Track objects for the identities
            track_identity_before = next(
                (t for t in labels.tracks if t.name == switch.identity_before), None
            )
            if track_identity_before is None:
                track_identity_before = sio.Track(name=switch.identity_before)
                labels.tracks.append(track_identity_before)

            track_identity_after = next(
                (t for t in labels.tracks if t.name == switch.identity_after), None
            )
            if track_identity_after is None:
                track_identity_after = sio.Track(name=switch.identity_after)
                labels.tracks.append(track_identity_after)

            # Build a set of (frame_idx, identity) pairs that are already assigned
            # to OTHER instances (not part of this tracklet being split)
            already_assigned: Dict[int, set] = {}  # frame_idx -> set of assigned identities
            tracklet_frame_set = set((f, i) for f, i in tracklet_frames)
            for lf in labels.labeled_frames:
                for i, inst in enumerate(lf.instances):
                    if (lf.frame_idx, i) not in tracklet_frame_set:
                        if inst.track is not None:
                            track_name = inst.track.name if hasattr(inst.track, 'name') else str(inst.track)
                            if lf.frame_idx not in already_assigned:
                                already_assigned[lf.frame_idx] = set()
                            already_assigned[lf.frame_idx].add(track_name)

            # Update instances in the before segment - assign identity_before directly
            # Skip frames where identity_before is already assigned to another instance
            skipped_before = 0
            assigned_before = 0
            for frame_idx, inst_idx in frames_before:
                # Check for conflict - skip if identity already assigned in this frame
                if frame_idx in already_assigned and switch.identity_before in already_assigned[frame_idx]:
                    skipped_before += 1
                    continue

                lf = self._frame_idx_to_lf.get(frame_idx)
                if lf is not None and inst_idx < len(lf.instances):
                    inst = lf.instances[inst_idx]
                    # Preserve existing history
                    existing_history = []
                    if isinstance(inst.track, TrackContext):
                        existing_history = inst.track.track_history.copy()

                    # Create new TrackContext with the identity (NOT temporary tracklet)
                    track_context = TrackContext(
                        priority=self.priority,
                        track=track_identity_before,
                        name=switch.identity_before,
                        temporary_track=False,  # This is now a permanent identity
                        valid=True,
                        track_history=existing_history,
                    )
                    track_context.add_history_entry(
                        layer_name=self.name,
                        old_track_name=tracklet_name,
                        new_track_name=switch.identity_before,
                        frame_idx=frame_idx,
                        reason=f"Split at frame {switch.switch_frame}: assigned {switch.identity_before} ({switch.votes_before} votes)",
                        conflict_resolved=False,
                        propagated_from_frame=None,
                    )
                    inst.track = track_context
                    assigned_before += 1

                    # Track that this identity is now assigned in this frame
                    if frame_idx not in already_assigned:
                        already_assigned[frame_idx] = set()
                    already_assigned[frame_idx].add(switch.identity_before)

            # Update instances in the after segment - assign identity_after directly
            # Skip frames where identity_after is already assigned to another instance
            skipped_after = 0
            assigned_after = 0
            for frame_idx, inst_idx in frames_after:
                # Check for conflict - skip if identity already assigned in this frame
                if frame_idx in already_assigned and switch.identity_after in already_assigned[frame_idx]:
                    skipped_after += 1
                    continue

                lf = self._frame_idx_to_lf.get(frame_idx)
                if lf is not None and inst_idx < len(lf.instances):
                    inst = lf.instances[inst_idx]
                    # Preserve existing history
                    existing_history = []
                    if isinstance(inst.track, TrackContext):
                        existing_history = inst.track.track_history.copy()

                    # Create new TrackContext with the identity (NOT temporary tracklet)
                    track_context = TrackContext(
                        priority=self.priority,
                        track=track_identity_after,
                        name=switch.identity_after,
                        temporary_track=False,  # This is now a permanent identity
                        valid=True,
                        track_history=existing_history,
                    )
                    track_context.add_history_entry(
                        layer_name=self.name,
                        old_track_name=tracklet_name,
                        new_track_name=switch.identity_after,
                        frame_idx=frame_idx,
                        reason=f"Split at frame {switch.switch_frame}: assigned {switch.identity_after} ({switch.votes_after} votes)",
                        conflict_resolved=False,
                        propagated_from_frame=None,
                    )
                    inst.track = track_context
                    assigned_after += 1

                    # Track that this identity is now assigned in this frame
                    if frame_idx not in already_assigned:
                        already_assigned[frame_idx] = set()
                    already_assigned[frame_idx].add(switch.identity_after)

            # Log skipped frames if any
            if skipped_before > 0 or skipped_after > 0:
                print(f"      (skipped {skipped_before} before, {skipped_after} after due to conflicts)")

            # Remove from tracklets dict - these are now assigned, not pending
            del tracklets[tracklet_name]
            # Note: We don't add the split tracklets to the dict because they're already assigned

            # Remove from votes dict
            if tracklet_name in votes:
                del votes[tracklet_name]

            # Remove from analyses dict - the split tracklets are already assigned
            del analyses[tracklet_name]

            # Log to explanation store
            if self._explanation_store is not None:
                # Log split decision for frames in the before segment
                for frame_idx, inst_idx in frames_before:
                    record = SwitchRepairDecisionRecord(
                        frame_idx=frame_idx,
                        instance_idx=inst_idx,
                        tracker_name=self.name,
                        tracker_priority=self.priority,
                        decision_type=DecisionType.MATCHED,
                        assigned_track_id=switch.identity_before,
                        previous_track_id=tracklet_name,
                        summary=f"Split at frame {switch.switch_frame}: assigned {switch.identity_before}",
                        reasons=[
                            f"Detected switch from {switch.identity_before} to {switch.identity_after}",
                            f"Votes before switch: {switch.votes_before}",
                            f"Votes after switch: {switch.votes_after}",
                            f"Switch confidence: {switch.confidence:.2f}",
                        ],
                        switch_frame=switch.switch_frame,
                        identity_before=switch.identity_before,
                        identity_after=switch.identity_after,
                        votes_before=switch.votes_before,
                        votes_after=switch.votes_after,
                        switch_confidence=switch.confidence,
                        partner_tracklet=switch.partner_tracklet,
                        partner_validated=switch.partner_tracklet is not None,
                        repair_action="split",
                        original_tracklet=tracklet_name,
                        resulting_tracks=[switch.identity_before, switch.identity_after],
                    )
                    self._explanation_store.add(record)

                # Log split decision for frames in the after segment
                for frame_idx, inst_idx in frames_after:
                    record = SwitchRepairDecisionRecord(
                        frame_idx=frame_idx,
                        instance_idx=inst_idx,
                        tracker_name=self.name,
                        tracker_priority=self.priority,
                        decision_type=DecisionType.MATCHED,
                        assigned_track_id=switch.identity_after,
                        previous_track_id=tracklet_name,
                        summary=f"Split at frame {switch.switch_frame}: assigned {switch.identity_after}",
                        reasons=[
                            f"Detected switch from {switch.identity_before} to {switch.identity_after}",
                            f"Votes before switch: {switch.votes_before}",
                            f"Votes after switch: {switch.votes_after}",
                            f"Switch confidence: {switch.confidence:.2f}",
                        ],
                        switch_frame=switch.switch_frame,
                        identity_before=switch.identity_before,
                        identity_after=switch.identity_after,
                        votes_before=switch.votes_before,
                        votes_after=switch.votes_after,
                        switch_confidence=switch.confidence,
                        partner_tracklet=switch.partner_tracklet,
                        partner_validated=switch.partner_tracklet is not None,
                        repair_action="split",
                        original_tracklet=tracklet_name,
                        resulting_tracks=[switch.identity_before, switch.identity_after],
                    )
                    self._explanation_store.add(record)

            split_count += 1

        return split_count

    def assign_identities_to_tracklets(
        self,
        labels: sio.Labels,
        tracklets: Dict[str, List[Tuple[int, int]]],
        votes: Dict[str, List[IdentityVote]],
        frame_tolerance: int = 10,
    ) -> Dict[str, str]:
        """Assign identities to tracklets with conflict resolution.

        This is the main entry point for identity assignment. It:
        1. Analyzes each tracklet for temporal consistency
        2. Builds proposed assignments with ranked identity candidates
        3. Detects conflicts (same identity for overlapping frames)
        4. Resolves conflicts recursively (winner keeps, loser tries next-best)
        5. Applies resolved assignments

        The conflict resolution ensures that no two instances in the same frame
        get assigned the same identity.

        Args:
            labels: SLEAP Labels object.
            tracklets: Dict mapping tracklet_name -> [(frame_idx, instance_idx), ...].
            votes: Dict mapping tracklet_name -> list of IdentityVote.
            frame_tolerance: Max frame difference to consider switches related.

        Returns:
            Dict mapping tracklet_name -> assigned_identity (or None if no assignment).
        """
        # Build frame mapping
        if not self._frame_idx_to_lf:
            self._build_frame_mapping(labels)

        # Step 1: Analyze each tracklet for consistency
        analyses: Dict[str, TrackletIdentityAnalysis] = {}
        for tracklet_name, tracklet_frames in tracklets.items():
            tracklet_votes = votes.get(tracklet_name, [])
            analysis = self.analyze_tracklet_consistency(
                tracklet_name=tracklet_name,
                votes=tracklet_votes,
                tracklet_frames=tracklet_frames,
            )
            analyses[tracklet_name] = analysis

        # Step 1.5: Apply switch splits if enabled
        # This modifies tracklets, votes, and analyses in place
        if self.split_on_switch:
            split_count = self._apply_switch_splits(
                labels=labels,
                tracklets=tracklets,
                votes=votes,
                analyses=analyses,
            )
            if split_count > 0:
                print(f"  Applied {split_count} tracklet splits at detected switch points")

        # Step 2: Build ranked identity candidates for each tracklet
        # Each tracklet gets a list of (identity, vote_count, avg_distance) sorted by preference
        tracklet_candidates: Dict[str, List[Tuple[str, int, float]]] = {}
        for tracklet_name in tracklets:
            tracklet_votes = votes.get(tracklet_name, [])
            candidates = self._get_identity_candidates(analyses[tracklet_name], tracklet_votes)
            tracklet_candidates[tracklet_name] = candidates

        # Step 3: Resolve conflicts with iterative assignment
        # Track: identity -> {frame_idx -> tracklet_name}
        assigned_frames: Dict[str, Dict[int, str]] = {}

        # Initialize assigned_frames with any RFID identities already assigned during split
        # This prevents conflict resolution from creating duplicates
        for lf in labels.labeled_frames:
            for inst_idx, inst in enumerate(lf.instances):
                if inst.track is not None:
                    track_name = inst.track.name if hasattr(inst.track, 'name') else str(inst.track)
                    # Only consider non-tracklet names (i.e., RFID identities from splits)
                    if not track_name.startswith("tracklet"):
                        if track_name not in assigned_frames:
                            assigned_frames[track_name] = {}
                        # Use a special marker to indicate this was from a split
                        assigned_frames[track_name][lf.frame_idx] = f"__split_{track_name}"

        # Track: tracklet_name -> assigned_identity
        final_assignments: Dict[str, str] = {}
        # Track: tracklet_name -> current candidate index
        candidate_index: Dict[str, int] = {t: 0 for t in tracklets}
        # Track: tracklets that still need assignment
        pending_tracklets = set(t for t in tracklets if tracklet_candidates.get(t))

        # Sort tracklets by their best candidate's vote count (descending)
        # This gives priority to tracklets with stronger claims
        def get_best_vote_count(t):
            cands = tracklet_candidates.get(t, [])
            return cands[0][1] if cands else 0

        max_iterations = len(tracklets) * 10  # Prevent infinite loops
        iteration = 0

        while pending_tracklets and iteration < max_iterations:
            iteration += 1

            # Sort pending tracklets by current candidate strength
            sorted_pending = sorted(
                pending_tracklets,
                key=lambda t: (
                    -tracklet_candidates[t][candidate_index[t]][1]  # vote count desc
                    if candidate_index[t] < len(tracklet_candidates[t]) else 0,
                    tracklet_candidates[t][candidate_index[t]][2]  # avg_distance asc
                    if candidate_index[t] < len(tracklet_candidates[t]) else float('inf'),
                )
            )

            made_progress = False

            for tracklet_name in sorted_pending:
                idx = candidate_index[tracklet_name]
                candidates = tracklet_candidates[tracklet_name]

                if idx >= len(candidates):
                    # No more candidates - leave as tracklet
                    pending_tracklets.discard(tracklet_name)
                    made_progress = True
                    continue

                identity, vote_count, avg_dist = candidates[idx]
                tracklet_frames = tracklets[tracklet_name]

                # DEBUG: Track specific tracklets
                debug_tracklets = {'tracklet_2452', 'tracklet_2491'}
                if tracklet_name in debug_tracklets:
                    print(f"  DEBUG: {tracklet_name} trying {identity} ({vote_count} votes)")

                # Check for conflict
                conflicting_tracklet = self._check_identity_conflict(
                    tracklet_frames, identity, assigned_frames
                )

                if tracklet_name in debug_tracklets:
                    print(f"    Conflict check: {conflicting_tracklet}")

                if conflicting_tracklet is None:
                    # No conflict - assign this identity
                    final_assignments[tracklet_name] = identity

                    if tracklet_name in debug_tracklets:
                        print(f"    ASSIGNED: {tracklet_name} -> {identity}")

                    # Record assigned frames
                    if identity not in assigned_frames:
                        assigned_frames[identity] = {}
                    for frame_idx, _ in tracklet_frames:
                        assigned_frames[identity][frame_idx] = tracklet_name

                    pending_tracklets.discard(tracklet_name)
                    made_progress = True
                elif conflicting_tracklet.startswith("__split_"):
                    # Conflict with a split assignment - split always wins
                    # This tracklet must try its next candidate
                    if tracklet_name in debug_tracklets:
                        print(f"    CONFLICT with split assignment: {tracklet_name} loses to {conflicting_tracklet}")
                    candidate_index[tracklet_name] += 1
                    made_progress = True
                else:
                    # Conflict exists - determine winner
                    winner = self._resolve_conflict(
                        tracklet_name,
                        conflicting_tracklet,
                        identity,
                        votes.get(tracklet_name, []),
                        votes.get(conflicting_tracklet, []),
                    )

                    if tracklet_name in debug_tracklets or conflicting_tracklet in debug_tracklets:
                        print(f"    CONFLICT: {tracklet_name} vs {conflicting_tracklet} for {identity}")
                        print(f"    Winner: {winner}")

                    if winner == tracklet_name:
                        # This tracklet wins - bump the other to next candidate
                        # First, unassign the conflicting tracklet
                        if conflicting_tracklet in final_assignments:
                            old_identity = final_assignments.pop(conflicting_tracklet)
                            # Remove its frame assignments
                            if old_identity in assigned_frames:
                                frames_to_remove = [
                                    f for f, t in assigned_frames[old_identity].items()
                                    if t == conflicting_tracklet
                                ]
                                for f in frames_to_remove:
                                    del assigned_frames[old_identity][f]

                        # Bump conflicting tracklet to next candidate
                        candidate_index[conflicting_tracklet] += 1
                        pending_tracklets.add(conflicting_tracklet)

                        # Assign this tracklet
                        final_assignments[tracklet_name] = identity

                        if tracklet_name in debug_tracklets:
                            print(f"    ASSIGNED (after winning): {tracklet_name} -> {identity}")

                        if identity not in assigned_frames:
                            assigned_frames[identity] = {}
                        for frame_idx, _ in tracklet_frames:
                            assigned_frames[identity][frame_idx] = tracklet_name

                        pending_tracklets.discard(tracklet_name)
                        made_progress = True
                    else:
                        # This tracklet loses - try next candidate
                        if tracklet_name in debug_tracklets:
                            print(f"    LOST: {tracklet_name} moving to next candidate")
                        candidate_index[tracklet_name] += 1
                        made_progress = True

            if not made_progress:
                # No progress made - break to avoid infinite loop
                break

        # Step 4: Apply the resolved assignments
        assignments: Dict[str, str] = {}
        conflict_stats = {"resolved": 0, "unassigned": 0}

        for tracklet_name, identity in final_assignments.items():
            tracklet_frames = tracklets[tracklet_name]
            tracklet_votes = votes.get(tracklet_name, [])

            # Get vote stats for this identity
            identity_votes = [v for v in tracklet_votes if v.identity == identity]
            vote_count = len(identity_votes)
            avg_dist = sum(v.source_info.get('distance', 50.0) for v in identity_votes) / max(vote_count, 1)

            reason = f"Assigned {identity} ({vote_count} votes, avg dist {avg_dist:.1f}px)"

            success = self.rename_track_globally(
                labels=labels,
                old_track_name=tracklet_name,
                new_track_name=identity,
                reason=reason,
            )

            if success:
                assignments[tracklet_name] = identity

                # Log to explanation store
                if self._explanation_store is not None:
                    for frame_idx, inst_idx in tracklet_frames:
                        record = SwitchRepairDecisionRecord(
                            frame_idx=frame_idx,
                            instance_idx=inst_idx,
                            tracker_name=self.name,
                            tracker_priority=self.priority,
                            decision_type=DecisionType.MATCHED,
                            assigned_track_id=identity,
                            previous_track_id=tracklet_name,
                            summary=reason,
                            reasons=[reason],
                            repair_action="conflict_resolved",
                            original_tracklet=tracklet_name,
                            resulting_tracks=[identity],
                            votes_before=vote_count,
                            switch_confidence=vote_count / max(len(tracklet_votes), 1),
                        )
                        self._explanation_store.add(record)

        # Log unassigned tracklets
        unassigned = set(tracklets.keys()) - set(assignments.keys())
        if unassigned:
            print(f"  Tracklets left unassigned (no valid identity): {len(unassigned)}")

        print(f"  Conflict resolution complete: {len(assignments)} assigned, {len(unassigned)} unassigned")

        return assignments