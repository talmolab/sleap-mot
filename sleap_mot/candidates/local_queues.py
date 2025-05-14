"""Module to generate Tracking local queue candidates."""

from typing import Dict, Optional, List, Deque, DefaultDict, Union
import numpy as np
import sleap_io as sio
from sleap_mot.track_instance import (
    TrackInstanceLocalQueue,
    TrackedInstanceFeature,
)
from collections import defaultdict, deque


class LocalQueueCandidates:
    """Track local queues method for candidate generation.

    This module handles `tracker_queue` using the local queues method, where track assignments
    are determined based on the last `window_size` instances for each track.

    Attributes:
        window_size: Number of previous frames to compare the current predicted instance with.
            Default: 5.
        max_tracks: Maximum number of new tracks that can be created. Default: None.
        instance_score_threshold: Instance score threshold for creating new tracks.
            Default: 0.0.
        tracker_queue: Dictionary that stores the past frames of all the tracks identified
            so far as `deque`.
        current_tracks: List of track IDs that are being tracked.
    """

    def __init__(
        self,
        window_size: int = 5,
        max_tracks: Optional[int] = None,
        instance_score_threshold: float = 0.0,
    ):
        """Initialize class variables."""
        self.window_size = window_size
        self.max_tracks = max_tracks
        self.instance_score_threshold = instance_score_threshold
        self.tracker_queue = defaultdict(Deque)
        self.current_tracks = []

    def get_track_instances(
        self,
        feature_list: List[Union[np.ndarray]],
        untracked_instances: List[sio.PredictedInstance],
        frame_idx: int,
        image: np.ndarray,
    ) -> List[TrackInstanceLocalQueue]:
        """Return a list of `TrackInstanceLocalQueue` instances for the `untracked_instances`."""
        track_instances = []
        for ind, (feat, instance) in enumerate(zip(feature_list, untracked_instances)):
            track_instance = TrackInstanceLocalQueue(
                src_instance=instance,
                src_instance_idx=ind,
                track_id=None,
                feature=feat,
                instance_score=instance.score,
                frame_idx=frame_idx,
                image=image,
            )
            track_instances.append(track_instance)
        return track_instances

    def get_features_from_track_id(
        self, track_id: int, candidates_list: Optional[DefaultDict[int, Deque]] = None
    ) -> List[TrackedInstanceFeature]:
        """Return list of `TrackedInstanceFeature` objects for instances in tracker queue with the given `track_id`.

        Note: If `candidates_list` is `None`, then features of all the instances in the
            tracker queue are returned by default. Else, only the features from the given
            candidates_list are returned.
        """
        tracked_instances = (
            candidates_list if candidates_list is not None else self.tracker_queue
        )
        output = []
        for t in tracked_instances[track_id]:
            tracked_instance_feature = TrackedInstanceFeature(
                feature=t.feature,
                src_predicted_instance=t.src_instance,
                frame_idx=t.frame_idx,
                tracking_score=t.tracking_score,
                instance_score=t.instance_score,
                shifted_keypoints=None,
            )
            output.append(tracked_instance_feature)
        return output

    def get_new_track_id(self, existing_track_ids: List[str] = []) -> int:
        """Return a new track_id."""
        if not existing_track_ids:
            new_track_id = "0"

        else:
            numeric_track_ids = [
                int(track_id) for track_id in existing_track_ids if track_id.isdigit()
            ]
            if not numeric_track_ids:
                new_track_id = "0"
            else:
                new_track_id = str(
                    min(set(range(max(numeric_track_ids) + 2)) - set(numeric_track_ids))
                )
            if (
                self.max_tracks is not None and int(new_track_id) > self.max_tracks
            ):  # TODO
                raise Exception("Exceeding max tracks")
        self.tracker_queue[new_track_id] = deque(maxlen=self.window_size)
        return new_track_id

    def add_new_tracks(
        self,
        current_instances: List[TrackInstanceLocalQueue],
        existing_track_ids: List[str] = [],
        maintain_track_ids: bool = False,
    ) -> List[TrackInstanceLocalQueue]:
        """Add new track IDs to the `TrackInstanceLocalQueue` objects and to the tracker queue."""
        track_instances = []
        for t in current_instances:
            if t.instance_score > self.instance_score_threshold:
                if maintain_track_ids and t.src_instance.track is not None:
                    track_name = t.src_instance.track.name
                    track_numbers = [
                        int(s) for s in track_name.split("_") if s.isdigit()
                    ]
                    new_track_id = "".join(map(str, track_numbers))
                else:
                    new_track_id = self.get_new_track_id(existing_track_ids)
                    existing_track_ids.append(new_track_id)
                t.track_id = new_track_id
                t.tracking_score = 1.0
                self.current_tracks.append(new_track_id)
                self.tracker_queue[new_track_id].append(t)
            track_instances.append(t)

        return track_instances

    def update_tracks(
        self,
        current_instances: List[TrackInstanceLocalQueue],
        row_inds: np.ndarray,
        col_inds: np.ndarray,
        tracking_scores: List[float],
        add_to_queue: bool = False,
        existing_track_ids: List[str] = [],
    ) -> List[TrackInstanceLocalQueue]:
        """Assign tracks to `TrackInstanceLocalQueue` objects based on the output of track matching algorithm.

        Args:
            current_instances: List of TrackInstanceLocalQueue objects with features and unassigned tracks.
            row_inds: List of indices for the  `current_instances` object that has an assigned
                track.
            col_inds: List of track IDs that have been assigned a new instance.
            tracking_scores: List of tracking scores from the cost matrix.
            add_to_queue: Boolean to add the `current_instances` to the tracker queue.
            existing_track_ids: List of existing track IDs.
        """
        if row_inds is not None and col_inds is not None:
            for idx, (row, col) in enumerate(zip(row_inds, col_inds)):
                current_instances[row].track_id = col
                current_instances[row].tracking_score = tracking_scores[idx]

        if add_to_queue:
            # Track which track_ids get updated
            updated_track_ids = set()

            for track_instance in current_instances:
                if track_instance.track_id is not None:
                    self.tracker_queue[track_instance.track_id].append(track_instance)
                    updated_track_ids.add(track_instance.track_id)

            # Add empty instances for tracks that weren't updated
            for track_id in self.current_tracks:
                if track_id not in updated_track_ids:
                    empty_instance = TrackInstanceLocalQueue(
                        src_instance=None,
                        src_instance_idx=None,
                        feature=None,
                        instance_score=0.0,
                        track_id=track_id,
                        tracking_score=0.0,
                        frame_idx=current_instances[0].frame_idx,
                        image=None,
                    )
                    self.tracker_queue[track_id].append(empty_instance)

            # Create new tracks for instances with unassigned tracks from track matching
            new_current_instances_inds = [
                x for x in range(len(current_instances)) if x not in row_inds
            ]
            if new_current_instances_inds:
                for ind in new_current_instances_inds:
                    self.add_new_tracks(
                        [current_instances[ind]],
                        existing_track_ids=existing_track_ids,
                    )

        return current_instances

    def get_instances_groupby_frame_idx(
        self, candidates_list: Optional[DefaultDict[int, Deque]]
    ) -> Dict[int, List[TrackInstanceLocalQueue]]:
        """Return dictionary with list of `TrackInstanceLocalQueue` objects grouped by frame index."""
        instances_dict = defaultdict(list)
        tracked_instances = (
            candidates_list if candidates_list is not None else self.tracker_queue
        )
        for _, instances in tracked_instances.items():
            for instance in instances:
                instances_dict[instance.frame_idx].append(instance)
        return instances_dict
