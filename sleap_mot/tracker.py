"""Module for tracking."""

from typing import Any, Dict, List, Union, Deque, DefaultDict, Optional, Tuple, Tuple
from collections import defaultdict
import attrs
import cv2
import numpy as np
from collections import deque
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import logging
from sklearn.preprocessing import StandardScaler

import sleap_io as sio
from sleap_mot.candidates.fixed_window import FixedWindowCandidates
from sleap_mot.candidates.local_queues import LocalQueueCandidates
from sleap_mot.track_instance import (
    TrackedInstanceFeature,
    TrackInstances,
    TrackInstanceLocalQueue,
)
from sleap_mot.utils import (
    hungarian_matching,
    greedy_matching,
    get_bbox,
    get_centroid,
    get_keypoints,
    compute_euclidean_distance,
    compute_iou,
    compute_cosine_sim,
    compute_oks,
)
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


@attrs.define
class Tracker:
    """Simple Pose Tracker.

    This is the base class for all Trackers. This module handles tracking instances
    across frames by creating new track IDs (or) assigning track IDs to each predicted
    instance when the `.track()` is called. This class is initialized in the `Predictor`
    classes.

    Attributes:
        candidate: Instance of either `FixedWindowCandidates` or `LocalQueueCandidates`.
        features: Feature representation for the candidates to update current detections.
            One of [`keypoints`, `centroids`, `bboxes`, `image`]. Default: `keypoints`.
        scoring_method: Method to compute association score between features from the
            current frame and the previous tracks. One of [`oks`, `cosine_sim`, `iou`,
            `euclidean_dist`]. Default: `oks`.
        scoring_reduction: Method to aggregate and reduce multiple scores if there are
            several detections associated with the same track. One of [`mean`, `max`,
            `weighted`]. Default: `mean`.
        track_matching_method: Track matching algorithm. One of `hungarian`, `greedy.
                Default: `hungarian`.
        use_flow: If True, `FlowShiftTracker` is used, where the poses are matched using
            optical flow shifts. Default: `False`.
        is_local_queue: `True` if `LocalQueueCandidates` is used else `False`.
        max_cost: Maximum cost threshold for track assignment. If the matching score is
            greater than this threshold, the track will not be assigned. Default: 0.5.

    """

    candidate: Union[FixedWindowCandidates, LocalQueueCandidates] = (
        FixedWindowCandidates()
    )
    features: str = "keypoints"
    scoring_method: str = "oks"
    scoring_reduction: str = "mean"
    track_matching_method: str = "hungarian"
    use_flow: bool = False
    is_local_queue: bool = False
    max_cost: float = 0.5
    max_cost: float = 0.5
    _scoring_functions: Dict[str, Any] = {
        "oks": compute_oks,
        "iou": compute_iou,
        "cosine_sim": compute_cosine_sim,
        "euclidean_dist": compute_euclidean_distance,
    }
    _scoring_reduction_methods: Dict[str, Any] = {
        "mean": np.nanmean,
        "max": np.nanmax,
        "weighted": lambda x: (
            np.average(
                np.array(x),
                weights=np.maximum(
                    1e-10,
                    1
                    - (
                        np.abs(np.arange(len(np.array(x))) - (len(np.array(x)) - 1))
                        / len(np.array(x))
                    ),
                ),
                axis=0,
            )
            if len(np.array(x).shape) > 0
            else x  # Return the scalar value directly if x is a scalar
        ),
    }
    _scoring_reduction_methods: Dict[str, Any] = {
        "mean": np.nanmean,
        "max": np.nanmax,
        "weighted": lambda x: (
            np.average(
                np.array(x),
                weights=np.maximum(
                    1e-10,
                    1
                    - (
                        np.abs(np.arange(len(np.array(x))) - (len(np.array(x)) - 1))
                        / len(np.array(x))
                    ),
                ),
                axis=0,
            )
            if len(np.array(x).shape) > 0
            else x  # Return the scalar value directly if x is a scalar
        ),
    }
    _feature_methods: Dict[str, Any] = {
        "keypoints": get_keypoints,
        "centroids": get_centroid,
        "bboxes": get_bbox,
    }
    _track_matching_methods: Dict[str, Any] = {
        "hungarian": hungarian_matching,
        "greedy": greedy_matching,
    }
    _track_objects: Dict[str, sio.Track] = attrs.field(factory=dict)
    global_track_ids: List[str] = attrs.field(factory=list)
    _track_objects: Dict[str, sio.Track] = attrs.field(factory=dict)
    global_track_ids: List[str] = attrs.field(factory=list)

    @classmethod
    def from_config(
        cls,
        window_size: int = 5,
        instance_score_threshold: float = 0.0,
        candidates_method: str = "fixed_window",
        features: str = "keypoints",
        scoring_method: str = "oks",
        scoring_reduction: str = "mean",
        track_matching_method: str = "hungarian",
        max_tracks: Optional[int] = None,
        use_flow: bool = False,
        of_img_scale: float = 1.0,
        of_window_size: int = 21,
        of_max_levels: int = 3,
        max_cost: float = None,
    ):
        """Create `Tracker` from config.

        Args:
            window_size: Number of frames to look for in the candidate instances to match
                with the current detections. Default: 5.
            instance_score_threshold: Instance score threshold for creating new tracks.
                Default: 0.0.
            candidates_method: Either of `fixed_window` or `local_queues`. In fixed window
                method, candidates from the last `window_size` frames. In local queues,
                last `window_size` instances for each track ID is considered for matching
                against the current detection. Default: `fixed_window`.
            features: Feature representation for the candidates to update current detections.
                One of [`keypoints`, `centroids`, `bboxes`, `image`]. Default: `keypoints`.
            scoring_method: Method to compute association score between features from the
                current frame and the previous tracks. One of [`oks`, `cosine_sim`, `iou`,
                `euclidean_dist`]. Default: `oks`.
            scoring_reduction: Method to aggregate and reduce multiple scores if there are
                several detections associated with the same track. One of [`mean`, `max`,
                `weighted`]. Default: `mean`.
            track_matching_method: Track matching algorithm. One of `hungarian`, `greedy.
                Default: `hungarian`.
            max_tracks: Meaximum number of new tracks to be created to avoid redundant tracks.
                (only for local queues candidate) Default: None.
            use_flow: If True, `FlowShiftTracker` is used, where the poses are matched using
            optical flow shifts. Default: `False`.
            of_img_scale: Factor to scale the images by when computing optical flow. Decrease
                this to increase performance at the cost of finer accuracy. Sometimes
                decreasing the image scale can improve performance with fast movements.
                Default: 1.0. (only if `use_flow` is True)
            of_window_size: Optical flow window size to consider at each pyramid scale
                level. Default: 21. (only if `use_flow` is True)
            of_max_levels: Number of pyramid scale levels to consider. This is different
                from the scale parameter, which determines the initial image scaling.
                Default: 3. (only if `use_flow` is True)
            max_cost: Maximum cost threshold for track assignment. If the matching score is
                greater than this threshold, the track will not be assigned. Default: None.
            max_cost: Maximum cost threshold for track assignment. If the matching score is
                greater than this threshold, the track will not be assigned. Default: None.

        """
        if candidates_method == "fixed_window":
            candidate = FixedWindowCandidates(
                window_size=window_size,
                instance_score_threshold=instance_score_threshold,
            )
            is_local_queue = False

        elif candidates_method == "local_queues":
            candidate = LocalQueueCandidates(
                window_size=window_size,
                max_tracks=max_tracks,
                instance_score_threshold=instance_score_threshold,
            )
            is_local_queue = True

        else:
            raise ValueError(
                f"{candidates_method} is not a valid method. Please choose one of [`fixed_window`, `local_queues`]"
            )

        if use_flow:
            return FlowShiftTracker(
                candidate=candidate,
                features=features,
                scoring_method=scoring_method,
                scoring_reduction=scoring_reduction,
                track_matching_method=track_matching_method,
                img_scale=of_img_scale,
                of_window_size=of_window_size,
                of_max_levels=of_max_levels,
                is_local_queue=is_local_queue,
                max_cost=max_cost,
                max_cost=max_cost,
            )

        tracker = cls(
            candidate=candidate,
            features=features,
            scoring_method=scoring_method,
            scoring_reduction=scoring_reduction,
            track_matching_method=track_matching_method,
            use_flow=use_flow,
            is_local_queue=is_local_queue,
            max_cost=max_cost,
            max_cost=max_cost,
        )
        return tracker

    def update_track_id_and_queue(
        self, tracklet_id, global_track_id, frame_idx, labels
    ):
        """Update track ID and queue for a tracklet.

        Updates the track ID of a tracklet and its corresponding queue entries to use a new global track ID.
        If the tracklet ID is not already in global_track_ids, updates the candidate's current tracks and
        tracker queue to use the new global ID. Otherwise, updates the tracker queue with instances from
        the recent frames.

        Args:
            tracklet_id (Track): The current track ID of the tracklet
            global_track_id (Track): The new global track ID to assign
            frame_idx (int): Current frame index
            labels (Labels): Labels object containing tracked instances
        """
        if tracklet_id.name not in self.global_track_ids:
            if tracklet_id.name in self.candidate.current_tracks:
                self.candidate.current_tracks.remove(tracklet_id.name)
            if global_track_id.name not in self.candidate.current_tracks:
                self.candidate.current_tracks.append(global_track_id.name)
            # self._track_objects.pop(tracklet_id.name)
            self._track_objects[global_track_id.name] = global_track_id

            self.candidate.tracker_queue[global_track_id.name] = (
                self.candidate.tracker_queue.pop(tracklet_id.name, [])
            )
            for val in self.candidate.tracker_queue[global_track_id.name]:
                val.track_id = global_track_id.name

        else:
            for lf in labels[frame_idx - self.candidate.window_size : frame_idx + 1]:
                updated_track_ids = set()
                for inst in lf.instances:
                    self.candidate.tracker_queue[inst.track.name].append(
                        self.get_features([inst], lf.frame_idx, lf.image)[0]
                    )
                    updated_track_ids.add(inst.track.name)

                for track_id in self.candidate.current_tracks:
                    if track_id not in updated_track_ids:
                        empty_instance = TrackInstanceLocalQueue(
                            src_instance=None,
                            src_instance_idx=None,
                            feature=None,
                            instance_score=0.0,
                            track_id=track_id,
                            tracking_score=0.0,
                            frame_idx=lf.frame_idx,
                            image=lf.image,
                        )
                        self.candidate.tracker_queue[track_id].append(empty_instance)

    def get_tracklet(self, matching_instance, global_track_id, labels, frame_idx):
        """Get the tracklet (sequence of instances) associated with a given instance.

        Given an instance and global track ID, finds all consecutive instances in previous frames
        that have the same track ID, up to 300 frames back. Stops if it encounters the global track ID
        or if the matching instance's tracklet ID is already in global_track_ids.

        Args:
            matching_instance (Instance): The instance to find the tracklet for
            global_track_id (Track): The global track ID being matched against
            labels (Labels): Labels object containing tracked instances
            frame_idx (int): Current frame index
            matching_instance (Instance): The instance to find the tracklet for
            global_track_id (Track): The global track ID being matched against
            labels (Labels): Labels object containing tracked instances
            frame_idx (int): Current frame index

        Returns:
            list[Instance] or None: List of instances in the tracklet if found, None if no matching instance
        """
        if matching_instance:
            tracklet_id = matching_instance.track
            before_frames = [
                labels.find(frame_idx=frame, video=labels.video, return_new=True)[0]
                for frame in range(frame_idx - 1, max(frame_idx - 300, -1), -1)
            ]

            tracklet = [matching_instance]

            for lf in before_frames:
                if any(
                    inst.track.name == global_track_id.name for inst in lf.instances
                ):
                    break
                for inst in lf.instances:
                    if inst.track.name == tracklet_id.name:
                        if inst.track.name in self.global_track_ids:
                            return tracklet
                        tracklet.append(inst)
            return tracklet
        return None

    def id_tracklet(self, tracked_instances, new_instances, labels, frame_idx):
        """Update track IDs for a set of tracked instances.

        Given a list of tracked instances with their global track IDs, updates the track IDs
        of matching instances in the current frame. For each tracked instance, finds any matching
        tracklets and updates their track IDs to use the global track ID.

        Args:
            tracked_instances (List[Tuple[np.ndarray, str]]): List of tuples containing instance
            coordinates and their global track IDs
            new_instances (List[Instance]): List of instances in current frame to update
            labels (Labels): Labels object containing all tracked instances
            frame_idx (int): Current frame index being processed

        Returns:
            List[Instance]: Updated list of instances with track IDs assigned
        """
        for inst_numpy, global_track_name in tracked_instances:
            global_track_id = self._track_objects.get(global_track_name)

            if not global_track_id:
                global_track_id = sio.Track(name=global_track_name)
                self._track_objects[global_track_name] = global_track_id

            if any(global_track_name == inst.track.name for inst in new_instances):
                continue
            matching_instance = next(
                i
                for i in new_instances
                if np.allclose(i.numpy(), inst_numpy, equal_nan=True)
            )
            tracklet_id = matching_instance.track
            tracklet = self.get_tracklet(
                matching_instance, global_track_id, labels, frame_idx
            )

            if tracklet:
                for inst in tracklet:
                    inst.track = global_track_id

            self.update_track_id_and_queue(
                tracklet_id, global_track_id, frame_idx, labels
            )

        return new_instances

    def track(
        self,
        labels: sio.Labels,
        max_dist: int = None,
        inplace: bool = False,
        model_config: Dict[str, Any] = None,
    ):
        """Track instances across frames.

        This method tracks instances across frames in the provided `sio.Labels` object.
        It supports single-video labels and allows for in-place updates of the labels.

        Args:
            labels (sio.Labels): The labeled frames to track.
            inplace (bool): If True, the labels are updated in place. Default: False.

        Returns:
            sio.Labels: The updated labeled frames with track IDs assigned.
        """
        if len(labels.videos) > 1:
            raise NotImplementedError("Multiple videos are not supported.")

        can_load_images = labels.video.exists()

        n_frames = labels.video.shape[0]
        sorted_labels = []
        for frame_idx in range(n_frames):
            found_label = labels.find(
                frame_idx=frame_idx, video=labels.video, return_new=True
            )
            inst_to_remove = []
            inst_to_add = []
            for inst in found_label[0]:
                if not isinstance(inst, sio.PredictedInstance):
                    new_inst = sio.PredictedInstance(
                        skeleton=inst.skeleton,
                        points={
                            node: sio.PredictedPoint(
                                x=point.x,
                                y=point.y,
                                visible=point.visible,
                                complete=point.complete,
                                score=1.0,
                            )
                            for node, point in inst.points.items()
                        },
                        track=inst.track,
                        score=1.00,
                        tracking_score=0,
                    )
                    inst_to_add.append(new_inst)
                    inst_to_remove.append(inst)
            for inst in inst_to_remove:
                found_label[0].instances.remove(inst)
            for inst in inst_to_add:
                found_label[0].instances.append(inst)

            sorted_labels.append(found_label[0])
            found_label = labels.find(
                frame_idx=frame_idx, video=labels.video, return_new=True
            )
            inst_to_remove = []
            inst_to_add = []
            for inst in found_label[0]:
                if not isinstance(inst, sio.PredictedInstance):
                    new_inst = sio.PredictedInstance(
                        skeleton=inst.skeleton,
                        points={
                            node: sio.PredictedPoint(
                                x=point.x,
                                y=point.y,
                                visible=point.visible,
                                complete=point.complete,
                                score=1.0,
                            )
                            for node, point in inst.points.items()
                        },
                        track=inst.track,
                        score=1.00,
                        tracking_score=0,
                    )
                    inst_to_add.append(new_inst)
                    inst_to_remove.append(inst)
            for inst in inst_to_remove:
                found_label[0].instances.remove(inst)
            for inst in inst_to_add:
                found_label[0].instances.append(inst)

            sorted_labels.append(found_label[0])
        labels.labeled_frames = sorted_labels

        self.global_track_ids = [t.name for t in labels.tracks]

        self.global_track_ids = [t.name for t in labels.tracks]

        for lf in labels:
            if lf.instances:
                tracked_instances = [
                    (inst.numpy(), inst.track.name)
                    for inst in lf.instances
                    if inst.track is not None
                ]

                self.track_frame(
                    lf.instances,
                    lf.frame_idx,
                    lf.image,
                    max_dist=max_dist,
                    add_to_queue=True,
                )
                if tracked_instances:
                    self.id_tracklet(
                        tracked_instances,
                        lf.instances,
                        labels,
                        lf.frame_idx,
                    )

        if model_config:
            global_identity_model, identity_model_scaler = self.train_identity_model(
                labels, model_config
            )
            self.infer_and_assign_untracked_identities(
                labels, global_identity_model, identity_model_scaler
            )

        labels.update()
        labels.tracks = labels.tracks

        # Create new list of unique tracks first
        unique_tracks = []
        seen_names = set()
        for track in labels.tracks:
            if track.name not in seen_names:
                unique_tracks.append(track)
                seen_names.add(track.name)

        # Assign unique tracks back to labels.tracks
        labels.tracks = unique_tracks

        # Create new list of unique tracks first
        unique_tracks = []
        seen_names = set()
        for track in labels.tracks:
            if track.name not in seen_names:
                unique_tracks.append(track)
                seen_names.add(track.name)

        # Assign unique tracks back to labels.tracks
        labels.tracks = unique_tracks

        for lf in labels:
            for inst in lf.instances:
                if inst.track is not None:
                    inst.track = next(
                        t for t in labels.tracks if t.name == inst.track.name
                    )
                inst.skeleton = labels.skeleton
                # inst.points = {
                #     i: point for i, (node, point) in enumerate(inst.points.items())
                # }
                inst.skeleton = labels.skeleton
                # inst.points = {
                #     i: point for i, (node, point) in enumerate(inst.points.items())
                # }

        labels.update()
        return labels
        labels.update()
        return labels

    def track_frame(
        self,
        instances: List[sio.PredictedInstance],
        frame_idx: int,
        image: np.ndarray = None,
        max_dist: int = None,
        add_to_queue: bool = False,
    ) -> List[sio.PredictedInstance]:
        """Assign track IDs to the untracked list of `sio.PredictedInstance` objects.

        This method assigns track IDs to the untracked instances in the provided list.
        It first generates candidate features, updates the candidates list, computes
        association scores, and then assigns track IDs using the Hungarian method.

        Args:
            instances (List[sio.PredictedInstance]): The list of instances to assign track IDs to.
            instances (List[sio.PredictedInstance]): The list of instances to assign track IDs to.
            frame_idx (int): The index of the current frame.
            image (np.ndarray): The image of the current frame.
            add_to_queue (bool): If True, the instances will be added to the tracker queue.
            max_dist (int): The maximum pixel distance between instances to consider a match.
            max_dist (int): The maximum pixel distance between instances to consider a match.

        Returns:
            List[sio.PredictedInstance]: The list of instances with assigned track IDs.
        """
        current_instances = self.get_features(instances, frame_idx, image)
        current_instances = self.get_features(instances, frame_idx, image)

        candidates_feature_dict = self.generate_candidates()
        candidates_feature_dict = self.generate_candidates()

        if candidates_feature_dict:
            scores = self.get_scores(
                current_instances, candidates_feature_dict, max_dist
            )
        if candidates_feature_dict:
            scores = self.get_scores(
                current_instances, candidates_feature_dict, max_dist
            )
            cost_matrix = self.scores_to_cost_matrix(scores)

            current_tracked_instances = self.assign_tracks(
                current_instances, cost_matrix, add_to_queue
            )

        else:
            current_tracked_instances = self.candidate.add_new_tracks(
                current_instances, existing_track_ids=list(self._track_objects.keys())
            )
            current_tracked_instances = self.candidate.add_new_tracks(
                current_instances, existing_track_ids=list(self._track_objects.keys())
            )

        # Convert the `current_instances` back to `List[sio.PredictedInstance]` objects.
        if self.is_local_queue:
            new_pred_instances = []
            for instance in current_tracked_instances:
                if instance.track_id is not None:
                    if instance.track_id not in self._track_objects:
                        self._track_objects[instance.track_id] = sio.Track(
                            instance.track_id
                        )
                    instance.src_instance.track = self._track_objects[instance.track_id]
                    instance.src_instance.tracking_score = instance.tracking_score
                else:
                    self.candidate.add_new_tracks(
                        [instance], existing_track_ids=self._track_objects.keys()
                    )
                    if instance.track_id not in self._track_objects:
                        self._track_objects[instance.track_id] = sio.Track(
                            instance.track_id
                        )
                    instance.src_instance.track = self._track_objects[instance.track_id]
                    instance.src_instance.tracking_score = instance.tracking_score
                new_pred_instances.append(instance.src_instance)

        else:
            new_pred_instances = []
            for idx, inst in enumerate(current_tracked_instances.src_instances):
                track_id = current_tracked_instances.track_ids[idx]
                if track_id is not None:
                    if track_id not in self._track_objects:
                        self._track_objects[track_id] = sio.Track(f"track_{track_id}")
                    inst.track = self._track_objects[track_id]
                    inst.tracking_score = current_tracked_instances.tracking_scores[idx]
                    new_pred_instances.append(inst)

        return new_pred_instances

    def get_features(
        self,
        untracked_instances: List[sio.PredictedInstance],
        frame_idx: int,
        image: np.ndarray = None,
    ) -> Union[TrackInstances, List[TrackInstanceLocalQueue]]:
        """Get features for the current untracked instances.

        The feature can either be an embedding of cropped image around each instance (visual feature),
        the bounding box coordinates, or centroids, or the poses as a feature.

        Args:
            untracked_instances: List of untracked `sio.PredictedInstance` objects.
            frame_idx: Frame index of the current untracked instances.
            image: Image of the current frame if visual features are to be used.

        Returns:
            `TrackInstances` object or `List[TrackInstanceLocalQueue]` with the features
            assigned for the untracked instances and track_id set as `None`.
        """
        if self.features not in self._feature_methods:
            raise ValueError(
                "Invalid `features` argument. Please provide one of `keypoints`, `centroids`, `bboxes` and `image`"
            )

        feature_method = self._feature_methods[self.features]
        feature_list = []
        for pred_instance in untracked_instances:
            feature_list.append(feature_method(pred_instance))

        current_instances = self.candidate.get_track_instances(
            feature_list, untracked_instances, frame_idx=frame_idx, image=image
        )

        return current_instances

    def generate_candidates(self):
        """Get the tracked instances from tracker queue."""
        return self.update_candidates(self.candidate.tracker_queue)
        return self.update_candidates(self.candidate.tracker_queue)

    def update_candidates(self, candidates_list: Union[Deque, DefaultDict[int, Deque]]):
        """Return dictionary with the features of tracked instances.

        Args:
            candidates_list: List of tracked instances from tracker queue to consider.
            image: Image of the current untracked frame. (used for flow shift tracker)

        Returns:
            Dictionary with keys as track IDs and values as the list of `TrackedInstanceFeature`.
        """
        if self.is_local_queue:
            candidates_feature_dict = defaultdict(list)
            for track_id in self.candidate.current_tracks:
                candidates_feature_dict[track_id].extend(
                    self.candidate.get_features_from_track_id(track_id, candidates_list)
                )
                if all(x.feature is None for x in candidates_feature_dict[track_id]):
                    self.candidate.current_tracks.remove(track_id)
                    del self.candidate.tracker_queue[track_id]
                    del candidates_feature_dict[track_id]
        else:
            candidates_feature_dict = deque()
            # For fixed window, candidates_list is a deque of TrackInstances
            for track_id in self.candidate.current_tracks:
                for track_instance in candidates_list:
                    if track_id in track_instance.track_ids:
                        track_idx = track_instance.track_ids.index(track_id)
                        tracked_instance_feature = TrackedInstanceFeature(
                            feature=track_instance.features[track_idx],
                            src_predicted_instance=track_instance.src_instances[
                                track_idx
                            ],
                            frame_idx=track_instance.frame_idx,
                            tracking_score=track_instance.tracking_scores[track_idx],
                            instance_score=track_instance.instance_scores[track_idx],
                            shifted_keypoints=None,
                        )
                        candidates_feature_dict.append(tracked_instance_feature)
        if self.is_local_queue:
            candidates_feature_dict = defaultdict(list)
            for track_id in self.candidate.current_tracks:
                candidates_feature_dict[track_id].extend(
                    self.candidate.get_features_from_track_id(track_id, candidates_list)
                )
                if all(x.feature is None for x in candidates_feature_dict[track_id]):
                    self.candidate.current_tracks.remove(track_id)
                    del self.candidate.tracker_queue[track_id]
                    del candidates_feature_dict[track_id]
        else:
            candidates_feature_dict = deque()
            # For fixed window, candidates_list is a deque of TrackInstances
            for track_id in self.candidate.current_tracks:
                for track_instance in candidates_list:
                    if track_id in track_instance.track_ids:
                        track_idx = track_instance.track_ids.index(track_id)
                        tracked_instance_feature = TrackedInstanceFeature(
                            feature=track_instance.features[track_idx],
                            src_predicted_instance=track_instance.src_instances[
                                track_idx
                            ],
                            frame_idx=track_instance.frame_idx,
                            tracking_score=track_instance.tracking_scores[track_idx],
                            instance_score=track_instance.instance_scores[track_idx],
                            shifted_keypoints=None,
                        )
                        candidates_feature_dict.append(tracked_instance_feature)
        return candidates_feature_dict

    def get_scores(
        self,
        current_instances: Union[TrackInstances, List[TrackInstanceLocalQueue]],
        candidates_feature_dict: Dict[int, TrackedInstanceFeature],
        max_dist: int = None,
    ):
        """Compute association score between untracked and tracked instances.

        For visual feature vectors, this can be `cosine_sim`, for bounding boxes
        it could be `iou`, for centroids it could be `euclidean_dist`, and for poses it
        could be `oks`.

        Args:
            current_instances: `TrackInstances` object or `List[TrackInstanceLocalQueue]`
                with features and unassigned tracks.
            candidates_feature_dict: Dictionary with keys as track IDs and values as the
                list of `TrackedInstanceFeature`.
            max_dist: Maximum distance (in pixels) between centroids to consider a match.
                If None, no distance constraint is applied.
            max_dist: Maximum distance (in pixels) between centroids to consider a match.
                If None, no distance constraint is applied.

        Returns:
            scores: Score matrix of shape (num_new_instances, num_existing_tracks)
        """
        if self.scoring_method not in self._scoring_functions:
            raise ValueError(
                "Invalid `scoring_method` argument. Please provide one of `oks`, `cosine_sim`, `iou`, and `euclidean_dist`."
            )

        if self.scoring_reduction not in self._scoring_reduction_methods:
            raise ValueError(
                "Invalid `scoring_reduction` argument. Please provide one of `mean`, `max`, and `weighted`."
            )

        scoring_method = self._scoring_functions[self.scoring_method]
        scoring_reduction = self._scoring_reduction_methods[self.scoring_reduction]

        # Get list of features for the `current_instances`.
        if self.is_local_queue:
            current_instances_features = [x.feature for x in current_instances]
        else:
            current_instances_features = [x for x in current_instances.features]

        scores = {
            track_id: np.zeros(len(current_instances_features))
            for track_id in self.candidate.current_tracks
        }

        for f_idx, f in enumerate(current_instances_features):
            for track_id in self.candidate.current_tracks:
                # Ensure features are numpy arrays
                f = np.array(f) if not isinstance(f, np.ndarray) else f

                # Process each candidate feature
                oks = []
                for x in candidates_feature_dict[track_id]:
                    if x.feature is not None:
                        # Ensure candidate feature is a numpy array
                        candidate_feature = (
                            np.array(x.feature)
                            if not isinstance(x.feature, np.ndarray)
                            else x.feature
                        )

                        # If max_dist is set, check the distance between current instance and last instance of the track
                        if max_dist is not None:
                            last_instance = next(
                                (
                                    x
                                    for x in reversed(candidates_feature_dict[track_id])
                                    if x.feature is not None
                                ),
                                None,
                            )
                            if last_instance is not None:
                                distance = np.linalg.norm(
                                    get_centroid(f)
                                    - get_centroid(last_instance.feature)
                                )
                                if distance > max_dist:
                                    oks.append(
                                        -1e10
                                    )  # Using a very large negative number instead of -inf
                                    continue

                        score = scoring_method(f, candidate_feature)
                        oks.append(score)
                    else:
                        if self.scoring_reduction == "weighted":
                            oks.append(-1e10)
                        else:
                            oks.append(np.nan)

                # Apply scoring reduction
                if oks:
                    if np.all(isinstance(x, np.ndarray) for x in oks):
                        oks = [x[0][0] if isinstance(x, np.ndarray) else x for x in oks]
                    oks = scoring_reduction(oks)  # scoring reduction
                    scores[track_id][f_idx] = oks
                else:
                    scores[track_id][f_idx] = -1e10
                # Ensure features are numpy arrays
                f = np.array(f) if not isinstance(f, np.ndarray) else f

                # Process each candidate feature
                oks = []
                for x in candidates_feature_dict[track_id]:
                    if x.feature is not None:
                        # Ensure candidate feature is a numpy array
                        candidate_feature = (
                            np.array(x.feature)
                            if not isinstance(x.feature, np.ndarray)
                            else x.feature
                        )

                        # If max_dist is set, check the distance between current instance and last instance of the track
                        if max_dist is not None:
                            last_instance = next(
                                (
                                    x
                                    for x in reversed(candidates_feature_dict[track_id])
                                    if x.feature is not None
                                ),
                                None,
                            )
                            if last_instance is not None:
                                distance = np.linalg.norm(
                                    get_centroid(f)
                                    - get_centroid(last_instance.feature)
                                )
                                if distance > max_dist:
                                    oks.append(
                                        -1e10
                                    )  # Using a very large negative number instead of -inf
                                    continue

                        score = scoring_method(f, candidate_feature)
                        oks.append(score)
                    else:
                        if self.scoring_reduction == "weighted":
                            oks.append(-1e10)
                        else:
                            oks.append(np.nan)

                # Apply scoring reduction
                if oks:
                    if np.all(isinstance(x, np.ndarray) for x in oks):
                        oks = [x[0][0] if isinstance(x, np.ndarray) else x for x in oks]
                    oks = scoring_reduction(oks)  # scoring reduction
                    scores[track_id][f_idx] = oks
                else:
                    scores[track_id][f_idx] = -1e10
        return scores

    def scores_to_cost_matrix(self, scores: np.ndarray):
        """Converts `scores` matrix to cost matrix for track assignments."""
        # Keep scores as dictionary but negate values for cost
        cost_matrix = {
            track_id: -scores[track_id] for track_id in self.candidate.current_tracks
        }


        return cost_matrix

    def assign_tracks(
        self,
        current_instances: Union[TrackInstances, List[TrackInstanceLocalQueue]],
        cost_matrix: np.ndarray,
        add_to_queue: bool = False,
    ) -> Union[TrackInstances, List[TrackInstanceLocalQueue]]:
        """Assign track IDs using Hungarian method.

        Args:
            current_instances: `TrackInstances` object or `List[TrackInstanceLocalQueue]`
                with features and unassigned tracks.
            cost_matrix: Cost matrix of shape (num_new_instances, num_existing_tracks).

        Returns:
            `TrackInstances` object or `List[TrackInstanceLocalQueue]`objects with
                track IDs assigned.
        """
        if self.track_matching_method not in self._track_matching_methods:
            raise ValueError(
                "Invalid `track_matching_method` argument. Please provide one of `hungarian`, and `greedy`."
            )

        # If cost matrix is empty, create new tracks for all instances
        if not cost_matrix:
            return self.candidate.add_new_tracks(
                current_instances, existing_track_ids=self._track_objects.keys()
            )

        # If cost matrix is empty, create new tracks for all instances
        if not cost_matrix:
            return self.candidate.add_new_tracks(
                current_instances, existing_track_ids=self._track_objects.keys()
            )

        # Get best track matches and scores directly from cost dictionary
        tracking_scores = []
        matched_track_ids = []
        matched_instance_indices = []
        matching_method = self._track_matching_methods[self.track_matching_method]

        # Convert cost_matrix dict to numpy array for Hungarian algorithm
        track_ids = list(cost_matrix.keys())
        costs_array = np.array([cost_matrix[tid] for tid in track_ids])

        # Use Hungarian algorithm to find optimal matching if there are valid tracks
        if costs_array.shape[0] > 0 and costs_array.shape[1] > 0:
            row_ind, col_ind = matching_method(costs_array)
        # Use Hungarian algorithm to find optimal matching if there are valid tracks
        if costs_array.shape[0] > 0 and costs_array.shape[1] > 0:
            row_ind, col_ind = matching_method(costs_array)

            for row, col in zip(row_ind, col_ind):
                score = -costs_array[row, col]  # Convert cost back to score
                if (
                    score > -1e10 and score < self.max_cost if self.max_cost else True
                ):  # Only assign track if score is below threshold
                    tracking_scores.append(score)
                    matched_track_ids.append(track_ids[row])
                    matched_instance_indices.append(col)
            for row, col in zip(row_ind, col_ind):
                score = -costs_array[row, col]  # Convert cost back to score
                if (
                    score > -1e10 and score < self.max_cost if self.max_cost else True
                ):  # Only assign track if score is below threshold
                    tracking_scores.append(score)
                    matched_track_ids.append(track_ids[row])
                    matched_instance_indices.append(col)

        # Update tracker queue and assign track IDs
        current_tracked_instances = self.candidate.update_tracks(
            current_instances,
            matched_instance_indices,
            matched_track_ids,
            tracking_scores,
            add_to_queue,
            existing_track_ids=list(self._track_objects.keys()),
            existing_track_ids=list(self._track_objects.keys()),
        )

        return current_tracked_instances

    def train_identity_model(self, labels: sio.Labels, model_config: Dict[str, Any]):
        """Train an identity assignment model using tracked instances.

        Args:
            labels: Labels object containing all instances.
            model_config: Dictionary specifying the model type and its parameters.
                          Example: {"type": "hmm", "params": {"n_components": 5, ...}}
        """

        # Group instances by global track ID
        tracked_sequences = defaultdict(list)

        # Extract features for each tracked instance
        for lf in labels:
            for inst in lf.instances:
                if inst.track is not None and inst.track.name in self.global_track_ids:
                    # Assuming self.extract_pose_features is available and appropriate
                    features = self.extract_pose_features(inst)
                    tracked_sequences[inst.track.name].append(features)

        # Convert sequences to numpy arrays
        sequences = []
        lengths = []
        for track_id, seq in tracked_sequences.items():
            if len(seq) > 0:
                sequences.append(np.array(seq))
                lengths.append(len(seq))

        if not sequences:
            logger.info("No tracked sequences found for model training.")
            global_identity_model = None
            identity_model_scaler = None
            return

        # Concatenate all sequences
        X = np.vstack(sequences)

        # Initialize and fit scaler
        feature_scaler = StandardScaler()
        X_scaled = feature_scaler.fit_transform(X)
        identity_model_scaler = feature_scaler  # Store the scaler for inference

        # Check for NaN values in scaled features
        if np.isnan(X_scaled).any():
            logger.warning(
                "Scaled features contain NaN values. Attempting to remove them."
            )
            logger.warning(f"Number of NaN values: {np.isnan(X_scaled).sum()}")

            valid_mask = ~np.isnan(X_scaled).any(axis=1)
            X_scaled = X_scaled[valid_mask]

            # Update lengths to match filtered data
            new_lengths = []
            current_pos = 0
            for length in lengths:
                original_segment = valid_mask[current_pos : current_pos + length]
                new_segment_length = original_segment.sum()
                if new_segment_length > 0:
                    new_lengths.append(new_segment_length)
                current_pos += length
            lengths = new_lengths

            if not X_scaled.size or not lengths:
                logger.warning(
                    "No valid sequences remaining after NaN removal for model training."
                )
                global_identity_model = None
                # Keep the scaler as it was fit on original data before NaN check on X_scaled
                return

        # Model Initialization and Training
        model_type = model_config.get("type")
        if not model_type:
            raise ValueError("Model type must be specified in model_config.")

        model_params = model_config.get("params", {})
        current_model = None

        if model_type.lower() == "hmm":
            try:
                from hmmlearn import hmm
            except ImportError:
                logger.error(
                    "hmmlearn package is required for HMM model. Please install it."
                )
                raise

            n_components = model_params.get("n_components", 5)
            covariance_type = model_params.get("covariance_type", "full")
            random_state = model_params.get("random_state", 42)
            # Add any other HMM parameters from model_params, e.g., n_iter, tol
            hmm_params = {
                "n_components": n_components,
                "covariance_type": covariance_type,
                "random_state": random_state,
                **{
                    k: v
                    for k, v in model_params.items()
                    if k not in ["n_components", "covariance_type", "random_state"]
                },
            }
            current_model = hmm.GaussianHMM(**hmm_params)
            logger.info(f"Initializing HMM with parameters: {hmm_params}")
        # Example for another model type:
        # elif model_type.lower() == "transformer":
        #     # from .models import TransformerIdentityModel # Example import
        #     # current_model = TransformerIdentityModel(**model_params)
        #     logger.info(f"Initializing Transformer model with parameters: {model_params}")
        #     raise NotImplementedError("Transformer model training not yet implemented.")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        try:
            logger.info(f"Fitting {model_type} model...")
            if model_type.lower() == "hmm":  # HMM's fit method uses lengths
                current_model.fit(X_scaled, lengths=lengths)
            else:
                # Assume other models might have a simpler fit or need X_scaled and labels/targets
                # This part needs to be adapted based on the common interface for your models
                current_model.fit(
                    X_scaled
                )  # Or current_model.fit(sequences_scaled, sequence_labels)

            global_identity_model = current_model
            logger.info(f"Successfully trained {model_type} model.")
            if model_type.lower() == "hmm":
                logger.info(
                    f"  HMM - n_components: {current_model.n_components}, converged: {current_model.monitor_.converged}"
                )

        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
            global_identity_model = None  # Ensure model is not set if training failed
            identity_model_scaler = None  # Also clear scaler if training failed
            # Optionally re-raise the exception if the caller should handle it
            # raise

        return global_identity_model, identity_model_scaler

    def infer_and_assign_untracked_identities(
        self, labels: sio.Labels, global_identity_model, identity_model_scaler
    ):
        """Infer and assign identities to untracked instances using the trained model.

        Args:
            labels: Labels object containing all instances.
        """

        if not global_identity_model or not identity_model_scaler:
            logger.warning(
                "Identity model or feature scaler is not available. "
                "Run train_identity_model first."
            )
            return

        if not self.global_track_ids:
            logger.warning("No global track IDs defined to assign. Skipping inference.")
            return

        tracklets = self.group_tracklets(labels)

        # Process each untracked tracklet
        for track_id, instances_in_tracklet_with_frame_idx in tracklets.items():
            is_local_tracklet = True
            if track_id in self.global_track_ids:
                is_local_tracklet = False  # This tracklet already has a global ID name

            if (
                is_local_tracklet
            ):  # Only process tracklets that don't already have a global ID
                if not instances_in_tracklet_with_frame_idx:
                    logger.debug(f"Tracklet {track_id} is empty, skipping.")
                    continue

                # Sort instances by frame index
                instances_in_tracklet_with_frame_idx.sort(key=lambda x: x[0])

                frames = [item[0] for item in instances_in_tracklet_with_frame_idx]
                actual_instances = [
                    item[1] for item in instances_in_tracklet_with_frame_idx
                ]

                if not actual_instances:
                    logger.debug(
                        f"Tracklet {track_id} has no actual instances after unpacking, skipping."
                    )
                    continue

                # Extract features
                # Assuming extract_pose_features is a static method or defined in the class
                features = np.array(
                    [self.extract_pose_features(inst) for inst in actual_instances]
                )

                if features.size == 0:
                    logger.debug(
                        f"No features extracted for tracklet {track_id}, skipping."
                    )
                    continue

                # Scale features
                features_scaled = identity_model_scaler.transform(features)

                # Check for NaN values in scaled features
                if np.isnan(features_scaled).any():
                    logger.warning(
                        f"Scaled features for track {track_id} contain NaN values. Imputing with mean."
                    )
                    # Impute NaNs with the mean of the respective column
                    for col in range(features_scaled.shape[1]):
                        col_mask = np.isnan(features_scaled[:, col])
                        if col_mask.any():
                            col_mean = np.nanmean(features_scaled[:, col])
                            if np.isnan(col_mean):
                                # If entire column was NaN, use 0 or another placeholder
                                col_mean = 0
                            features_scaled[col_mask, col] = col_mean
                    if np.isnan(features_scaled).any():
                        logger.error(
                            f"Features for tracklet {track_id} still contain NaNs after imputation. Skipping."
                        )
                        continue  # Skip this tracklet if NaNs persist

                if features_scaled.shape[0] == 0:
                    logger.debug(
                        f"Tracklet {track_id} has no features after scaling/NaN handling, skipping."
                    )
                    continue

                # Predict global identity for the tracklet
                # The model's predict_identity method should handle its internal logic
                # and return a proposed global ID name from the available self.global_track_ids.
                try:
                    # We pass self.global_track_ids so the model knows the candidate pool.
                    # The model's predict_identity should return the *name* of the chosen global ID.
                    predicted_global_id_name = global_identity_model.predict(
                        features_scaled
                    )
                except AttributeError as e:
                    logger.error(
                        f"The trained model does not have a 'predict_identity' method or it failed: {e}"
                    )
                    continue  # Skip to next tracklet
                except Exception as e:
                    logger.error(
                        f"Error during model prediction for tracklet {track_id}: {e}"
                    )
                    continue

                if (
                    not predicted_global_id_name
                    or predicted_global_id_name not in self._track_objects
                ):
                    logger.warning(
                        f"Model predicted an invalid or unknown global ID '{predicted_global_id_name}' for tracklet {track_id}. Skipping."
                    )
                    continue

                assigned_global_track_object = self._track_objects[
                    predicted_global_id_name
                ]

                # Check if the proposed global track is already used in any frame of this tracklet by another instance
                is_track_available = True
                for frame_idx, inst_to_assign in zip(frames, actual_instances):
                    # labels[frame_idx] should give a LabeledFrame object
                    lf = labels.find(frame_idx=frame_idx, video=labels.video)[0]
                    if lf:
                        for existing_inst in lf.instances:
                            if (
                                existing_inst != inst_to_assign
                                and existing_inst.track == assigned_global_track_object
                            ):
                                is_track_available = False
                                break
                if not is_track_available:
                    break

                if not is_track_available:
                    # Simple strategy: if predicted ID is taken, try to find an alternative
                    # This could be made more sophisticated (e.g. using model scores for alternatives)
                    logger.warning(
                        f"Predicted global ID {predicted_global_id_name} for tracklet {track_id} is unavailable in some frames."
                    )
                    # For now, we just skip assignment if the preferred one is taken.
                    # A more advanced strategy could be implemented here, e.g., trying next best prediction from model.
                    # Or, iterate through self.global_track_ids to find one that *is* available.
                    # This part of the logic from the notebook is complex and highly dependent on the model's output.
                    # Simplified: if primary is unavailable, we skip for now.
                    logger.info(
                        f"Skipping assignment for tracklet {track_id} as preferred ID {predicted_global_id_name} is unavailable."
                    )
                    continue  # Or implement alternative assignment logic here

                # Update track references for all instances in this tracklet
                for inst_to_assign in actual_instances:
                    inst_to_assign.track = assigned_global_track_object

                logger.info(
                    f"Assigned tracklet {track_id} to global identity {assigned_global_track_object.name}"
                )

    def extract_pose_features(self, instance: sio.PredictedInstance) -> np.ndarray:
        """Extract pose features from an instance for HMM input.

        Args:
            instance: The instance to extract features from.

        Returns:
            np.ndarray: Flattened feature vector containing:
                - Centroid coordinates (x, y)
                - Keypoint coordinates (x, y for each keypoint)
                - Bounding box coordinates (x1, y1, x2, y2)
        """
        # Get centroid
        centroid = get_centroid(instance)

        # Get keypoints
        keypoints = get_keypoints(instance)
        # Replace NaN values with mean of non-NaN values
        keypoints = np.where(np.isnan(keypoints), np.nanmean(keypoints), keypoints)

        # Get bounding box
        bbox = get_bbox(instance)

        # Concatenate all features
        features = np.concatenate(
            [
                centroid,  # [x, y]
                keypoints.flatten(),  # [x1, y1, x2, y2, ...]
                bbox,  # [x1, y1, x2, y2]
            ]
        )

        return features

    def group_tracklets(
        self, labels: sio.Labels
    ) -> Dict[str, List[sio.PredictedInstance]]:
        """Group instances into tracklets based on their track IDs.

        Args:
            labels: Labels object containing all instances.

        Returns:
            Dict mapping track IDs to lists of instances in that tracklet.
        """
        tracklets = defaultdict(list)

        for frame_idx, lf in enumerate(labels):
            for inst in lf.instances:
                if inst.track is not None:
                    tracklets[inst.track.name].append((frame_idx, inst))

        return tracklets


@attrs.define
class FlowShiftTracker(Tracker):
    """Module for tracking using optical flow shift matching.

    This module handles tracking instances across frames by creating new track IDs (or)
    assigning track IDs to each instance when the `.track()` is called using optical flow
    based track matching. This is a sub-class of the `Tracker` module, which configures
    the `update_candidates()` method specific to optical flow shift matching. This class is
    initialized in the `Tracker.from_config()` method.

    Attributes:
        candidates: Either `FixedWindowCandidates` or `LocalQueueCandidates` object.
        features: One of [`keypoints`, `centroids`, `bboxes`, `image`].
            Default: `keypoints`.
        scoring_method: Method to compute association score between features from the
            current frame and the previous tracks. One of [`oks`, `cosine_sim`, `iou`,
            `euclidean_dist`]. Default: `oks`.
        scoring_reduction: Method to aggregate and reduce multiple scores if there are
            several detections associated with the same track. One of [`mean`, `max`,
            `weighted`]. Default: `mean`.
        track_matching_method: track matching algorithm. One of `hungarian`, `greedy.
                Default: `hungarian`.
        use_flow: If True, `FlowShiftTracker` is used, where the poses are matched using
            optical flow. Default: `False`.
        is_local_queue: `True` if `LocalQueueCandidates` is used else `False`.
        img_scale: Factor to scale the images by when computing optical flow. Decrease
            this to increase performance at the cost of finer accuracy. Sometimes
            decreasing the image scale can improve performance with fast movements.
            Default: 1.0.
        of_window_size: Optical flow window size to consider at each pyramid scale
            level. Default: 21.
        of_max_levels: Number of pyramid scale levels to consider. This is different
            from the scale parameter, which determines the initial image scaling.
            Default: 3

    """

    img_scale: float = 1.0
    of_window_size: int = 21
    of_max_levels: int = 3

    def _compute_optical_flow(
        self, ref_pts: np.ndarray, ref_img: np.ndarray, new_img: np.ndarray
    ):
        """Compute instances on new frame using optical flow displacements."""
        ref_img, new_img = self._preprocess_imgs(ref_img, new_img)
        shifted_pts, status, errs = cv2.calcOpticalFlowPyrLK(
            ref_img,
            new_img,
            (np.concatenate(ref_pts, axis=0)).astype("float32") * self.img_scale,
            None,
            winSize=(self.of_window_size, self.of_window_size),
            maxLevel=self.of_max_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        shifted_pts /= self.img_scale
        return shifted_pts, status, errs

    def _preprocess_imgs(self, ref_img: np.ndarray, new_img: np.ndarray):
        """Pre-process images for optical flow."""
        # Convert to uint8 for cv2.calcOpticalFlowPyrLK
        if np.issubdtype(ref_img.dtype, np.floating):
            ref_img = ref_img.astype("uint8")
        if np.issubdtype(new_img.dtype, np.floating):
            new_img = new_img.astype("uint8")

        # Ensure images are rank 2 in case there is a singleton channel dimension.
        if ref_img.ndim > 3:
            ref_img = np.squeeze(ref_img)
            new_img = np.squeeze(new_img)

        # Convert RGB to grayscale.
        if ref_img.ndim > 2 and ref_img.shape[0] == 3:
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

        # Input image scaling.
        if self.img_scale != 1:
            ref_img = cv2.resize(ref_img, None, None, self.img_scale, self.img_scale)
            new_img = cv2.resize(new_img, None, None, self.img_scale, self.img_scale)

        return ref_img, new_img

    def get_shifted_instances_from_prv_frames(
        self,
        candidates_list: Union[Deque, DefaultDict[int, Deque]],
        new_img: np.ndarray,
        feature_method,
    ) -> Dict[int, List[TrackedInstanceFeature]]:
        """Generate shifted instances onto the new frame by applying optical flow."""
        shifted_instances_prv_frames = defaultdict(list)

        if self.is_local_queue:
            # for local queue
            ref_candidates = self.candidate.get_instances_groupby_frame_idx(
                candidates_list
            )
            for fidx, ref_candidate_list in ref_candidates.items():
                ref_pts = [x.src_instance.numpy() for x in ref_candidate_list]
                shifted_pts, status, errs = self._compute_optical_flow(
                    ref_pts=ref_pts,
                    ref_img=ref_candidate_list[0].image,
                    new_img=new_img,
                )

                sections = np.cumsum([len(x) for x in ref_pts])[:-1]
                shifted_pts = np.split(shifted_pts, sections, axis=0)
                status = np.split(status, sections, axis=0)
                errs = np.split(errs, sections, axis=0)

                # Create shifted instances.
                for idx, (ref_candidate, pts, found) in enumerate(
                    zip(ref_candidate_list, shifted_pts, status)
                ):
                    # Exclude points that weren't found by optical flow.
                    found = found.squeeze().astype(bool)
                    pts[~found] = np.nan

                    # Create a shifted instance.
                    shifted_instances_prv_frames[ref_candidate.track_id].append(
                        TrackedInstanceFeature(
                            feature=feature_method(pts),
                            src_predicted_instance=ref_candidate.src_instance,
                            frame_idx=fidx,
                            tracking_score=ref_candidate.tracking_score,
                            instance_score=ref_candidate.instance_score,
                            shifted_keypoints=pts,
                        )
                    )

        else:
            # for fixed window
            candidates_list = (
                candidates_list
                if candidates_list is not None
                else self.candidate.tracker_queue
            )
            for ref_candidate in candidates_list:
                ref_pts = [x.numpy() for x in ref_candidate.src_instances]
                shifted_pts, status, errs = self._compute_optical_flow(
                    ref_pts=ref_pts, ref_img=ref_candidate.image, new_img=new_img
                )

                sections = np.cumsum([len(x) for x in ref_pts])[:-1]
                shifted_pts = np.split(shifted_pts, sections, axis=0)
                status = np.split(status, sections, axis=0)
                errs = np.split(errs, sections, axis=0)

                # Create shifted instances.
                for idx, (pts, found) in enumerate(zip(shifted_pts, status)):
                    # Exclude points that weren't found by optical flow.
                    found = found.squeeze().astype(bool)
                    pts[~found] = np.nan

                    # Create a shifted instance.
                    shifted_instances_prv_frames[ref_candidate.track_ids[idx]].append(
                        TrackedInstanceFeature(
                            feature=feature_method(pts),
                            src_predicted_instance=ref_candidate.src_instances[idx],
                            frame_idx=ref_candidate.frame_idx,
                            tracking_score=ref_candidate.tracking_scores[idx],
                            instance_score=ref_candidate.instance_scores[idx],
                            shifted_keypoints=pts,
                        )
                    )

        return shifted_instances_prv_frames

    def update_candidates(
        self,
        candidates_list: Union[Deque, DefaultDict[int, Deque]],
        image: np.ndarray,
    ) -> Dict[int, TrackedInstanceFeature]:
        """Return dictionary with the features of tracked instances.

        In this method, the tracked instances in the tracker queue are shifted on to the
        current frame using optical flow. The features are then computed from the shifted
        instances.

        Args:
            candidates_list: Tracker queue from the candidate class.
            image: Image of the current untracked frame. (used for flow shift tracker)

        Returns:
            Dictionary with keys as track IDs and values as the list of `TrackedInstanceFeature`.
        """
        # get feature method for the shifted instances
        if self.features not in self._feature_methods:
            raise ValueError(
                "Invalid `features` argument. Please provide one of `keypoints`, `centroids`, `bboxes` and `image`"
            )
        feature_method = self._feature_methods[self.features]

        # get shifted instances from optical flow
        shifted_instances_prv_frames = self.get_shifted_instances_from_prv_frames(
            candidates_list=candidates_list,
            new_img=image,
            feature_method=feature_method,
        )

        return shifted_instances_prv_frames
