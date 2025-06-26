"""Module for tracking."""

from typing import Any, Dict, List, Union, Deque, DefaultDict, Optional, Tuple
from collections import defaultdict
import attrs
import cv2
import numpy as np
from collections import deque
from tqdm import tqdm
import logging
import imageio.v3 as iio

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
    get_next_instance,
    get_bbox_pixel_intensities,
    get_pos,
    is_track_available,
    calc_seq_cost,
    combine_cost_dicts,
)
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
            self._track_objects.pop(tracklet_id.name)
            self._track_objects[global_track_id.name] = global_track_id

            self.candidate.tracker_queue[global_track_id.name] = (
                self.candidate.tracker_queue.pop(tracklet_id.name, [])
            )
            for val in self.candidate.tracker_queue[global_track_id.name]:
                val.track_id = global_track_id.name

        else:
            # for lf in labels[frame_idx - self.candidate.window_size : frame_idx + 1]:
            frames_to_process = labels[
                frame_idx - self.candidate.window_size : frame_idx + 1
            ]

            for lf in frames_to_process:
                instances = [inst for inst in lf.instances]
                features = self.get_features(instances, frame_idx)  # , lf.image)

                updated_track_ids = set()
                for inst, feature in zip(instances, features):
                    if inst.track.name not in self.candidate.tracker_queue:
                        self.candidate.tracker_queue[inst.track.name] = deque(
                            maxlen=self.candidate.window_size
                        )
                    self.candidate.tracker_queue[inst.track.name].append(feature)
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
                            # image=lf.image,
                        )
                        self.candidate.tracker_queue[track_id].append(empty_instance)

        try:
            if any(self._track_objects[key].name != key for key in self._track_objects):
                print("track object name does not match queue key")
                print(frame_idx)
            if any(v.maxlen != 5 for v in self.candidate.tracker_queue.values()):
                print(frame_idx)
        except:
            print(frame_idx)
            raise ValueError("Queue maxlen is not 5")

        # logger.debug(f"Current queue sizes: {[(k, len(v)) for k, v in self.candidate.tracker_queue.items()]}")

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
                for frame in range(frame_idx - 1, max(frame_idx - 30, -1), -1)
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

            matching_instance = next(
                i
                for i in new_instances
                if np.allclose(i.numpy(), inst_numpy, equal_nan=True)
            )

            if matching_instance.track.name == global_track_name:
                continue

            if matching_instance.track.name != global_track_name:
                for curr in new_instances:
                    if curr.track.name == global_track_name:
                        curr.track = matching_instance.track
                        matching_instance.track = self._track_objects[global_track_name]
                        break

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

    def sort_labels(self, labels):
        """Sort labels by frame index.

        Args:
            labels (Labels): Labels object containing all tracked instances

        Returns:
            sio.Labels: Labels object with frames sorted by frame index
        """
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

        return sorted_labels

    def track(
        self,
        labels: sio.Labels,
        max_dist: int = None,
        generate_new_tracks: bool = False,
    ):
        """Track instances across frames.

        This method tracks instances across frames in the provided `sio.Labels` object.
        It supports single-video labels and allows for in-place updates of the labels.

        Args:
            labels (sio.Labels): The labeled frames to track.
            generate_new_tracks (bool): If True, new tracks are generated for each frame. Recommended for videos that are challenging to track Default: False.

        Returns:
            sio.Labels: The updated labeled frames with track IDs assigned.
        """
        if len(labels.videos) > 1:
            raise NotImplementedError("Multiple videos are not supported.")

        can_load_images = labels.video.exists()

        sorted_labels = self.sort_labels(labels)
        labels.labeled_frames = sorted_labels

        self.global_track_ids = {t.name: t for t in labels.tracks}

        for lf in labels:
            if any(self._track_objects[key].name != key for key in self._track_objects):
                print(lf.frame_idx)
            if lf.instances:
                tracked_instances = [
                    (inst.numpy(), inst.track.name)
                    for inst in lf.instances
                    if inst.track is not None
                ]

                self.track_frame(
                    lf.instances,
                    lf.frame_idx,
                    generate_new_tracks=generate_new_tracks,
                    # lf.image,
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

        if generate_new_tracks:
            grid_size = (15, 15)
            transition_matrices = self.build_transition_matrices(labels, grid_size)
            averaged_visual_features = self.extract_visual_features(
                labels, patch_dimension=50
            )

            self.assign_tracklets_to_global_tracks(
                labels,
                transition_matrices,
                averaged_visual_features,
                grid_size,
                alpha=0.3,
                beta=0.7,
                patch_dimension=50,
            )

        labels.update()

        # Create new list of unique tracks first
        unique_tracks = []
        seen_names = set()
        for track in labels.tracks:
            if track.name not in seen_names:
                unique_tracks.append(track)
                seen_names.add(track.name)

        # Assign unique tracks back to labels.tracks
        labels.tracks = unique_tracks

        # Consolidate skeleton assignment
        for lf in labels:
            for inst in lf.instances:
                if inst.track is not None:
                    # Ensure inst.track points to the unique track object from the updated labels.tracks list
                    try:
                        inst.track = next(
                            t for t in labels.tracks if t.name == inst.track.name
                        )
                    except StopIteration:
                        logger.warning(
                            f"Track with name {inst.track.name} not found in unique tracks. Instance will have no track."
                        )
                        inst.track = None  # Or handle as an error
                inst.skeleton = labels.skeleton
                # inst.points = {
                #     i: point for i, (node, point) in enumerate(inst.points.items())
                # }

        labels.update()  # Call update again after potentially re-assigning track objects
        return labels

    def track_frame(
        self,
        instances: List[sio.PredictedInstance],
        frame_idx: int,
        generate_new_tracks: bool = False,
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
            frame_idx (int): The index of the current frame.
            image (np.ndarray): The image of the current frame.
            add_to_queue (bool): If True, the instances will be added to the tracker queue.
            max_dist (int): The maximum pixel distance between instances to consider a match.

        Returns:
            List[sio.PredictedInstance]: The list of instances with assigned track IDs.
        """
        current_instances = self.get_features(instances, frame_idx, image)

        candidates_feature_dict = self.generate_candidates(generate_new_tracks)

        if candidates_feature_dict:
            scores = self.get_scores(
                current_instances, candidates_feature_dict, max_dist
            )

            cost_matrix = self.scores_to_cost_matrix(scores)

            current_tracked_instances = self.assign_tracks(
                current_instances, cost_matrix, add_to_queue, generate_new_tracks
            )
        else:
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

    def generate_candidates(self, generate_new_tracks: bool = False):
        """Get the tracked instances from tracker queue."""
        return self.update_candidates(self.candidate.tracker_queue, generate_new_tracks)

    def update_candidates(
        self,
        candidates_list: Union[Deque, DefaultDict[int, Deque]],
        generate_new_tracks: bool = False,
    ):
        """Return dictionary with the features of tracked instances.

        Args:
            candidates_list: List of tracked instances from tracker queue to consider.
            image: Image of the current untracked frame. (used for flow shift tracker)

        Returns:
            Dictionary with keys as track IDs and values as the list of `TrackedInstanceFeature`.
        """
        if self.is_local_queue:
            candidates_feature_dict = defaultdict(list)
            for track_id in list(
                self.candidate.current_tracks
            ):  # Iterate over a copy if modifying during iteration
                track_features = self.candidate.get_features_from_track_id(
                    track_id, candidates_list
                )
                if (
                    all(x.feature is None for x in track_features)
                    and generate_new_tracks
                ):
                    if (
                        track_id in self.candidate.current_tracks
                    ):  # Check if still present before removing
                        self.candidate.current_tracks.remove(track_id)
                    if track_id in self.candidate.tracker_queue:
                        del self.candidate.tracker_queue[track_id]
                    # No need to delete from candidates_feature_dict as it's being built
                else:
                    candidates_feature_dict[track_id].extend(track_features)
        else:
            candidates_feature_dict = deque()
            # For fixed window, candidates_list is a deque of TrackInstances
            # This part assumes candidates_list is a deque of TrackInstances objects
            # and self.candidate.current_tracks contains the track_ids to look for.
            active_track_ids_in_deque = set()
            for (
                track_instance_container
            ) in candidates_list:  # This is a Deque[TrackInstances]
                for idx, track_id in enumerate(track_instance_container.track_ids):
                    if track_id in self.candidate.current_tracks:
                        tracked_instance_feature = TrackedInstanceFeature(
                            feature=track_instance_container.features[idx],
                            src_predicted_instance=track_instance_container.src_instances[
                                idx
                            ],
                            frame_idx=track_instance_container.frame_idx,
                            tracking_score=track_instance_container.tracking_scores[
                                idx
                            ],
                            instance_score=track_instance_container.instance_scores[
                                idx
                            ],
                            shifted_keypoints=None,  # This is not FlowShiftTracker
                        )
                        candidates_feature_dict.append(tracked_instance_feature)
                        active_track_ids_in_deque.add(track_id)
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
        generate_new_tracks: bool = False,
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

        tracking_scores = []
        matched_track_ids = []
        matched_instance_indices = []
        matching_method = self._track_matching_methods[self.track_matching_method]

        # Convert cost_matrix dict to numpy array for Hungarian algorithm
        # Filter out tracks from cost_matrix that have empty cost lists, as they can cause issues
        valid_track_ids = [tid for tid, costs in cost_matrix.items() if costs.size > 0]
        if not valid_track_ids:
            # All tracks had empty cost lists, so no matching is possible.
            return self.candidate.add_new_tracks(
                current_instances, existing_track_ids=list(self._track_objects.keys())
            )

        costs_array = np.array([cost_matrix[tid] for tid in valid_track_ids])

        # Use Hungarian algorithm to find optimal matching if there are valid tracks
        if (
            costs_array.ndim == 1
        ):  # Handle case where there's only one valid track_id with costs
            costs_array = costs_array.reshape(1, -1)

        if costs_array.shape[0] > 0 and costs_array.shape[1] > 0:
            row_ind, col_ind = matching_method(costs_array)

            for row, col in zip(row_ind, col_ind):
                score = -costs_array[row, col]  # Convert cost back to score
                if score > -1e10 and (
                    self.max_cost is None or score < self.max_cost
                ):  # Only assign track if score is below threshold or no threshold
                    tracking_scores.append(score)
                    matched_track_ids.append(
                        valid_track_ids[row]
                    )  # Use valid_track_ids here
                    matched_instance_indices.append(col)

        # Update tracker queue and assign track IDs
        current_tracked_instances = self.candidate.update_tracks(
            current_instances,
            matched_instance_indices,
            matched_track_ids,
            tracking_scores,
            add_to_queue=add_to_queue,
            generate_new_tracks=generate_new_tracks,
            existing_track_ids=list(self._track_objects.keys()),
        )

        return current_tracked_instances

    def build_transition_matrices(
        self, labels: sio.Labels, grid_size: Tuple[int, int]
    ) -> Dict:
        """Build transition probability matrices for tracked instances.

        Args:
            labels: Labels object
            grid_size: Grid dimensions

        Returns:
            Dict: Dictionary containing transition matrices for forward and backward directions
        """
        arena_width, arena_height = labels.video.shape[2], labels.video.shape[1]

        transitions = defaultdict(lambda: np.zeros(grid_size))
        reverse_transitions = defaultdict(lambda: np.zeros(grid_size))
        transition_matrices = {1: transitions, -1: reverse_transitions}
        prev_instances = {}

        for lf in labels:
            for inst in lf:
                if inst.track.name in self.global_track_ids:
                    i_curr, j_curr = get_pos(inst, grid_size, arena_width, arena_height)
                    if inst.track.name in prev_instances:
                        prev_centroid = prev_instances[inst.track.name]
                        i_prev, j_prev = prev_centroid[0], prev_centroid[1]
                        transitions[(i_prev, j_prev)][i_curr, j_curr] += 1
                        reverse_transitions[(i_curr, j_curr)][i_prev, j_prev] += 1
                    prev_instances[inst.track.name] = [i_curr, j_curr]

        # Normalize transition matrices
        for key in transitions:
            heatmap = transitions[key]
            total = heatmap.sum()
            if total > 0:
                transitions[key] = heatmap / total

        for key in reverse_transitions:
            heatmap = reverse_transitions[key]
            total = heatmap.sum()
            if total > 0:
                reverse_transitions[key] = heatmap / total

        return transition_matrices

    def extract_visual_features(
        self, labels: sio.Labels, patch_dimension: int = 50
    ) -> Dict[str, np.ndarray]:
        """Extract and average visual features for each global track.

        Args:
            labels: Labels object
            patch_dimension: Size of image patches to extract

        Returns:
            Dict: Dictionary mapping track names to averaged visual features
        """

        def full_pose_available(inst: sio.PredictedInstance) -> bool:
            """Check if all keypoints in an instance are available (not NaN).

            Args:
                inst: Instance to check

            Returns:
                bool: True if all keypoints are available, False otherwise
            """
            for point in inst.points.values():
                if np.isnan(point.x) or np.isnan(point.y):
                    return False
            return True

        reader = iio.imiter(labels.video.filename)
        tracked_sequences = defaultdict(list)

        for lf in labels:
            image = next(reader)
            for inst in lf.instances:
                if inst.track.name in self.global_track_ids and full_pose_available(
                    inst
                ):
                    visual_features = get_bbox_pixel_intensities(
                        inst, image, patch_dimension
                    )
                    tracked_sequences[inst.track.name].append(visual_features)

        # Average features across all frames for each track
        averaged_tracked_sequences = {}
        for global_track, seq in tracked_sequences.items():
            if len(seq) > 0:
                avg_array = np.mean(np.stack(seq, axis=0), axis=0)
                averaged_tracked_sequences[global_track] = avg_array

        return averaged_tracked_sequences

    def calculate_tracklet_costs(
        self,
        tracklet: List[Tuple[int, sio.PredictedInstance]],
        labels: sio.Labels,
        transition_matrices: Dict,
        averaged_visual_features: Dict[str, np.ndarray],
        grid_size: Tuple[int, int],
        patch_dimension: int = 50,
    ) -> Tuple[Dict, Dict]:
        """Calculate positional and visual costs for assigning a tracklet to each global track.

        Args:
            tracklet: List of (frame_idx, instance) tuples
            labels: Labels object
            transition_matrices: Dictionary of transition matrices
            averaged_visual_features: Dictionary of averaged visual features
            grid_size: Grid dimensions
            patch_dimension: Size of image patches

        Returns:
            Tuple of (positional_costs, visual_costs) dictionaries
        """
        arena_width, arena_height = labels.video.shape[2], labels.video.shape[1]
        curr_frames = [inst[0] for inst in tracklet]
        curr_instances = [inst[1] for inst in tracklet]
        first_frame_idx = curr_frames[0]
        last_frame_idx = curr_frames[-1]

        cost_dict_pos = {}
        cost_dict_vis = {}

        # Extract visual features for current tracklet
        images = [labels.video[frame] for frame in curr_frames]
        visual_features = np.array(
            [
                get_bbox_pixel_intensities(inst, image, patch_dimension)
                for inst, image in zip(curr_instances, images)
            ]
        )

        for global_track in self.global_track_ids:
            if is_track_available(curr_frames, curr_instances, global_track, labels):
                prev_inst, prev_frame, prev_direction = get_next_instance(
                    labels, first_frame_idx, global_track, -1
                )
                seq_inst, seq_frame, seq_direction = get_next_instance(
                    labels, last_frame_idx, global_track, 1
                )

                if len(curr_instances) < 2:
                    prev_dist = (
                        first_frame_idx - prev_frame
                        if prev_frame is not None
                        else float("inf")
                    )
                    seq_dist = (
                        seq_frame - last_frame_idx
                        if seq_frame is not None
                        else float("inf")
                    )

                    if prev_dist > seq_dist:
                        prev_inst, prev_frame = seq_inst, seq_frame
                        prev_direction = 1
                    else:
                        seq_inst, seq_frame = prev_inst, prev_frame
                        seq_direction = -1

                if prev_frame is None:
                    prev_inst = seq_inst
                    prev_frame = seq_frame
                    prev_direction = 1

                if seq_frame is None:
                    seq_inst = prev_inst
                    seq_frame = prev_frame
                    seq_direction = -1

                if prev_direction == seq_direction:
                    cost = calc_seq_cost(
                        prev_inst,
                        tracklet,
                        prev_direction,
                        transition_matrices[prev_direction],
                        grid_size,
                        arena_width,
                        arena_height,
                    )

                    # Create weighting vector that emphasizes ends of tracklet
                    # weights = np.ones(len(tracklet))
                    # for i in range(len(tracklet)):
                    #     if prev_direction > 0:
                    #         dist_from_end = (len(tracklet) - i - 1) / len(tracklet)
                    #     else:
                    #         dist_from_end = i / len(tracklet)
                    #     weights[i] = 1.0 - (dist_from_end * 0.8)
                else:
                    # Split tracklet into two halves
                    mid_idx = len(tracklet) // 2
                    first_half = tracklet[:mid_idx]
                    second_half = tracklet[mid_idx:]

                    first_cost = calc_seq_cost(
                        prev_inst,
                        first_half,
                        prev_direction,
                        transition_matrices[prev_direction],
                        grid_size,
                        arena_width,
                        arena_height,
                    )
                    second_cost = calc_seq_cost(
                        seq_inst,
                        second_half,
                        seq_direction,
                        transition_matrices[seq_direction],
                        grid_size,
                        arena_width,
                        arena_height,
                    )

                    cost = np.mean([first_cost, second_cost])
                    # np.concatenate([first_cost, second_cost])

                    # Create weighting vector that emphasizes ends of tracklet
                    # weights = np.ones(len(tracklet))
                    # mid_point = len(tracklet) // 2
                    # for i in range(len(tracklet)):
                    #     end_weight = 1.8
                    #     mid_weight = 0.4
                    #     dist_from_mid = abs(i - mid_point) / mid_point
                    #     weights[i] = mid_weight + (
                    #         dist_from_mid * (end_weight - mid_weight)
                    #     )

                # Apply weights and calculate final positional cost
                # cost_dist = cost * weights
                # cost_dist = np.mean(cost_dist) * 0.5
                cost_dist = cost

                # Calculate visual cost
                if global_track in averaged_visual_features:
                    global_image = averaged_visual_features[global_track]
                    flat_global = global_image.flatten()
                    cost_array = []
                    for curr_feature in visual_features:
                        flat_curr = curr_feature.flatten()
                        cost = 1 - np.dot(flat_curr, flat_global) / (
                            np.linalg.norm(flat_curr) * np.linalg.norm(flat_global)
                        )
                        cost_array.append(cost)
                    cost_vis = np.mean(cost_array) * 0.5
                else:
                    cost_vis = float("inf")
            else:
                cost_dist = float("inf")
                cost_vis = float("inf")

            cost_dict_pos[global_track] = cost_dist
            cost_dict_vis[global_track] = cost_vis

        return cost_dict_pos, cost_dict_vis

    def assign_tracklets_to_global_tracks(
        self,
        labels: sio.Labels,
        transition_matrices: Dict,
        averaged_visual_features: Dict[str, np.ndarray],
        grid_size: Tuple[int, int],
        alpha: float = 0.3,
        beta: float = 0.7,
        patch_dimension: int = 50,
    ) -> None:
        """Assign untracked tracklets to global track IDs.

        Args:
            labels: Labels object
            transition_matrices: Dictionary of transition matrices
            averaged_visual_features: Dictionary of averaged visual features
            grid_size: Grid dimensions
            alpha: Weight for positional costs
            beta: Weight for visual costs
            patch_dimension: Size of image patches
        """

        def group_tracklets(
            labels: sio.Labels,
        ) -> Dict[str, List[Tuple[int, sio.PredictedInstance]]]:
            """Group instances into tracklets based on their track IDs.

            Args:
                labels: Labels object containing all instances.

            Returns:
                Dict mapping track IDs to lists of (frame_idx, instance) tuples.
            """
            tracklets = defaultdict(list)

            for frame_idx, lf in enumerate(labels):
                for inst in lf.instances:
                    if inst.track is not None:
                        tracklets[inst.track.name].append((frame_idx, inst))

            return tracklets

        arena_width, arena_height = labels.video.shape[2], labels.video.shape[1]
        tracklets = group_tracklets(labels)

        for track_id, instances in tracklets.items():
            if track_id not in self.global_track_ids:
                cost_dict_pos, cost_dict_vis = self.calculate_tracklet_costs(
                    instances,
                    labels,
                    transition_matrices,
                    averaged_visual_features,
                    grid_size,
                    patch_dimension,
                )

                print(cost_dict_pos)
                print(cost_dict_vis)

                cost_dict = combine_cost_dicts(
                    cost_dict_pos, cost_dict_vis, alpha, beta
                )
                track_to_assign = min(cost_dict.items(), key=lambda x: x[1])[0]

                print(f"tracklet {track_id} given ID {track_to_assign}")

                for frame, inst in instances:
                    inst.track = self.global_track_ids[track_to_assign]


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
