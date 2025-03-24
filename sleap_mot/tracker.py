"""Module for tracking."""

from typing import Any, Dict, List, Union, Deque, DefaultDict, Optional
from collections import defaultdict
import attrs
import cv2
import numpy as np
from collections import deque
import threading

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
    _scoring_functions: Dict[str, Any] = {
        "oks": compute_oks,
        "iou": compute_iou,
        "cosine_sim": compute_cosine_sim,
        "euclidean_dist": compute_euclidean_distance,
    }
    _scoring_reduction_methods: Dict[str, Any] = {"mean": np.nanmean, "max": np.nanmax}
    _feature_methods: Dict[str, Any] = {
        "keypoints": get_keypoints,
        "centroids": get_centroid,
        "bboxes": get_bbox,
    }
    _track_matching_methods: Dict[str, Any] = {
        "hungarian": hungarian_matching,
        "greedy": greedy_matching,
    }
    _track_objects: Dict[int, sio.Track] = {}

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
            )

        tracker = cls(
            candidate=candidate,
            features=features,
            scoring_method=scoring_method,
            scoring_reduction=scoring_reduction,
            track_matching_method=track_matching_method,
            use_flow=use_flow,
            is_local_queue=is_local_queue,
        )
        return tracker

    def initialize_tracker(self, context_frames: List[sio.Labels]):
        """Initialize the tracker with context frames.

        This method clears the existing tracker queue and initializes the tracker with the provided context frames.
        The context frames are used to establish initial tracks and their features.

        Args:
            context_frames (List[sio.Labels]): A list of labeled frames containing tracked instances to initialize
                the tracker with. Each frame should have instances with track IDs already assigned.

        Returns:
            None
        """
        self.candidate.tracker_queue.clear()

        current_instances = []
        for lf in context_frames:
            untracked_instances = lf.instances
            current_instances.extend(
                self.get_features(untracked_instances, lf.frame_idx, lf.image)
            )

        for inst in current_instances:
            track_name = int(inst.src_instance.track.name.split("_")[1])

            inst.track_id = track_name
            if track_name not in self.candidate.tracker_queue:
                self.candidate.tracker_queue[track_name] = deque(
                    maxlen=self.candidate.window_size
                )
                self._track_objects[track_name] = inst.src_instance.track
            self.candidate.tracker_queue[track_name].append(inst)

            if track_name not in self.candidate.current_tracks:
                self.candidate.current_tracks.append(track_name)

    def track(self, labels: sio.Labels, inplace: bool = False):
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
            sorted_labels.append(
                labels.find(frame_idx=frame_idx, video=labels.video, return_new=True)[0]
            )

        labels.labeled_frames = sorted_labels

        def initialize_and_track(bout, untracked_frames, start, end):
            """Initialize the tracker and track instances across frames.

            This helper function initializes the tracker with the provided context frames
            and tracks instances across the untracked frames.

            Args:
                bout (List[sio.Labels]): The context frames to initialize the tracker with.
                untracked_frames (List[sio.Labels]): The untracked frames to track.
                start (int): The start index of the untracked frames.
                end (int): The end index of the untracked frames.
            """
            # Initialize the tracker with context frames
            self.initialize_tracker(bout)

            # For large gaps between tracked frames, enable queue addition to improve tracking
            if len(untracked_frames) > 1000 or len(untracked_frames) == len(labels):
                add_to_queue = True
            else:
                add_to_queue = False

            if (
                len(self.candidate.tracker_queue) > 0
                and 0 in self.candidate.tracker_queue
                and len(self.candidate.tracker_queue[0]) > 0
            ):
                # Compare frame indices to determine tracking direction
                if (
                    self.candidate.tracker_queue[0][0].frame_idx
                    > untracked_frames[end - 1].frame_idx
                ):
                    # Track frames in reverse order from end to start
                    untracked_frames = reversed(untracked_frames[start:end])
                else:
                    untracked_frames = untracked_frames[start:end]

            # Process each untracked frame
            for untracked_lf in untracked_frames:
                img = untracked_lf.image if can_load_images else None
                # if untracked_lf.frame_idx % 1000 == 0:
                #     logger.info(f"Tracking frame: {untracked_lf.frame_idx}")

                untracked_lf.instances = self.track_frame(
                    untracked_lf.instances,
                    untracked_lf.frame_idx,
                    image=img,
                    add_to_queue=add_to_queue,
                )

        # Initialize lists to store tracked and untracked frames
        tracked_frames, untracked_frames = [], []
        untracked_frames_grouped = []
        tracked_frames_grouped = []
        prev_frame_tracked = True

        # Iterate through all frames and group them into tracked and untracked sequences
        for lf in labels:
            # Check if frame has instances and all instances are tracked
            if lf.instances and all(inst.track is not None for inst in lf.instances):
                # If previous frame was untracked, start new tracked group
                if not prev_frame_tracked:
                    tracked_frames_grouped.append(tracked_frames)
                    tracked_frames = []

                tracked_frames.append(lf)
                prev_frame_tracked = True

            else:
                # If previous frame was tracked and we have untracked frames,
                # add them to grouped list and start new untracked group
                if prev_frame_tracked and untracked_frames:
                    untracked_frames_grouped.append(untracked_frames)
                    untracked_frames = []

                untracked_frames.append(lf)
                prev_frame_tracked = False

        tracked_frames_grouped.append(tracked_frames)
        untracked_frames_grouped.append(untracked_frames)

        # Process each group of untracked frames
        for i in range(len(untracked_frames_grouped)):
            # Get the tracked frames before this untracked group (if any)
            first_bout = (
                tracked_frames_grouped[i] if i < len(tracked_frames_grouped) else []
            )
            # Get the tracked frames after this untracked group (if any)
            second_bout = (
                tracked_frames_grouped[i + 1]
                if i + 1 < len(tracked_frames_grouped)
                else []
            )
            # Get the current group of untracked frames
            untracked_bout = untracked_frames_grouped[i]

            # Determine where to split the untracked frames for bidirectional tracking
            if first_bout and second_bout:
                # If we have tracked frames on both sides, split in middle
                half_idx = len(untracked_bout) // 2
            elif first_bout:
                # If we only have tracked frames before, process all frames forward
                half_idx = len(untracked_bout)
            else:
                # If we only have tracked frames after, process all frames backward
                half_idx = 0

            # Track forward from previous tracked frames
            if first_bout:
                initialize_and_track(
                    # Use up to 5 previous tracked frames for initialization
                    first_bout[-5:] if len(first_bout) >= 5 else first_bout,
                    untracked_bout,
                    0,  # Start from beginning of untracked bout
                    half_idx,  # Track up to middle or end
                )

            # Track backward from next tracked frames
            if second_bout:
                initialize_and_track(
                    # Use up to 5 next tracked frames (reversed) for initialization
                    list(
                        reversed(
                            second_bout[:5] if len(second_bout) >= 5 else second_bout
                        )
                    ),
                    untracked_bout,
                    half_idx,  # Start from middle
                    len(untracked_bout),  # Track to end
                )

            # If no tracked frames on either side, track all frames forward
            if not first_bout and not second_bout:
                initialize_and_track([], untracked_bout, 0, None)

        # Update the labels object with new tracking information
        labels.update()
        return labels

    def track_frame(
        self,
        untracked_instances: List[sio.PredictedInstance],
        frame_idx: int,
        image: np.ndarray = None,
        add_to_queue: bool = False,
    ) -> List[sio.PredictedInstance]:
        """Assign track IDs to the untracked list of `sio.PredictedInstance` objects.

        This method assigns track IDs to the untracked instances in the provided list.
        It first generates candidate features, updates the candidates list, computes
        association scores, and then assigns track IDs using the Hungarian method.

        Args:
            untracked_instances (List[sio.PredictedInstance]): The list of untracked instances from frame_idx to assign track IDs to.
            frame_idx (int): The index of the current frame.
            image (np.ndarray): The image of the current frame.
            add_to_queue (bool): If True, the instances will be added to the tracker queue.

        Returns:
            List[sio.PredictedInstance]: The list of instances with assigned track IDs.
        """
        current_instances = self.get_features(untracked_instances, frame_idx, image)

        candidates_list = self.generate_candidates()

        if candidates_list:
            candidates_feature_dict = self.update_candidates(candidates_list, image)

            scores = self.get_scores(current_instances, candidates_feature_dict)
            cost_matrix = self.scores_to_cost_matrix(scores)

            current_tracked_instances = self.assign_tracks(
                current_instances, cost_matrix, add_to_queue
            )

        else:
            current_tracked_instances = self.candidate.add_new_tracks(current_instances)

        # Convert the `current_instances` back to `List[sio.PredictedInstance]` objects.
        if self.is_local_queue:
            new_pred_instances = []
            for instance in current_tracked_instances:
                if instance.track_id is not None:
                    if instance.track_id not in self._track_objects:
                        self._track_objects[instance.track_id] = sio.Track(
                            f"track_{instance.track_id}"
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
        return self.candidate.tracker_queue

    def update_candidates(
        self, candidates_list: Union[Deque, DefaultDict[int, Deque]], image: np.ndarray
    ) -> Dict[int, TrackedInstanceFeature]:
        """Return dictionary with the features of tracked instances.

        Args:
            candidates_list: List of tracked instances from tracker queue to consider.
            image: Image of the current untracked frame. (used for flow shift tracker)

        Returns:
            Dictionary with keys as track IDs and values as the list of `TrackedInstanceFeature`.
        """
        candidates_feature_dict = defaultdict(list)
        for track_id in self.candidate.current_tracks:
            candidates_feature_dict[track_id].extend(
                self.candidate.get_features_from_track_id(track_id, candidates_list)
            )
        return candidates_feature_dict

    def get_scores(
        self,
        current_instances: Union[TrackInstances, List[TrackInstanceLocalQueue]],
        candidates_feature_dict: Dict[int, TrackedInstanceFeature],
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

        scores = np.zeros(
            (len(current_instances_features), len(self.candidate.current_tracks))
        )

        for f_idx, f in enumerate(current_instances_features):
            for track_id in self.candidate.current_tracks:
                oks = [
                    scoring_method(f, x.feature)
                    for x in candidates_feature_dict[track_id]
                ]
                oks = scoring_reduction(oks)  # scoring reduction
                scores[f_idx][track_id] = oks

        return scores

    def scores_to_cost_matrix(self, scores: np.ndarray):
        """Converts `scores` matrix to cost matrix for track assignments."""
        cost_matrix = -scores
        cost_matrix[np.isnan(cost_matrix)] = np.inf
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

        matching_method = self._track_matching_methods[self.track_matching_method]

        row_inds, col_inds = matching_method(cost_matrix)
        tracking_scores = [
            -cost_matrix[row, col] for row, col in zip(row_inds, col_inds)
        ]

        # update the candidates tracker queue with the newly tracked instances and assign
        # track IDs to `current_instances`.
        current_tracked_instances = self.candidate.update_tracks(
            current_instances, row_inds, col_inds, tracking_scores, add_to_queue
        )

        return current_tracked_instances


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
