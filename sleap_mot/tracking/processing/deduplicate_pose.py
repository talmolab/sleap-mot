from sleap_mot.tracking.base import IdTrackLayer
import sleap_io as sio
import numpy as np

from sleap_mot.utils import (
    get_bbox,
    compute_iou,
    get_bbox_centroid,
    compute_euclidean_distance,
    get_centroid,
    get_pairwise_distance,
)

class DeduplicatePose(IdTrackLayer):
    def __init__(self, method: str, threshold: float, operator: callable):
        """
        Args:
            method (str): Method used for deduplication. Must be one of:
                "bbox_iou", "bbox_distance", "centroid_distance", "pairwise_distance".
            threshold (float): Threshold value for the deduplication method.
            operator (callable): Comparison operator, such as operator.lt, operator.gt, etc.
        """
        valid_methods = {
            "bbox_iou",
            "bbox_distance",
            "centroid_distance",
            "pairwise_distance",
        }
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of {valid_methods}."
            )
        if not callable(operator):
            raise ValueError(
                "Operator must be a callable, e.g. operator.lt, operator.ge."
            )
        self.method = method
        self.threshold = threshold
        self.operator = operator

        # Map methods to their corresponding functions. When you define more, add here.
        self.compare_functions = {
            "bbox_iou": self.bbox_iou,
            "bbox_distance": self.bbox_distance,
            "centroid_distance": self.centroid_distance,
            "pairwise_distance": self.pairwise_distance,
        }

    @staticmethod
    def bbox_iou(self, pose_a, pose_b):
        bbox_a = get_bbox(pose_a)
        bbox_b = get_bbox(pose_b)
        return compute_iou(bbox_a, bbox_b)

    def bbox_distance(self, pose_a, pose_b):
        bbox_centroid_a = get_bbox_centroid(get_bbox(pose_a))
        bbox_centroid_b = get_bbox_centroid(get_bbox(pose_b))
        return compute_euclidean_distance(bbox_centroid_a, bbox_centroid_b)

    def centroid_distance(self, pose_a, pose_b):
        centroid_a = get_centroid(pose_a)
        centroid_b = get_centroid(pose_b)
        return compute_euclidean_distance(centroid_a, centroid_b)

    def pairwise_distance(self, pose_a, pose_b):
        pairwise_distance = get_pairwise_distance(pose_a, pose_b)
        return pairwise_distance

    def track(self, labels: sio.Labels, priority: int):
        for lf_idx, labeled_frame in enumerate(labels.labeled_frames):
            instances = [
                inst
                for inst in labeled_frame.instances
                if self.check_priority(inst, priority)
            ]
            n_poses = len(instances)
            if n_poses < 2:
                continue

            # Select the compare function for the chosen method, fallback to attribute if not defined.
            compare_fn = self.compare_functions.get(self.method, None)
            if compare_fn is None:
                # If not provided in self.compare_functions, expect an instance method with the same name
                compare_fn = getattr(self, self.method)

            for i in range(n_poses):
                for j in range(i + 1, n_poses):
                    val = compare_fn(instances[i], instances[j])
                    if self.operator(np.abs(val), self.threshold):
                        # Conflict; choose which to remove by comparing inst.score
                        if hasattr(instances[i], "score") and hasattr(
                            instances[j], "score"
                        ):
                            score_i = instances[i].score
                            score_j = instances[j].score
                            if score_i < score_j:
                                instances[i].track.valid = False
                            else:
                                instances[j].track.valid = False
                        else:
                            max(instances[i], instances[j]).track.valid = False
        return labels