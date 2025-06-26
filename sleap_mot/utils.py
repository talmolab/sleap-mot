"""Helper functions for Tracker module."""

from scipy.optimize import linear_sum_assignment

import numpy as np
import sleap_io as sio
from typing import List, Tuple, Dict
import cv2


def compute_instance_area(points: np.ndarray) -> np.ndarray:
    """Compute the area of the bounding box of a set of keypoints.

    Args:
        points: A numpy array of coordinates.

    Returns:
        The area of the bounding box of the points.
    """
    if points.ndim == 2:
        points = np.expand_dims(points, axis=0)

    min_pt = np.nanmin(points)
    max_pt = np.nanmax(points)

    return np.prod(max_pt - min_pt, axis=-1)


def compute_oks(
    points_gt: np.ndarray,
    points_pr: np.ndarray,
    scale: float | None = None,
    stddev: float = 0.025,
    use_cocoeval: bool = True,
) -> np.ndarray:
    """Compute the object keypoints similarity between sets of points.

    Args:
        points_gt: Ground truth instances of shape (n_gt, n_nodes, n_ed),
            where n_nodes is the number of body parts/keypoint types, and n_ed
            is the number of Euclidean dimensions (typically 2 or 3). Keypoints
            that are missing/not visible should be represented as NaNs.
        points_pr: Predicted instance of shape (n_pr, n_nodes, n_ed).
        use_cocoeval: Indicates whether the OKS score is calculated like cocoeval
            method or not. True indicating the score is calculated using the
            cocoeval method (widely used and the code can be found here at
            https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L192C5-L233C20)
            and False indicating the score is calculated using the method exactly
            as given in the paper referenced in the Notes below.
        scale: Size scaling factor to use when weighing the scores, typically
            the area of the bounding box of the instance (in pixels). This
            should be of the length n_gt. If a scalar is provided, the same
            number is used for all ground truth instances. If set to None, the
            bounding box area of the ground truth instances will be calculated.
        stddev: The standard deviation associated with the spread in the
            localization accuracy of each node/keypoint type. This should be of
            the length n_nodes. "Easier" keypoint types will have lower values
            to reflect the smaller spread expected in localizing it.

    Returns:
        The object keypoints similarity between every pair of ground truth and
        predicted instance, a numpy array of of shape (n_gt, n_pr) in the range
        of [0, 1.0], with 1.0 denoting a perfect match.

    Notes:
        It's important to set the stddev appropriately when accounting for the
        difficulty of each keypoint type. For reference, the median value for
        all keypoint types in COCO is 0.072. The "easiest" keypoint is the left
        eye, with stddev of 0.025, since it is easy to precisely locate the
        eyes when labeling. The "hardest" keypoint is the left hip, with stddev
        of 0.107, since it's hard to locate the left hip bone without external
        anatomical features and since it is often occluded by clothing.

        The implementation here is based off of the descriptions in:
        Ronch & Perona. "Benchmarking and Error Diagnosis in Multi-Instance Pose
        Estimation." ICCV (2017).
    """
    if points_gt.ndim == 2:
        points_gt = np.expand_dims(points_gt, axis=0)
    if points_pr.ndim == 2:
        points_pr = np.expand_dims(points_pr, axis=0)

    if scale is None:
        scale = compute_instance_area(points_gt)

    n_gt, n_nodes, n_ed = points_gt.shape  # n_ed = 2 or 3 (euclidean dimensions)
    n_pr = points_pr.shape[0]

    # If scalar scale was provided, use the same for each ground truth instance.
    if np.isscalar(scale):
        scale = np.full(n_gt, scale)

    # If scalar standard deviation was provided, use the same for each node.
    if np.isscalar(stddev):
        stddev = np.full(n_nodes, stddev)

    # Compute displacement between each pair.
    displacement = np.reshape(points_gt, (n_gt, 1, n_nodes, n_ed)) - np.reshape(
        points_pr, (1, n_pr, n_nodes, n_ed)
    )
    assert displacement.shape == (n_gt, n_pr, n_nodes, n_ed)

    # Convert to pairwise Euclidean distances.
    distance = (displacement**2).sum(axis=-1)  # (n_gt, n_pr, n_nodes)
    assert distance.shape == (n_gt, n_pr, n_nodes)

    # Compute the normalization factor per keypoint.
    if use_cocoeval:
        # If use_cocoeval is True, then compute normalization factor according to cocoeval.
        spread_factor = (2 * stddev) ** 2
        scale_factor = 2 * (scale + np.spacing(1))
    else:
        # If use_cocoeval is False, then compute normalization factor according to the paper.
        spread_factor = stddev**2
        scale_factor = 2 * ((scale + np.spacing(1)) ** 2)
    normalization_factor = np.reshape(spread_factor, (1, 1, n_nodes)) * np.reshape(
        scale_factor, (n_gt, 1, 1)
    )
    assert normalization_factor.shape == (n_gt, 1, n_nodes)

    # Since a "miss" is considered as KS < 0.5, we'll set the
    # distances for predicted points that are missing to inf.
    missing_pr = np.any(np.isnan(points_pr), axis=-1)  # (n_pr, n_nodes)
    assert missing_pr.shape == (n_pr, n_nodes)
    distance[:, missing_pr] = np.inf

    # Compute the keypoint similarity as per the top of Eq. 1.
    ks = np.exp(-(distance / normalization_factor))  # (n_gt, n_pr, n_nodes)
    assert ks.shape == (n_gt, n_pr, n_nodes)

    # Set the KS for missing ground truth points to 0.
    # This is equivalent to the visibility delta function of the bottom
    # of Eq. 1.
    missing_gt = np.any(np.isnan(points_gt), axis=-1)  # (n_gt, n_nodes)
    assert missing_gt.shape == (n_gt, n_nodes)
    ks[np.expand_dims(missing_gt, axis=1)] = 0

    # Compute the OKS.
    n_visible_gt = np.sum(
        (~missing_gt).astype("float32"), axis=-1, keepdims=True
    )  # (n_gt, 1)
    oks = np.sum(ks, axis=-1) / n_visible_gt
    assert oks.shape == (n_gt, n_pr)

    return oks


def hungarian_matching(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Match new instances to existing tracks using Hungarian matching."""
    row_ids, col_ids = linear_sum_assignment(cost_matrix)
    return row_ids, col_ids


def greedy_matching(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Match new instances to existing tracks using greedy bipartite matching."""
    # Sort edges by ascending cost.
    rows, cols = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
    unassigned_edges = list(zip(rows, cols))

    # Greedily assign edges.
    row_inds, col_inds = [], []
    while len(unassigned_edges) > 0:
        # Assign the lowest cost edge.
        row_ind, col_ind = unassigned_edges.pop(0)
        row_inds.append(row_ind)
        col_inds.append(col_ind)

        # Remove all other edges that contain either node (in reverse order).
        for i in range(len(unassigned_edges) - 1, -1, -1):
            if unassigned_edges[i][0] == row_ind or unassigned_edges[i][1] == col_ind:
                del unassigned_edges[i]

    return row_inds, col_inds


def get_keypoints(pred_instance: sio.PredictedInstance | np.ndarray):
    """Return keypoints as np.array from the `PredictedInstance` object."""
    if isinstance(pred_instance, np.ndarray):
        return pred_instance
    return pred_instance.numpy()


def get_centroid(pred_instance: sio.PredictedInstance | np.ndarray):
    """Return the centroid of the `PredictedInstance` object."""
    pts = pred_instance
    if not isinstance(pred_instance, np.ndarray):
        pts = pred_instance.numpy()
    centroid = np.nanmedian(pts, axis=0)
    return centroid


def get_bbox(pred_instance: sio.PredictedInstance | np.ndarray):
    """Return the bounding box coordinates for the `PredictedInstance` object."""
    points = (
        pred_instance.numpy()
        if not isinstance(pred_instance, np.ndarray)
        else pred_instance
    )
    bbox = np.concatenate(
        [
            np.nanmin(points, axis=0),
            np.nanmax(points, axis=0),
        ]  # [xmin, ymin, xmax, ymax]
    )
    return bbox


def compute_euclidean_distance(a, b):
    """Return the negative euclidean distance between a and b points."""
    return -np.linalg.norm(a - b)


def compute_iou(a, b):
    """Return the intersection over union for given a and b bounding boxes [xmin, ymin, xmax, ymax]."""
    (xmin1, ymin1, xmax1, ymax1), (xmin2, ymin2, xmax2, ymax2) = a, b

    xmin_intersection = max(xmin1, xmin2)
    ymin_intersection = max(ymin1, ymin2)
    xmax_intersection = min(xmax1, xmax2)
    ymax_intersection = min(ymax1, ymax2)

    intersection_area = max(0, xmax_intersection - xmin_intersection + 1) * max(
        0, ymax_intersection - ymin_intersection + 1
    )
    bbox1_area = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    bbox2_area = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area
    return iou


def compute_cosine_sim(a, b):
    """Return cosine simalirity between a and b vectors."""
    numer = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cosine_sim = numer / denom
    return cosine_sim


def get_pos(inst, grid_size, arena_width, arena_height):
    centroid = get_centroid(inst)
    i = int(centroid[0] / arena_width * grid_size[0])
    j = int(centroid[1] / arena_height * grid_size[1])
    return i, j


def get_patch(
    instance: sio.PredictedInstance, image: np.ndarray, dimensions: int = 20
) -> np.ndarray:
    """Extract a square patch centered around the instance's centroid and rotate it so the nose points left.

    Args:
        instance: The instance to extract the patch from
        image: The video frame containing the instance
        dimensions: Size of the square patch (width and height)

    Returns:
        np.ndarray: Square patch of the image centered on the instance, rotated so the nose points left
    """
    # Get instance centroid and keypoints
    centroid = get_centroid(instance)
    keypoints = get_keypoints(instance)
    if centroid is None or image is None or keypoints is None or len(keypoints) < 2:
        return np.zeros((dimensions, dimensions))

    # Convert centroid to integers
    cx, cy = int(centroid[0]), int(centroid[1])

    # Calculate angle to rotate
    nose = keypoints[0]  # Assuming the first keypoint is the nose
    tail = keypoints[-1]  # Assuming the last keypoint is the tail
    angle = np.degrees(np.arctan2(nose[1] - tail[1], nose[0] - tail[0]))

    # Rotate the entire image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), angle - 180, 1)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    # Calculate patch boundaries on the rotated image
    half_dim = dimensions // 2
    x1 = cx - half_dim
    x2 = cx + half_dim
    y1 = cy - half_dim
    y2 = cy + half_dim

    # Create empty patch
    patch = np.zeros(
        (dimensions, dimensions, 3)
        if len(image.shape) == 3
        else (dimensions, dimensions)
    )

    # Calculate valid source and destination regions
    src_x1 = max(0, x1)
    src_x2 = min(w, x2)
    src_y1 = max(0, y1)
    src_y2 = min(h, y2)

    dst_x1 = max(0, -x1)
    dst_x2 = dimensions - max(0, x2 - w)
    dst_y1 = max(0, -y1)
    dst_y2 = dimensions - max(0, y2 - h)

    # Copy valid region from rotated image to patch
    patch[dst_y1:dst_y2, dst_x1:dst_x2] = rotated_image[src_y1:src_y2, src_x1:src_x2]

    return patch


def get_bbox_pixel_intensities(
    instance: sio.PredictedInstance, image, dimension: int = 20
) -> np.ndarray:
    """Extract pixel intensity values from within an instance's bounding box.

    Args:
        instance: The instance to extract features from
        image: The video frame containing the instance

    Returns:
        np.ndarray: Flattened array of pixel intensity values from within the bounding box,
                resized to a fixed size (32x32) for consistent feature dimensions
    """
    bbox_pixels = get_patch(instance, image, dimension)

    # Convert to grayscale if image is RGB
    if len(bbox_pixels.shape) == 3:
        bbox_pixels = np.mean(bbox_pixels, axis=2)

    # Resize to fixed dimensions for consistent feature size
    try:
        bbox_pixels_resized = cv2.resize(bbox_pixels, (dimension, dimension))
        # Flatten and normalize pixel values
        return bbox_pixels_resized
    except:
        # Return zeros if resize fails
        return np.zeros(dimension * dimension)


def is_track_available(
    frames: List[int],
    instances: List[sio.PredictedInstance],
    global_track: str,
    labels: sio.Labels,
) -> bool:
    """Check if a global track is available across all frames for the given instances.

    Args:
        frames: List of frame indices
        instances: List of instances corresponding to frames
        global_track: Global track name to check
        labels: Labels object

    Returns:
        bool: True if track is available, False otherwise
    """
    for frame, inst in zip(frames, instances):
        existing_instances = [i for i in labels[frame].instances if i != inst]
        if any(i.track.name == global_track for i in existing_instances):
            return False
    return True


def get_next_instance(labels, frame_idx, track_name, direction):
    next_frame = frame_idx + direction
    while next_frame >= 0 and next_frame < labels.video.shape[0]:
        frame = labels.find(frame_idx=next_frame, video=labels.video, return_new=True)[
            0
        ]
        for inst in frame:
            if inst.track is not None and inst.track.name == track_name:
                return inst, next_frame, direction
        next_frame += direction
    return None, None, None


def calc_seq_cost(
    prev_inst: sio.PredictedInstance,
    tracklet: List[Tuple[int, sio.PredictedInstance]],
    direction: int,
    transition_matrix: Dict,
    grid_size: Tuple[int, int],
    arena_width: int,
    arena_height: int,
) -> np.ndarray:
    """Calculate sequence cost for a tracklet based on transition probabilities.

    Args:
        prev_inst: Previous instance
        tracklet: List of (frame_idx, instance) tuples
        direction: Direction to process tracklet (1 for forward, -1 for backward)
        transition_matrix: Dictionary of transition probability matrices
        grid_size: Grid dimensions
        arena_width: Arena width
        arena_height: Arena height

    Returns:
        np.ndarray: Cost array for each instance in tracklet
    """
    i_prev, j_prev = get_pos(prev_inst, grid_size, arena_width, arena_height)
    # cost = np.zeros(len(tracklet))

    if direction > 0:
        ind = 0
    elif direction < 0:
        ind = len(tracklet) - 1

    # while 0 <= ind <= len(tracklet) - 1:
    curr_inst = tracklet[ind][1]
    key = (i_prev, j_prev)
    heat_map = (
        transition_matrix[key] if key in transition_matrix else np.zeros(grid_size)
    )
    i_curr, j_curr = get_pos(curr_inst, grid_size, arena_width, arena_height)
    prob = heat_map[i_curr, j_curr]
    # cost[ind] = -np.log(prob + 1e-10)
    cost = -np.log(prob + 1e-10)
    # i_prev, j_prev = i_curr, j_curr
    # ind += direction
    # prev_inst = curr_inst

    return cost


def combine_cost_dicts(
    cost_dict_pos: Dict, cost_dict_vis: Dict, alpha: float = 0.5, beta: float = 0.5
) -> Dict:
    """Combine positional and visual cost dictionaries.

    Args:
        cost_dict_pos: Dictionary of positional costs
        cost_dict_vis: Dictionary of visual costs
        alpha: Weight for positional costs
        beta: Weight for visual costs

    Returns:
        Dict: Combined cost dictionary
    """
    combined = {}
    all_keys = set(cost_dict_pos) | set(cost_dict_vis)

    for key in all_keys:
        pos_cost = cost_dict_pos.get(key, 0.0)
        vis_cost = cost_dict_vis.get(key, 0.0)
        combined[key] = alpha * pos_cost + beta * vis_cost

    return combined
