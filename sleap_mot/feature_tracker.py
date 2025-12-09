"""Feature-based tracking algorithms for SLEAP-MOT.

This module provides abstract base classes and concrete implementations for
feature-based tracking algorithms in SLEAP-MOT (SLEAP Multi-Object Tracking).
The module includes trackers that use various features such as RFID data,
fur color, and motion models to track multiple objects across video frames.

Classes
-------
FeatureTracker : ABC
    Abstract base class for feature-based tracking algorithms.
RFIDFeatureTracker : FeatureTracker
    RFID-based tracking using spatial heatmaps and RFID ping data.
FurColorFeatureTracker : FeatureTracker
    Fur color-based tracking using PCA and KNN clustering.

Examples
--------
>>> from sleap_mot.feature_tracker import RFIDFeatureTracker
>>> tracker = RFIDFeatureTracker()
>>> tracker.generate_heatmaps(rfid_data, slp_files, video_files)
>>> tracked_labels = tracker.track(labels, video_path, output_path, rfid_data)
"""

from abc import ABC, abstractmethod
import sleap_io as sio
from pathlib import Path
import pandas as pd
import tqdm
import numpy as np
import h5py
import shapely
import joblib
from sklearn.neighbors import KernelDensity
from collections import Counter
import math
from functools import partial
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
from collections import defaultdict
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

from sleap_mot.utils import (
    get_bbox,
    get_centroid,
    tri_point_motion_model,
    check_bbox_overlap,
    rotate_points,
)


class FeatureTracker(ABC):
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
        self.motion_kde_paths = None

    def load_and_preprocess_labels(self, labels: sio.Labels, video_path: str):
        """Load and preprocess SLEAP labels."""
        # Replace video paths
        labels.replace_filenames(prefix_map={labels.videos[0].filename: video_path})

        # Convert instances to PredictedInstance
        # n_frames = labels.video.shape[0]
        n_frames = len(labels)
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

        labels.labeled_frames = sorted_labels
        return labels

    def get_iou(self, trx):
        """Calculate IoU for each frame and pose."""
        # Calculate IoU for each frame and pose
        n_frames = trx.shape[0]
        n_poses = trx.shape[1]
        iou_per_pose = np.zeros((n_frames, n_poses))

        for frame_idx in range(n_frames):
            for pose_idx in range(n_poses):
                # Get bounding box for current pose
                pose_pts = trx[frame_idx, pose_idx]
                if np.all(np.isnan(pose_pts)):
                    continue

                x0y0_pose = np.nanmin(pose_pts, axis=0)
                x1y1_pose = np.nanmax(pose_pts, axis=0)
                pose_area = np.prod(x1y1_pose - x0y0_pose)

                # Compare with other poses in same frame
                max_iou = 0
                for other_idx in range(n_poses):
                    if other_idx == pose_idx:
                        continue

                    other_pts = trx[frame_idx, other_idx]
                    if np.all(np.isnan(other_pts)):
                        continue

                    # Get bounding box for other pose
                    x0y0_other = np.nanmin(other_pts, axis=0)
                    x1y1_other = np.nanmax(other_pts, axis=0)
                    other_area = np.prod(x1y1_other - x0y0_other)

                    # Calculate intersection coordinates
                    ix_x0y0 = np.maximum(x0y0_pose, x0y0_other)
                    ix_x1y1 = np.minimum(x1y1_pose, x1y1_other)

                    # Calculate intersection area
                    ix_area = np.prod(np.maximum(ix_x1y1 - ix_x0y0, 0))

                    # Calculate union area
                    un_area = pose_area + other_area - ix_area

                    # Calculate IoU
                    iou = ix_area / un_area if un_area > 0 else 0
                    max_iou = max(max_iou, iou)

                iou_per_pose[frame_idx, pose_idx] = max_iou

        return iou_per_pose

    def extract_tracking_data(self, labels):
        """Extract tracking data from labels into DataFrame and array format."""
        node_names = labels.skeleton.node_names
        columns = ["frame_idx", "pose_idx"] + [
            f"{node}.{coord}" for node in node_names for coord in ["x", "y", "score"]
        ]

        # Convert to array format
        track_names = list(set(track.name for track in labels.tracks))
        n_tracks = len(track_names)
        n_frames = int(len(labels))
        n_nodes = len(node_names)

        trx = np.full((n_frames, n_tracks, n_nodes, 2), np.nan)

        for frame_idx in range(n_frames):
            lf = labels.find(frame_idx=frame_idx, video=labels.video, return_new=True)
            for pose_idx, inst in enumerate(lf[0].instances):
                pts = inst.numpy()
                trx[frame_idx, pose_idx] = pts[:, :2]

        iou_per_pose = self.get_iou(trx)

        return trx, track_names, iou_per_pose

    def find_close_poses(self, pose_instance, labels, frame_idx, threshold):
        """Find poses with bounding boxes that are close to each other."""
        close_poses = []
        frame_poses = []

        # Get bounding box for input pose
        if pose_instance is not None:
            x0y0_pose, x1y1_pose = get_bbox(pose_instance)

            # Compare with other poses in same frame
            frame_labels = labels[frame_idx]
            for other_idx, other_instance in enumerate(frame_labels.instances):
                if other_instance is None:
                    continue

                # Get bounding box for other pose
                x0y0_other, x1y1_other = get_bbox(other_instance)

                # Calculate centers of bounding boxes
                center_pose = (x0y0_pose + x1y1_pose) / 2
                center_other = (x0y0_other + x1y1_other) / 2

                # Calculate displacement between centers
                displacement = np.linalg.norm(center_pose - center_other)

                # If centers are within threshold distance
                if displacement <= threshold:
                    frame_poses.append(other_idx)

            for pose_idx in frame_poses:
                close_poses.append((frame_idx, pose_idx))

        return close_poses

    def get_bbox_tracklets(
        self,
        labels: sio.Labels,
        trx: np.ndarray,
        iou_per_pose: np.ndarray,
        iou_thresh: float,
        dist_thresh: float,
    ):
        """Get tracklets based on bounding box overlap."""
        tracklets = []
        curr_tracklet = []
        n_frames = trx.shape[0]
        n_tracks = trx.shape[1]

        tracklet_matrix = np.full((n_frames, n_tracks), False)

        for global_frame_idx in tqdm.tqdm(range(n_frames)):
            lf = labels[global_frame_idx]
            for pose_idx, pose in enumerate(lf.instances):
                curr_tracklet = []
                if tracklet_matrix[global_frame_idx, pose_idx]:
                    continue

                curr_frame_idx = global_frame_idx

                while iou_per_pose[curr_frame_idx, pose_idx] <= iou_thresh:
                    curr_tracklet.append((curr_frame_idx, pose_idx))
                    tracklet_matrix[curr_frame_idx, pose_idx] = True
                    if curr_frame_idx == n_frames - 1:
                        break
                    close_poses = self.find_close_poses(
                        labels[curr_frame_idx].instances[pose_idx],
                        labels,
                        curr_frame_idx + 1,
                        dist_thresh,
                    )

                    # Filter out poses that are already in other tracklets
                    if (
                        len(close_poses) == 1
                        and not tracklet_matrix[close_poses[0][0], close_poses[0][1]]
                    ):
                        curr_frame_idx, pose_idx = close_poses[0]

                    else:
                        break

                if len(curr_tracklet) > 0:
                    tracklets.append(curr_tracklet)

        return tracklets

    def get_motion_tracklets(
        self,
        labels: sio.Labels,
        long_kde_path: str,
        short_kde_path: str,
        iou_per_pose: np.ndarray,
        iou_thresh: float,
        max_motion_gap: int,
        KDE_samples: int,
        percentile: int,
    ):
        """Get tracklets based on motion model.

        Args:
            labels: SLEAP labels object containing instances to track
            long_kde_path: Path to long distance KDE model
            short_kde_path: Path to short distance KDE model
            iou_per_pose: IoU per pose
            iou_thresh: IoU threshold
            max_motion_gap: Maximum motion gap to consider (px)
            KDE_samples: Number of samples for KDE
            percentile: Percentile for KDE
        """

        class KDE:
            """Kernel Density Estimation class."""

            def __init__(self, kde, bounds):
                """Initialize the KDE class."""
                x_min, x_max, y_min, y_max = bounds
                # Store the KDE model
                self.kde = kde
                # Get min/max bounds from provided bounds
                self.x_min = x_min
                self.x_max = x_max
                self.y_min = y_min
                self.y_max = y_max

                # Create grid of points
                x = np.linspace(self.x_min, self.x_max, 100)
                y = np.linspace(self.y_min, self.y_max, 100)
                X_grid, Y_grid = np.meshgrid(x, y)
                xy_grid = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

                # Get density values and find maximum
                density = np.exp(kde.score_samples(xy_grid))
                self.max_density = density.max()

            def get_probability(self, point):
                """Get probability density at a point, normalized by maximum density.

                Args:
                    point: Array-like of shape (2,) containing x,y coordinates

                Returns:
                    float: Probability density at the point normalized between 0 and 1
                """
                x, y = point

                if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
                    return 0.0
                # Reshape point to 2D array expected by KDE
                point = np.array(point).reshape(1, -1)

                # Get density at point
                density = np.exp(self.kde.score_samples(point))[0]

                # Normalize by maximum density
                prob = density / self.max_density

                return prob

        def motion_model(
            labels,
            frame_idx,
            curr_tracklet,
            close_KDE: KDE,
            far_KDE: KDE,
            max_motion_gap,
        ):
            """Predict next pose using motion model based on KDE."""
            # Need at least 2 frames to calculate velocity
            if len(curr_tracklet) < 2:
                return None

            prev2_frame_idx, prev2_pose_idx = curr_tracklet[-2]
            prev_frame_idx, prev_pose_idx = curr_tracklet[-1]

            prev2_pose = labels[prev2_frame_idx].instances[prev2_pose_idx]
            prev_pose = labels[prev_frame_idx].instances[prev_pose_idx]

            # Skip if any poses are None
            if prev2_pose is None or prev_pose is None:
                return None

            # Calculate centers of poses
            prev2_points = np.array(
                [[point["xy"][0], point["xy"][1]] for point in prev2_pose.points]
            )
            prev_points = np.array(
                [[point["xy"][0], point["xy"][1]] for point in prev_pose.points]
            )
            prev2_center = np.nanmean(prev2_points, axis=0)
            prev_center = np.nanmean(prev_points, axis=0)

            best_pose_idx = None

            # Check each pose in next frame
            x1, y1 = prev2_center
            x2, y2 = prev_center

            p2_hat = (0, 0)  # Origin point
            p1_hat = (x1 - x2, y1 - y2)

            j = (0, -1)
            # Find angle between j and p1_hat using dot product formula
            theta = math.atan2(-p1_hat[0], p1_hat[1])
            p1_hat = rotate_points(p1_hat, theta)

            frame_instances = labels[frame_idx].instances
            for pose_idx, next_pose in enumerate(frame_instances):
                if next_pose is None:
                    continue

                # Calculate center of next pose
                next_points = next_pose.numpy()
                next_center = np.nanmean(next_points, axis=0)
                x3, y3 = next_center
                p3_hat = (x3 - x2, y3 - y2)
                p3_hat = rotate_points(p3_hat, theta)

                # Calculate distance between previous centers
                dist_between_prev = np.linalg.norm(np.array(p2_hat) - np.array(p1_hat))

                # Skip if distance between previous poses is too large
                if dist_between_prev >= max_motion_gap:
                    prob = far_KDE.get_probability(p3_hat)
                else:
                    prob = close_KDE.get_probability(p3_hat)

                if prob > 0:
                    if best_pose_idx is not None:
                        # If we already found a pose, only keep the one with highest probability
                        if prob > best_prob:
                            best_pose_idx = pose_idx
                            best_prob = prob
                    else:
                        best_pose_idx = pose_idx
                        best_prob = prob

            if best_pose_idx is not None:
                print(
                    f"Returning best pose index {frame_idx} for track {labels[frame_idx].instances[best_pose_idx].track}"
                )
            return best_pose_idx

        long_motion_kde = joblib.load(long_kde_path)
        short_motion_kde = joblib.load(short_kde_path)

        min_percentile = 100 - percentile
        max_percentile = percentile

        long_kde_samples = long_motion_kde.sample(KDE_samples)
        x_vals = long_kde_samples[:, 0]
        y_vals = long_kde_samples[:, 1]
        x_min, x_max = np.percentile(x_vals, [min_percentile, max_percentile])
        y_min, y_max = np.percentile(y_vals, [min_percentile, max_percentile])
        long_kde_stats = KDE(long_motion_kde, (x_min, x_max, y_min, y_max))

        short_kde_samples = short_motion_kde.sample(KDE_samples)
        x_vals = short_kde_samples[:, 0]
        y_vals = short_kde_samples[:, 1]
        x_min, x_max = np.percentile(x_vals, [min_percentile, max_percentile])
        y_min, y_max = np.percentile(y_vals, [min_percentile, max_percentile])
        short_kde_stats = KDE(short_motion_kde, (x_min, x_max, y_min, y_max))

        n_frames = len(labels.labeled_frames)
        n_poses = len(labels.tracks)

        tracklets = []
        is_already_in_tracklet = np.full((n_frames, n_poses), False)

        for global_frame_idx in range(n_frames):
            lf = labels[global_frame_idx]
            for pose_idx, pose in enumerate(lf.instances):
                curr_tracklet = []
                if is_already_in_tracklet[global_frame_idx, pose_idx]:
                    continue

                curr_frame_idx = global_frame_idx

                while iou_per_pose[curr_frame_idx, pose_idx] <= iou_thresh:
                    curr_tracklet.append((curr_frame_idx, pose_idx))
                    is_already_in_tracklet[curr_frame_idx, pose_idx] = True
                    if curr_frame_idx == n_frames - 1:
                        print(
                            f"Ending tracklet at frame {curr_frame_idx, pose_idx}: Reached end of video"
                        )
                        break
                    close_poses = self.find_close_poses(
                        labels[curr_frame_idx].instances[pose_idx],
                        labels,
                        curr_frame_idx + 1,
                        max_motion_gap,
                    )

                    # Filter out poses that are already in other tracklets
                    if (
                        len(close_poses) == 1
                        and not is_already_in_tracklet[
                            close_poses[0][0], close_poses[0][1]
                        ]
                    ):
                        curr_frame_idx, pose_idx = close_poses[0]

                    elif len(close_poses) < 1 and len(curr_tracklet) >= 2:
                        velocity_pose_idx = motion_model(
                            labels,
                            curr_frame_idx + 1,
                            curr_tracklet,
                            short_kde_stats,
                            long_kde_stats,
                            max_motion_gap,
                        )
                        if (
                            velocity_pose_idx
                            and not is_already_in_tracklet[
                                curr_frame_idx + 1, velocity_pose_idx
                            ]
                        ):
                            curr_frame_idx, pose_idx = (
                                curr_frame_idx + 1,
                                velocity_pose_idx,
                            )
                        else:
                            print(
                                f"Ending tracklet at frame {curr_frame_idx, pose_idx}: No valid velocity-based prediction"
                            )
                            break

                    else:
                        if len(close_poses) == 0:
                            print(
                                f"Ending tracklet at frame {curr_frame_idx, pose_idx}: Tracklet not long enough to predict velocity"
                            )
                        else:
                            print(
                                f"Ending tracklet at frame {curr_frame_idx, pose_idx}: Multiple close poses ({len(close_poses)}) or pose already tracked"
                            )
                        break

                if len(curr_tracklet) > 1:
                    tracklets.append(curr_tracklet)

        return tracklets

    def assign_track_ids(self, tracklet_id_pairs, labels):
        """Assign track IDs to tracklets based on RFID assignments."""

        def set_track_id(tracklet, id, labels):
            """Set track ID for all poses in a tracklet."""
            for frame_idx, pose_idx in tracklet:
                labels[frame_idx].instances[pose_idx].track = id

        # Reset all tracks to None
        for lf in labels:
            for inst in lf.instances:
                inst.track = None

        # Generate new track IDs for each tracklet
        for index, (tracklet, track_id_list) in enumerate(tracklet_id_pairs):

            if len(set(track_id_list)) > 1:
                # Count occurrences of each ID
                id_counts = Counter(track_id_list)

                # Find the ID(s) with maximum occurrences
                # Exclude None from id_counts
                filtered_id_counts = {
                    id: count for id, count in id_counts.items() if id is not None
                }
                if filtered_id_counts:
                    max_count = max(filtered_id_counts.values())
                    most_common_ids = [
                        id
                        for id, count in filtered_id_counts.items()
                        if count == max_count
                    ]
                else:
                    max_count = 0
                    most_common_ids = []

                # If there's a single most common ID, use it, otherwise empty the list
                if len(most_common_ids) == 1:
                    track_id_list = [most_common_ids[0]] * len(track_id_list)
                else:
                    track_id_list = []
                tracklet_id_pairs[index] = (tracklet, track_id_list)

            if len(track_id_list) > 0:
                track_id = track_id_list[0]
                current_frames = set(frame for frame, _ in tracklet)
                # Check if this tracklet overlaps with any other tracklets that have the same track_id
                for other_tracklet, other_track_ids in tracklet_id_pairs:
                    if (
                        len(other_track_ids) > 0
                        and other_track_ids[0] == track_id
                        and other_tracklet != tracklet
                    ):
                        # Get frame numbers for both tracklets
                        other_frames = set(frame for frame, _ in other_tracklet)

                        # Check for overlap
                        if current_frames & other_frames:
                            # If overlap exists, compare track_id_list lengths
                            if len(track_id_list) > len(other_track_ids):
                                # Current tracklet has more ID assignments, clear the other one
                                set_track_id(other_tracklet, None, labels)
                            elif len(track_id_list) < len(other_track_ids):
                                # Other tracklet has more ID assignments, clear current one
                                set_track_id(tracklet, None, labels)
                                track_id = None
                                break
                            else:
                                # Equal lengths, clear both
                                set_track_id(other_tracklet, None, labels)
                                set_track_id(tracklet, None, labels)
                                track_id = None
                                break
                # Check if track with this ID already exists
                existing_track = next(
                    (t for t in labels.tracks if t.name == track_id), None
                )
                if track_id == None:
                    set_track_id(tracklet, None, labels)
                    continue
                track = existing_track if existing_track else sio.Track(name=track_id)
                if not existing_track:
                    labels.tracks.append(track)
                # Label all poses in the tracklet with the same track ID
                set_track_id(tracklet, track, labels)

        # unique_tracks = []
        # seen_names = set()
        # for lf in labels:
        #     for inst in lf.instances:
        #         if inst.track is not None:
        #             if inst.track.name not in seen_names:
        #                 unique_tracks.append(inst.track)
        #                 seen_names.add(inst.track.name)

        # # Assign unique tracks back to labels.tracks
        # labels.tracks = unique_tracks

    def get_motion_sequences(self, slp_file, video_file, max_motion_gap):
        """Extract motion sequences from a single video file."""
        idle_count = 0
        motion_count = 0

        labels = sio.load_file(slp_file)
        labels.replace_filenames(
            prefix_map={
                str(Path(labels.videos[0].backend_metadata["filename"]).parent): str(
                    Path(video_file).parent
                )
            }
        )

        frames_with_instances = []
        motion_sequences = []
        thetas = []

        for frame in labels:
            frame_idx = frame.frame_idx
            if len(frame.instances) > 0 and not np.all(
                np.isnan(frame.instances[0].numpy())
            ):
                # Get center point of bounding box
                instance = frame.instances[0]
                frames_with_instances.append((frame_idx, instance))

        # Find sequences of 3 consecutive frames
        for i in range(len(frames_with_instances) - 2):
            f1_idx, pose1 = frames_with_instances[i]
            f2_idx, pose2 = frames_with_instances[i + 1]
            f3_idx, pose3 = frames_with_instances[i + 2]

            if f2_idx == f1_idx + 1 and f3_idx == f2_idx + 1:
                # Get bounding boxes for each pose
                bbox1 = get_bbox(frames_with_instances[i][1])
                bbox2 = get_bbox(frames_with_instances[i + 1][1])
                bbox3 = get_bbox(frames_with_instances[i + 2][1])

                # Run motion model on the three points
                p1 = get_centroid(pose1)
                p2 = get_centroid(pose2)
                p3 = get_centroid(pose3)
                theta, p1_hat, p2_hat, p3_hat = tri_point_motion_model(p1, p2, p3)

                # Skip if distance between p1 and p3 is too large
                if (
                    np.linalg.norm(p2 - p1) > max_motion_gap
                    or np.linalg.norm(p3 - p2) > max_motion_gap
                ):
                    continue

                if (
                    check_bbox_overlap(bbox1, bbox2)
                    or check_bbox_overlap(bbox2, bbox3)
                    or check_bbox_overlap(bbox1, bbox3)
                ):
                    idle_count += 1
                else:
                    motion_count += 1

                motion_sequences.append(
                    (p1_hat, p2_hat, p3_hat, f1_idx, f2_idx, f3_idx)
                )
                thetas.append(theta)

        return motion_sequences, thetas, idle_count, motion_count

    def generate_motion_kde_models(
        self,
        slp_folder,
        video_folder,
        output_dir="./",
        max_motion_gap=400,
        short_distance_threshold=80,
    ):
        """Generate long and short distance KDE motion models from SLEAP tracking data.

        Args:
            slp_folder (str or Path): Path to folder containing .slp files
            video_folder (str or Path): Path to folder containing corresponding video files
            output_dir (str or Path): Directory to save the joblib files
            max_motion_gap: Maximum motion gap to consider (px)
            short_distance_threshold: Threshold to split long and short distance sequences (px)

        Returns:
            tuple: Paths to the generated long and short KDE joblib files
        """
        slp_folder = Path(slp_folder)
        video_folder = Path(video_folder)
        output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all .slp files
        slp_files = sorted(slp_folder.glob("*.slp"))

        # Create a list to store matching pairs
        file_pairs = []
        for slp_file in slp_files:
            # Extract the timestamp from the .slp filename
            timestamp = slp_file.stem  # e.g. "2025-01-22T18_20_25"

            # Construct the expected video filename
            video_filename = f"Oryx_chunked{timestamp}.avi"
            video_path = video_folder / video_filename

            if video_path.exists():
                file_pairs.append({"slp": slp_file, "video": video_path})
            else:
                print(f"Warning: No matching video file found for {slp_file.name}")

        print(f"Found {len(file_pairs)} file pairs to process")

        # Initialize list to store all motion sequences
        motion_sequences = []
        thetas = []
        idle_count = 0
        motion_count = 0

        # Process each file pair
        print("Processing motion sequences...")
        for pair in tqdm.tqdm(file_pairs):
            # Load and process the SLP file
            slp_file = pair["slp"]
            video_file = pair["video"]

            # Get motion sequences for this video
            sequences, theta, curr_idle_count, curr_motion_count = (
                self.get_motion_sequences(slp_file, video_file, max_motion_gap)
            )
            motion_sequences.append(sequences)
            thetas.append(theta)
            idle_count += curr_idle_count
            motion_count += curr_motion_count

        # print(f"Found {len(motion_sequences)} total motion sequences across all videos")
        print(f"Idle sequences: {idle_count}, Motion sequences: {motion_count}")

        # Calculate total sequences across all videos
        total_sequences = sum(len(sequences) for sequences in motion_sequences)
        print(f"Total number of individual motion sequences: {total_sequences}")

        # Convert sequences to numpy arrays with consistent shape (n,3,2)
        print("Converting sequences to arrays...")
        processed_motion_sequences = np.array([])

        for sequences in motion_sequences:
            # Take first 3 sequences and convert to array with shape (3,2)
            for seq in sequences:
                seq = np.array(seq[:3])
                if processed_motion_sequences.size == 0:
                    processed_motion_sequences = np.array([seq])
                else:
                    processed_motion_sequences = np.vstack(
                        (processed_motion_sequences, [seq])
                    )

        print(f"Processed motion sequences shape: {processed_motion_sequences.shape}")

        # Calculate distances between p1 and p2 for all sequences
        p1p2_distances = np.sqrt(
            np.sum(
                (
                    processed_motion_sequences[:, 1, :]
                    - processed_motion_sequences[:, 0, :]
                )
                ** 2,
                axis=1,
            )
        )

        # Create boolean masks for filtering
        long_mask = p1p2_distances > short_distance_threshold
        short_mask = p1p2_distances <= short_distance_threshold

        # Split sequences using boolean masks
        long_sequences = processed_motion_sequences[long_mask]
        short_sequences = processed_motion_sequences[short_mask]

        print(
            f"Long sequences: {len(long_sequences)}, Short sequences: {len(short_sequences)}"
        )

        if len(long_sequences) == 0 or len(short_sequences) == 0:
            error_msg = (
                f"Cannot generate KDE models: "
                f"Long sequences: {len(long_sequences)}, Short sequences: {len(short_sequences)}.\n"
            )
            if len(long_sequences) == 0:
                error_msg += (
                    f"No long-distance sequences found (movements > {short_distance_threshold}px). "
                    f"Try decreasing the 'short_distance_threshold' parameter (current: {short_distance_threshold}px). "
                    f"Typical values: 40-100px depending on your video resolution and animal movement patterns.\n"
                )
            if len(short_sequences) == 0:
                error_msg += (
                    f"No short-distance sequences found (movements <= {short_distance_threshold}px). "
                    f"Try increasing the 'short_distance_threshold' parameter (current: {short_distance_threshold}px).\n"
                )
            raise ValueError(error_msg)

        # Create long distance KDE
        print("Creating long distance KDE...")
        long_points = long_sequences[:, 2]  # Third point (p3) from each sequence

        long_kde = KernelDensity(bandwidth=10.0, kernel="gaussian")
        long_kde.fit(long_points)

        # Create short distance KDE
        print("Creating short distance KDE...")
        short_points = short_sequences[:, 2]  # Third point (p3) from each sequence

        short_kde = KernelDensity(bandwidth=15.0, kernel="gaussian")
        short_kde.fit(short_points)

        # Save the KDE models
        long_kde_path = output_dir / "motion_kde_long.joblib"
        short_kde_path = output_dir / "motion_kde_short.joblib"

        print(f"Saving KDE models to {output_dir}...")
        joblib.dump(long_kde, long_kde_path)
        joblib.dump(short_kde, short_kde_path)

        print("KDE models exported successfully!")

        self.motion_kde_paths = (str(long_kde_path), str(short_kde_path))

        return str(long_kde_path), str(short_kde_path)

    def get_tracklets(
        self,
        labels,
        method,
        trx,
        iou_per_pose,
        iou_thresh,
        dist_thresh,
        KDE_samples,
        percentile,
    ):
        """Get tracklets from labels using specified method."""
        if method == "bbox":
            tracklets = self.get_bbox_tracklets(
                labels, trx, iou_per_pose, iou_thresh, dist_thresh
            )
        elif method == "motion":
            if self.motion_kde_paths is None:
                raise ValueError(
                    "motion_kde_paths must be generated before using motion tracking"
                )
            long_kde_path, short_kde_path = self.motion_kde_paths
            tracklets = self.get_motion_tracklets(
                labels,
                long_kde_path,
                short_kde_path,
                iou_per_pose,
                iou_thresh,
                dist_thresh,
                KDE_samples,
                percentile,
            )
        else:
            raise ValueError("method must be either 'bbox' or 'motion'")

        return tracklets

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

    def get_confidence_vector(self, tracklets, n_frames, n_tracks):
        """Get vector specifying which poses from which frames are included in tracklets."""
        if not tracklets:
            return None, tracklets

        confidence_vector = np.full((n_frames, n_tracks), False)

        for tracklet in tracklets:
            for frame_idx, pose_idx in tracklet:
                confidence_vector[frame_idx, pose_idx] = True

        return confidence_vector

    def get_tracklet_id_pairs_from_knn(self, tracklets, G_mapped, track_names):
        """Get tracklet id pairs from KNN results."""
        tracklet_id_pairs = []

        for t_idx, tracklet in enumerate(tracklets):
            # print(f"Processing tracklet {t_idx}: {tracklet}")
            curr_tracks = []
            for f_idx, (frame_idx, pose_idx) in enumerate(tracklet):
                track_idx = int(G_mapped[frame_idx, pose_idx])
                # print(f"  Frame {frame_idx}, Pose {pose_idx}: G_mapped={track_idx}")
                if track_idx == -1:
                    # print(f"    -> Ambiguous or unassigned, appending None")
                    curr_tracks.append(None)
                    continue
                # print(f"    -> Assigned to track name: {track_names[track_idx]}")
                curr_tracks.append(track_names[track_idx])
            # print(f"  Resulting track IDs for tracklet {t_idx}: {curr_tracks}")
            tracklet_id_pairs.append((tracklet, curr_tracks))

        return tracklet_id_pairs

    @abstractmethod
    def track(self, *args, **kwargs):
        """Abstract method to be implemented by subclasses."""
        pass


class RFIDFeatureTracker(FeatureTracker):
    """Feature tracker for RFID-based tracking.

    This class implements a feature tracker that uses RFID data to track
    animals in videos. It generates heatmaps from RFID pings and uses them
    to assign tracks to animals.
    """

    def __init__(self):
        """Initialize the RFIDFeatureTracker."""
        self.heatmaps = None
        self.heatmaps_path = None

    def _get_hull_polygons(self, inst, body_inds, pad):
        """Get convex hull polygons for an instance."""
        pts = inst.numpy()[body_inds]
        is_nan = np.isnan(pts).all(axis=-1)
        if (~is_nan).sum() < 3:
            return None
        pts = pts[(~np.isnan(pts)).any(axis=-1)]
        pts = shapely.MultiPoint(pts)
        hull = pts.convex_hull
        if pad > 0:
            hull = hull.buffer(pad)
        return hull

    def _get_bounding_box_polygons(self, inst, body_inds, pad):
        """Get bounding box polygons for an instance."""
        pts = inst.numpy()[body_inds]
        if np.isnan(pts).any():
            return None
        pts = shapely.MultiPoint(pts)
        bbox = pts.envelope
        if pad > 0:
            bbox = bbox.buffer(pad)
        return bbox

    def _get_ellipse_polygons(self, inst, body_inds, pad):
        """Get ellipse polygons for an instance."""
        pts = inst.numpy()[body_inds]
        if np.isnan(pts).any():
            return None
        pts = pts[(~np.isnan(pts)).any(axis=-1)]
        if len(pts) < 3:
            return None
        pts = shapely.MultiPoint(pts)
        # Calculate the minimum rotated rectangle (which is an ellipse approximation)
        min_rotated_rect = pts.minimum_rotated_rectangle
        if pad > 0:
            min_rotated_rect = min_rotated_rect.buffer(pad)
        return min_rotated_rect

    def _get_unit_label_polygons(
        self,
        unit_label,
        rfid_pings,
        labels,
        body_inds,
        video_timestamp,
        polygon_method,
        fps,
        pad,
    ):
        """Get all instacne polygons for a specific unit label RFID reciever."""
        # Method mapping
        methods = {
            "convex_hull": self._get_hull_polygons,
            "bounding_box": self._get_bounding_box_polygons,
            "ellipse": self._get_ellipse_polygons,
        }
        polygon_func = methods.get(polygon_method, self._get_hull_polygons)
        # Filter DataFrame for the specified unit label
        unit_label_df = rfid_pings[rfid_pings["unitLabel"] == unit_label]

        # Check if the DataFrame for the unit label is empty
        if unit_label_df.empty:
            # print(f"No data found for unit label {unit_label}.")
            return []

        # List to hold polygons for return
        polygons = []

        # Iterate through each row in the filtered DataFrame
        for index, row in unit_label_df.iterrows():
            start_frame = int(row["frame_number"])
            duration = int(row["eventDuration"])
            end_frame = int(start_frame + ((duration / 1000) * fps))

            for frame in range(start_frame, end_frame + 1):
                # Check if frame is in labels for the specified video
                if (
                    row["video_start_DateTime"] != video_timestamp
                ):  # Skip frames not labeled in `labels`
                    continue

                # Get the labeled instance for the current frame
                # lf = labels[(labels.video, frame)]
                lf = labels.find(video=labels.video, frame_idx=frame, return_new=True)[
                    0
                ]
                # print(f"Frame {frame} found in labels with {len(lf)} instances.")

                # Ensure we have an instance to process
                for inst in lf:
                    # Retrieve the points for this instance and compute the convex hull
                    polygon = polygon_func(inst, body_inds, pad)
                    if polygon is not None:
                        polygons.append(polygon)

        return polygons

    def _rasterize_polygon(self, polygon, image_width, image_height, bin_size):
        """Rasterize a polygon into a binary mask."""
        XX, YY = np.meshgrid(
            np.arange(0, image_width, bin_size), np.arange(0, image_height, bin_size)
        )
        shapely.prepare(polygon)
        BW = shapely.contains_xy(polygon, XX, YY)
        return BW

    def generate_heatmaps(
        self,
        rfid_pings_path,
        slp_paths,
        video_paths,
        output_path="rfid_heatmaps.h5",
        body_nodes=None,
        camera_filter=None,
        video_number_filter=None,
        bin_size=1,
        polygon_method="convex_hull",
        fps=5,
        pad=0,
    ):
        """Generate RFID heatmaps for each unit and save to H5 file.

        Args:
        rfid_pings_path : str or Path
            Path to the CSV file containing RFID ping data
        slp_paths : list of str or Path
            List of paths to SLEAP (.slp) files
        video_paths : list of str or Path
            List of paths to video files (must align with slp_paths)
        output_path : str or Path, optional
            Path for the output H5 file (default: "rfid_heatmaps.h5")
        body_nodes : list, optional
            List of body node names to use for polygon generation.
            If None, uses ['Nose', 'Head', 'Upper_back', 'Lower_back', 'Tailbase ']
        camera_filter : str, optional
            Camera name to filter RFID pings (default: "Oryx")
        video_number_filter : int, optional
            Video number to filter RFID pings. If None, uses all videos
        bin_size : int, optional
            Size of bins for rasterization (default: 1)
        polygon_method : str, optional
            Method to use for polygon generation (default: "convex_hull")

        Returns:
        tuple:
            (plots_by_unit, unique_units) where plots_by_unit is a list of heatmap arrays
            and unique_units is an array of unit labels
        """
        # Convert paths to Path objects
        rfid_pings_path = Path(rfid_pings_path)
        slp_paths = [Path(p) for p in slp_paths]
        video_paths = [Path(p) for p in video_paths]
        output_path = Path(output_path)

        # Validate that slp_paths and video_paths have the same length
        if len(slp_paths) != len(video_paths):
            raise ValueError("slp_paths and video_paths must have the same length")

        # Load RFID pings data
        print("Loading RFID pings data...")
        rfid_pings = pd.read_csv(rfid_pings_path)

        # Filter RFID pings based on camera and video number
        if camera_filter:
            rfid_pings = rfid_pings[rfid_pings["Camera"] == camera_filter]

        if video_number_filter is not None:
            rfid_pings = rfid_pings[rfid_pings["video_number"] == video_number_filter]

        # Get unique units
        unique_units = rfid_pings["unitLabel"].unique()
        print(f"Found {len(unique_units)} unique RFID units: {unique_units}")

        # Set default body nodes if not provided
        if body_nodes is None:
            body_nodes = ["Nose", "Head", "Upper_back", "Lower_back", "Tailbase "]

        # Create a list to store matching pairs
        file_pairs = []

        for slp_path, video_path in zip(slp_paths, video_paths):
            if slp_path.exists() and video_path.exists():
                file_pairs.append({"slp": slp_path, "video": video_path})
            else:
                print(
                    f"Warning: Skipping pair - slp exists: {slp_path.exists()}, video exists: {video_path.exists()}"
                )

        print(f"Found {len(file_pairs)} valid slp-video pairs")

        if len(file_pairs) == 0:
            raise ValueError("No valid slp-video pairs found. Check your file paths.")

        # Load first slp file to get skeleton information
        import logging

        logging.getLogger("cv2").setLevel(logging.ERROR)
        first_slp = sio.load_file(file_pairs[0]["slp"])
        first_slp.replace_filenames(
            prefix_map={
                str(Path(first_slp.videos[0].backend_metadata["filename"]).parent): str(
                    Path(file_pairs[0]["video"]).parent
                )
            }
        )

        # Get body indices from skeleton
        try:
            body_inds = [first_slp.skeleton.index(node) for node in body_nodes]
        except ValueError as e:
            print(
                f"Error: Could not find all body nodes in skeleton. Available nodes: {first_slp.skeleton.node_names}"
            )
            raise e

        print(f"Using body nodes: {body_nodes} (indices: {body_inds})")

        # Generate heatmaps for each unit
        plots_by_unit = []

        for rfid_unit_label in tqdm.tqdm(unique_units):
            curr_plots = []

            for pair in tqdm.tqdm(file_pairs):

                slp = sio.load_file(pair["slp"])
                slp.replace_filenames(
                    prefix_map={
                        str(
                            Path(slp.videos[0].backend_metadata["filename"]).parent
                        ): str(Path(pair["video"]).parent)
                    }
                )
                video = slp.videos[0]
                video_timestamp = (
                    slp.videos[0]
                    .backend_metadata["filename"]
                    .split("/")[-1]
                    .replace("Oryx_chunked", "")
                    .replace(".avi", "")
                )
                video_height, video_width = video.shape[1], video.shape[2]

                polygons = self._get_unit_label_polygons(
                    rfid_unit_label,
                    rfid_pings,
                    slp,
                    body_inds,
                    video_timestamp,
                    polygon_method,
                    fps,
                    pad,
                )
                curr_plots.extend(polygons)

            if len(curr_plots) == 0:
                print(f"Warning: No polygons found for unit {rfid_unit_label}")
                # Create empty heatmap with default dimensions
                video_height, video_width = (
                    video.shape[1],
                    video.shape[2],
                )  # Default dimensions
                freq = np.zeros((video_height, video_width))
            else:
                # Get video dimensions from first valid polygon
                for pair in file_pairs:
                    try:
                        slp = sio.load_file(pair["slp"])
                        video = slp.videos[0]
                        video_height, video_width = video.shape[1], video.shape[2]
                        break
                    except:
                        continue

                # Generate heatmap
                count = np.zeros((video_height, video_width))
                for polygon in curr_plots:
                    BW_poly = self._rasterize_polygon(
                        polygon,
                        image_width=video_width,
                        image_height=video_height,
                        bin_size=bin_size,
                    )
                    count += BW_poly.astype(float)

                freq = count / len(curr_plots)

            plots_by_unit.append(freq)

        self.heatmaps = plots_by_unit
        self.heatmaps_path = output_path

        # Save to H5 file
        print(f"Saving heatmaps to {output_path}...")
        import gc

        gc.collect()
        with h5py.File(output_path, "w") as f:
            f.create_dataset("plots_by_unit", data=np.array(plots_by_unit))
            # Convert string array to fixed-length bytes for HDF5 compatibility
            unit_names = np.array([unit.encode("utf-8") for unit in unique_units])
            f.create_dataset("unique_units", data=unit_names)

            # Add metadata
            f.attrs["body_nodes"] = str(body_nodes)
            f.attrs["camera_filter"] = (
                camera_filter if camera_filter is not None else ""
            )
            f.attrs["video_number_filter"] = (
                video_number_filter if video_number_filter is not None else -1
            )
            f.attrs["bin_size"] = bin_size
            f.attrs["video_height"] = video_height
            f.attrs["video_width"] = video_width
            f.attrs["num_units"] = len(unique_units)
            f.attrs["num_files"] = len(file_pairs)

        print(f"Successfully generated heatmaps for {len(unique_units)} units")
        print(f"Output saved to: {output_path}")

        return plots_by_unit, unique_units

    def assign_rfid_to_tracklet(
        self,
        rfid_ping,
        unique_units,
        plots_by_unit,
        labels,
        tracklet_id_pairs,
        window_size,
    ):
        """Assign RFID tags to tracklets based on spatial heatmaps and RFID ping data."""
        unit_label = rfid_ping["unitLabel"]

        if unit_label in unique_units:
            # Find index of this unit's heatmap in plots_by_unit
            unit_idx = np.where(unique_units == unit_label)[0][0]
            heatmap = plots_by_unit[unit_idx]

            # Get frame number for this ping
            frame_num = rfid_ping["frame_number"]
            # Get frame range centered on ping, bounded by video limits
            # Divide window_size into two parts: before and after
            before_window = window_size // 2
            after_window = window_size - before_window
            frame_range = range(
                max(0, frame_num - before_window),
                min(frame_num + after_window, len(labels)),
            )

            # Store probabilities for each instance in each frame
            inst_probs = []

            # Iterate through frame range
            for frame_idx in frame_range:
                instances = labels[frame_idx].instances

                # Calculate probability for each pose in this frame
                for pose_idx, pose in enumerate(instances):
                    # Get center coordinates
                    centroid = get_centroid(pose)
                    center_x = int(centroid[0])
                    center_y = int(centroid[1])

                    # Get probability from heatmap at this location
                    if (
                        0 <= center_y < heatmap.shape[0]
                        and 0 <= center_x < heatmap.shape[1]
                    ):
                        prob = heatmap[center_y, center_x]
                        if prob > 0:
                            inst_probs.append((frame_idx, pose_idx, prob))

            best_tracklet_ind = None
            best_prob = 0
            for prob in inst_probs:
                for ind, (track, track_id_list) in enumerate(tracklet_id_pairs):
                    if any((fi[0] == prob[0] and fi[1] == prob[1]) for fi in track):
                        if best_tracklet_ind == ind:
                            best_prob += prob[2]
                            continue

                        if prob[2] > best_prob:
                            best_tracklet_ind = ind
                            best_prob = prob[2]

            if best_tracklet_ind is not None:
                print(
                    f"Found match for RFID {rfid_ping['IdRFID'], int(frame_num)}: {tracklet_id_pairs[best_tracklet_ind][0]}"
                )

                tracklet_id_pairs[best_tracklet_ind][1].append(rfid_ping["IdRFID"])
            else:
                print(f"No match found for RFID {rfid_ping['IdRFID'], int(frame_num)}")

        return tracklet_id_pairs

    def track(
        self,
        labels: sio.Labels,
        video_path: str,
        output_path: str,
        rfid_pings_path: str,
        method: str = "bbox",
        iou_thresh: float = 0.1,
        dist_thresh: int = 30,
        KDE_samples: int = 100000,
        window_size: int = 5,
        percentile: int = 95,
    ):
        """Track instances across frames using either bbox or motion-based tracking.

        Args:
            labels: SLEAP labels object containing instances to track
            video_path: Path to the video file
            output_path: Path to save tracking results
            rfid_pings_path: Path to RFID ping data
            method: Tracking method to use - either "bbox" or "motion" (default: "bbox")
            iou_thresh (float, optional): IOU threshold for tracklet extraction. If an instance is within this threshold across two frames a tracklet is created. Lower values create more strict tracklets. Used for bounding box method only.Defaults to 0.3.


            dist_thresh (float, optional): Distance threshold for tracklet extraction. If the distance between two instances in the same frame is less than this value, NO tracklet is created. Used for bounding box method only. Defaults to 30.
            KDE_samples (int, optional): Number of samples for KDE estimation. Used for motion model method only. Defaults to 100000.
            window_size (int, optional): Frame window to look for a matching instance to an RFID ping. Defaults to 5.
            percentile (int, optional): Percentile for KDE thresholding. Used for motion model method only. Defaults to 95.
        """
        with h5py.File(self.heatmaps_path, "r") as f:

            plots_by_unit = list(f["plots_by_unit"])
            # Convert bytes back to strings
            unique_units = np.array(
                [name.decode("utf-8") for name in f["unique_units"]]
            )

        labels = self.load_and_preprocess_labels(labels, video_path)
        trx, track_names, iou_per_pose = self.extract_tracking_data(labels)

        tracklets = self.get_tracklets(
            labels,
            method,
            trx,
            iou_per_pose,
            iou_thresh,
            dist_thresh,
            KDE_samples,
            percentile,
        )

        rfid_pings = pd.read_csv(rfid_pings_path)
        tracklet_id_pairs = [(tracklet, []) for tracklet in tracklets]

        for row in rfid_pings.iterrows():
            rfid_ping = rfid_pings.loc[row[0]]
            tracklet_id_pairs = self.assign_rfid_to_tracklet(
                rfid_ping,
                unique_units,
                plots_by_unit,
                labels,
                tracklet_id_pairs,
                window_size,
            )

        self.assign_track_ids(tracklet_id_pairs, labels)

        sio.save_file(labels, output_path)

        return labels


class FurColorFeatureTracker(FeatureTracker):
    """Feature tracker for fur color-based tracking.

    This class implements a feature tracker that uses fur color data to track
    animals in videos. It generates heatmaps from fur color data and uses them
    to assign tracks to animals.
    """

    def __init__(self):
        """Initialize the FurColorFeatureTracker."""
        pass

    def get_patches(
        self, lf: sio.LabeledFrame, patch_size: int, pose_percentage_threshold: float
    ):
        """Get patches from a labeled frame."""
        xv = np.arange(-patch_size // 2 + 1, patch_size // 2 + 1)
        yv = np.arange(-patch_size // 2 + 1, patch_size // 2 + 1)
        grid = np.stack(np.meshgrid(xv, yv), axis=-1)

        poses = lf.numpy()
        for pose in poses:
            mask = np.isnan(pose).any(axis=-1)
            missing_nodes_count = np.sum(mask)
            total_nodes = pose.shape[0]

            if (missing_nodes_count / total_nodes) >= pose_percentage_threshold:
                patches = np.full(
                    (poses.shape[0], poses.shape[1], patch_size, patch_size, 1), -1
                )
                return patches, mask

        mask = np.isnan(poses).any(axis=-1)

        centers = np.round(poses)
        centers = np.where(np.isnan(centers), 0, centers)

        img = lf.image

        patch_inds = centers.reshape(centers.shape[0], centers.shape[1], 1, 1, 2) + grid
        patch_inds[..., 1] = np.clip(patch_inds[..., 1], 0, img.shape[0] - 1)
        patch_inds[..., 0] = np.clip(patch_inds[..., 0], 0, img.shape[1] - 1)
        patch_inds = patch_inds.astype(int)

        patches = img[patch_inds[..., 1], patch_inds[..., 0]]

        patches = np.where(
            mask.reshape(poses.shape[0], poses.shape[1], 1, 1, 1), -1, patches
        )

        valid_pixels = patches[patches != -1]
        avg_pixel_value = int(np.mean(valid_pixels))

        patches = np.where(patches == -1, avg_pixel_value, patches)

        return patches, mask

    def get_binned_freqs(self, patches, n_bins, mask=None):
        """Get binned pixel frequencies from patches."""
        bins = np.arange(0, 255, 256 // n_bins)[1:]

        binned = np.digitize(patches, bins)
        binned = binned.reshape(binned.shape[0] * binned.shape[1], -1)
        count_fn = partial(np.bincount, minlength=n_bins)

        counts = np.apply_along_axis(count_fn, axis=1, arr=binned)
        freqs = counts / counts.sum(axis=1, keepdims=True)
        freqs = freqs.reshape(patches.shape[0], patches.shape[1], n_bins)

        if mask is not None:
            freqs = np.where(mask.reshape(*mask.shape, 1), 0, freqs)
        return freqs

    def feature_extraction(
        self,
        labels,
        n_tracks,
        n_frames,
        n_bins,
        patch_size,
        pose_percentage_threshold,
        output_features_path=None,
    ):
        """Extract features from labels."""
        skeleton_length = len(labels.skeleton)
        # Initialize arrays to store patches and frequencies
        patches = np.full(
            (n_frames, n_tracks, skeleton_length, patch_size, patch_size, 1), -1
        )
        freqs = np.full((n_frames, n_tracks, skeleton_length, n_bins), 0)

        for lf_idx, lf in enumerate(labels):
            if len(lf.instances) > 0:
                patches_, mask = self.get_patches(
                    lf, patch_size, pose_percentage_threshold
                )
                freqs_ = self.get_binned_freqs(patches_, n_bins, mask)
            else:
                # If no instances, fill with -1 and 0 for all tracks
                patches_ = np.full(
                    (n_tracks, skeleton_length, patch_size, patch_size, 1), -1
                )
                freqs_ = np.zeros((n_tracks, skeleton_length, n_bins))

            # If only one instance, but patches_ and freqs_ are shape (1, ...), pad to (n_tracks, ...)
            if patches_.shape[0] < n_tracks:
                padded_patches = np.full(
                    (n_tracks, skeleton_length, patch_size, patch_size, 1), -1
                )
                padded_freqs = np.zeros((n_tracks, skeleton_length, n_bins))
                padded_patches[: patches_.shape[0]] = patches_
                padded_freqs[: freqs_.shape[0]] = freqs_
                patches_ = padded_patches
                freqs_ = padded_freqs

            patches[lf_idx] = patches_
            freqs[lf_idx] = freqs_

        if output_features_path is not None:
            with h5py.File(output_features_path, "w") as f:
                f.create_dataset("frequencies", data=freqs)

        return freqs

    def track(
        self,
        labels: sio.Labels,
        video_path: str,
        output_path: str,
        method: str = "bbox",
        output_features_path=None,
        iou_thresh: float = 0.3,
        dist_thresh: float = 30,
        KDE_samples: int = 100000,
        n_bins: int = 4,
        patch_size: int = 5,
        pose_percentage_threshold: float = 0.6,
        n_neighbors: int = 5,
        max_instances: int = 2,
        n_components: int = 10,
        percentile: int = 95,
    ):
        """Track instances across frames using either bbox or motion-based tracking."""
        # features = h5py.File(output_features_path, "r")
        # freqs = features["frequencies"]

        labels = self.load_and_preprocess_labels(labels, video_path)
        trx, track_names, iou_per_pose = self.extract_tracking_data(labels)
        n_tracks = len(track_names)
        n_frames = len(labels)

        tracklets = self.get_tracklets(
            labels,
            method,
            trx,
            iou_per_pose,
            iou_thresh,
            dist_thresh,
            KDE_samples,
            percentile,
        )

        freqs = self.feature_extraction(
            labels,
            n_tracks,
            n_frames,
            n_bins,
            patch_size,
            pose_percentage_threshold,
            output_features_path,
        )

        confidence_vector = self.get_confidence_vector(tracklets, n_frames, n_tracks)

        G, X, Z = self.run_pca(freqs, confidence_vector, max_instances, n_tracks)
        G_mapped = self.run_knn(
            X, G, Z, confidence_vector, max_instances, n_neighbors, n_components
        )

        tracklet_id_pairs = self.get_tracklet_id_pairs_from_knn(
            tracklets, G_mapped, track_names
        )

        self.assign_track_ids(tracklet_id_pairs, labels)

        sio.save_file(labels, output_path)

        return labels


class TailTattooFeatureTracker(FeatureTracker):
    """Feature tracker for tail tattoo-based tracking.

    This class implements a feature tracker that uses tail tattoo data to track
    animals in videos. It generates heatmaps from tail tattoo data and uses them
    to assign tracks to animals.
    """

    def __init__(self):
        """Initialize the TailTattooFeatureTracker."""

    def adjust_contrast(self, img, contrast=1.5, brightness=0):
        """Adjust contrast and brightness of grayscale image."""
        # Apply contrast adjustment
        # Center image, scale, then shift back to preserve min/max spread
        img_float = img.astype(np.float32)
        mean = np.mean(img_float)
        adjusted = np.clip(
            (img_float - mean) * contrast + mean + brightness, 0, 255
        ).astype(np.uint8)

        # Apply adaptive histogram equalization for better local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(adjusted)

        return enhanced

    def get_tail_segments(
        self, tail_nodes, lf: sio.LabeledFrame, angle_threshold_degrees=50
    ):
        """Extract tail segments from a labeled frame.

        Return None if any angle between consecutive segments exceeds the given angle threshold (in degrees).
        """
        tail_data = []  # {instance: [{start, end, magnitude, direction}]}
        angle_threshold = np.deg2rad(angle_threshold_degrees)

        for pose_idx, instance in enumerate(lf.instances):
            curr_tail = []
            # Find the maximum tail node present
            max_tail_idx = -1
            for node1, node2 in instance.skeleton.edges:
                if node1.name in tail_nodes and node2.name in tail_nodes:
                    max_tail_idx = max(
                        max_tail_idx,
                        tail_nodes.index(node1.name),
                        tail_nodes.index(node2.name),
                    )

            # Collect points for the present tail nodes in order
            present_nodes = []
            for i in range(max_tail_idx + 1):
                node_name = tail_nodes[i]
                # Find the point for this node by name
                idx = next(
                    (
                        j
                        for j, pt in enumerate(instance.points)
                        if pt["name"] == node_name
                    ),
                    None,
                )
                if idx is not None:
                    pt = instance.points[idx]["xy"]
                    present_nodes.append((pt[0], pt[1]))
                else:
                    present_nodes.append((np.nan, np.nan))

            # Check for NaNs in present_nodes, skip if any are missing
            if any(np.isnan(x) or np.isnan(y) for x, y in present_nodes):
                tail_data.append(None)
                continue

            # Compute angles between consecutive segments
            angles = []
            for i in range(len(present_nodes) - 2):
                p0 = np.array(present_nodes[i])
                p1 = np.array(present_nodes[i + 1])
                p2 = np.array(present_nodes[i + 2])
                v1 = p1 - p0
                v2 = p2 - p1
                # Normalize
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 == 0 or norm2 == 0:
                    continue
                v1 /= norm1
                v2 /= norm2
                dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(dot)
                angles.append(angle)
            # If any angle is greater than threshold, return None
            if any(a > angle_threshold for a in angles):
                tail_data.append(None)
                continue

            # Otherwise, collect tail segments
            for i in range(0, max_tail_idx):
                node1_name = tail_nodes[i]
                node2_name = tail_nodes[i + 1]
                idx1 = next(
                    j
                    for j, pt in enumerate(instance.points)
                    if pt["name"] == node1_name
                )
                idx2 = next(
                    j
                    for j, pt in enumerate(instance.points)
                    if pt["name"] == node2_name
                )
                point1 = instance.points[idx1]["xy"]
                point2 = instance.points[idx2]["xy"]

                magnitude = np.sqrt(
                    (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2
                )
                direction = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
                if not np.isnan(magnitude) and not np.isnan(direction):
                    curr_tail.append(
                        {
                            "start": (
                                (point1.x, point1.y)
                                if hasattr(point1, "x")
                                else (point1[0], point1[1])
                            ),
                            "end": (
                                (point2.x, point2.y)
                                if hasattr(point2, "x")
                                else (point2[0], point2[1])
                            ),
                            "magnitude": magnitude,
                            "direction": direction,
                        }
                    )
            tail_data.append(curr_tail)
        return tail_data

    def generate_bounding_boxes(
        self, lf, tail_nodes, width, contrast, brightness, fixed_length
    ):
        """Generate normalized rectangular segments for each tail section.

        Args:
            lf: LabeledFrame containing the image and tracking data
            tail_nodes: List of node names corresponding to the tail
            width: Width of the bounding boxes
            contrast: Contrast adjustment factor
            brightness: Brightness adjustment
            fixed_length: If provided, all segments will be resized to this length
        """
        tail_segments = self.get_tail_segments(tail_nodes, lf)

        img = lf.image
        # Convert image to grayscale if it has 3 or more channels, otherwise use as-is
        if len(img.shape) >= 3 and img.shape[-1] >= 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img

        # Apply contrast adjustment
        img_enhanced = self.adjust_contrast(img_gray, contrast, brightness)

        normalized_segments = []

        for tail in tail_segments:
            curr_segments = []
            if tail is None:
                normalized_segments.append(None)
                continue
            for vec in tail:
                x1, y1 = vec["start"]
                x2, y2 = vec["end"]

                # Skip invalid segments
                if np.isnan([x1, y1, x2, y2]).any():
                    normalized_segments.append(None)
                    continue

                length = vec["magnitude"]
                angle = vec["direction"]

                # Skip zero-length segments
                if length < 1:
                    normalized_segments.append(None)
                    continue

                # Calculate corners of the bounding box
                dx = width / 2 * math.sin(angle)
                dy = width / 2 * -math.cos(angle)

                # original segment corners
                src_points = np.float32(
                    [
                        [x1 - dx, y1 - dy],
                        [x1 + dx, y1 + dy],
                        [x2 + dx, y2 + dy],
                        [x2 - dx, y2 - dy],
                    ]
                )

                # Use fixed length if provided, otherwise use actual length
                height = fixed_length if fixed_length is not None else math.ceil(length)

                # destination points (normalized rectangle)
                dst_points = np.float32(
                    [[0, 0], [0, width], [height, width], [height, 0]]
                )

                # Calculate rotation matrix
                A = cv2.getPerspectiveTransform(src_points, dst_points)
                normalized = cv2.warpPerspective(img_enhanced, A, (height, width))

                # Convert to 3 channels for visualization while keeping it grayscale
                normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)

                curr_segments.append(
                    {
                        "segment": normalized,
                        "start": vec["start"],
                        "end": vec["end"],
                        "direction": vec["direction"],
                        "src_points": src_points,  # Save for visualization
                        "original_length": length,  # Save original length for reference
                    }
                )
            normalized_segments.append(curr_segments)

        return normalized_segments

    def smooth_tail(self, segment, weight_power=2, black_threshold=90, gap_threshold=3):
        """Smooth a tail segment and generate a barcode by applying weighted averaging."""
        segment = np.asarray(segment)
        smoothed = np.zeros_like(segment, dtype=np.float32)

        for col in range(segment.shape[1]):
            column = segment[:, col]
            # Give dark pixels a much heavier weighting: use exponential or quadratic weighting
            # Here, we use quadratic weighting for strong emphasis
            inv = np.max(column) - column + 1e-3
            weights = inv**weight_power
            weights = weights / np.sum(weights)
            weighted_avg = np.sum(column * weights)
            smoothed[:, col] = weighted_avg

        # Stretch values to span the full range [0, 255]
        min_val = np.min(smoothed)
        max_val = np.max(smoothed)
        if max_val > min_val:
            smoothed = (smoothed - min_val) / (max_val - min_val) * 255
        else:
            smoothed = np.zeros_like(smoothed)  # all values are the same

        smoothed[smoothed > black_threshold] = 255

        # Post-process to fill small white gaps with black, and small black gaps with white
        # Since all pixel values in a column are the same, we only need to process the first row of smoothed.
        # We'll process smoothed[0, :] as a 1D array, then broadcast the result to all rows.

        col_data = smoothed[0, :]
        # Binarize: black = 0, white = 255
        binary = (col_data > black_threshold).astype(np.uint8)

        # 1. Fill small white gaps (white runs of length < gap_param) with black
        in_white = False
        start = 0
        for i in range(len(binary)):
            if binary[i] == 1 and not in_white:
                in_white = True
                start = i
            elif binary[i] == 0 and in_white:
                end = i
                if (end - start) < gap_threshold:
                    binary[start:end] = 0
                in_white = False
        # Handle run to end
        if in_white:
            end = len(binary)
            if (end - start) < gap_threshold:
                binary[start:end] = 0

        # 2. Fill small black gaps (black runs of length < gap_param) with white
        in_black = False
        start = 0
        for i in range(len(binary)):
            if binary[i] == 0 and not in_black:
                in_black = True
                start = i
            elif binary[i] == 1 and in_black:
                end = i
                if (end - start) < gap_threshold:
                    binary[start:end] = 1
                in_black = False
        # Handle run to end
        if in_black:
            end = len(binary)
            if (end - start) < gap_threshold:
                binary[start:end] = 1

        # Broadcast the processed binary mask to all rows
        smoothed[:, :] = (binary * 255)[None, :]

        return smoothed

    def collect_tail_features(
        self,
        labels,
        tail_nodes,
        fixed_length=50,
        width=25,
        contrast=4.5,
        brightness=40,
        all_instances=False,
    ):
        """Collect tail segments across multiple frames and convert to feature vectors.

        Each track's tail segments are concatenated horizontally before being flattened.

        Returns:
            features: Array of flattened concatenated tail segments
            frame_ids: List of frame numbers
        """
        features = []
        frame_ids = []
        pose_inds = []

        # for frame_idx in range(7608, 7615):
        for frame_idx in tqdm.tqdm(range(0, len(labels.labeled_frames))):
            lf = labels.labeled_frames[frame_idx]
            segments = self.generate_bounding_boxes(
                lf,
                tail_nodes,
                width=width,
                contrast=contrast,
                brightness=brightness,
                fixed_length=fixed_length,
            )

            if len(segments) == 0:
                continue
            curr_instances = []

            for pose_idx, segment in enumerate(segments):
                if (
                    segment is not None and len(segment) >= len(tail_nodes) - 1
                ):  # Only process tracks with at least 3 segments
                    # Take first 3 segments and concatenate horizontally
                    concat_segments = []
                    for i, seg in enumerate(segment[: len(tail_nodes) - 1]):
                        gray_segment = cv2.cvtColor(seg["segment"], cv2.COLOR_RGB2GRAY)
                        concat_segments.append(gray_segment)

                    # Horizontal concatenation of the 3 segments
                    if len(concat_segments) == len(tail_nodes) - 1:
                        full_tail = np.hstack(concat_segments)
                        smoothed_tail = self.smooth_tail(full_tail)
                        curr_instances.append([smoothed_tail[0], frame_idx, pose_idx])
                elif all_instances:
                    break
            features.extend(inst[0] for inst in curr_instances)
            frame_ids.extend(inst[1] for inst in curr_instances)
            pose_inds.extend(inst[2] for inst in curr_instances)

        return np.array(features), frame_ids, pose_inds

    def analyze_tail_markings(
        self,
        labels,
        tail_nodes,
        n_components=3,
        n_clusters=3,
        fixed_length=50,
        width=25,
        contrast=4.5,
        brightness=40,
        all_instances=False,
    ):
        """Perform PCA and KMeans clustering analysis on tail segments.

        Args:
            labels: SLEAP labels object
            tail_nodes: List of node names corresponding to the tail
            n_components: Number of PCA components to keep
            n_clusters: Number of KMeans clusters
            fixed_length: Length to which tail segments are resampled (px)
            width: Width parameter for tail cropping (px)
            contrast: Contrast factor for adjusting tail segment appearance
            brightness: Brightness factor for adjusting tail segment appearance
            all_instances: If True, requires all instances to be present in a frame to analyze the tail markings of any of them. Defaults to False.
        """
        # Collect features
        features, frame_ids, pose_inds = self.collect_tail_features(
            labels,
            tail_nodes,
            fixed_length=fixed_length,
            width=width,
            contrast=contrast,
            brightness=brightness,
            all_instances=all_instances,
        )

        if len(features) == 0:
            print("No valid tail segments found in the specified frames")
            return

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Run PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(features_scaled)

        # Normalize PCA results to help KMeans find better clusters
        pca_result_scaled = StandardScaler().fit_transform(pca_result)
        # Remove outlier data points using z-score thresholding
        zscores = np.abs(zscore(pca_result_scaled, axis=0))
        # Consider a point an outlier if any of its PC z-scores > 3
        non_outlier_mask = (zscores < 3).all(axis=1)
        features = features[non_outlier_mask]
        features_scaled = features_scaled[non_outlier_mask]
        pca_result = pca_result[non_outlier_mask]
        pca_result_scaled = pca_result_scaled[non_outlier_mask]
        frame_ids = [frame_ids[i] for i, keep in enumerate(non_outlier_mask) if keep]
        pose_inds = [pose_inds[i] for i, keep in enumerate(non_outlier_mask) if keep]

        # Run KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(pca_result_scaled)

        # Create DataFrame for plotting
        df = pd.DataFrame(
            {
                "PC1": pca_result[:, 0],
                "PC2": pca_result[:, 1],
                "PC3": (
                    pca_result[:, 2] if n_components > 2 else np.zeros(len(pca_result))
                ),
                "Pose Ind": pose_inds,
                "Frame": frame_ids,
                "Cluster": [f"Cluster {c}" for c in clusters],
            }
        )
        return df

    def knn(self, results_df, n_neighbors=150):
        """Filter results_df using kNN to keep only points that have all their k neighbors in the same cluster."""
        # Use the PCA columns and cluster labels from results_df for kNN
        pca_cols = [col for col in results_df.columns if col.startswith("PC")]
        X = results_df[pca_cols].values
        cluster_labels = results_df["Cluster"].values

        # Fit NearestNeighbors on the PCA features
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(
            X
        )  # +1 to include the point itself
        distances, indices = nbrs.kneighbors(X)

        # For each point, check if all k neighbors (excluding itself) are in the same cluster
        to_keep = []
        for i, neighbors in enumerate(indices):
            neighbor_idxs = neighbors[1:]  # exclude self
            neighbor_clusters = cluster_labels[neighbor_idxs]
            # Count how many neighbors are in the same cluster as the point
            same_cluster_count = np.sum(neighbor_clusters == cluster_labels[i])
            # Keep the point only if all k neighbors are in the same cluster
            if same_cluster_count == n_neighbors:
                to_keep.append(True)
            else:
                to_keep.append(False)

        filtered_df = results_df.iloc[np.where(to_keep)[0]]
        return filtered_df

    def get_G_mapped(self, filtered_df, num_frames, num_tracks):
        """Map frame and pose indices to cluster labels."""
        G_mapped = pd.DataFrame(np.full((num_frames, num_tracks), -1))

        for row_ind, row in filtered_df.iterrows():
            frame = row["Frame"]
            pose_ind = row["Pose Ind"]
            cluster = int(row["Cluster"].replace("Cluster ", ""))
            G_mapped.loc[frame, pose_ind] = cluster

        G_mapped = np.array(G_mapped)

        # For each frame, if any value appears more than once (ignoring -1), set the whole frame to -1
        for i in range(G_mapped.shape[0]):
            frame_vals = G_mapped[i]
            # Exclude -1 values
            valid_vals = frame_vals[frame_vals != -1]
            if len(valid_vals) > 0:
                unique, counts = np.unique(valid_vals, return_counts=True)
                if np.any(counts > 1):
                    G_mapped[i, :] = -1

        return G_mapped

    def track(
        self,
        labels: sio.Labels,
        video_path: str,
        output_path: str,
        method: str = "bbox",
        tail_nodes: list = ["tti", "t0", "t1", "t2"],
        iou_thresh: float = 0.3,
        dist_thresh: float = 30,
        KDE_samples: int = 100000,
        percentile: int = 95,
        fixed_length: int = 70,
        width: int = 25,
        contrast: float = 4.5,
        brightness: int = 40,
        max_instances: int = None,
        n_neighbors: int = 10,
        n_components: int = 6,
        all_instances: bool = False,
    ):
        """Perform feature-based tracking on SLEAP labels using a combination of tail appearance and spatial information.

        This method processes the input labels and video to extract tracklets, analyze tail markings,
        filter features using k-nearest neighbors, and assign consistent track IDs across frames.

        Args:
            labels (sio.Labels): SLEAP labels object containing pose and instance data.
            video_path (str): Path to the video file associated with the labels.
            output_path (str): Path to save the output results.
            method (str, optional): Method for extracting tracklets. Defaults to "bbox".
            tail_nodes (list, optional): List of node names corresponding to the tail. Defaults to ["tti", "t0", "t1", "t2"].
            iou_thresh (float, optional): IOU threshold for tracklet extraction. If an instance is within this threshold across two frames a tracklet is created. Lower values create more strict tracklets. Used for bounding box method only.Defaults to 0.3.
            dist_thresh (float, optional): Distance threshold for tracklet extraction. If the distance between two instances in the same frame is less than this value, NO tracklet is created. Used for bounding box method only. Defaults to 30.
            KDE_samples (int, optional): Number of samples for KDE estimation. Used for motion model method only. Defaults to 100000.
            percentile (int, optional): Percentile for KDE thresholding. Used for motion model method only. Defaults to 95.
            fixed_length (int, optional): Length to which tail segments are resampled. Defaults to 70.
            width (int, optional): Width parameter for tail cropping. Defaults to 25.
            contrast (float, optional): Contrast factor for adjusting tail segment appearance. Defaults to 4.5.
            brightness (int, optional): Brightness factor for adjusting tail segment appearance. Defaults to 40.
            max_instances (int, optional): Maximum number of instances to track. Defaults to None (all).
            n_neighbors (int, optional): Number of neighbors for kNN filtering. Defaults to 10.
            n_components (int, optional): Number of PCA components for feature analysis. Defaults to 6.
            all_instances (bool, optional): If True, requires all instances to be present in a frame to analyze the tail markings of any of them. Defaults to False.

        Returns:
            None. The function processes the data and assigns track IDs in-place.
        """
        labels = self.load_and_preprocess_labels(labels, video_path)
        trx, track_names, iou_per_pose = self.extract_tracking_data(labels)

        n_tracks = max_instances if max_instances is not None else len(track_names)
        n_frames = len(labels)

        print("extracting tracklets using method: ", method)

        tracklets = self.get_tracklets(
            labels,
            method,
            trx,
            iou_per_pose,
            iou_thresh,
            dist_thresh,
            KDE_samples,
            percentile,
        )

        print("extracting features")

        results_df = self.analyze_tail_markings(
            labels,
            tail_nodes,
            n_components=n_components,
            n_clusters=max_instances,
            fixed_length=fixed_length,
            width=width,
            contrast=contrast,
            brightness=brightness,
            all_instances=all_instances,
        )

        filtered_df = self.knn(results_df, n_neighbors=n_neighbors)
        # INSERT_YOUR_CODE
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        unique_clusters = filtered_df["Cluster"].unique()
        colors = plt.cm.get_cmap("tab10", len(unique_clusters))

        for idx, cluster in enumerate(unique_clusters):
            cluster_df = filtered_df[filtered_df["Cluster"] == cluster]
            plt.scatter(
                cluster_df["PC1"],
                cluster_df["PC2"],
                label=cluster,
                alpha=0.7,
                color=colors(idx),
                edgecolor="k",
                s=40,
            )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Filtered PCA Clusters (after kNN)")
        plt.legend(title="Cluster")
        plt.tight_layout()
        plt.show()
        G_mapped = self.get_G_mapped(filtered_df, n_frames, n_tracks)

        print(G_mapped[0][0])

        print("assigning track ids")

        tracklet_id_pairs = self.get_tracklet_id_pairs_from_knn(
            tracklets, G_mapped, track_names
        )

        self.assign_track_ids(tracklet_id_pairs, labels)

        print("saving labels")

        sio.save_file(labels, output_path)

        return labels
