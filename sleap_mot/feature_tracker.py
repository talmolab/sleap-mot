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
from scipy.stats import kde
import math
from functools import partial
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
        labels.replace_filenames(
            prefix_map={
                Path(labels.videos[0].backend_metadata["filename"])
                .parent: Path(video_path)
                .parent
            }
        )

        # Convert instances to PredictedInstance
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
        columns = ["frame_idx", "track"] + [
            f"{node}.{coord}" for node in node_names for coord in ["x", "y", "score"]
        ]

        data = []
        for frame_data in labels:
            if not frame_data:  # Skip empty frames
                continue

            # Iterate through each instance (track) in the frame
            for inst in frame_data:
                # Initialize a dictionary for this row with 'frame_idx' and 'track'
                row = {"frame_idx": frame_data.frame_idx, "track": inst.track.name}
                for point in inst.points:
                    if (
                        point is not None
                        and type(point) == dict
                        or type(point) == np.void
                    ):
                        row[f"{point['name']}.x"] = point["xy"][0]
                        row[f"{point['name']}.y"] = point["xy"][1]
                        row[f"{point['name']}.score"] = (
                            point["score"] if hasattr(point, "score") else 1
                        )
                    elif point is not None:
                        row[f"{point.name}.x"] = inst.points[point].x
                        row[f"{point.name}.y"] = inst.points[point].y
                        row[f"{point.name}.score"] = inst.points[point].score
                    else:
                        row[f"{point['name']}.x"] = "NaN"
                        row[f"{point['name']}.y"] = "NaN"
                        row[f"{point['name']}.score"] = "NaN"
                data.append(row)

        tracks = pd.DataFrame(data, columns=columns)

        # Convert to array format
        track_names = tracks["track"].unique().tolist()
        n_tracks = len(track_names)
        n_frames = int(tracks["frame_idx"].max() + 1)
        n_nodes = len(node_names)

        trx = np.full((n_frames, n_tracks, n_nodes, 2), np.nan)
        track_index = {name: idx for idx, name in enumerate(track_names)}

        for _, row in tracks.iterrows():
            frame_idx = row["frame_idx"]
            track_name = row["track"]

            # Get current tracks at this frame
            existing_tracks = ~np.all(np.isnan(trx[frame_idx]), axis=(1, 2))
            track_idx = np.sum(existing_tracks)  # Next available index

            # Extract x, y coordinates and scores
            pts = (
                row[2:].to_numpy().reshape(n_nodes, 3)
            )  # Assuming x, y, score for each node
            trx[frame_idx, track_idx] = pts[:, :2]

        iou_per_pose = self.get_iou(trx)

        return trx, track_names, iou_per_pose

    def find_close_poses(self, pose_instance, labels, frame_idx, threshold=15):
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
        threshold: float = 0.1,
    ):
        """Get tracklets based on bounding box overlap."""
        tracklets = []
        curr_tracklet = []
        n_frames = trx.shape[0]
        n_poses = trx.shape[1]

        tracklet_matrix = np.full((n_frames, n_poses), False)

        for global_frame_idx in range(n_frames):
            lf = labels[global_frame_idx]
            for pose_idx, pose in enumerate(lf.instances):
                curr_tracklet = []
                if tracklet_matrix[global_frame_idx, pose_idx]:
                    continue

                curr_frame_idx = global_frame_idx

                while iou_per_pose[curr_frame_idx, pose_idx] <= threshold:
                    curr_tracklet.append((curr_frame_idx, pose_idx))
                    tracklet_matrix[curr_frame_idx, pose_idx] = True
                    if curr_frame_idx == n_frames - 1:
                        break
                    close_poses = self.find_close_poses(
                        labels[curr_frame_idx].instances[pose_idx],
                        labels,
                        curr_frame_idx + 1,
                        threshold=30,
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
        threshold: float = 0.1,
    ):
        """Get tracklets based on motion model."""

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
            labels, frame_idx, curr_tracklet, close_KDE: KDE, far_KDE: KDE
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
                if dist_between_prev >= 80:
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

        long_kde_samples = long_motion_kde.sample(100000)
        x_vals = long_kde_samples[:, 0]
        y_vals = long_kde_samples[:, 1]
        x_min, x_max = np.percentile(x_vals, [5, 95])
        y_min, y_max = np.percentile(y_vals, [5, 95])
        long_kde_stats = KDE(long_motion_kde, (x_min, x_max, y_min, y_max))
        # long_kde_stats = KDE(long_motion_kde, (-40, 40, -175, 25))

        short_kde_samples = short_motion_kde.sample(100000)
        x_vals = short_kde_samples[:, 0]
        y_vals = short_kde_samples[:, 1]
        x_min, x_max = np.percentile(x_vals, [5, 95])
        y_min, y_max = np.percentile(y_vals, [5, 95])
        short_kde_stats = KDE(short_motion_kde, (x_min, x_max, y_min, y_max))
        # short_kde_stats = KDE(short_motion_kde, (-100, 100, -100, 100))

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

                while iou_per_pose[curr_frame_idx, pose_idx] <= 0.1:
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
                        threshold=40,
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
        for tracklet, track_id_list in tracklet_id_pairs:

            if len(set(track_id_list)) > 1:
                # Count occurrences of each ID
                id_counts = Counter(track_id_list)

                # Find the ID(s) with maximum occurrences
                max_count = max(id_counts.values())
                most_common_ids = [
                    id for id, count in id_counts.items() if count == max_count
                ]

                # If there's a single most common ID, use it, otherwise empty the list
                if len(most_common_ids) == 1:
                    track_id_list = [
                        id for id in track_id_list if id == most_common_ids[0]
                    ]
                else:
                    track_id_list = []

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
                                break
                            else:
                                # Equal lengths, clear both
                                set_track_id(other_tracklet, None, labels)
                                set_track_id(tracklet, None, labels)
                                break
                # Check if track with this ID already exists
                existing_track = next(
                    (t for t in labels.tracks if t.name == track_id), None
                )
                track = existing_track if existing_track else sio.Track(name=track_id)
                if not existing_track:
                    labels.tracks.append(track)
                # Label all poses in the tracklet with the same track ID
                set_track_id(tracklet, track, labels)

        unique_tracks = []
        seen_names = set()
        for lf in labels:
            for inst in lf.instances:
                if inst.track is not None:
                    if inst.track.name not in seen_names:
                        unique_tracks.append(inst.track)
                        seen_names.add(inst.track.name)

        # Assign unique tracks back to labels.tracks
        labels.tracks = unique_tracks

    def get_motion_sequences(self, slp_file, video_file):
        """Extract motion sequences from a single video file."""
        idle_count = 0
        motion_count = 0

        labels = sio.load_file(slp_file)
        labels.replace_filenames(
            prefix_map={
                Path(labels.videos[0].backend_metadata["filename"])
                .parent: Path(video_file)
                .parent
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
                if np.linalg.norm(p2 - p1) > 400 or np.linalg.norm(p3 - p2) > 400:
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

    def generate_motion_kde_models(self, slp_folder, video_folder, output_dir="./"):
        """Generate long and short distance KDE motion models from SLEAP tracking data.

        Args:
            slp_folder (str or Path): Path to folder containing .slp files
            video_folder (str or Path): Path to folder containing corresponding video files
            output_dir (str or Path): Directory to save the joblib files

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
        for pair in file_pairs:
            # Load and process the SLP file
            slp_file = pair["slp"]
            video_file = pair["video"]

            # Get motion sequences for this video
            sequences, theta, curr_idle_count, curr_motion_count = (
                self.get_motion_sequences(slp_file, video_file)
            )
            motion_sequences.append(sequences)
            thetas.append(theta)
            idle_count += curr_idle_count
            motion_count += curr_motion_count

        print(f"Found {len(motion_sequences)} total motion sequences across all videos")
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
        long_mask = p1p2_distances > 80
        short_mask = p1p2_distances <= 80

        # Split sequences using boolean masks
        long_sequences = processed_motion_sequences[long_mask]
        short_sequences = processed_motion_sequences[short_mask]

        print(
            f"Long sequences: {len(long_sequences)}, Short sequences: {len(short_sequences)}"
        )

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

    def get_tracklets(self, labels, method, trx, iou_per_pose):
        """Get tracklets from labels using specified method."""
        if method == "bbox":
            tracklets = self.get_bbox_tracklets(labels, trx, iou_per_pose)
        elif method == "motion":
            if self.motion_kde_paths is None:
                raise ValueError(
                    "motion_kde_paths must be generated before using motion tracking"
                )
            long_kde_path, short_kde_path = self.motion_kde_paths
            tracklets = self.get_motion_tracklets(
                labels, long_kde_path, short_kde_path, iou_per_pose
            )
        else:
            raise ValueError("method must be either 'bbox' or 'motion'")

        return tracklets

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

    def _get_hull_polygons(self, inst, body_inds, pad=0):
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

    def _get_bounding_box_polygons(self, inst, body_inds, pad=0):
        """Get bounding box polygons for an instance."""
        pts = inst.numpy()[body_inds]
        if np.isnan(pts).any():
            return None
        pts = shapely.MultiPoint(pts)
        bbox = pts.envelope
        if pad > 0:
            bbox = bbox.buffer(pad)
        return bbox

    def _get_ellipse_polygons(self, inst, body_inds, pad=0):
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
        polygon_method="convex_hull",
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
            end_frame = int(start_frame + ((duration / 1000) * 5))

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
                    polygon = polygon_func(inst, body_inds)
                    if polygon is not None:
                        polygons.append(polygon)

        return polygons

    def _rasterize_polygon(self, polygon, image_width, image_height, bin_size=1):
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
                Path(first_slp.videos[0].backend_metadata["filename"])
                .parent: Path(file_pairs[0]["video"])
                .parent
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

        for rfid_unit_label in unique_units:
            curr_plots = []

            for pair in file_pairs:

                slp = sio.load_file(pair["slp"])
                slp.replace_filenames(
                    prefix_map={
                        Path(slp.videos[0].backend_metadata["filename"])
                        .parent: Path(pair["video"])
                        .parent
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
            f.attrs["camera_filter"] = camera_filter
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
        self, rfid_ping, unique_units, plots_by_unit, labels, tracklet_id_pairs
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
            frame_range = range(max(0, frame_num - 2), min(frame_num + 3, len(labels)))

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
    ):
        """Track instances across frames using either bbox or motion-based tracking.

        Args:
            labels: SLEAP labels object containing instances to track
            video_path: Path to the video file
            output_path: Path to save tracking results
            rfid_pings_path: Path to RFID ping data
            method: Tracking method to use - either "bbox" or "motion" (default: "bbox")
        """
        with h5py.File(self.heatmaps_path, "r") as f:
            plots_by_unit = list(f["plots_by_unit"])
            # Convert bytes back to strings
            unique_units = np.array(
                [name.decode("utf-8") for name in f["unique_units"]]
            )

        labels = self.load_and_preprocess_labels(labels, video_path)
        trx, track_names, iou_per_pose = self.extract_tracking_data(labels)

        tracklets = self.get_tracklets(labels, method, trx, iou_per_pose)

        rfid_pings = pd.read_csv(rfid_pings_path)
        tracklet_id_pairs = [(tracklet, []) for tracklet in tracklets]

        for row in rfid_pings.iterrows():
            rfid_ping = rfid_pings.loc[row[0]]
            tracklet_id_pairs = self.assign_rfid_to_tracklet(
                rfid_ping, unique_units, plots_by_unit, labels, tracklet_id_pairs
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

    def get_patches(self, lf: sio.LabeledFrame, patch_size: int = 5):
        """Get patches from a labeled frame."""
        xv = np.arange(-patch_size // 2 + 1, patch_size // 2 + 1)
        yv = np.arange(-patch_size // 2 + 1, patch_size // 2 + 1)
        grid = np.stack(np.meshgrid(xv, yv), axis=-1)

        poses = lf.numpy()
        for pose in poses:
            mask = np.isnan(pose).any(axis=-1)
            missing_nodes_count = np.sum(mask)
            total_nodes = pose.shape[0]

            if (missing_nodes_count / total_nodes) >= 0.6:
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

    def get_binned_freqs(self, patches, mask=None, n_bins=4):
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

    def feature_extraction(self, labels, n_tracks, n_frames):
        """Extract features from labels."""
        patches = np.full((n_frames, n_tracks, 15, 5, 5, 1), -1)
        freqs = np.full((n_frames, n_tracks, 15, 4), 0)

        for lf_idx, lf in enumerate(labels):
            if len(lf.instances) > 0:
                patches_, mask = self.get_patches(lf)
                freqs_ = self.get_binned_freqs(patches_, mask)
            else:
                # If no instances, fill with -1 and 0 for all tracks
                patches_ = np.full((n_tracks, 15, 5, 5, 1), -1)
                freqs_ = np.zeros((n_tracks, 15, 4))

            # If only one instance, but patches_ and freqs_ are shape (1, ...), pad to (n_tracks, ...)
            if patches_.shape[0] < n_tracks:
                padded_patches = np.full((n_tracks, 15, 5, 5, 1), -1)
                padded_freqs = np.zeros((n_tracks, 15, 4))
                padded_patches[: patches_.shape[0]] = patches_
                padded_freqs[: freqs_.shape[0]] = freqs_
                patches_ = padded_patches
                freqs_ = padded_freqs

            patches[lf_idx] = patches_
            freqs[lf_idx] = freqs_

        return freqs

    def run_pca(self, freqs, confidence_vector):
        """Run PCA on features and confidence vector."""
        freqs = np.array(freqs)
        freqs_reshaped = freqs.reshape(
            freqs.shape[0] * 2, freqs.shape[2] * freqs.shape[3]
        )

        # Flatten confidence vector to align with reshaped features
        conf_flat = confidence_vector.flatten()

        # Select only valid poses using boolean indexing
        X = freqs_reshaped[conf_flat]

        # Fit PCA and transform data
        pcs = PCA()
        pcs = pcs.fit(X)
        Z = pcs.transform(X)

        # Cluster the transformed data
        kmeans = KMeans(n_clusters=2).fit(X)
        G = kmeans.labels_

        return G, X, Z

    def run_knn(self, X, G, Z, freqs, confidence_vector):
        """Run KNN on features and confidence vector."""
        n = int(freqs.shape[0] * 0.005)
        if n < 2:
            n = 2

        knn = NearestNeighbors(n_neighbors=300, radius=0.5, metric="cosine")
        knn.fit(X)
        distances, indices = knn.kneighbors(X)

        nn_G = G[indices]
        is_unambiguous = (nn_G == G.reshape(-1, 1)).all(axis=1)

        kmeans = KMeans(n_clusters=2).fit(Z[:, :10])
        G = kmeans.labels_

        # Set ambiguous points to -1 in G
        G[~is_unambiguous] = -1

        # Create array same size as confidence_vector filled with -1
        G_mapped = np.full(confidence_vector.shape, -1)

        true_positions = np.where(confidence_vector)
        G_mapped[true_positions] = G

        return G_mapped

    def get_tracklet_id_pairs_from_knn(self, tracklets, G_mapped, track_names):
        """Get tracklet id pairs from KNN results."""
        tracklet_id_pairs = []

        for tracklet in tracklets:
            curr_tracks = []
            for frame_idx, pose_idx in tracklet:
                track_idx = G_mapped[frame_idx, pose_idx]
                curr_tracks.append(track_names[track_idx])
            tracklet_id_pairs.append((tracklet, curr_tracks))

        return tracklet_id_pairs

    def get_confidence_vector(self, tracklets, n_frames, n_tracks):
        """Get vector specifying which poses from which frames are included in tracklets."""
        if not tracklets:
            return None, tracklets

        confidence_vector = np.full((n_frames, n_tracks), False)

        for tracklet in tracklets:
            for frame_idx, pose_idx in tracklet:
                confidence_vector[frame_idx, pose_idx] = True

        return confidence_vector

    def track(
        self,
        labels: sio.Labels,
        video_path: str,
        output_path: str,
        method: str = "bbox",
    ):
        """Track instances across frames using either bbox or motion-based tracking."""
        output_features_path = "/root/vast/elise/sleap-mot-elise/notebooks/mov.00001.predictions.features.h5"
        features = h5py.File(output_features_path, "r")
        freqs = features["frequencies"]

        labels = self.load_and_preprocess_labels(labels, video_path)
        trx, track_names, iou_per_pose = self.extract_tracking_data(labels)
        n_tracks = len(track_names)
        n_frames = len(labels)

        tracklets = self.get_tracklets(labels, method, trx, iou_per_pose)

        # freqs = self.feature_extraction(labels, n_tracks, n_frames)
        print("Loaded features")

        confidence_vector = self.get_confidence_vector(tracklets, n_frames, n_tracks)

        G, X, Z = self.run_pca(freqs, confidence_vector)
        G_mapped = self.run_knn(X, G, Z, freqs, confidence_vector)

        tracklet_id_pairs = self.get_tracklet_id_pairs_from_knn(
            tracklets, G_mapped, track_names
        )

        self.assign_track_ids(tracklet_id_pairs, labels)

        sio.save_file(labels, output_path)

        return labels
