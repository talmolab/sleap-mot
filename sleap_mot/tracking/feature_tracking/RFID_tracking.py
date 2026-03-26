from sleap_mot.tracking.feature_tracking.base import FeatureTracker
from sleap_mot.tracking.explanations import RFIDExplanationGenerator
from sleap_mot.tracking.instance_explanations import (
    RFIDDecisionRecord,
    CandidateScore,
    DecisionType,
)
from sleap_mot.tracking.base import TrackContext
import numpy as np
import sleap_io as sio
import pandas as pd
import h5py
import shapely
import tqdm
from sleap_mot.utils import get_centroid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

class RFIDFeatureTracker(FeatureTracker):
    """Feature tracker for RFID-based tracking.

    This class implements a feature tracker that uses RFID data to track
    animals in videos. It generates heatmaps from RFID pings and uses them
    to assign tracks to animals.
    """

    def __init__(self, priority: int = 10, name: str = "RFIDFeatureTracker"):
        """Initialize the RFIDFeatureTracker.

        Args:
            priority: Priority level for conflict resolution (higher = more authoritative).
                      RFID tracking typically has high priority (default: 10).
            name: Name of this tracking layer.
        """
        super().__init__(priority=priority, name=name)
        self.heatmaps = None
        self.heatmaps_path = None

        # Use RFID-specific explanation generator
        self._explanation_generator = RFIDExplanationGenerator()

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
        rfid_id,
        rfid_pings,
        labels,
        body_inds,
        video_timestamp,
        polygon_method,
        fps,
        pad,
    ):
        """Get all instance polygons for a specific RFID chip ID (animal).

        Args:
            rfid_id: The RFID chip ID (IdRFID) to filter for - this identifies the animal.
            rfid_pings: DataFrame containing RFID ping data.
            labels: SLEAP Labels object.
            body_inds: Indices of body nodes to use for polygon generation.
            video_timestamp: Video timestamp to match.
            polygon_method: Method to use for polygon generation.
            fps: Frames per second for duration calculation.
            pad: Padding for polygon expansion.
        """
        # Method mapping
        methods = {
            "convex_hull": self._get_hull_polygons,
            "bounding_box": self._get_bounding_box_polygons,
            "ellipse": self._get_ellipse_polygons,
        }
        polygon_func = methods.get(polygon_method, self._get_hull_polygons)
        # Filter DataFrame for the specified RFID ID (animal chip)
        rfid_id_df = rfid_pings[rfid_pings["IdRFID"] == rfid_id]

        # Check if the DataFrame for the RFID ID is empty
        if rfid_id_df.empty:
            # print(f"No data found for RFID ID {rfid_id}.")
            return []

        # List to hold polygons for return
        polygons = []

        # Iterate through each row in the filtered DataFrame
        for index, row in rfid_id_df.iterrows():
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

        # Get unique RFID IDs (animal chip IDs, not receiver locations)
        unique_units = rfid_pings["IdRFID"].unique()
        print(f"Found {len(unique_units)} unique RFID IDs: {unique_units}")

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

    def process_rfid_pings(
        self,
        rfid_pings,
        unique_units,
        plots_by_unit,
        labels,
        window_size,
        max_instances: int,
        frame_column_title="frame_number",
        unit_column_title="unitLabel",
        id_column_title="IdRFID",
    ):
        """Assign RFID tags to tracklets based on spatial heatmaps and RFID ping data.

        Args:
            rfid_pings: DataFrame containing RFID ping data
            unique_units: Array of unique RFID unit labels
            plots_by_unit: List of heatmap arrays for each unit
            labels: SLEAP Labels object
            window_size: Frame window to look for matching instances
            max_instances: Maximum number of instances per frame (determines DataFrame columns)
            frame_column_title: Column name for frame numbers in rfid_pings
            unit_column_title: Column name for unit labels in rfid_pings
            id_column_title: Column name for RFID IDs in rfid_pings

        Returns:
            DataFrame of shape (num_frames, max_instances) where each cell contains
            {rfid_id: probability} dict or None
        """
        # Build frame_idx -> LabeledFrame mapping
        # IMPORTANT: labels[x] returns the x-th labeled frame, NOT the frame at index x
        frame_idx_to_lf = {lf.frame_idx: lf for lf in labels.labeled_frames}
        labeled_frame_indices = sorted(frame_idx_to_lf.keys())
        max_frame_idx = max(labeled_frame_indices) if labeled_frame_indices else 0

        # Create a DataFrame indexed by actual frame indices
        # Use sparse representation - only include labeled frames
        probabilities_df = pd.DataFrame(
            data=np.full((len(labeled_frame_indices), max_instances), None, dtype=object),
            index=labeled_frame_indices,
            columns=np.arange(max_instances)
        )

        # Track no-match situations for broadcasting
        no_match_threshold = 0.0  # Minimum probability to consider as a potential match

        # Iterate through each RFID ping
        for row in rfid_pings.iterrows():
            rfid_ping = rfid_pings.loc[row[0]]
            # FIX: Look up heatmap by IdRFID (animal chip) not unitLabel (receiver)
            rfid_id = rfid_ping[id_column_title]
            rfid_unit = rfid_ping.get(unit_column_title)

            if rfid_id not in unique_units:
                # RFID ID not found in unique_units - broadcast no-match
                frame_num = int(rfid_ping[frame_column_title])
                if self._explanation_store is not None and frame_num in frame_idx_to_lf:
                    self._broadcast_rfid_no_match(
                        labels=labels,
                        frame_idx=frame_num,
                        rfid_id=rfid_id,
                        no_match_reason=f"RFID ID {rfid_id} not found in heatmap database",
                        instance_probabilities={},
                        max_probability_found=0.0,
                        rfid_unit_label=rfid_unit,
                    )
                continue

            # Find index of this animal's heatmap in plots_by_unit
            unit_idx = np.where(unique_units == rfid_id)[0][0]
            heatmap = plots_by_unit[unit_idx]

            # Get frame number for this ping
            frame_num = int(rfid_ping[frame_column_title])
            # Get frame range centered on ping
            # Divide window_size into two parts: before and after
            before_window = window_size // 2
            after_window = window_size - before_window
            frame_range = range(
                max(0, frame_num - before_window),
                frame_num + after_window,  # Don't cap - we check frame existence below
            )

            # Track probabilities per frame for no-match detection
            frame_probabilities = {}

            # Iterate through frame range
            for frame_idx in frame_range:
                # Use the mapping to get labeled frames (skip unlabeled frames)
                lf = frame_idx_to_lf.get(frame_idx)
                if lf is None:
                    continue
                instances = lf.instances
                frame_probabilities[frame_idx] = {}

                # Calculate probability for each pose in this frame
                for pose_idx, pose in enumerate(instances):
                    # Skip if pose_idx exceeds max_instances
                    if pose_idx >= max_instances:
                        continue

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
                        frame_probabilities[frame_idx][pose_idx] = prob

                        if probabilities_df.loc[frame_idx, pose_idx] is None:
                            probabilities_df.at[frame_idx, pose_idx] = {rfid_ping[id_column_title]: prob}
                        elif rfid_ping[id_column_title] in probabilities_df.loc[frame_idx, pose_idx]:
                            # Pick the greater probability between prob and the current value of this rfid ping name
                            existing_prob = probabilities_df.loc[frame_idx, pose_idx][rfid_ping[id_column_title]]
                            probabilities_df.loc[frame_idx, pose_idx][rfid_ping[id_column_title]] = max(prob, existing_prob)
                        else:
                            # Add or update the dictionary with new RFID/prob pair
                            probabilities_df.loc[frame_idx, pose_idx][rfid_ping[id_column_title]] = prob
                    else:
                        frame_probabilities[frame_idx][pose_idx] = 0.0

            # Check if this RFID ping had no viable matches (all zero probability)
            all_probs = [p for frame_probs in frame_probabilities.values() for p in frame_probs.values()]
            max_prob = max(all_probs) if all_probs else 0.0

            if max_prob <= no_match_threshold and self._explanation_store is not None:
                # Broadcast no-match to the center frame
                self._broadcast_rfid_no_match(
                    labels=labels,
                    frame_idx=frame_num,
                    rfid_id=rfid_id,
                    no_match_reason=f"All instances had zero probability for RFID {rfid_id}",
                    instance_probabilities=frame_probabilities.get(frame_num, {}),
                    max_probability_found=max_prob,
                    rfid_unit_label=rfid_unit,
                )

        return probabilities_df
            # best_tracklet_ind = None
            # best_prob = 0
            # for frame_index, pose_index, prob in inst_probs:
            #     for ind, (track, track_id_list) in enumerate(tracklet_id_pairs):
            #         if any((fi[0] == frame_index and fi[1] == pose_index) for fi in track):
            #             if best_tracklet_ind == ind:
            #                 best_prob += prob
            #                 continue

            #             if prob > best_prob:
            #                 best_tracklet_ind = ind
            #                 best_prob = prob

        #     if best_tracklet_ind is not None:
        #         print(
        #             f"Found match for RFID {rfid_ping[id_column_title], int(frame_num)}: {tracklet_id_pairs[best_tracklet_ind][0]}"
        #         )

        #         tracklet_id_pairs[best_tracklet_ind][1].append(rfid_ping[id_column_title])
        #     else:
        #         print(f"No match found for RFID {rfid_ping[id_column_title], int(frame_num)}")

        # return tracklet_id_pairs
        

    def track(
        self,
        labels: sio.Labels,
        max_instances: int = None,
        video_path: str = None,
        output_path: str = None,
        rfid_pings_path: str = None,
        window_size: int = 5,
        clear_existing_tracks: bool = True,
    ):
        """Track instances across frames using RFID-based tracking.

        Args:
            labels: SLEAP labels object containing instances to track
            max_instances: Maximum number of instances per frame. If None,
                calculated automatically as the max instances across all frames.
            video_path: Path to the video file (optional, for path replacement)
            output_path: Path to save tracking results (optional)
            rfid_pings_path: Path to RFID ping data CSV file
            window_size (int, optional): Frame window to look for a matching
                instance to an RFID ping. Defaults to 5.
            clear_existing_tracks: If True, clear all existing track assignments
                before assigning new ones. If False, preserve existing tracks and
                only assign RFID identities to instances with probability data.
                Defaults to True.

        Returns:
            labels: The Labels object with track assignments
        """
        # Calculate max_instances if not provided
        if max_instances is None:
            max_instances = self._calculate_max_instances(labels)
            print(f"Calculated max_instances: {max_instances}")
        if self.heatmaps_path is None:
            raise ValueError(
                "Heatmaps path not set. Call generate_heatmaps() first or set "
                "heatmaps_path manually."
            )

        if rfid_pings_path is None:
            raise ValueError("rfid_pings_path is required")

        # Load heatmaps from H5 file
        print(f"Loading heatmaps from {self.heatmaps_path}...")
        with h5py.File(self.heatmaps_path, "r") as f:
            plots_by_unit = list(f["plots_by_unit"])
            # Convert bytes back to strings
            unique_units = np.array(
                [name.decode("utf-8") for name in f["unique_units"]]
            )
        print(f"Loaded {len(unique_units)} unit heatmaps")

        # Load RFID pings
        print(f"Loading RFID pings from {rfid_pings_path}...")
        rfid_pings = pd.read_csv(rfid_pings_path)
        print(f"Loaded {len(rfid_pings)} RFID pings")

        # Check for existing tracklets
        tracklets = self.get_current_tracklets(labels)
        if tracklets is None:
            print("No existing tracklets found, using window_size=1")
            window_size = 1
            # Disable tracklet-only mode since there are no tracklets
            self.only_apply_to_tracklets = False
        else:
            print(f"Found {len(tracklets)} existing tracklets")

        # Process RFID pings to generate probabilities
        print("Processing RFID pings...")
        probabilities_df = self.process_rfid_pings(
            rfid_pings, unique_units, plots_by_unit, labels, window_size, max_instances
        )

        # Count non-null cells
        non_null_count = probabilities_df.notna().sum().sum()
        print(f"Generated probabilities DataFrame: {probabilities_df.shape}, "
              f"{non_null_count} non-null cells")

        # Assign track IDs based on probabilities
        print("Assigning track IDs...")
        self.assign_track_ids(probabilities_df, labels, clear_existing_tracks=clear_existing_tracks)

        # Count assigned tracks
        assigned_count = sum(
            1 for lf in labels for inst in lf.instances if inst.track is not None
        )
        unique_tracks = set(
            inst.track.name for lf in labels for inst in lf.instances
            if inst.track is not None
        )
        print(f"Assigned {assigned_count} instances to {len(unique_tracks)} unique tracks")

        # Save if output path provided
        if output_path is not None:
            print(f"Saving results to {output_path}...")
            sio.save_file(labels, output_path)
            print("Done!")

        return labels

    def get_config(self) -> Dict[str, Any]:
        """Get RFID feature tracker configuration.

        Returns:
            Dict of configuration parameters for this RFID tracker.
        """
        config = super().get_config()
        config.update({
            "heatmaps_path": str(self.heatmaps_path) if self.heatmaps_path else None,
        })
        return config


class CoordinateRFIDTracker(FeatureTracker):
    """RFID tracker using coordinate-based matching.

    This tracker uses direct coordinate matching between RFID ping locations
    and instance centroids, achieving significantly higher accuracy than
    heatmap-based approaches (~95% vs ~31%).

    The tracker supports two operating modes:
    1. With tracklets: Matches RFID pings to tracklets and propagates the
       assignment to all instances in the tracklet.
    2. Without tracklets: Matches RFID pings to individual instances.

    Identity Switch Detection:
    When `detect_switches=True`, the tracker can detect and repair identity
    switches within tracklets. This is useful when a motion tracker has
    incorrectly swapped the identities of two animals.

    Required RFID CSV columns: frame_number, IdRFID, center.x, center.y

    Example:
        >>> from sleap_mot.tracking.feature_tracking.RFID_tracking import CoordinateRFIDTracker
        >>> tracker = CoordinateRFIDTracker(priority=10)
        >>> result = tracker.track(labels=labels, rfid_pings_path="rfid_pings.csv")

        # With switch detection enabled:
        >>> tracker = CoordinateRFIDTracker(
        ...     priority=10,
        ...     min_votes_for_identity=2,
        ...     switch_confidence_threshold=0.8,
        ...     split_on_switch=True,
        ... )
        >>> result = tracker.track(labels=labels, rfid_pings_path="rfid_pings.csv", detect_switches=True)

    Attributes:
        max_distance_from_rfid: Maximum distance in pixels for a valid match.
        min_votes_for_identity: Minimum votes required on each side of a switch.
        switch_confidence_threshold: Confidence required to confirm a switch.
        split_on_switch: Whether to split tracklets at detected switch points.
        require_partner_validation: Only split/swap if partner tracklet confirms.
    """

    def __init__(
        self,
        priority: int = 10,
        name: str = "CoordinateRFIDTracker",
        max_distance_from_rfid: float = 100.0,
        min_votes_for_identity: int = 2,
        switch_confidence_threshold: float = 0.8,
        split_on_switch: bool = True,
        require_partner_validation: bool = False,
    ):
        """Initialize the CoordinateRFIDTracker.

        Args:
            priority: Priority level for conflict resolution (higher = more authoritative).
                      RFID tracking typically has high priority (default: 10).
            name: Name of this tracking layer.
            max_distance_from_rfid: Maximum distance in pixels between RFID ping
                location and instance centroid for a valid match (default: 100.0).
            min_votes_for_identity: Minimum votes required on each side of a switch
                to consider it valid. Default: 2.
            switch_confidence_threshold: Confidence required to confirm a switch
                (fraction of votes agreeing on each side). Default: 0.8.
            split_on_switch: Whether to split tracklets at detected switch points.
                Default: True.
            require_partner_validation: Only split/swap if partner tracklet confirms
                the switch (cross-validation). Default: False.
        """
        super().__init__(
            priority=priority,
            name=name,
            min_votes_for_identity=min_votes_for_identity,
            switch_confidence_threshold=switch_confidence_threshold,
            split_on_switch=split_on_switch,
            require_partner_validation=require_partner_validation,
        )
        self.max_distance_from_rfid = max_distance_from_rfid
        self._frame_idx_to_lf: Dict[int, sio.LabeledFrame] = {}

    def track(
        self,
        labels: sio.Labels,
        rfid_pings_path: str,
        max_distance_from_rfid: Optional[float] = None,
        window_size: int = 1,
        clear_existing_tracks: bool = True,
        output_path: Optional[str] = None,
        detect_switches: bool = False,
        frame_tolerance: int = 10,
    ) -> sio.Labels:
        """Track instances using coordinate-based RFID matching.

        Args:
            labels: SLEAP Labels object containing instances to track.
            rfid_pings_path: Path to RFID ping data CSV file. Must contain columns:
                frame_number, IdRFID, center.x, center.y
            max_distance_from_rfid: Maximum distance for valid match. If None,
                uses the value from constructor (default: 100.0).
            window_size: Number of frames to search around each RFID ping (default: 1).
            clear_existing_tracks: If True, clear all existing track assignments
                before assigning new ones. If False, preserve existing tracklets
                and only assign RFID identities (default: True).
            output_path: Optional path to save the tracked labels.
            detect_switches: If True, use the new voting-based switch detection
                and repair system. This collects ALL RFID matches (not just closest)
                and analyzes temporal consistency to detect and repair identity
                switches. Default: False (uses legacy behavior).
            frame_tolerance: Max frame difference to consider switches related
                (only used when detect_switches=True). Default: 10.

        Returns:
            Labels object with track assignments based on RFID coordinate matching.
        """
        if max_distance_from_rfid is not None:
            self.max_distance_from_rfid = max_distance_from_rfid

        # Build frame_idx -> LabeledFrame mapping
        # CRITICAL: labels[frame_idx] creates a new object, so we must use labeled_frames directly
        self._frame_idx_to_lf = {lf.frame_idx: lf for lf in labels.labeled_frames}
        self._build_frame_mapping(labels)

        # Load RFID pings
        print(f"Loading RFID pings from {rfid_pings_path}...")
        rfid_pings = pd.read_csv(rfid_pings_path)
        print(f"  Loaded {len(rfid_pings)} pings")

        # Validate required columns
        required_cols = ["frame_number", "IdRFID", "center.x", "center.y"]
        missing_cols = [col for col in required_cols if col not in rfid_pings.columns]
        if missing_cols:
            raise ValueError(
                f"RFID pings CSV missing required columns: {missing_cols}. "
                f"Required columns: {required_cols}"
            )

        # Get existing tracklets
        tracklets = self._get_tracklets(labels)
        has_tracklets = len(tracklets) > 0
        print(f"  Found {len(tracklets)} existing tracklets")

        if has_tracklets and detect_switches:
            # New mode: Use voting-based switch detection
            print("\n  Using switch detection mode...")
            votes = self.collect_identity_votes(
                labels=labels,
                tracklets=tracklets,
                rfid_pings=rfid_pings,
                window_size=window_size,
            )
            print(f"  Collected votes for {len(votes)} tracklets")

            # Use the new assign_identities_to_tracklets with switch detection
            assignments = self.assign_identities_to_tracklets(
                labels=labels,
                tracklets=tracklets,
                votes=votes,
                frame_tolerance=frame_tolerance,
            )
            print(f"  Assigned identities to {len(assignments)} tracklets")

        elif has_tracklets:
            # Legacy mode: Match RFID to tracklets and propagate (closest match only)
            tracklet_matches = self._find_tracklet_matches(
                labels, rfid_pings, tracklets, window_size
            )
            self._assign_rfid_to_tracklets(labels, tracklet_matches, tracklets)
        else:
            # Mode 2: Match RFID to individual instances
            instance_matches = self._find_instance_matches(labels, rfid_pings, window_size)
            self._assign_rfid_to_instances(labels, instance_matches)

        # Count results
        tracked_count = sum(
            1 for lf in labels for inst in lf.instances if inst.track is not None
        )
        unique_tracks = set(
            inst.track.name
            for lf in labels
            for inst in lf.instances
            if inst.track is not None
        )
        rfid_tracks = [t for t in unique_tracks if not t.startswith("tracklet")]
        tracklet_tracks = [t for t in unique_tracks if t.startswith("tracklet")]

        print(f"\nResults:")
        print(f"  Total tracked instances: {tracked_count}")
        print(f"  RFID tracks assigned: {len(rfid_tracks)}")
        print(f"  Tracklets preserved: {len(tracklet_tracks)}")

        # Save if output path provided
        if output_path is not None:
            print(f"Saving results to {output_path}...")
            sio.save_file(labels, output_path)
            print("Done!")

        return labels

    def _get_tracklets(self, labels: sio.Labels) -> Dict[str, List[Tuple[int, int]]]:
        """Get dictionary of tracklet_name -> [(frame_idx, instance_idx), ...].

        Args:
            labels: SLEAP Labels object.

        Returns:
            Dict mapping tracklet names (starting with "tracklet") to lists of
            (frame_idx, instance_idx) tuples.
        """
        tracklets: Dict[str, List[Tuple[int, int]]] = {}
        for lf in labels:
            for i, inst in enumerate(lf.instances):
                if inst.track and inst.track.name.startswith("tracklet"):
                    name = inst.track.name
                    if name not in tracklets:
                        tracklets[name] = []
                    tracklets[name].append((lf.frame_idx, i))
        return tracklets

    def collect_identity_votes(
        self,
        labels: sio.Labels,
        tracklets: Dict[str, List[Tuple[int, int]]],
        rfid_pings: pd.DataFrame,
        window_size: int = 5,
    ) -> Dict[str, List["IdentityVote"]]:
        """Collect RFID identity votes for each tracklet using closest-only matching.

        For each RFID ping and each frame in the search window:
        1. Find ALL instances and their distances to the ping location
        2. Only the CLOSEST instance gets a vote (not all within threshold)
        3. The closest instance must still be within max_distance_from_rfid
        4. Vote confidence = 1 - (distance / max_distance_from_rfid)

        IMPORTANT: Only the closest instance per frame gets a vote. This prevents
        incorrect votes when multiple animals are near the same RFID antenna,
        reducing false positive rate from ~17% to ~8%.

        Args:
            labels: SLEAP Labels object.
            tracklets: Dict mapping tracklet_name -> [(frame_idx, instance_idx), ...].
            rfid_pings: DataFrame with columns: frame_number, IdRFID, center.x, center.y.
            window_size: Frames to search around each ping.

        Returns:
            Dict mapping tracklet_name -> list of IdentityVote.
        """
        from sleap_mot.tracking.instance_explanations import IdentityVote

        votes: Dict[str, List[IdentityVote]] = {name: [] for name in tracklets}

        # Build reverse lookup: (frame_idx, instance_idx) -> tracklet_name
        instance_to_tracklet: Dict[Tuple[int, int], str] = {}
        for tracklet_name, frame_instances in tracklets.items():
            for frame_idx, inst_idx in frame_instances:
                instance_to_tracklet[(frame_idx, inst_idx)] = tracklet_name

        print("\n  Collecting identity votes (closest-only mode)...")
        total_votes = 0
        skipped_too_far = 0

        for _, ping in rfid_pings.iterrows():
            frame_num = int(ping["frame_number"])
            rfid_id = str(ping["IdRFID"])
            ping_x = ping["center.x"]
            ping_y = ping["center.y"]

            half_window = window_size // 2
            for frame_idx in range(
                max(0, frame_num - half_window),
                frame_num + half_window + 1,
            ):
                lf = self._frame_idx_to_lf.get(frame_idx)
                if lf is None:
                    continue

                # Find ALL instances and their distances
                candidates = []
                for inst_idx, inst in enumerate(lf.instances):
                    tracklet_name = instance_to_tracklet.get((frame_idx, inst_idx))
                    if tracklet_name is None:
                        continue

                    centroid = get_centroid(inst)
                    if centroid is None or np.isnan(centroid).any():
                        continue

                    distance = np.sqrt(
                        (centroid[0] - ping_x) ** 2 + (centroid[1] - ping_y) ** 2
                    )

                    candidates.append({
                        'inst_idx': inst_idx,
                        'tracklet_name': tracklet_name,
                        'distance': distance,
                        'centroid': centroid,
                    })

                if not candidates:
                    continue

                # Only the CLOSEST instance gets a vote
                closest = min(candidates, key=lambda x: x['distance'])

                if closest['distance'] < self.max_distance_from_rfid:
                    confidence = 1.0 - (closest['distance'] / self.max_distance_from_rfid)
                    vote = IdentityVote(
                        frame_idx=frame_idx,
                        instance_idx=closest['inst_idx'],
                        identity=rfid_id,
                        confidence=confidence,
                        source_info={
                            "distance": float(closest['distance']),
                            "ping_frame": frame_num,
                            "ping_x": float(ping_x),
                            "ping_y": float(ping_y),
                            "num_candidates": len(candidates),
                        },
                    )
                    votes[closest['tracklet_name']].append(vote)
                    total_votes += 1
                else:
                    skipped_too_far += 1

        print(f"  Collected {total_votes} votes (closest-only)")
        print(f"  Skipped {skipped_too_far} frames (closest instance too far)")

        # Log vote distribution
        votes_per_tracklet = {name: len(v) for name, v in votes.items() if v}
        if votes_per_tracklet:
            avg_votes = sum(votes_per_tracklet.values()) / len(votes_per_tracklet)
            print(f"  Tracklets with votes: {len(votes_per_tracklet)}, avg votes: {avg_votes:.1f}")

        return votes

    def _find_tracklet_matches(
        self,
        labels: sio.Labels,
        rfid_pings: pd.DataFrame,
        tracklets: Dict[str, List[Tuple[int, int]]],
        window_size: int,
    ) -> Dict[str, Tuple[str, float, int, int]]:
        """Find the best RFID match for each tracklet.

        For each RFID ping, searches within a window of frames for the closest
        instance that belongs to a tracklet. Returns the best match per tracklet.

        Args:
            labels: SLEAP Labels object.
            rfid_pings: DataFrame with RFID ping data.
            tracklets: Dict mapping tracklet names to instance locations.
            window_size: Number of frames to search around each ping.

        Returns:
            Dict mapping tracklet_name -> (rfid_id, distance, frame_idx, instance_idx)
        """
        tracklet_matches: Dict[str, Tuple[str, float, int, int]] = {}

        # Build reverse lookup: (frame_idx, instance_idx) -> tracklet_name
        instance_to_tracklet: Dict[Tuple[int, int], str] = {}
        for tracklet_name, frame_instances in tracklets.items():
            for frame_idx, inst_idx in frame_instances:
                instance_to_tracklet[(frame_idx, inst_idx)] = tracklet_name

        print("\nMatching RFID pings to instances...")
        pings_matched = 0
        pings_no_match = 0

        for _, ping in rfid_pings.iterrows():
            frame_num = int(ping["frame_number"])
            rfid_id = ping["IdRFID"]
            ping_x = ping["center.x"]
            ping_y = ping["center.y"]

            best_distance = float("inf")
            best_tracklet = None
            best_frame = None
            best_inst_idx = None

            half_window = window_size // 2
            for frame_idx in range(
                max(0, frame_num - half_window),
                frame_num + half_window + 1,
            ):
                # Use the cached LabeledFrame to avoid creating new objects
                lf = self._frame_idx_to_lf.get(frame_idx)
                if lf is None:
                    continue

                for inst_idx, inst in enumerate(lf.instances):
                    tracklet_name = instance_to_tracklet.get((frame_idx, inst_idx))
                    if tracklet_name is None:
                        continue

                    centroid = get_centroid(inst)
                    if centroid is None or np.isnan(centroid).any():
                        continue

                    distance = np.sqrt(
                        (centroid[0] - ping_x) ** 2 + (centroid[1] - ping_y) ** 2
                    )

                    if distance < best_distance and distance < self.max_distance_from_rfid:
                        best_distance = distance
                        best_tracklet = tracklet_name
                        best_frame = frame_idx
                        best_inst_idx = inst_idx

            if best_tracklet is not None:
                pings_matched += 1
                if best_tracklet not in tracklet_matches:
                    tracklet_matches[best_tracklet] = (
                        rfid_id,
                        best_distance,
                        best_frame,
                        best_inst_idx,
                    )
                else:
                    existing_rfid, existing_dist, _, _ = tracklet_matches[best_tracklet]
                    # Keep the closer match (prefer same RFID if closer)
                    if rfid_id == existing_rfid:
                        if best_distance < existing_dist:
                            tracklet_matches[best_tracklet] = (
                                rfid_id,
                                best_distance,
                                best_frame,
                                best_inst_idx,
                            )
                    elif best_distance < existing_dist:
                        tracklet_matches[best_tracklet] = (
                            rfid_id,
                            best_distance,
                            best_frame,
                            best_inst_idx,
                        )
            else:
                pings_no_match += 1

        print(f"  Pings matched to tracklets: {pings_matched}")
        print(f"  Pings with no match (>{self.max_distance_from_rfid}px): {pings_no_match}")
        print(f"  Tracklets with RFID match: {len(tracklet_matches)}")

        return tracklet_matches

    def _find_instance_matches(
        self,
        labels: sio.Labels,
        rfid_pings: pd.DataFrame,
        window_size: int,
    ) -> Dict[Tuple[int, int], Tuple[str, float]]:
        """Find RFID matches for individual instances (no tracklet mode).

        Args:
            labels: SLEAP Labels object.
            rfid_pings: DataFrame with RFID ping data.
            window_size: Number of frames to search around each ping.

        Returns:
            Dict mapping (frame_idx, instance_idx) -> (rfid_id, distance)
        """
        instance_matches: Dict[Tuple[int, int], Tuple[str, float]] = {}

        print("\nMatching RFID pings to individual instances...")
        pings_matched = 0
        pings_no_match = 0

        for _, ping in rfid_pings.iterrows():
            frame_num = int(ping["frame_number"])
            rfid_id = ping["IdRFID"]
            ping_x = ping["center.x"]
            ping_y = ping["center.y"]

            best_distance = float("inf")
            best_frame = None
            best_inst_idx = None

            half_window = window_size // 2
            for frame_idx in range(
                max(0, frame_num - half_window),
                frame_num + half_window + 1,
            ):
                lf = self._frame_idx_to_lf.get(frame_idx)
                if lf is None:
                    continue

                for inst_idx, inst in enumerate(lf.instances):
                    centroid = get_centroid(inst)
                    if centroid is None or np.isnan(centroid).any():
                        continue

                    distance = np.sqrt(
                        (centroid[0] - ping_x) ** 2 + (centroid[1] - ping_y) ** 2
                    )

                    if distance < best_distance and distance < self.max_distance_from_rfid:
                        best_distance = distance
                        best_frame = frame_idx
                        best_inst_idx = inst_idx

            if best_frame is not None and best_inst_idx is not None:
                pings_matched += 1
                key = (best_frame, best_inst_idx)
                if key not in instance_matches:
                    instance_matches[key] = (rfid_id, best_distance)
                else:
                    existing_rfid, existing_dist = instance_matches[key]
                    # Keep closer match
                    if best_distance < existing_dist:
                        instance_matches[key] = (rfid_id, best_distance)
            else:
                pings_no_match += 1

        print(f"  Pings matched: {pings_matched}")
        print(f"  Pings with no match (>{self.max_distance_from_rfid}px): {pings_no_match}")
        print(f"  Instances with RFID match: {len(instance_matches)}")

        return instance_matches

    def _assign_rfid_to_tracklets(
        self,
        labels: sio.Labels,
        tracklet_matches: Dict[str, Tuple[str, float, int, int]],
        tracklets: Dict[str, List[Tuple[int, int]]],
    ) -> None:
        """Assign RFID IDs to tracklets with conflict resolution.

        Groups tracklets by matched RFID, resolves conflicts by preferring
        closer matches and checking for frame overlap.

        Args:
            labels: SLEAP Labels object.
            tracklet_matches: Dict mapping tracklet_name -> (rfid_id, distance, frame_idx, inst_idx)
            tracklets: Dict mapping tracklet names to instance locations.
        """
        # Group by RFID ID
        rfid_to_tracklets: Dict[str, List[Tuple[str, float, int, int]]] = defaultdict(list)
        for tracklet_name, (rfid_id, distance, frame_idx, inst_idx) in tracklet_matches.items():
            rfid_to_tracklets[rfid_id].append((tracklet_name, distance, frame_idx, inst_idx))

        assigned_tracklets = set()
        conflict_losers = set()
        track_cache: Dict[str, sio.Track] = {}

        for rfid_id, candidates in rfid_to_tracklets.items():
            if len(candidates) == 1:
                # No conflict - assign directly
                tracklet_name, distance, frame_idx, inst_idx = candidates[0]
                self._assign_single_tracklet(
                    labels,
                    tracklet_name,
                    tracklets[tracklet_name],
                    rfid_id,
                    distance,
                    frame_idx,
                    inst_idx,
                    track_cache,
                )
                assigned_tracklets.add(tracklet_name)
            else:
                # Multiple tracklets want this RFID - resolve conflicts
                candidates_sorted = sorted(candidates, key=lambda x: x[1])  # Sort by distance

                # Build frame sets for each tracklet
                tracklet_frames = {}
                for tracklet_name, _, _, _ in candidates:
                    tracklet_frames[tracklet_name] = set(
                        f for f, _ in tracklets[tracklet_name]
                    )

                assigned_frames = set()

                for tracklet_name, distance, frame_idx, inst_idx in candidates_sorted:
                    frames = tracklet_frames[tracklet_name]
                    if frames & assigned_frames:
                        # Overlapping frames - this tracklet loses
                        conflict_losers.add(tracklet_name)
                    else:
                        # No overlap - assign
                        self._assign_single_tracklet(
                            labels,
                            tracklet_name,
                            tracklets[tracklet_name],
                            rfid_id,
                            distance,
                            frame_idx,
                            inst_idx,
                            track_cache,
                        )
                        assigned_tracklets.add(tracklet_name)
                        assigned_frames.update(frames)

        total_tracklets = len(tracklets)
        no_match = total_tracklets - len(assigned_tracklets) - len(conflict_losers)

        print(f"\nAssignment results:")
        print(f"  Tracklets assigned RFID: {len(assigned_tracklets)}")
        print(f"  Tracklets lost conflict (preserved): {len(conflict_losers)}")
        print(f"  Tracklets no match (preserved): {no_match}")

    def _assign_rfid_to_instances(
        self,
        labels: sio.Labels,
        instance_matches: Dict[Tuple[int, int], Tuple[str, float]],
    ) -> None:
        """Assign RFID IDs to individual instances (no tracklet mode).

        Args:
            labels: SLEAP Labels object.
            instance_matches: Dict mapping (frame_idx, instance_idx) -> (rfid_id, distance)
        """
        track_cache: Dict[str, sio.Track] = {}

        for (frame_idx, inst_idx), (rfid_id, distance) in instance_matches.items():
            lf = self._frame_idx_to_lf.get(frame_idx)
            if lf is None or inst_idx >= len(lf.instances):
                continue

            inst = lf.instances[inst_idx]
            old_track_name = inst.track.name if inst.track else None

            # Get or create track
            if rfid_id not in track_cache:
                existing = next((t for t in labels.tracks if t.name == rfid_id), None)
                if existing:
                    track_cache[rfid_id] = existing
                else:
                    new_track = sio.Track(name=rfid_id)
                    labels.tracks.append(new_track)
                    track_cache[rfid_id] = new_track

            base_track = track_cache[rfid_id]
            reason = f"RFID coordinate match (dist={distance:.1f}px)"

            track_context = TrackContext(
                priority=self.priority,
                track=base_track,
                name=rfid_id,
                temporary_track=False,
                valid=True,
                track_history=[],
            )

            track_context.add_history_entry(
                layer_name=self.name,
                old_track_name=old_track_name,
                new_track_name=rfid_id,
                frame_idx=frame_idx,
                reason=reason,
                conflict_resolved=False,
                propagated_from_frame=None,
            )

            inst.track = track_context

            # Log to explanation store
            if self._explanation_store is not None:
                centroid = get_centroid(inst)
                centroid_tuple = (
                    tuple(centroid.tolist()) if centroid is not None else None
                )

                candidate_scores = [
                    CandidateScore(
                        candidate_id=rfid_id,
                        score=1.0 - (distance / self.max_distance_from_rfid),
                        passed_thresholds=True,
                        threshold_results={
                            "distance": {
                                "value": distance,
                                "threshold": self.max_distance_from_rfid,
                                "passed": True,
                            }
                        },
                        rejection_reason=None,
                    )
                ]

                record = RFIDDecisionRecord(
                    frame_idx=frame_idx,
                    instance_idx=inst_idx,
                    tracker_name=self.name,
                    tracker_priority=self.priority,
                    decision_type=DecisionType.MATCHED,
                    assigned_track_id=rfid_id,
                    previous_track_id=old_track_name,
                    summary=reason,
                    reasons=[f"Distance to ping: {distance:.1f}px"],
                    rfid_ping_present=True,
                    candidate_rfids=candidate_scores,
                    winning_rfid_id=rfid_id,
                    winning_probability=1.0 - (distance / self.max_distance_from_rfid),
                    is_propagation=False,
                    propagation_source_frame=None,
                    heatmap_probability=None,
                    instance_centroid=centroid_tuple,
                )
                self._explanation_store.add(record)

        print(f"  Assigned RFID to {len(instance_matches)} instances")

    def _assign_single_tracklet(
        self,
        labels: sio.Labels,
        tracklet_name: str,
        frame_instances: List[Tuple[int, int]],
        rfid_id: str,
        distance: float,
        source_frame: int,
        source_inst_idx: int,
        track_cache: Dict[str, sio.Track],
    ) -> None:
        """Assign RFID ID to all instances in a tracklet.

        Args:
            labels: SLEAP Labels object.
            tracklet_name: Original tracklet name.
            frame_instances: List of (frame_idx, instance_idx) tuples in the tracklet.
            rfid_id: RFID ID to assign.
            distance: Distance of the match.
            source_frame: Frame index where the match was found.
            source_inst_idx: Instance index of the match.
            track_cache: Cache of track_name -> sio.Track objects.
        """
        # Get or create track
        if rfid_id not in track_cache:
            existing = next((t for t in labels.tracks if t.name == rfid_id), None)
            if existing:
                track_cache[rfid_id] = existing
            else:
                new_track = sio.Track(name=rfid_id)
                labels.tracks.append(new_track)
                track_cache[rfid_id] = new_track

        base_track = track_cache[rfid_id]

        # Assign to all instances in the tracklet
        for frame_idx, inst_idx in frame_instances:
            # Use the cached LabeledFrame
            lf = self._frame_idx_to_lf.get(frame_idx)
            if lf is None:
                continue
            if inst_idx >= len(lf.instances):
                continue

            inst = lf.instances[inst_idx]

            is_source = frame_idx == source_frame and inst_idx == source_inst_idx
            reason = (
                f"RFID coordinate match (dist={distance:.1f}px)"
                if is_source
                else f"Propagated from frame {source_frame} (dist={distance:.1f}px)"
            )

            # Preserve existing history if available
            existing_history = []
            if isinstance(inst.track, TrackContext):
                existing_history = inst.track.track_history.copy()

            track_context = TrackContext(
                priority=self.priority,
                track=base_track,
                name=rfid_id,
                temporary_track=False,
                valid=True,
                track_history=existing_history,
            )

            track_context.add_history_entry(
                layer_name=self.name,
                old_track_name=tracklet_name,
                new_track_name=rfid_id,
                frame_idx=frame_idx,
                reason=reason,
                conflict_resolved=False,
                propagated_from_frame=None if is_source else source_frame,
            )

            inst.track = track_context

            # Log to explanation store
            if self._explanation_store is not None:
                centroid = get_centroid(inst)
                centroid_tuple = (
                    tuple(centroid.tolist()) if centroid is not None else None
                )

                candidate_scores = [
                    CandidateScore(
                        candidate_id=rfid_id,
                        score=1.0 - (distance / self.max_distance_from_rfid),
                        passed_thresholds=True,
                        threshold_results={
                            "distance": {
                                "value": distance,
                                "threshold": self.max_distance_from_rfid,
                                "passed": True,
                            }
                        },
                        rejection_reason=None,
                    )
                ]

                record = RFIDDecisionRecord(
                    frame_idx=frame_idx,
                    instance_idx=inst_idx,
                    tracker_name=self.name,
                    tracker_priority=self.priority,
                    decision_type=DecisionType.MATCHED if is_source else DecisionType.PROPAGATED,
                    assigned_track_id=rfid_id,
                    previous_track_id=tracklet_name,
                    summary=reason,
                    reasons=[f"Distance to ping: {distance:.1f}px"],
                    rfid_ping_present=is_source,
                    candidate_rfids=candidate_scores,
                    winning_rfid_id=rfid_id,
                    winning_probability=1.0 - (distance / self.max_distance_from_rfid),
                    is_propagation=not is_source,
                    propagation_source_frame=None if is_source else source_frame,
                    heatmap_probability=None,
                    instance_centroid=centroid_tuple,
                )
                self._explanation_store.add(record)

    def get_config(self) -> Dict[str, Any]:
        """Get coordinate RFID tracker configuration.

        Returns:
            Dict of configuration parameters for this coordinate RFID tracker.
        """
        config = super().get_config()
        config.update({
            "max_distance_from_rfid": self.max_distance_from_rfid,
        })
        return config