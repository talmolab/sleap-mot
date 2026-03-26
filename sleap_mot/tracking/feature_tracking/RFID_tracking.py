from sleap_mot.tracking.feature_tracking.base import FeatureTracker
import numpy as np
import sleap_io as sio
import pandas as pd
import h5py
import shapely
import tqdm
from sleap_mot.utils import get_centroid
from pathlib import Path

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
        # Create a DataFrame of shape (num_frames, max_instances)
        num_frames = len(labels)
        probabilities_df = pd.DataFrame(
            data=np.full((num_frames, max_instances), None, dtype=object),
            index=np.arange(num_frames),
            columns=np.arange(max_instances)
        )

        # Iterate through each RFID ping
        for row in rfid_pings.iterrows():
            rfid_ping = rfid_pings.loc[row[0]]
            # FIX: Look up heatmap by IdRFID (animal chip) not unitLabel (receiver)
            rfid_id = rfid_ping[id_column_title]

            if rfid_id in unique_units:
                # Find index of this animal's heatmap in plots_by_unit
                unit_idx = np.where(unique_units == rfid_id)[0][0]
                heatmap = plots_by_unit[unit_idx]

                # Get frame number for this ping
                frame_num = rfid_ping[frame_column_title]
                # Get frame range centered on ping, bounded by video limits
                # Divide window_size into two parts: before and after
                before_window = window_size // 2
                after_window = window_size - before_window
                frame_range = range(
                    max(0, frame_num - before_window),
                    min(frame_num + after_window, len(labels)),
                )

                # Iterate through frame range
                for frame_idx in frame_range:
                    instances = labels.find(frame_idx=frame_idx, video=labels.video, return_new=True)[0].instances

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
                            if probabilities_df.loc[frame_idx, pose_idx] is None:
                                probabilities_df.at[frame_idx, pose_idx] = {rfid_ping[id_column_title]: prob}
                            elif rfid_ping[id_column_title] in probabilities_df.loc[frame_idx, pose_idx]:
                                # Pick the greater probability between prob and the current value of this rfid ping name
                                existing_prob = probabilities_df.loc[frame_idx, pose_idx][rfid_ping[id_column_title]]
                                probabilities_df.loc[frame_idx, pose_idx][rfid_ping[id_column_title]] = max(prob, existing_prob)
                            else:
                                # Add or update the dictionary with new RFID/prob pair
                                probabilities_df.loc[frame_idx, pose_idx][rfid_ping[id_column_title]] = prob
        
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
        self.assign_track_ids(probabilities_df, labels)

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