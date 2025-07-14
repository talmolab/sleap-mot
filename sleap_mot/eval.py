"""Module for evaluation."""

import numpy as np
import motmetrics as mm
import pandas as pd


def get_df(df, track_key):
    """Get a dataframe from a list of labeled frames.

    Args:
        df (Labels): Labeled frames.
        track_key (str): Key to use for the track ID.

    Returns:
        df (pd.DataFrame): Dataframe with the labeled frames.
    """
    gt_frame_meta_list = []

    # loop through the labeled frames
    for lf in df:
        # in each frame, loop through instances
        for inst in lf:
            # make a dictionary for each instance
            frame_meta = {
                "frame_id": lf.frame_idx,
                track_key: inst.track.name if inst.track is not None else None,
            }
            points = inst.points
            for point in points:
                if type(point) == dict or type(point) == np.void:
                    frame_meta[point["name"]] = (point["xy"][0], point["xy"][1])
                else:
                    node = points[point]
                    frame_meta[point.name] = (node.x, node.y)

            gt_frame_meta_list.append(frame_meta)
    return_df = pd.DataFrame(gt_frame_meta_list)

    # Create a new DataFrame to store the expanded columns
    df_expanded = pd.DataFrame()

    # Copy non-tuple columns as is
    for col in return_df.columns:
        if return_df[col].dtype != "object" or not isinstance(
            return_df[col].iloc[0], tuple
        ):
            df_expanded[col] = return_df[col]

    # Expand tuple columns
    for col in return_df.columns:
        if return_df[col].dtype == "object" and isinstance(
            return_df[col].iloc[0], tuple
        ):
            # Create two new columns with suffixes _x and _y
            df_expanded[f"{col}_x"] = return_df[col].apply(
                lambda x: x[0] if isinstance(x, tuple) else None
            )
            df_expanded[f"{col}_y"] = return_df[col].apply(
                lambda x: x[1] if isinstance(x, tuple) else None
            )

    # Replace the original df_gt with the expanded version
    return_df = df_expanded.fillna(0)
    return return_df


def get_metrics(df_gt_in, df_pred_in, track_dict=None):
    """Get metrics for tracking using a ground truth (proofread) file and a predicted file.

    Args:
        df_gt_in (Labels): Labeled frames from a proofread file.
        df_pred_in (Labels): Labeled frames from a predicted file.

    Returns:
        summary (pd.DataFrame): Summary of the tracking metrics.
        total_mislabeled_frames (int): Total number of mislabeled frames.
        group_lengths (list): Lengths of consecutive mislabeled frames.
    """
    df_gt = get_df(df_gt_in, track_key="gt_track_id")
    df_pred = get_df(df_pred_in, track_key="pred_track_id")

    # Get common columns between gt and pred dataframes
    gt_cols = set(df_gt.columns)
    pred_cols = set(df_pred.columns)
    common_cols = list(gt_cols.intersection(pred_cols))

    # Make sure we have at least one common column to merge on
    if not common_cols:
        raise ValueError(
            "No common columns found between ground truth and predicted dataframes"
        )

    df_merged = pd.merge(
        df_gt,
        df_pred,
        left_on=common_cols,
        right_on=common_cols,
        how="inner",
    )

    # Initialize MOT metrics accumulator and tracking variables
    acc = mm.MOTAccumulator(auto_id=True)
    total_mislabeled_identities = 0
    total_correct_identities = 0
    mislabeled_frames = []
    correct_frames = []

    all_track_ids = set(df_merged["gt_track_id"].unique()) | set(
        df_merged["pred_track_id"].unique()
    )
    track_id_map = {track_id: i for i, track_id in enumerate(all_track_ids)}

    # Process each frame in the merged dataframe, limiting to first 10,000 frames
    for frame, framedf in df_merged.groupby("frame_id"):
        # Get ground truth and predicted track IDs for this frame
        gt_ids = framedf["gt_track_id"].values
        pred_tracks = framedf["pred_track_id"].values

        # Check for any mismatches between ground truth and predictions
        for idx, gt_id in enumerate(gt_ids):
            correct_id = True
            pred_id = pred_tracks[idx]
            if track_dict is not None and pred_id in track_dict:
                pred_tracks[idx] = track_dict[pred_id]
                pred_id = pred_tracks[idx]
            if gt_id != pred_id:
                correct_id = False

            if not correct_id:
                total_mislabeled_identities += 1
                mislabeled_frames.append(frame)
            else:
                total_correct_identities += 1
                correct_frames.append(frame)
        # Create cost matrix for MOT metrics
        # NaN indicates no association, 1 indicates perfect match
        gt_ids = np.array([track_id_map[id] for id in gt_ids])
        pred_tracks = np.array([track_id_map[id] for id in pred_tracks])
        cost_gt_pred = np.full((len(gt_ids), len(pred_tracks)), np.nan)
        np.fill_diagonal(cost_gt_pred, 1)

        # Update MOT accumulator with frame data
        acc.update(
            oids=gt_ids,  # Ground truth object IDs
            hids=pred_tracks,  # Hypothesis (predicted) IDs
            dists=cost_gt_pred,  # Distance/cost matrix
        )

    # Compute MOT metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, name="acc").transpose()

    # Group consecutive mislabeled frames to analyze error patterns
    grouped_mislabeled_frames = []
    mislabeled_group_lengths = []
    if mislabeled_frames:
        current_group = [mislabeled_frames[0]]
        for frame in mislabeled_frames[1:]:
            if frame == current_group[-1] + 1:
                # Add to current group if frames are consecutive
                current_group.append(frame)
            else:
                # Start new group if frames are not consecutive
                grouped_mislabeled_frames.append(current_group)
                mislabeled_group_lengths.append(len(current_group))
                current_group = [frame]
        # Add final group
        grouped_mislabeled_frames.append(current_group)
        mislabeled_group_lengths.append(len(current_group))

    # Group consecutive correct frames to analyze error patterns
    grouped_correct_frames = []
    group_lengths_correct = []
    if correct_frames:
        current_group = [correct_frames[0]]
        for frame in correct_frames[1:]:
            if frame == current_group[-1]:
                continue
            if frame == current_group[-1] + 1:
                current_group.append(frame)
            else:
                grouped_correct_frames.append(current_group)
                group_lengths_correct.append(len(current_group))
                current_group = [frame]
        # Add final group
        grouped_correct_frames.append(current_group)

    # Add final group length if there's a current group
    if current_group:
        group_lengths_correct.append(len(current_group))

    # Calculate mean of correct group lengths
    mean_mislabeled_length = (
        np.mean(mislabeled_group_lengths) if mislabeled_group_lengths else 0
    )
    mean_correct_length = np.mean(group_lengths_correct) if group_lengths_correct else 0

    return {
        "total_mislabeled_identities": total_mislabeled_identities,
        "mislabeled_group_lengths": mislabeled_group_lengths,
        "total_correct_identities": total_correct_identities,
        "summary": summary,
        "grouped_mislabeled_frames": grouped_mislabeled_frames,
        "mean_mislabeled_length": mean_mislabeled_length,
        "grouped_correct_frames": grouped_correct_frames,
        "group_lengths_correct": group_lengths_correct,
        "mean_correct_length": mean_correct_length,
    }
