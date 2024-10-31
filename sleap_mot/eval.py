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
                track_key: inst.track.name[-1] if inst.track is not None else None,
            }
            points = inst.points
            for key in points:
                frame_meta[key.name] = (points[key].x, points[key].y)

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


def get_metrics(df_gt_in, df_pred_in):
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

    df_merged = pd.merge(
        df_gt,
        df_pred,
        left_on=[
            "frame_id",
            "Nose_x",
            "Ear_R_x",
            "Ear_L_x",
            "TTI_x",
            "TailTip_x",
            "Head_x",
            "Trunk_x",
            "Tail_0_x",
            "Tail_1_x",
            "Tail_2_x",
            "Shoulder_left_x",
            "Shoulder_right_x",
            "Haunch_left_x",
            "Haunch_right_x",
            "Neck_x",
            "Nose_y",
            "Ear_R_y",
            "Ear_L_y",
            "TTI_y",
            "TailTip_y",
            "Head_y",
            "Trunk_y",
            "Tail_0_y",
            "Tail_1_y",
            "Tail_2_y",
            "Shoulder_left_y",
            "Shoulder_right_y",
            "Haunch_left_y",
            "Haunch_right_y",
            "Neck_y",
        ],
        right_on=[
            "frame_id",
            "Nose_x",
            "Ear_R_x",
            "Ear_L_x",
            "TTI_x",
            "TailTip_x",
            "Head_x",
            "Trunk_x",
            "Tail_0_x",
            "Tail_1_x",
            "Tail_2_x",
            "Shoulder_left_x",
            "Shoulder_right_x",
            "Haunch_left_x",
            "Haunch_right_x",
            "Neck_x",
            "Nose_y",
            "Ear_R_y",
            "Ear_L_y",
            "TTI_y",
            "TailTip_y",
            "Head_y",
            "Trunk_y",
            "Tail_0_y",
            "Tail_1_y",
            "Tail_2_y",
            "Shoulder_left_y",
            "Shoulder_right_y",
            "Haunch_left_y",
            "Haunch_right_y",
            "Neck_y",
        ],
        how="inner",
    )

    # initialize separate pymm accumulators for dreem and trackmate
    acc = mm.MOTAccumulator(auto_id=True)
    total_mislabeled_frames = 0
    mislabeled_frames = []
    # now iterate over each frame and pass in gt,dreem,tm pred tracks into pymotmetrics
    for frame, framedf in df_merged.groupby("frame_id"):
        gt_ids = framedf["gt_track_id"].values
        pred_tracks = framedf["pred_track_id"].values

        # pass into pymotmetrics
        for idx, gt_id in enumerate(gt_ids):
            if gt_id != pred_tracks[idx]:
                total_mislabeled_frames += 1
                mislabeled_frames.append(frame)
                break

        # define cost matrix of size num_gt_tracks x num_gt_tracks for gt vs dreem
        # since our preds match with gt ids, so if dreem is correct, dreem id = gt id
        cost_gt_pred = np.full((len(gt_ids), len(gt_ids)), np.nan)
        np.fill_diagonal(cost_gt_pred, 1)

        acc.update(
            oids=gt_ids,
            hids=pred_tracks,
            dists=cost_gt_pred,
        )
    # Get the events log for all frames after processing
    # all_events = acc.events

    # Check for 'SWITCH' events in all frames
    # switch_events = all_events[all_events['Type'] == 'SWITCH']
    # get pymotmetrics summary
    mh = mm.metrics.create()
    summary = mh.compute(acc, name="acc").transpose()

    if total_mislabeled_frames > len(df_merged["frame_id"].unique()) / 2:
        total_mislabeled_frames = (
            len(df_merged["frame_id"].unique()) - total_mislabeled_frames
        )
        # Get all unique frame IDs
        all_frames = set(df_merged["frame_id"].unique())

        # Convert mislabeled_frames to a set for efficient difference operation
        mislabeled_frames_set = set(mislabeled_frames)

        # Find frames that are not in mislabeled_frames
        correctly_labeled_frames = list(all_frames - mislabeled_frames_set)

        # Sort the list to maintain order
        correctly_labeled_frames.sort()

        # Update mislabeled_frames to be the list of correctly labeled frames
        mislabeled_frames = correctly_labeled_frames
    # Group consecutive mislabeled frames
    grouped_mislabeled_frames = []
    group_lengths = []
    if mislabeled_frames:
        current_group = [mislabeled_frames[0]]
        for frame in mislabeled_frames[1:]:
            if frame == current_group[-1] + 1:
                current_group.append(frame)
            else:
                grouped_mislabeled_frames.append(current_group)
                group_lengths.append(len(current_group))
                current_group = [frame]
        grouped_mislabeled_frames.append(current_group)
        group_lengths.append(len(current_group))

    return summary, total_mislabeled_frames, group_lengths
