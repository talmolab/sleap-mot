import numpy as np
import motmetrics as mm
import pandas as pd

def get_df(df):
    gt_frame_meta_list = []

    # loop through the labeled frames
    for lf in df:
        # in each frame, loop through instances
        for inst in lf:
            # make a dictionary for each instance
            frame_meta = {
                "frame_id": lf.frame_idx,
                "gt_track_id": inst.track.name[-1] if inst.track is not None else None,
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
        if return_df[col].dtype != 'object' or not isinstance(return_df[col].iloc[0], tuple):
            df_expanded[col] = return_df[col]

    # Expand tuple columns
    for col in return_df.columns:
        if return_df[col].dtype == 'object' and isinstance(return_df[col].iloc[0], tuple):
            # Create two new columns with suffixes _x and _y
            df_expanded[f'{col}_x'] = return_df[col].apply(lambda x: x[0] if isinstance(x, tuple) else None)
            df_expanded[f'{col}_y'] = return_df[col].apply(lambda x: x[1] if isinstance(x, tuple) else None)

    # Replace the original df_gt with the expanded version
    return_df = df_expanded.fillna(0)
    return return_df


def get_metrics(df_gt, df_pred):

    df_gt = get_df(df_gt)
    df_pred = get_df(df_pred)

    df_merged = pd.merge(df_gt, df_pred, left_on=["frame_id", "Nose_x", "Ear_R_x", "Ear_L_x", "TTI_x", "TailTip_x",
         "Head_x", "Trunk_x", "Tail_0_x", "Tail_1_x", "Tail_2_x", "Shoulder_left_x",
         "Shoulder_right_x", "Haunch_left_x", "Haunch_right_x", "Neck_x", "Nose_y", "Ear_R_y", "Ear_L_y", "TTI_y", "TailTip_y",
         "Head_y", "Trunk_y", "Tail_0_y", "Tail_1_y", "Tail_2_y", "Shoulder_left_y",
         "Shoulder_right_y", "Haunch_left_y", "Haunch_right_y", "Neck_y"], right_on=["frame_id", "Nose_x", "Ear_R_x", "Ear_L_x", "TTI_x", "TailTip_x",
         "Head_x", "Trunk_x", "Tail_0_x", "Tail_1_x", "Tail_2_x", "Shoulder_left_x",
         "Shoulder_right_x", "Haunch_left_x", "Haunch_right_x", "Neck_x", "Nose_y", "Ear_R_y", "Ear_L_y", "TTI_y", "TailTip_y",
         "Head_y", "Trunk_y", "Tail_0_y", "Tail_1_y", "Tail_2_y", "Shoulder_left_y",
         "Shoulder_right_y", "Haunch_left_y", "Haunch_right_y", "Neck_y"], how="outer")

    # initialize separate pymm accumulators for dreem and trackmate
    acc = mm.MOTAccumulator(auto_id=True)
    # now iterate over each frame and pass in gt,dreem,tm pred tracks into pymotmetrics
    for frame, framedf in df_merged.groupby('frame_id'):
        gt_ids = framedf['gt_track_id'].values
        pred_tracks = framedf['pred_track_id'].values
        
        # pass into pymotmetrics
        
        # define cost matrix of size num_gt_tracks x num_gt_tracks for gt vs dreem
        # since our preds match with gt ids, so if dreem is correct, dreem id = gt id
        cost_gt_pred = np.full((len(gt_ids), len(gt_ids)), np.nan)
        np.fill_diagonal(cost_gt_pred, 1)
        
        acc.update(
            oids=gt_ids,
            hids=pred_tracks,
            dists=cost_gt_pred,
        )
        # save the frame-by-frame mot events log for TrackMate for validation
        #if frame == df_merged.frame_id.max(): 
            # acc_dreem.mot_events.to_csv(f"/home/jovyan/vast/mustafa/lysosomes-airyscan-proofread/dream-track-motmetrics/{stem}-motmetrics.csv")
            #acc.mot_events.to_csv(os.path.join(csv_path, "motmetrics", f"{stem}-motmetrics.csv"))
        
    # get pymotmetrics summary    
    mh = mm.metrics.create()
    summary = mh.compute(acc, name="acc").transpose()

    return summary