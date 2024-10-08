import pandas as pd
import numpy as np
from functools import partial
import sleap_io
import h5py
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import imageio.v3 as iio
from pathlib import Path


def get_patches(lf: sleap_io.LabeledFrame, patch_size: int = 5):

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


def get_binned_freqs(patches, mask=None, n_bins=4):
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


def feature_extraction(input_slp, video_path, output_file):

    labels = sleap_io.load_file(input_slp)

    labels.replace_filenames(
        prefix_map={
            Path(labels.videos[0].backend_metadata["filename"])
            .parent: Path(video_path)
            .parent
        }
    )

    patches, freqs = [], []
    for lf in labels:
        if len(lf.instances) == 2:
            patches_, mask = get_patches(lf)
            freqs_ = get_binned_freqs(patches_, mask)
        else:
            patches_ = np.full((2, 15, 5, 5, 1), -1)
            freqs_ = np.zeros((2, 15, 4))

        patches.append(patches_)
        freqs.append(freqs_)

    index = list(range(len(freqs)))
    patches = np.stack(patches, axis=0)
    freqs = np.stack(freqs, axis=0)

    with h5py.File(output_file, "w") as hdf5_file:
        hdf5_file.create_dataset("frequencies", data=freqs, compression=1)
        hdf5_file.create_dataset("patches", data=patches, compression=1)
        hdf5_file.create_dataset("indices", data=index, compression=1)


def get_data(labels, columns, node_names):
    data = []
    for frame_data in labels:
        if not frame_data:  # Skip empty frames
            continue

        # Iterate through each instance (track) in the frame
        for inst in frame_data:

            # Initialize a dictionary for this row with "frame_idx" and "track"
            row = {"frame_idx": frame_data.frame_idx, "track": inst.track.name}
            for node, point in inst.points.items():
                if point is not None:
                    row[f"{node.name}.x"] = point.x
                    row[f"{node.name}.y"] = point.y
                    row[f"{node.name}.score"] = point.score
                else:
                    row[f"{node}.x"] = "NaN"
                    row[f"{node}.y"] = "NaN"
                    row[f"{node}.score"] = "NaN"
            data.append(row)

    tracks = pd.DataFrame(data, columns=columns)

    track_names = tracks["track"].unique().tolist()
    n_tracks = len(track_names)
    n_frames = int(tracks["frame_idx"].max() + 1)
    n_nodes = len(node_names)

    trx = np.full((n_frames, n_tracks, n_nodes, 2), np.nan)
    trx_scores = np.full((n_frames, n_tracks, n_nodes), np.nan)

    track_names = sorted(tracks["track"].unique())
    track_index = {name: idx for idx, name in enumerate(track_names)}

    for _, row in tracks.iterrows():
        frame_idx = row["frame_idx"]
        track_name = row["track"]
        track_idx = track_index[track_name]

        # Extract x, y coordinates and scores
        pts = (
            row[2:].to_numpy().reshape(n_nodes, 3)
        )  # Assuming x, y, score for each node
        trx[frame_idx, track_idx] = pts[:, :2]
        trx_scores[frame_idx, track_idx] = pts[:, 2]

    return trx


def get_iou(trx, features):
    x0y0 = np.nanmin(trx, axis=-2)
    x1y1 = np.nanmax(trx, axis=-2)

    ix_x0y0 = np.maximum(x0y0[:, 0], x0y0[:, 1])
    ix_x1y1 = np.minimum(x1y1[:, 0], x1y1[:, 1])

    bbox_area = np.prod(x1y1 - x0y0, axis=-1)

    ix_area = np.prod(np.maximum(ix_x1y1 - ix_x0y0, 0), axis=1)
    ix_area = np.where(np.isnan(ix_area), 0, ix_area)
    un_area = np.nansum(bbox_area, axis=-1) - ix_area
    iou = ix_area / un_area

    confidence_vector = iou == 0

    patches = np.array(features["patches"])
    mask = np.all(patches == -1, axis=(1, 2, 3, 4))
    mask = np.squeeze(mask)
    confidence_vector[np.logical_and(confidence_vector, mask)] = False

    return confidence_vector


def run_PCA(features, confidence_vector):

    feats = features["frequencies"]
    feats = feats[confidence_vector]

    X0 = np.stack([feat[0].flatten() for feat in feats], axis=0)
    X1 = np.stack([feat[1].flatten() for feat in feats], axis=0)
    X = np.concatenate([X0, X1], axis=0)

    pcs = PCA()
    pcs = pcs.fit(X)
    Z = pcs.transform(X)

    kmeans = KMeans(n_clusters=2).fit(X)
    G = kmeans.labels_

    return G, X, Z


def run_knn(X, G, Z, features):
    n = int(features["frequencies"].shape[0] * 0.005)
    if n < 2:
        n = 2

    knn = NearestNeighbors(n_neighbors=n, radius=0.5, metric="cosine")
    knn.fit(X)
    distances, indices = knn.kneighbors(X)

    nn_G = G[indices]
    is_unambiguous = (nn_G == G.reshape(-1, 1)).all(axis=1)

    kmeans = KMeans(n_clusters=2).fit(Z[:, :10])
    G = kmeans.labels_

    return is_unambiguous, G


def process_ambiguous_frames(is_unambiguous, confidence_vector, G):
    ambig_counter, G_counter = 0, 0
    G_split = int(len(G) / 2) - 1

    for i in range(len(confidence_vector)):
        if confidence_vector[i]:
            if (
                not (
                    is_unambiguous[ambig_counter] and is_unambiguous[ambig_counter + 1]
                )
                or G[G_counter] == G[G_counter + G_split]
            ):
                confidence_vector[i] = False
            ambig_counter += 2
            G_counter += 1

    return confidence_vector


def relabel_tracks(X, G, labels, features, confidence_vector):
    indices = features["indices"]
    indices = indices[confidence_vector]

    luminance_bins = np.tile(np.arange(4, dtype=float) / 3, 15)

    (X[G == 0].mean(axis=0) * luminance_bins).mean(), (
        X[G == 1].mean(axis=0) * luminance_bins
    ).mean()

    confident_lf = [lf.instances for lf in labels[indices]]
    feats_instance_inds = np.array([[0, 1] for instance in confident_lf])

    X_instance_inds = np.concatenate(
        [feats_instance_inds[:, 0], feats_instance_inds[:, 1]], axis=0
    )
    X_frame_inds = np.concatenate([indices, indices], axis=0)

    avg_luminance_0 = (X[G == 0].mean(axis=0) * luminance_bins).mean()
    avg_luminance_1 = (X[G == 1].mean(axis=0) * luminance_bins).mean()

    if avg_luminance_0 < avg_luminance_1:
        cluster_to_track_map = {
            0: labels.tracks[0],
            1: labels.tracks[1],
        }
    else:
        cluster_to_track_map = {
            0: labels.tracks[1],
            1: labels.tracks[0],
        }

    for cluster, frame_idx, instance_ind in zip(G, X_frame_inds, X_instance_inds):
        track = cluster_to_track_map[cluster]
        lf = labels[(labels.video, frame_idx)]
        lf.instances[instance_ind].track = track

    for i in range(len(confidence_vector)):
        if not confidence_vector[i]:
            for instance in labels[i].instances:
                instance.track = None

    return labels


def global_track(input_slp_path, input_vid_path, output_features_path=None):
    if output_features_path == None:
        output_features_path = f"{input_slp_path}.features.hdf5"

    feature_extraction(input_slp_path, input_vid_path, output_features_path)

    print("Features saved to: ", output_features_path)

    features = h5py.File(output_features_path, "r")
    labels = sleap_io.load_file(input_slp_path)
    node_names = labels.skeleton.node_names
    columns = ["frame_idx", "track"] + [
        f"{node}.{coord}" for node in node_names for coord in ["x", "y", "score"]
    ]

    trx = get_data(labels, columns, node_names)

    confidence_vector = get_iou(trx, features)

    G, X, Z = run_PCA(features, confidence_vector)
    is_unambiguous, G = run_knn(X, G, Z, features)

    confidence_vector = process_ambiguous_frames(is_unambiguous, confidence_vector, G)
    G, X, Z = run_PCA(features, confidence_vector)

    labels = relabel_tracks(X, G, labels, features, confidence_vector)

    features.close()

    return labels, output_features_path
