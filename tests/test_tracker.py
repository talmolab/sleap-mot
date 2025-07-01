import pytest
import numpy as np

# from sleap_nn.inference.predictors import main
from sleap_mot.tracker import Tracker, FlowShiftTracker
from sleap_mot.track_instance import (
    TrackedInstanceFeature,
    TrackInstanceLocalQueue,
    TrackInstances,
)
import math


def test_high_threshold(centered_pair_predictions):
    # Test for the first two instances (high instance threshold)
    # no new tracks should be created

    pred_instances = centered_pair_predictions[0].instances
    for inst in pred_instances:
        inst.track = None
        assert inst.track is None

    tracker = Tracker.from_config(instance_score_threshold=999)
    assert isinstance(tracker, Tracker)
    assert not isinstance(tracker, FlowShiftTracker)

    tracked_instances = tracker.track_frame(pred_instances, 0)
    for inst in tracked_instances:
        assert inst.track is None
    assert len(tracker.candidate.current_tracks) == 0


def test_fixed_window(centered_pair_predictions):
    # Test Fixed-window method
    # pose as feature, oks scoring method, avg score reduction, hungarian matching
    # Test for the first two instances (tracks assigned to each of the new instances)

    pred_instances = centered_pair_predictions[0].instances
    for inst in pred_instances:
        inst.track = None

    tracker = Tracker.from_config(
        instance_score_threshold=0.0, candidates_method="fixed_window"
    )
    for inst in pred_instances:
        assert inst.track is None

    tracked_instances = tracker.track_frame(pred_instances, 0)  # 2 tracks are created
    for inst in tracked_instances:
        assert inst.track is not None
    assert (
        tracked_instances[0].track.name == "track_0"
        and tracked_instances[1].track.name == "track_1"
    )
    assert len(tracker.candidate.tracker_queue) == 1
    assert tracker.candidate.current_tracks == [0, 1]
    assert tracker.candidate.tracker_queue[0].track_ids == [0, 1]


def test_local_queue(centered_pair_predictions):
    # Test local queue method
    # pose as feature, oks scoring method, max score reduction, hungarian matching
    # Test for the first two instances (tracks assigned to each of the new instances)

    pred_instances = centered_pair_predictions[0].instances
    for inst in pred_instances:
        inst.track = None

    tracker = Tracker.from_config(
        instance_score_threshold=0.0, candidates_method="local_queues"
    )
    tracked_instances = tracker.track_frame(pred_instances, 0)  # 2 tracks are created
    for inst in tracked_instances:
        assert inst.track is not None
    assert len(tracker.candidate.tracker_queue) == 2
    assert tracker.candidate.current_tracks == ["0", "1"]
    assert (
        tracked_instances[0].track.name == "0"
        and tracked_instances[1].track.name == "1"
    )


def test_fixed_window(centered_pair_predictions):
    # Test indv. functions for fixed window
    # with 2 existing tracks in the queue
    pred_instances = centered_pair_predictions[0].instances
    for inst in pred_instances:
        inst.track = None

    tracker = Tracker.from_config(
        instance_score_threshold=0.0,
        candidates_method="fixed_window",
        scoring_reduction="max",
        track_matching_method="greedy",
    )
    tracked_instances = tracker.track_frame(pred_instances, 0)
    assert tracker.candidate.current_tracks == ["0", "1"]
    assert (
        tracked_instances[0].track.name == "track_0"
        and tracked_instances[1].track.name == "track_1"
    )


def test_point_features(centered_pair_predictions):
    pred_instances = centered_pair_predictions[0].instances
    for inst in pred_instances:
        inst.track = None

    tracker = Tracker.from_config(
        instance_score_threshold=0.0,
        candidates_method="fixed_window",
        scoring_reduction="max",
        track_matching_method="greedy",
    )

    track_instances = tracker.get_features(pred_instances, 0, None)
    assert isinstance(track_instances, TrackInstances)
    for p, t in zip(pred_instances, track_instances.features):
        np.testing.assert_array_almost_equal(p.numpy(), t)


# def test_score_and_assign(centered_pair_predictions):
#     pred_instances = centered_pair_predictions[0].instances
#     for inst in pred_instances:
#         inst.track = None

#     tracker = Tracker.from_config(
#         instance_score_threshold=0.0,
#         candidates_method="fixed_window",
#         scoring_reduction="max",
#         track_matching_method="greedy",
#     )
#     tracked_instances = tracker.track_frame(pred_instances, 0)

#     # Test get_scores(), oks as scoring
#     track_instances = tracker.get_features(pred_instances, 0, None)
#     candidates_list = tracker.generate_candidates()
#     candidate_feature_dict = tracker.update_candidates(candidates_list)
#     scores = tracker.get_scores(track_instances, candidate_feature_dict)
#     # Convert scores dictionary to 2D array and ensure integer values
#     scores = np.array([scores[i] for i in range(len(scores))])
#     np.testing.assert_array_almost_equal(scores, [[1, 0], [0, 1]])

#     # Test assign_tracks()
#     cost = tracker.scores_to_cost_matrix(scores)
#     track_instances = tracker.assign_tracks(track_instances, cost)
#     assert track_instances.track_ids[0] == "0" and track_instances.track_ids[1] == "1"
#     assert np.all(track_instances.features[0] == pred_instances[0].numpy())
#     assert len(tracker.candidate.current_tracks) == 2
#     assert len(tracker.candidate.tracker_queue) == 2
#     assert track_instances.tracking_scores == [1.0, 1.0]

#     tracked_instances = tracker.track_frame(pred_instances, 0)
#     assert len(tracker.candidate.tracker_queue) == 3
#     assert len(tracker.candidate.current_tracks) == 2
#     assert np.all(
#         tracker.candidate.tracker_queue[0].features[0]
#         == tracker.candidate.tracker_queue[2].features[0]
#     )
#     assert (
#         tracker.candidate.tracker_queue[0].track_ids[1]
#         == tracker.candidate.tracker_queue[2].track_ids[1]
#     )


def test_tracker_track(noisy_clip_predictions_untracked):
    tracker = Tracker.from_config(candidates_method="local_queues", max_tracks=2)

    tracked_labels = tracker.track(noisy_clip_predictions_untracked)
    assert len(tracked_labels) == len(noisy_clip_predictions_untracked)
    assert len(tracked_labels.tracks) == 2
