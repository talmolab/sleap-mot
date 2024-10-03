from collections import deque
from sleap_mot.track_instance import TrackedInstanceFeature
from sleap_mot.candidates.fixed_window import FixedWindowCandidates
from sleap_mot.tracker import Tracker


def test_fixed_window_candidates(centered_pair_predictions):

    pred_instances = centered_pair_predictions[0].instances

    tracker = Tracker.from_config()
    track_instances = tracker.get_features(pred_instances, 0)

    fixed_window_candidates = FixedWindowCandidates(3)
    assert isinstance(fixed_window_candidates.tracker_queue, deque)
    fixed_window_candidates.update_tracks(track_instances, None, None, None)
    # (tracks are assigned only if row/ col ids exists)
    assert not fixed_window_candidates.tracker_queue

    track_instances = fixed_window_candidates.add_new_tracks(track_instances)
    assert len(fixed_window_candidates.tracker_queue) == 1

    new_track_id = fixed_window_candidates.get_new_track_id()
    assert new_track_id == 2

    track_instances = tracker.get_features(pred_instances, 0)
    tracked_instances = fixed_window_candidates.add_new_tracks(track_instances)
    assert tracked_instances.track_ids == [2, 3]
    assert tracked_instances.tracking_scores == [1.0, 1.0]

    features_track_id = fixed_window_candidates.get_features_from_track_id(0)
    assert isinstance(features_track_id, list)
    assert len(features_track_id) == 1
