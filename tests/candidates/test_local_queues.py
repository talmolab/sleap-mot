from collections import deque, defaultdict
from sleap_mot.candidates.local_queues import LocalQueueCandidates
from sleap_mot.tracker import Tracker


def test_local_queues_candidates(centered_pair_predictions):

    pred_instances = centered_pair_predictions[0].instances
    tracker = Tracker.from_config(candidates_method="local_queues")
    track_instances = tracker.get_features(pred_instances, 0)

    local_queues_candidates = LocalQueueCandidates(3, 20)
    assert isinstance(local_queues_candidates.tracker_queue, defaultdict)
    # assert isinstance(local_queues_candidates.tracker_queue[0], deque)
    local_queues_candidates.update_tracks(track_instances, None, None, None)
    # (tracks are assigned only if row/ col ids exists)
    # assert not local_queues_candidates.tracker_queue[0]

    track_instances = local_queues_candidates.add_new_tracks(track_instances)
    assert len(local_queues_candidates.tracker_queue) == 2
    assert len(local_queues_candidates.tracker_queue["0"]) == 1
    assert len(local_queues_candidates.tracker_queue["1"]) == 1

    new_track_id = local_queues_candidates.get_new_track_id(
        existing_track_ids=local_queues_candidates.tracker_queue.keys()
    )
    assert new_track_id == "2"

    track_instances = tracker.get_features(pred_instances, 0)
    tracked_instances = local_queues_candidates.add_new_tracks(track_instances)
    assert tracked_instances[0].track_id == "2"
    assert tracked_instances[1].track_id == "3"

    features_track_id = local_queues_candidates.get_features_from_track_id("0")
    assert isinstance(features_track_id, list)
    assert len(features_track_id) == 1
