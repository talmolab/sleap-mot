"""Tests for the new tracking architecture (sleap_mot.tracking)."""

import pytest
import numpy as np
import sleap_io as sio

from sleap_mot.tracking.base import TrackContext, extract_all_track_histories
from sleap_mot.tracking.instance_explanations import (
    InstanceExplanationStore,
    DecisionType,
    MotionDecisionRecord,
    CandidateScore,
)


class TestTrackContext:
    """Tests for TrackContext wrapper."""

    def test_create_track_context(self):
        track = sio.Track(name="test_track")
        ctx = TrackContext(
            priority=5,
            track=track,
            name="test_track",
        )
        assert ctx.priority == 5
        assert ctx.name == "test_track"
        assert ctx.temporary_track is False
        assert ctx.valid is True
        assert ctx.track_history == []

    def test_add_history_entry(self):
        track = sio.Track(name="animal_A")
        ctx = TrackContext(priority=5, track=track, name="animal_A")
        ctx.add_history_entry(
            layer_name="MotionTracker",
            old_track_name=None,
            new_track_name="animal_A",
            frame_idx=0,
            reason="Initial assignment",
        )
        assert len(ctx.track_history) == 1
        assert ctx.track_history[0]["layer_name"] == "MotionTracker"
        assert ctx.track_history[0]["frame_idx"] == 0

    def test_temporary_track(self):
        track = sio.Track(name="tracklet_1")
        ctx = TrackContext(
            priority=5,
            track=track,
            name="tracklet_1",
            temporary_track=True,
        )
        assert ctx.temporary_track is True

    def test_is_valid(self):
        track = sio.Track(name="t")
        ctx = TrackContext(priority=1, track=track, name="t", valid=False)
        assert ctx.is_valid() is False


class TestInstanceExplanationStore:
    """Tests for the explanation store."""

    def test_create_store(self):
        store = InstanceExplanationStore()
        assert store is not None

    def test_add_and_retrieve_record(self):
        store = InstanceExplanationStore()
        record = MotionDecisionRecord(
            frame_idx=0,
            instance_idx=0,
            tracker_name="MotionTracker",
            tracker_priority=5,
            decision_type=DecisionType.MATCHED,
            assigned_track_id="track_0",
            previous_track_id=None,
            summary="Matched to track_0",
            candidate_scores=[
                CandidateScore(
                    candidate_id="track_0",
                    score=0.95,
                    passed_thresholds=True,
                ),
            ],
        )
        store.add(record)
        records = store.get_records_for_frame(0)
        assert len(records) > 0

    def test_get_records_for_instance(self):
        store = InstanceExplanationStore()
        record = MotionDecisionRecord(
            frame_idx=10,
            instance_idx=1,
            tracker_name="MotionTracker",
            tracker_priority=5,
            decision_type=DecisionType.MATCHED,
            assigned_track_id="track_1",
            previous_track_id=None,
            summary="Matched to track_1",
        )
        store.add(record)
        records = store.get_records(10, 1)
        assert len(records) == 1


class TestExtractTrackHistories:
    """Tests for track history extraction."""

    def test_extract_from_labels_with_contexts(self):
        """Test extracting histories from labels with TrackContext objects."""
        skeleton = sio.Skeleton(nodes=["A", "B"])
        track = sio.Track(name="animal_1")
        ctx = TrackContext(priority=5, track=track, name="animal_1")
        ctx.add_history_entry(
            layer_name="TestLayer",
            old_track_name=None,
            new_track_name="animal_1",
            frame_idx=0,
            reason="test",
        )

        inst = sio.PredictedInstance.from_numpy(
            points_data=np.array([[10.0, 20.0], [30.0, 40.0]]),
            skeleton=skeleton,
            point_scores=np.array([1.0, 1.0]),
            score=1.0,
        )
        inst.track = ctx

        lf = sio.LabeledFrame(
            video=sio.Video(filename="test.mp4"),
            frame_idx=0,
            instances=[inst],
        )
        labels = sio.Labels(labeled_frames=[lf])

        histories = extract_all_track_histories(labels)
        assert "animal_1" in histories
        assert len(histories["animal_1"]) == 1

    def test_extract_from_labels_without_contexts(self):
        """No TrackContext objects means empty histories."""
        skeleton = sio.Skeleton(nodes=["A"])
        track = sio.Track(name="t")
        inst = sio.PredictedInstance.from_numpy(
            points_data=np.array([[0.0, 0.0]]),
            skeleton=skeleton,
            point_scores=np.array([1.0]),
            score=1.0,
        )
        inst.track = track
        lf = sio.LabeledFrame(
            video=sio.Video(filename="test.mp4"),
            frame_idx=0,
            instances=[inst],
        )
        labels = sio.Labels(labeled_frames=[lf])
        histories = extract_all_track_histories(labels)
        assert histories == {}


class TestOnlineTrackingLayer:
    """Tests for online tracking (MotionTracker etc.) using real test data."""

    def test_motion_tracker_creates_tracks(self, noisy_clip_predictions_untracked):
        """MotionTracker should assign tracks to untracked instances."""
        from sleap_mot.tracking.online_tracking import MotionTracker

        labels = noisy_clip_predictions_untracked
        tracker = MotionTracker(priority=5, name="TestMotion")
        store = InstanceExplanationStore()
        tracker.explanation_store = store

        result = tracker.track(labels)

        # Should have assigned some tracks
        tracked = sum(
            1 for lf in result for inst in lf.instances if inst.track is not None
        )
        assert tracked > 0

    def test_motion_tracker_with_explanation_store(
        self, noisy_clip_predictions_untracked
    ):
        """MotionTracker should populate the explanation store."""
        from sleap_mot.tracking.online_tracking import MotionTracker

        labels = noisy_clip_predictions_untracked
        tracker = MotionTracker(priority=5, name="TestMotion")
        store = InstanceExplanationStore()
        tracker.explanation_store = store

        tracker.track(labels)

        # Store should have records
        all_records = store.get_all_records()
        assert len(all_records) > 0

    def test_tracklet_stitcher(self, noisy_clip_predictions):
        """TrackletStitcher should run without errors on tracked data."""
        from sleap_mot.tracking.online_tracking import TrackletStitcher

        labels = noisy_clip_predictions
        stitcher = TrackletStitcher(priority=3, name="TestStitcher")
        store = InstanceExplanationStore()
        stitcher.explanation_store = store

        result = stitcher.track(labels)
        assert result is not None

    def test_general_tracker(self, noisy_clip_predictions_untracked):
        """GeneralOnlineTracker should run with default settings."""
        from sleap_mot.tracking.online_tracking import GeneralOnlineTracker

        labels = noisy_clip_predictions_untracked
        tracker = GeneralOnlineTracker(priority=5, name="TestGeneral")
        store = InstanceExplanationStore()
        tracker.explanation_store = store

        result = tracker.track(labels)
        tracked = sum(
            1 for lf in result for inst in lf.instances if inst.track is not None
        )
        assert tracked > 0
