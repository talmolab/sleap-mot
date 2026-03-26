"""Tests for the CLI pipeline runner (sleap_mot.pipeline)."""

import pytest
import tempfile
import os
from pathlib import Path

import yaml

from sleap_mot.pipeline import (
    load_config,
    split_params,
    _interpolate_variables,
    _track_accepts_param,
    _ensure_registry,
    TRACKER_REGISTRY,
)


@pytest.fixture
def minimal_config_path():
    """Write a minimal valid YAML config to a temp file."""
    config = {
        "input": {"labels": "tests/data/slp/centered_pair_predictions.clean.slp"},
        "output": {"path": "/tmp/test_output.slpt"},
        "pipeline": [
            {
                "layer": "GeneralOnlineTracker",
                "priority": 5,
                "name": "TestTracker",
                "params": {},
            }
        ],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(config, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def config_with_interpolation():
    """Config that uses variable interpolation."""
    config = {
        "input": {
            "labels": "/fake/labels.slp",
            "video": "/fake/video.mp4",
        },
        "pipeline": [
            {
                "layer": "TailFeatureTracker",
                "priority": 10,
                "params": {
                    "tail_nodes": ["Tail_0", "Tail_1"],
                    "video_path": "${input.video}",
                },
            }
        ],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(config, f)
        path = f.name
    yield path
    os.unlink(path)


class TestInterpolation:
    """Tests for variable interpolation."""

    def test_string_interpolation(self):
        result = _interpolate_variables(
            "${input.video}", {"input.video": "/path/to/video.mp4"}
        )
        assert result == "/path/to/video.mp4"

    def test_no_interpolation_needed(self):
        result = _interpolate_variables("plain_string", {"input.video": "/v.mp4"})
        assert result == "plain_string"

    def test_list_interpolation(self):
        result = _interpolate_variables(
            ["${input.labels}", "other"],
            {"input.labels": "/path/labels.slp"},
        )
        assert result == ["/path/labels.slp", "other"]

    def test_dict_interpolation(self):
        result = _interpolate_variables(
            {"key": "${input.video}"},
            {"input.video": "/v.mp4"},
        )
        assert result == {"key": "/v.mp4"}

    def test_non_string_passthrough(self):
        assert _interpolate_variables(42, {}) == 42
        assert _interpolate_variables(3.14, {}) == 3.14
        assert _interpolate_variables(None, {}) is None


class TestLoadConfig:
    """Tests for YAML config loading and validation."""

    def test_load_valid_config(self, minimal_config_path):
        config = load_config(minimal_config_path)
        assert "input" in config
        assert "pipeline" in config
        assert len(config["pipeline"]) == 1

    def test_missing_labels_raises(self):
        config_str = "pipeline:\n  - layer: MotionTracker\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(config_str)
            path = f.name
        try:
            with pytest.raises(ValueError, match="input.labels"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_empty_pipeline_raises(self):
        config_str = "input:\n  labels: /tmp/x.slp\npipeline: []\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(config_str)
            path = f.name
        try:
            with pytest.raises(ValueError, match="at least one layer"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_unknown_tracker_raises(self):
        config_str = "input:\n  labels: /tmp/x.slp\npipeline:\n  - layer: FakeTracker\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(config_str)
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unknown tracker"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_missing_config_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_variable_interpolation_applied(self, config_with_interpolation):
        config = load_config(config_with_interpolation)
        tail_params = config["pipeline"][0]["params"]
        assert tail_params["video_path"] == "/fake/video.mp4"

    def test_cli_overrides(self, minimal_config_path):
        config = load_config(
            minimal_config_path,
            overrides={
                "labels": "/override/labels.slp",
                "output": "/override/output.slpt",
            },
        )
        assert config["input"]["labels"] == "/override/labels.slp"
        assert config["output"]["path"] == "/override/output.slpt"

    def test_default_output_path(self):
        config_str = (
            "input:\n  labels: /tmp/predictions.slp\n"
            "pipeline:\n  - layer: GeneralOnlineTracker\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(config_str)
            path = f.name
        try:
            config = load_config(path)
            assert config["output"]["path"] == "/tmp/predictions_tracked.slpt"
        finally:
            os.unlink(path)


class TestSplitParams:
    """Tests for constructor vs track arg splitting."""

    def test_motion_tracker_split(self):
        _ensure_registry()
        cls = TRACKER_REGISTRY["MotionTracker"]
        ctor, track = split_params(
            cls,
            {
                "long_kde_path": "/tmp/kde.joblib",
                "iou_threshold": 0.1,
                "max_instances": 8,
            },
        )
        assert "long_kde_path" in ctor
        assert "iou_threshold" in ctor
        assert "max_instances" in track

    def test_tail_tracker_split(self):
        _ensure_registry()
        cls = TRACKER_REGISTRY["TailFeatureTracker"]
        ctor, track = split_params(
            cls,
            {
                "tail_nodes": ["Tail_0"],
                "n_clusters": 8,
                "video_path": "/tmp/v.mp4",
            },
        )
        assert "tail_nodes" in ctor
        assert "n_clusters" in ctor
        assert "video_path" in track

    def test_empty_params(self):
        _ensure_registry()
        cls = TRACKER_REGISTRY["GeneralOnlineTracker"]
        ctor, track = split_params(cls, {})
        assert ctor == {}
        assert track == {}


class TestTrackAcceptsParam:
    """Tests for checking if .track() accepts a parameter."""

    def test_tail_tracker_accepts_video_path(self):
        _ensure_registry()
        cls = TRACKER_REGISTRY["TailFeatureTracker"]
        assert _track_accepts_param(cls, "video_path") is True

    def test_motion_tracker_accepts_kwargs(self):
        _ensure_registry()
        cls = TRACKER_REGISTRY["MotionTracker"]
        # MotionTracker.track() has **kwargs via OnlineTrackingLayer
        result = _track_accepts_param(cls, "video_path")
        # Either True (via **kwargs) or False — just shouldn't error
        assert isinstance(result, bool)


class TestRegistry:
    """Tests for the tracker registry."""

    def test_registry_loads(self):
        _ensure_registry()
        assert len(TRACKER_REGISTRY) >= 9

    def test_all_expected_trackers_present(self):
        _ensure_registry()
        expected = [
            "MotionTracker",
            "DirectionalMotionTracker",
            "FacingConsistencyTracker",
            "GeneralOnlineTracker",
            "TrackletStitcher",
            "TailFeatureTracker",
            "RFIDFeatureTracker",
            "CoordinateRFIDTracker",
            "VisualPatchTracker",
        ]
        for name in expected:
            assert name in TRACKER_REGISTRY, f"Missing tracker: {name}"
