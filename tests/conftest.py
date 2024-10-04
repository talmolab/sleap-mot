import pytest
import sleap_io as sio


@pytest.fixture
def centered_pair_predictions():
    return sio.load_slp("tests/data/slp/centered_pair_predictions.clean.slp")


@pytest.fixture
def noisy_clip_predictions():
    """Clip with predictions from noisy model.

    Clip is 96 frames long.

    Pose predictions are from a model trained on a noisy dataset, so there are frequent
    chimeras, missing nodes and identity switches.

    There are only 2 tracks.
    """
    return sio.load_slp(
        "tests/data/clips/cohort2.220516_093004.Camera0.mov.00000.t0=20068.t1=20165/sleap_tracked.slp"
    )


@pytest.fixture
def noisy_clip_predictions_untracked():
    """Clip with predictions from noisy model without tracks assigned.

    This is the same clip as `noisy_clip_predictions` but with all tracks removed.
    """
    labels = sio.load_slp(
        "tests/data/clips/cohort2.220516_093004.Camera0.mov.00000.t0=20068.t1=20165/sleap_tracked.slp"
    )

    for lf in labels:
        for inst in lf.instances:
            inst.track = None

    labels.tracks = []

    return labels


@pytest.fixture
def noisy_clip_feature_tracked():
    """Clip with "reliable" frames with tracks refined via reliable feature tracking.

    This is the same clip as `noisy_clip_predictions` but only reliable frames have
    tracks assigned. Predictions still exist for remaining frames, but they do not have
    tracks.
    """
    return sio.load_slp(
        "tests/data/clips/cohort2.220516_093004.Camera0.mov.00000.t0=20068.t1=20165/feature_tracked.slp"
    )


@pytest.fixture
def noisy_clip_proofread():
    """Clip with proofread tracks.

    This is the same clip as `noisy_clip_predictions` but it has been manually proofread
    to remove (most) chimeras and identity switches.
    """
    return sio.load_slp(
        "tests/data/clips/cohort2.220516_093004.Camera0.mov.00000.t0=20068.t1=20165/proofread.slp"
    )


@pytest.fixture
def noisy_clip_video():
    """Video for clip with noisy predictions."""
    return sio.load_video(
        "tests/data/clips/cohort2.220516_093004.Camera0.mov.00000.t0=20068.t1=20165/video.mp4"
    )
