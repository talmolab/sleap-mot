import pytest
import sleap_io as sio


@pytest.fixture
def centered_pair_predictions():
    return sio.load_slp("tests/data/slp/centered_pair_predictions.clean.slp")
