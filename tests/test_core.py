import pytest

from src.deepspeech_tts import deepspeech_predict


@pytest.fixture
def sample_wav():
    return "samples/test.wav"


def test_core(sample_wav):
    assert deepspeech_predict(sample_wav) == "test one two three test one two three"
