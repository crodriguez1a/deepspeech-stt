import pytest

from src.deepspeech_stt import deepspeech_predict


@pytest.fixture
def sample_wav():
    return "samples/test.wav"


def test_core(sample_wav):
    assert deepspeech_predict(sample_wav) == "test one two three test one two three"


def test_filters(sample_wav):
    all_filters = [
        "",
        "high_pass_filter",
        "low_pass_filter",
        "butter_bandpass_filter",
        "logmmse_denoise",
    ]
    assert deepspeech_predict(sample_wav, filters=all_filters) is not None
