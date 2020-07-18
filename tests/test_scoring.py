import pytest

from src.scoring import english_scoring


@pytest.fixture
def sentences():
    return


def test_scoring(sentences):
    assert english_scoring(["hello world", "hello world"]) == [1.0, 1.0]


def test_scoring_empty_strings(sentences):
    assert english_scoring(["", ""]) == [0.0, 0.0]
