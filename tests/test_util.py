import numpy as np
import pytest

try:
    from src import bitermplus as btm
except ImportError:
    import bitermplus as btm


def test_vectorized_docs_use_default_count_vectorizer_analysis():
    docs = ["Machine, learning!"]
    _, vocabulary, _ = btm.get_words_freqs(docs)

    result = btm.get_vectorized_docs(docs, vocabulary)

    assert result[0].size == 2


def test_biterms_preserve_document_alignment():
    docs = [
        np.array([0, 1], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([1, 2], dtype=np.int32),
    ]

    result = btm.get_biterms(docs, win=2)

    assert result == [[[0, 1]], [], [[1, 2]]]


def test_biterm_window_matches_reference_width_semantics():
    document = np.array([0, 1, 2, 3], dtype=np.int32)

    result = btm.get_biterms([document], win=3)

    assert result == [[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]]


def test_biterm_window_requires_two_positions():
    with pytest.raises(ValueError, match="at least 2"):
        btm.get_biterms([np.array([0, 1], dtype=np.int32)], win=1)


@pytest.mark.parametrize(
    "document",
    [np.array([-1, 0]), np.array([0.5, 1.0])],
)
def test_biterms_reject_invalid_word_ids(document):
    with pytest.raises((TypeError, ValueError)):
        btm.get_biterms([document])
