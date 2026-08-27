"""Regression tests for bugs found in the 0.12 audit."""

import numpy as np
import pytest
from scipy.sparse import csr_array, csr_matrix

try:
    from src import bitermplus as btm
except ImportError:
    import bitermplus as btm

TEXTS = [
    "machine learning algorithms are powerful tools",
    "deep learning neural networks process data",
    "natural language processing understands text",
    "machine learning models need training data",
    "neural networks learn representations from data",
]


def vectorize(texts=None):
    texts = TEXTS if texts is None else texts
    X, vocab, _ = btm.get_words_freqs(texts)
    docs_vec = btm.get_vectorized_docs(texts, vocab)
    return X, vocab, docs_vec


def make_classifier(**kwargs):
    params = {"n_topics": 2, "random_state": 42, "max_iter": 20}
    params.update(kwargs)
    return btm.BTMClassifier(**params)


def test_transform_is_pure():
    """transform() must not write model state (sklearn: state belongs to fit)."""
    model = make_classifier().fit(TEXTS)
    labels = model.labels_.copy()
    training = model.matrix_docs_topics_.copy()

    model.transform(["completely unrelated held out document text"])
    model.transform(TEXTS[:2], infer_type="mix")
    model.transform([])

    np.testing.assert_array_equal(model.labels_, labels)
    np.testing.assert_array_equal(model.matrix_docs_topics_, training)


def test_removed_btm_properties_explain_the_migration():
    """The p_zd-derived BTM properties were removed in 1.0."""
    model = make_classifier().fit(TEXTS)

    for name in ("labels_", "matrix_docs_topics_", "matrix_topics_docs_", "perplexity_"):
        with pytest.raises(AttributeError, match=r"removed in 1\.0"):
            getattr(model.model_, name)


def test_perplexity_accepts_explicit_documents():
    model = make_classifier().fit(TEXTS)

    assert model.perplexity() == model.perplexity_
    assert model.perplexity(TEXTS) > 0
    # Held-out documents (in-vocabulary, unseen combination) score on their own
    assert model.perplexity(["neural machine text data"]) > 0


def test_perplexity_needs_in_vocabulary_tokens():
    """Documents with no vocabulary overlap make perplexity undefined (0/0)."""
    model = make_classifier().fit(TEXTS)

    with pytest.raises(ValueError, match="at least one token"):
        model.perplexity(["zzz qqq vvv"])


def test_transform_empty_document_list():
    """cython.view.array rejects a zero-length axis; transform([]) crashed."""
    model = make_classifier().fit(TEXTS)

    result = model.transform([])

    assert result.shape == (0, 2)


def test_sparse_array_input_matches_sparse_matrix():
    """csr_array sums to a 1-D array, csr_matrix to a 2-D np.matrix."""
    X, vocab, docs_vec = vectorize()
    biterms = btm.get_biterms(docs_vec)

    from_matrix = btm.BTM(csr_matrix(X), vocab, T=2, seed=42)
    from_array = btm.BTM(csr_array(X), vocab, T=2, seed=42)
    from_matrix.fit(biterms, iterations=20, verbose=False)
    from_array.fit(biterms, iterations=20, verbose=False)

    np.testing.assert_allclose(
        from_matrix.matrix_topics_words_, from_array.matrix_topics_words_
    )


def test_empty_topics_idx_returns_empty_frame():
    """concat() raises on an empty sequence."""
    model = make_classifier().fit(TEXTS)
    p_zd = model.transform(TEXTS)

    assert btm.get_top_topic_words(model.model_, topics_idx=[]).empty
    assert btm.get_top_topic_docs(TEXTS, p_zd, topics_idx=[]).empty


@pytest.mark.parametrize("kwargs", [{"words_num": 0}, {"words_num": -1}])
def test_top_topic_words_rejects_bad_count(kwargs):
    model = make_classifier().fit(TEXTS)

    with pytest.raises(ValueError, match="words_num must be a positive integer"):
        btm.get_top_topic_words(model.model_, **kwargs)


def test_perplexity_property_does_not_mutate_model():
    """Reading perplexity_ used to re-run transform and overwrite p_zd."""
    model = make_classifier().fit(TEXTS)
    model.transform(TEXTS[:2])
    labels_before = model.labels_.copy()

    assert model.perplexity_ > 0

    np.testing.assert_array_equal(model.labels_, labels_before)


def test_get_biterms_as_array_matches_list_form():
    """The array fast path must produce the same pairs in the same order."""
    docs = [
        np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([7], dtype=np.int32),
        np.array([2, 2], dtype=np.int32),
    ]

    as_lists = btm.get_biterms(docs, win=4)
    as_arrays = btm.get_biterms(docs, win=4, as_array=True)

    assert [a.tolist() for a in as_arrays] == as_lists


def test_fit_identical_for_list_and_array_biterms():
    X, vocab, docs_vec = vectorize()

    def fit_topics(as_array):
        biterms = btm.get_biterms(docs_vec, as_array=as_array)
        model = btm.BTM(X, vocab, T=3, seed=42)
        model.fit(biterms, iterations=30, verbose=False)
        return model.matrix_topics_words_

    np.testing.assert_array_equal(fit_topics(False), fit_topics(True))


def test_background_topic_is_pinned_to_corpus_distribution():
    """The has_background branch had no test coverage at all."""
    X, vocab, docs_vec = vectorize()
    biterms = btm.get_biterms(docs_vec)

    model = btm.BTM(X, vocab, T=3, seed=42, has_background=True)
    model.fit(biterms, iterations=30, verbose=False)

    background = np.asarray(X.sum(axis=0), dtype=float).ravel() / X.sum()
    np.testing.assert_allclose(model.matrix_topics_words_[0], background)

    p_zd = model.transform(docs_vec, verbose=False)
    np.testing.assert_allclose(p_zd.sum(axis=1), 1.0)
    assert np.all(np.isfinite(model.matrix_topics_words_))
