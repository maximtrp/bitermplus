"""Tests for BTMClassifier sklearn-style API and dtype compatibility."""
import unittest
import numpy as np
import pandas as pd

try:
    from src import bitermplus as btm
except ImportError:
    import bitermplus as btm


class TestBTMClassifier(unittest.TestCase):
    """Test BTMClassifier API including dtype compatibility."""

    @classmethod
    def setUpClass(cls):
        """Set up test data used by all test methods."""
        # Create simple test documents
        cls.texts = [
            "machine learning models are powerful",
            "deep learning neural networks",
            "artificial intelligence methods",
            "machine learning algorithms",
            "data science techniques",
            "neural networks training",
            "deep learning frameworks",
            "machine learning systems",
        ]

        cls.n_topics = 3
        cls.max_iter = 20
        cls.random_state = 12345

    def test_btm_classifier_fit(self):
        """Test BTMClassifier fit method."""
        model = btm.BTMClassifier(
            n_topics=self.n_topics,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        model.fit(self.texts)

        # Check that model was fitted
        self.assertTrue(hasattr(model, "model_"))
        self.assertTrue(hasattr(model, "vocabulary_"))
        self.assertGreater(len(model.vocabulary_), 0)

    def test_btm_classifier_transform(self):
        """Test BTMClassifier transform method."""
        model = btm.BTMClassifier(
            n_topics=self.n_topics,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        model.fit(self.texts)

        # Transform documents
        doc_topics = model.transform(self.texts)

        # Check output shape and values
        self.assertEqual(doc_topics.shape, (len(self.texts), self.n_topics))
        self.assertTrue(np.all(doc_topics >= 0))
        self.assertTrue(np.all(doc_topics <= 1))
        # Check that each row sums to approximately 1
        row_sums = doc_topics.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(self.texts)))

    def test_btm_classifier_coherence_property(self):
        """Test BTMClassifier coherence_ property for dtype compatibility.

        This test specifically checks for the buffer dtype mismatch error
        that occurred on Windows with 'long' vs 'long long' types.
        """
        model = btm.BTMClassifier(
            n_topics=self.n_topics,
            coherence_window=5,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        model.fit(self.texts)

        # This should not raise ValueError: Buffer dtype mismatch
        coherence_scores = model.coherence_

        # Check coherence scores
        self.assertIsInstance(coherence_scores, np.ndarray)
        self.assertEqual(len(coherence_scores), self.n_topics)
        self.assertTrue(np.all(np.isfinite(coherence_scores)))

    def test_btm_classifier_perplexity_property(self):
        """Test BTMClassifier perplexity_ property for dtype compatibility.

        This test specifically checks for the buffer dtype mismatch error
        that occurred on Windows with 'long' vs 'long long' types.
        """
        model = btm.BTMClassifier(
            n_topics=self.n_topics,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        model.fit(self.texts)

        # Must call transform before perplexity can be calculated
        _ = model.transform(self.texts[:5])

        # This should not raise ValueError: Buffer dtype mismatch
        perplexity = model.perplexity_

        # Check perplexity
        self.assertIsInstance(perplexity, (float, np.floating))
        self.assertGreater(perplexity, 0)
        self.assertTrue(np.isfinite(perplexity))

    def test_btm_classifier_score_method(self):
        """Test BTMClassifier score method for dtype compatibility.

        This test specifically checks for the buffer dtype mismatch error
        that occurred on Windows with 'long' vs 'long long' types.
        The score() method internally calls coherence_.
        """
        model = btm.BTMClassifier(
            n_topics=self.n_topics,
            coherence_window=5,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        model.fit(self.texts)

        # This should not raise ValueError: Buffer dtype mismatch
        mean_coherence = model.score(self.texts)

        # Check score
        self.assertIsInstance(mean_coherence, (float, np.floating))
        self.assertTrue(np.isfinite(mean_coherence))

    def test_btm_classifier_with_pandas_series(self):
        """Test BTMClassifier with pandas Series input."""
        texts_series = pd.Series(self.texts)

        model = btm.BTMClassifier(
            n_topics=self.n_topics,
            coherence_window=5,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        model.fit(texts_series)

        # Test all methods that had dtype issues
        coherence_scores = model.coherence_
        self.assertEqual(len(coherence_scores), self.n_topics)

        _ = model.transform(texts_series[:5])
        perplexity = model.perplexity_
        self.assertGreater(perplexity, 0)

        mean_coherence = model.score(texts_series)
        self.assertTrue(np.isfinite(mean_coherence))

    def test_btm_low_level_api_coherence(self):
        """Test low-level BTM API coherence for dtype compatibility."""
        # Use low-level API
        X, vocabulary, vocab_dict = btm.get_words_freqs(self.texts)
        docs_vec = btm.get_vectorized_docs(self.texts, vocabulary)
        biterms = btm.get_biterms(docs_vec)

        model = btm.BTM(
            X,
            vocabulary,
            seed=self.random_state,
            T=self.n_topics,
            M=5,
            alpha=50 / self.n_topics,
            beta=0.01,
        )
        model.fit(biterms, iterations=self.max_iter)

        # This should not raise ValueError: Buffer dtype mismatch
        coherence_scores = model.coherence_
        self.assertEqual(len(coherence_scores), self.n_topics)

    def test_btm_low_level_api_perplexity(self):
        """Test low-level BTM API perplexity for dtype compatibility."""
        # Use low-level API
        X, vocabulary, vocab_dict = btm.get_words_freqs(self.texts)
        docs_vec = btm.get_vectorized_docs(self.texts, vocabulary)
        biterms = btm.get_biterms(docs_vec)

        model = btm.BTM(
            X,
            vocabulary,
            seed=self.random_state,
            T=self.n_topics,
            M=5,
            alpha=50 / self.n_topics,
            beta=0.01,
        )
        model.fit(biterms, iterations=self.max_iter)

        # Transform for perplexity calculation
        p_zd = model.transform(docs_vec[:5])

        # This should not raise ValueError: Buffer dtype mismatch
        perplexity = model.perplexity_
        self.assertGreater(perplexity, 0)

    def test_sparse_matrix_dtype_compatibility(self):
        """Test that both int32 and int64 sparse matrices work.

        This specifically tests for the Windows dtype issue where
        scipy sparse matrices can have different index dtypes.
        """
        from scipy.sparse import csr_matrix

        X, vocabulary, vocab_dict = btm.get_words_freqs(self.texts)
        docs_vec = btm.get_vectorized_docs(self.texts, vocabulary)
        biterms = btm.get_biterms(docs_vec)

        # Test with int32 indices (common on some platforms)
        X_int32 = csr_matrix((
            X.data.astype(np.int64),
            X.indices.astype(np.int32),
            X.indptr.astype(np.int32)
        ), shape=X.shape)

        model_int32 = btm.BTM(
            X_int32,
            vocabulary,
            seed=self.random_state,
            T=self.n_topics,
            M=5,
            alpha=50 / self.n_topics,
            beta=0.01,
        )
        model_int32.fit(biterms, iterations=self.max_iter)
        coherence_int32 = model_int32.coherence_
        self.assertEqual(len(coherence_int32), self.n_topics)

        # Test with int64 indices (common on other platforms)
        X_int64 = csr_matrix((
            X.data.astype(np.int64),
            X.indices.astype(np.int64),
            X.indptr.astype(np.int64)
        ), shape=X.shape)

        model_int64 = btm.BTM(
            X_int64,
            vocabulary,
            seed=self.random_state,
            T=self.n_topics,
            M=5,
            alpha=50 / self.n_topics,
            beta=0.01,
        )
        model_int64.fit(biterms, iterations=self.max_iter)
        coherence_int64 = model_int64.coherence_
        self.assertEqual(len(coherence_int64), self.n_topics)


if __name__ == "__main__":
    unittest.main()
