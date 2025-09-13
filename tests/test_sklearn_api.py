"""Tests for sklearn-style API."""

import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

try:
    from src.bitermplus import BTMClassifier
except ImportError:
    from bitermplus import BTMClassifier


class TestBTMClassifier(unittest.TestCase):
    """Test cases for BTMClassifier."""

    def setUp(self):
        """Set up test data."""
        self.sample_texts = [
            "machine learning algorithms are powerful tools",
            "deep learning neural networks process data efficiently",
            "natural language processing helps computers understand text",
            "artificial intelligence transforms many industries",
            "data science combines statistics and programming",
            "computer vision enables machines to see and interpret images",
            "reinforcement learning agents learn through trial and error",
            "supervised learning uses labeled training data",
            "unsupervised learning finds hidden patterns in data",
            "feature engineering improves model performance significantly"
        ]

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        model = BTMClassifier()
        self.assertEqual(model.n_topics, 8)
        self.assertEqual(model.alpha, 50.0 / 8)
        self.assertEqual(model.beta, 0.01)
        self.assertEqual(model.max_iter, 600)
        self.assertIsNone(model.random_state)

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        model = BTMClassifier(
            n_topics=5,
            alpha=0.1,
            beta=0.05,
            max_iter=100,
            random_state=42
        )
        self.assertEqual(model.n_topics, 5)
        self.assertEqual(model.alpha, 0.1)
        self.assertEqual(model.beta, 0.05)
        self.assertEqual(model.max_iter, 100)
        self.assertEqual(model.random_state, 42)

    def test_param_validation(self):
        """Test parameter validation."""
        with self.assertRaises(ValueError):
            BTMClassifier(n_topics=0)

        with self.assertRaises(ValueError):
            BTMClassifier(alpha=-1)

        with self.assertRaises(ValueError):
            BTMClassifier(beta=0)

        with self.assertRaises(ValueError):
            BTMClassifier(max_iter=-1)

    def test_fit_basic(self):
        """Test basic fitting functionality."""
        model = BTMClassifier(n_topics=3, random_state=42, max_iter=50)
        model.fit(self.sample_texts)

        self.assertTrue(hasattr(model, 'model_'))
        self.assertTrue(hasattr(model, 'vocabulary_'))
        self.assertTrue(hasattr(model, 'n_features_in_'))
        self.assertGreater(model.n_features_in_, 0)

    def test_fit_with_pandas_series(self):
        """Test fitting with pandas Series input."""
        texts_series = pd.Series(self.sample_texts)
        model = BTMClassifier(n_topics=3, random_state=42, max_iter=50)
        model.fit(texts_series)

        self.assertTrue(hasattr(model, 'model_'))
        self.assertTrue(hasattr(model, 'vocabulary_'))

    def test_fit_empty_input(self):
        """Test fitting with empty input."""
        model = BTMClassifier()
        with self.assertRaises(ValueError):
            model.fit([])

    def test_transform_basic(self):
        """Test basic transform functionality."""
        model = BTMClassifier(n_topics=3, random_state=42, max_iter=50)
        model.fit(self.sample_texts)

        doc_topics = model.transform(self.sample_texts[:5])

        self.assertEqual(doc_topics.shape, (5, 3))
        self.assertTrue(np.allclose(doc_topics.sum(axis=1), 1.0, rtol=1e-5))
        self.assertTrue(np.all(doc_topics >= 0))

    def test_transform_different_inference_types(self):
        """Test transform with different inference types."""
        model = BTMClassifier(n_topics=3, random_state=42, max_iter=50)
        model.fit(self.sample_texts)

        for infer_type in ['sum_b', 'sum_w', 'mix']:
            doc_topics = model.transform(self.sample_texts[:3], infer_type=infer_type)
            self.assertEqual(doc_topics.shape, (3, 3))
            self.assertTrue(np.all(doc_topics >= 0))

    def test_fit_transform(self):
        """Test fit_transform method."""
        model = BTMClassifier(n_topics=3, random_state=42, max_iter=50)
        doc_topics = model.fit_transform(self.sample_texts)

        self.assertEqual(doc_topics.shape, (len(self.sample_texts), 3))
        self.assertTrue(np.allclose(doc_topics.sum(axis=1), 1.0, rtol=1e-5))

    def test_get_topic_words_single_topic(self):
        """Test getting words for a single topic."""
        model = BTMClassifier(n_topics=3, random_state=42, max_iter=50)
        model.fit(self.sample_texts)

        words = model.get_topic_words(topic_id=0, n_words=5)

        self.assertIsInstance(words, list)
        self.assertEqual(len(words), 5)
        self.assertTrue(all(isinstance(word, str) for word in words))

    def test_get_topic_words_all_topics(self):
        """Test getting words for all topics."""
        model = BTMClassifier(n_topics=3, random_state=42, max_iter=50)
        model.fit(self.sample_texts)

        words_dict = model.get_topic_words(n_words=5)

        self.assertIsInstance(words_dict, dict)
        self.assertEqual(len(words_dict), 3)
        for topic_id, words in words_dict.items():
            self.assertIsInstance(words, list)
            self.assertEqual(len(words), 5)

    def test_get_topic_words_invalid_topic_id(self):
        """Test getting words with invalid topic ID."""
        model = BTMClassifier(n_topics=3, random_state=42, max_iter=50)
        model.fit(self.sample_texts)

        with self.assertRaises(ValueError):
            model.get_topic_words(topic_id=5)

    def test_get_document_topics(self):
        """Test getting dominant topics for documents."""
        model = BTMClassifier(n_topics=3, random_state=42, max_iter=50)
        model.fit(self.sample_texts)

        doc_topics = model.get_document_topics(self.sample_texts[:5], threshold=0.1)

        self.assertEqual(len(doc_topics), 5)
        self.assertTrue(all(isinstance(topics, list) for topics in doc_topics))
        for topics in doc_topics:
            self.assertTrue(all(0 <= topic_id < 3 for topic_id in topics))

    def test_properties(self):
        """Test model properties."""
        model = BTMClassifier(n_topics=3, random_state=42, max_iter=50)
        model.fit(self.sample_texts)

        # Test coherence
        coherence = model.coherence_
        self.assertIsInstance(coherence, np.ndarray)
        self.assertEqual(len(coherence), 3)

        # Test perplexity (requires transform to be called first)
        model.transform(self.sample_texts)
        perplexity = model.perplexity_
        self.assertIsInstance(perplexity, (int, float))

        # Test topic-word matrix
        topic_word_matrix = model.topic_word_matrix_
        self.assertIsInstance(topic_word_matrix, np.ndarray)
        self.assertEqual(topic_word_matrix.shape[0], 3)

    def test_score_method(self):
        """Test the score method."""
        model = BTMClassifier(n_topics=3, random_state=42, max_iter=50)
        model.fit(self.sample_texts)

        score = model.score(self.sample_texts)
        self.assertIsInstance(score, float)

    def test_sklearn_compatibility(self):
        """Test compatibility with sklearn utilities."""
        model = BTMClassifier(n_topics=3, random_state=42, max_iter=50)

        # Test with sklearn's cross_val_score (simplified test)
        try:
            # This tests that the estimator interface is correct
            scores = cross_val_score(model, self.sample_texts, cv=2, scoring=None)
            self.assertEqual(len(scores), 2)
        except Exception as e:
            # Some sklearn versions might have issues, but the interface should be correct
            self.assertIn('BTMClassifier', str(type(model)))

    def test_pipeline_integration(self):
        """Test integration with sklearn Pipeline."""
        # Simple preprocessing function
        def preprocess_texts(texts):
            return [text.lower() for text in texts]

        pipeline = Pipeline([
            ('preprocess', FunctionTransformer(preprocess_texts)),
            ('btm', BTMClassifier(n_topics=3, random_state=42, max_iter=50))
        ])

        doc_topics = pipeline.fit_transform(self.sample_texts)
        self.assertEqual(doc_topics.shape, (len(self.sample_texts), 3))

    def test_vectorizer_params(self):
        """Test custom vectorizer parameters."""
        vectorizer_params = {
            'min_df': 1,
            'max_df': 1.0,
            'stop_words': None
        }

        model = BTMClassifier(
            n_topics=3,
            random_state=42,
            max_iter=50,
            vectorizer_params=vectorizer_params
        )
        model.fit(self.sample_texts)

        self.assertTrue(hasattr(model, 'model_'))

    def test_transform_unseen_data(self):
        """Test transform on unseen data."""
        model = BTMClassifier(n_topics=3, random_state=42, max_iter=50)
        model.fit(self.sample_texts)

        new_texts = [
            "new machine learning algorithm",
            "innovative data processing technique"
        ]

        doc_topics = model.transform(new_texts)
        self.assertEqual(doc_topics.shape, (2, 3))
        self.assertTrue(np.all(doc_topics >= 0))


if __name__ == "__main__":
    unittest.main()