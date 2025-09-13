"""Sklearn-style API for Biterm Topic Model."""

__all__ = ['BTMClassifier']

from typing import List, Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted

from ._btm import BTM
from ._util import get_words_freqs, get_vectorized_docs, get_biterms


class BTMClassifier(BaseEstimator, TransformerMixin):
    """Sklearn-style Biterm Topic Model classifier.

    This class provides a scikit-learn compatible interface for the Biterm Topic Model,
    making it easy to integrate into existing ML pipelines and use familiar methods
    like fit() and transform().

    Parameters
    ----------
    n_topics : int, default=8
        Number of topics to extract.
    alpha : float, default=None
        Dirichlet prior parameter for topic distribution.
        If None, uses 50/n_topics as recommended.
    beta : float, default=0.01
        Dirichlet prior parameter for word distribution.
    max_iter : int, default=600
        Maximum number of iterations for model training.
    random_state : int, default=None
        Random seed for reproducible results.
    window_size : int, default=15
        Window size for biterm generation.
    has_background : bool, default=False
        Whether to use background topic for frequent words.
    coherence_window : int, default=20
        Number of top words for coherence calculation.
    vectorizer_params : dict, default=None
        Parameters to pass to CountVectorizer for preprocessing.

    Attributes
    ----------
    model_ : BTM
        The fitted BTM model instance.
    vocabulary_ : np.ndarray
        Vocabulary learned from training data.
    feature_names_out_ : np.ndarray
        Alias for vocabulary_ for sklearn compatibility.
    n_features_in_ : int
        Number of features (vocabulary size).
    vectorizer_ : CountVectorizer
        The fitted vectorizer used for preprocessing.

    Examples
    --------
    >>> import bitermplus as btm
    >>> texts = ["machine learning is great", "I love natural language processing"]
    >>> model = btm.BTMClassifier(n_topics=2, random_state=42)
    >>> model.fit(texts)
    >>> doc_topics = model.transform(texts)
    >>> topic_words = model.get_topic_words(n_words=5)
    """

    def __init__(
        self,
        n_topics: int = 8,
        alpha: Optional[float] = None,
        beta: float = 0.01,
        max_iter: int = 600,
        random_state: Optional[int] = None,
        window_size: int = 15,
        has_background: bool = False,
        coherence_window: int = 20,
        vectorizer_params: Optional[Dict[str, Any]] = None
    ):
        self.n_topics = n_topics
        self.beta = beta
        self.max_iter = max_iter
        self.random_state = random_state
        self.window_size = window_size
        self.has_background = has_background
        self.coherence_window = coherence_window
        self.vectorizer_params = vectorizer_params or {}

        # Validate parameters before calculating alpha
        self._validate_params()
        self.alpha = alpha if alpha is not None else 50.0 / n_topics

        # Validate alpha after calculation
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

    def _validate_params(self):
        """Validate model parameters."""
        if self.n_topics <= 0:
            raise ValueError("n_topics must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.coherence_window <= 0:
            raise ValueError("coherence_window must be positive")

    def _setup_vectorizer(self):
        """Initialize the vectorizer with default parameters."""
        default_params = {
            'lowercase': True,
            'token_pattern': r'\b[a-zA-Z][a-zA-Z0-9]*\b',
            'min_df': 2,
            'max_df': 0.95,
            'stop_words': 'english'
        }
        default_params.update(self.vectorizer_params)
        return CountVectorizer(**default_params)

    def fit(self, X: Union[List[str], pd.Series], y=None):
        """Fit the BTM model to documents.

        Parameters
        ----------
        X : array-like of shape (n_documents,)
            Documents to fit the model on. Each element should be a string.
        y : Ignored
            Not used, present for sklearn compatibility.

        Returns
        -------
        self : BTMClassifier
            Returns the instance itself.
        """
        self._validate_params()

        # Convert input to list of strings
        if isinstance(X, pd.Series):
            X = X.tolist()
        elif not isinstance(X, list):
            X = list(X)

        if len(X) == 0:
            raise ValueError("Input documents cannot be empty")

        # Vectorize documents
        self.vectorizer_ = self._setup_vectorizer()
        doc_term_matrix, vocabulary, vocab_dict = get_words_freqs(X, **self.vectorizer_params)

        # Store vocabulary information
        self.vocabulary_ = vocabulary
        self.feature_names_out_ = vocabulary
        self.n_features_in_ = len(vocabulary)

        # Prepare documents and biterms
        docs_vec = get_vectorized_docs(X, vocabulary)
        biterms = get_biterms(docs_vec, win=self.window_size)

        # Adjust coherence window to not exceed vocabulary size
        effective_coherence_window = min(self.coherence_window, len(vocabulary))

        # Initialize and fit BTM model
        self.model_ = BTM(
            doc_term_matrix,
            vocabulary,
            T=self.n_topics,
            M=effective_coherence_window,
            alpha=self.alpha,
            beta=self.beta,
            seed=self.random_state or 0,
            win=self.window_size,
            has_background=self.has_background
        )

        self.model_.fit(biterms, iterations=self.max_iter, verbose=True)

        return self

    def transform(self, X: Union[List[str], pd.Series], infer_type: str = 'sum_b') -> np.ndarray:
        """Transform documents to topic distribution.

        Parameters
        ----------
        X : array-like of shape (n_documents,)
            Documents to transform.
        infer_type : str, default='sum_b'
            Inference method. Options: 'sum_b', 'sum_w', 'mix'.

        Returns
        -------
        doc_topic_matrix : np.ndarray of shape (n_documents, n_topics)
            Document-topic probability matrix.
        """
        check_is_fitted(self, 'model_')

        # Convert input to list of strings
        if isinstance(X, pd.Series):
            X = X.tolist()
        elif not isinstance(X, list):
            X = list(X)

        # Vectorize documents using fitted vocabulary
        docs_vec = get_vectorized_docs(X, self.vocabulary_)

        # Transform using BTM model
        return self.model_.transform(docs_vec, infer_type=infer_type, verbose=False)

    def fit_transform(self, X: Union[List[str], pd.Series], y=None, infer_type: str = 'sum_b') -> np.ndarray:
        """Fit model and transform documents in one step.

        Parameters
        ----------
        X : array-like of shape (n_documents,)
            Documents to fit and transform.
        y : Ignored
            Not used, present for sklearn compatibility.
        infer_type : str, default='sum_b'
            Inference method. Options: 'sum_b', 'sum_w', 'mix'.

        Returns
        -------
        doc_topic_matrix : np.ndarray of shape (n_documents, n_topics)
            Document-topic probability matrix.
        """
        return self.fit(X).transform(X, infer_type=infer_type)

    def get_topic_words(self, topic_id: Optional[int] = None, n_words: int = 10) -> Union[List[str], Dict[int, List[str]]]:
        """Get top words for topics.

        Parameters
        ----------
        topic_id : int, optional
            If provided, return words for this topic only.
            If None, return words for all topics.
        n_words : int, default=10
            Number of top words to return per topic.

        Returns
        -------
        topic_words : list or dict
            If topic_id is provided, returns list of top words for that topic.
            Otherwise, returns dict mapping topic_id to list of words.
        """
        check_is_fitted(self, 'model_')

        topic_word_matrix = self.model_.matrix_topics_words_

        if topic_id is not None:
            if not 0 <= topic_id < self.n_topics:
                raise ValueError(f"topic_id must be between 0 and {self.n_topics - 1}")
            word_indices = np.argsort(topic_word_matrix[topic_id])[-n_words:][::-1]
            return self.vocabulary_[word_indices].tolist()
        else:
            result = {}
            for t in range(self.n_topics):
                word_indices = np.argsort(topic_word_matrix[t])[-n_words:][::-1]
                result[t] = self.vocabulary_[word_indices].tolist()
            return result

    def get_document_topics(self, X: Union[List[str], pd.Series], threshold: float = 0.1) -> List[List[int]]:
        """Get dominant topics for documents.

        Parameters
        ----------
        X : array-like of shape (n_documents,)
            Documents to analyze.
        threshold : float, default=0.1
            Minimum probability threshold for topic assignment.

        Returns
        -------
        doc_topics : list of list of int
            For each document, list of topic IDs above threshold.
        """
        doc_topic_probs = self.transform(X)
        doc_topics = []

        for doc_probs in doc_topic_probs:
            topics = [i for i, prob in enumerate(doc_probs) if prob >= threshold]
            doc_topics.append(topics)

        return doc_topics

    @property
    def coherence_(self) -> np.ndarray:
        """Topic coherence scores."""
        check_is_fitted(self, 'model_')
        return self.model_.coherence_

    @property
    def perplexity_(self) -> float:
        """Model perplexity."""
        check_is_fitted(self, 'model_')
        return self.model_.perplexity_

    @property
    def topic_word_matrix_(self) -> np.ndarray:
        """Topic-word probability matrix."""
        check_is_fitted(self, 'model_')
        return self.model_.matrix_topics_words_

    def score(self, X: Union[List[str], pd.Series], y=None) -> float:
        """Return the mean coherence score.

        Parameters
        ----------
        X : array-like of shape (n_documents,)
            Documents to score.
        y : Ignored
            Not used, present for sklearn compatibility.

        Returns
        -------
        score : float
            Mean coherence score across topics.
        """
        check_is_fitted(self, 'model_')
        return float(np.mean(self.coherence_))