"""Sklearn-style API for Biterm Topic Model."""

__all__ = ["BTMClassifier"]

from typing import List, Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted

from ._btm import BTM
from ._util import get_words_freqs, get_vectorized_docs, get_biterms


class BTMClassifier(BaseEstimator, TransformerMixin):
    """Sklearn-compatible Biterm Topic Model for short text analysis.

    This class provides a scikit-learn compatible interface for the Biterm Topic Model,
    designed specifically for short text analysis such as tweets, reviews, and messages.
    Unlike traditional topic models like LDA, BTM extracts biterms (word pairs) from
    the entire corpus to overcome data sparsity issues in short texts.

    The BTMClassifier automatically handles text preprocessing, vectorization, biterm
    generation, model training, and inference, making topic modeling as simple as
    calling fit() and transform().

    Parameters
    ----------
    n_topics : int, default=8
        Number of topics to extract from the corpus.
    alpha : float, default=None
        Dirichlet prior parameter for topic distribution. Controls topic sparsity
        in documents. Higher values create more uniform topic distributions.
        If None, uses 50/n_topics as recommended in the original paper.
    beta : float, default=0.01
        Dirichlet prior parameter for word distribution within topics. Controls
        topic-word sparsity. Lower values create more focused topics.
    max_iter : int, default=600
        Maximum number of Gibbs sampling iterations for model training.
        More iterations generally improve convergence but increase training time.
    random_state : int, default=None
        Random seed for reproducible results. Set to an integer for consistent
        results across runs.
    window_size : int, default=15
        Window size for biterm generation. Biterms are extracted from word pairs
        within this window distance in each document.
    has_background : bool, default=False
        Whether to use a background topic to model highly frequent words that
        appear across many topics (e.g., stop words).
    coherence_window : int, default=20
        Number of top words used for coherence calculation. This affects the
        semantic coherence metric computation.
    vectorizer_params : dict, default=None
        Additional parameters to pass to the internal CountVectorizer for text
        preprocessing. Common options include min_df, max_df, stop_words, etc.
    epsilon : float, default=1e-10
        Small numerical constant to prevent division by zero and improve
        numerical stability in probability calculations.

    Attributes
    ----------
    model_ : BTM
        The fitted BTM model instance containing learned parameters.
    vocabulary_ : numpy.ndarray
        Vocabulary learned from training data (words corresponding to features).
    feature_names_out_ : numpy.ndarray
        Alias for vocabulary_ for sklearn compatibility.
    n_features_in_ : int
        Number of features (vocabulary size) after preprocessing.
    vectorizer_ : CountVectorizer
        The fitted vectorizer used for text preprocessing.

    Methods
    -------
    fit(X, y=None)
        Fit the BTM model to documents.
    transform(X, infer_type='sum_b')
        Transform documents to topic probability distributions.
    fit_transform(X, y=None, infer_type='sum_b')
        Fit model and transform documents in one step.
    get_topic_words(topic_id=None, n_words=10)
        Get top words for topics.
    get_document_topics(X, threshold=0.1)
        Get dominant topics for documents.
    score(X, y=None)
        Return mean coherence score across topics.

    Examples
    --------
    Basic usage:

    >>> import bitermplus as btm
    >>> texts = [
    ...     "machine learning algorithms are powerful",
    ...     "deep learning neural networks process data",
    ...     "natural language processing understands text"
    ... ]
    >>> model = btm.BTMClassifier(n_topics=2, random_state=42)
    >>> model.fit(texts)
    BTMClassifier(n_topics=2, random_state=42)
    >>> doc_topics = model.transform(texts)
    >>> print(f"Shape: {doc_topics.shape}")
    Shape: (3, 2)

    Getting topic words:

    >>> topic_words = model.get_topic_words(n_words=5)
    >>> for topic_id, words in topic_words.items():
    ...     print(f"Topic {topic_id}: {', '.join(words)}")

    Using with sklearn pipelines:

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import FunctionTransformer
    >>> pipeline = Pipeline([
    ...     ('preprocess', FunctionTransformer(lambda x: [s.lower() for s in x])),
    ...     ('btm', btm.BTMClassifier(n_topics=3, random_state=42))
    ... ])
    >>> topics = pipeline.fit_transform(texts)

    References
    ----------
    Yan, X., Guo, J., Lan, Y., & Cheng, X. (2013). A biterm topic model for
    short texts. In Proceedings of the 22nd international conference on World
    Wide Web (pp. 1445-1456).

    See Also
    --------
    BTM : Low-level BTM implementation
    get_words_freqs : Extract word frequencies from documents
    get_biterms : Generate biterms from vectorized documents
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
        vectorizer_params: Optional[Dict[str, Any]] = None,
        epsilon: float = 1e-10,
    ):
        self.n_topics = n_topics
        self.beta = beta
        self.max_iter = max_iter
        self.random_state = random_state
        self.window_size = window_size
        self.has_background = has_background
        self.coherence_window = coherence_window
        self.vectorizer_params = vectorizer_params or {}
        self.epsilon = epsilon

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
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    def _setup_vectorizer(self):
        """Initialize the vectorizer with default parameters."""
        default_params = {
            "lowercase": True,
            "token_pattern": r"\b[a-zA-Z][a-zA-Z0-9]*\b",
            "min_df": 1,  # Changed from 2 to work with small datasets
            "max_df": 0.95,
            "stop_words": "english",
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

        # Vectorize documents using the configured vectorizer
        self.vectorizer_ = self._setup_vectorizer()
        doc_term_matrix = self.vectorizer_.fit_transform(X)
        vocabulary = np.array(self.vectorizer_.get_feature_names_out())
        vocab_dict = self.vectorizer_.vocabulary_

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
            has_background=self.has_background,
            epsilon=self.epsilon,
        )

        self.model_.fit(biterms, iterations=self.max_iter, verbose=True)

        return self

    def transform(
        self, X: Union[List[str], pd.Series], infer_type: str = "sum_b"
    ) -> np.ndarray:
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
        check_is_fitted(self, "model_")

        # Convert input to list of strings
        if isinstance(X, pd.Series):
            X = X.tolist()
        elif not isinstance(X, list):
            X = list(X)

        # Vectorize documents using fitted vocabulary
        docs_vec = get_vectorized_docs(X, self.vocabulary_)

        # Transform using BTM model
        return self.model_.transform(docs_vec, infer_type=infer_type, verbose=False)

    def fit_transform(
        self, X: Union[List[str], pd.Series], y=None, infer_type: str = "sum_b"
    ) -> np.ndarray:
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

    def get_topic_words(
        self, topic_id: Optional[int] = None, n_words: int = 10
    ) -> Union[List[str], Dict[int, List[str]]]:
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
        check_is_fitted(self, "model_")

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

    def get_document_topics(
        self, X: Union[List[str], pd.Series], threshold: float = 0.1
    ) -> List[List[int]]:
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
        check_is_fitted(self, "model_")
        return self.model_.coherence_

    @property
    def perplexity_(self) -> float:
        """Model perplexity."""
        check_is_fitted(self, "model_")
        return self.model_.perplexity_

    @property
    def topic_word_matrix_(self) -> np.ndarray:
        """Topic-word probability matrix."""
        check_is_fitted(self, "model_")
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
        check_is_fitted(self, "model_")
        return float(np.mean(self.coherence_))

