"""Sklearn-style API for Biterm Topic Model."""

__all__ = ["BTMClassifier"]

from numbers import Integral, Real
from typing import List, Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted

from ._btm import BTM
from ._metrics import coherence, perplexity
from ._util import get_biterms


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
        Window width for biterm generation. The maximum positional offset is
        ``window_size - 1``, matching the reference BTM implementation.
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
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.random_state = random_state
        self.window_size = window_size
        self.has_background = has_background
        self.coherence_window = coherence_window
        self.vectorizer_params = vectorizer_params
        self.epsilon = epsilon

        # Validate parameters before calculating alpha
        self._validate_params()

    def _validate_params(self):
        """Validate model parameters."""
        if isinstance(self.n_topics, bool) or not isinstance(self.n_topics, Real):
            raise TypeError("n_topics must be numeric")
        if not np.isfinite(self.n_topics):
            raise ValueError("n_topics must be finite")
        effective_n_topics = int(self.n_topics)
        if effective_n_topics <= 0:
            raise ValueError("n_topics must be positive")
        if self.alpha is not None:
            if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
                raise TypeError("alpha must be numeric or None")
            if not np.isfinite(self.alpha) or self.alpha <= 0:
                raise ValueError("alpha must be finite and positive")
        for name in ("beta", "epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be numeric")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_iter", "window_size", "coherence_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if self.random_state is not None and (
            isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral)
        ):
            raise TypeError("random_state must be an integer or None")
        if not isinstance(self.has_background, (bool, np.bool_)):
            raise TypeError("has_background must be boolean")
        if self.vectorizer_params is not None and not isinstance(self.vectorizer_params, dict):
            raise TypeError("vectorizer_params must be a dictionary or None")

    @staticmethod
    def _validate_documents(X, allow_empty: bool = False) -> List[str]:
        if isinstance(X, (str, bytes)):
            raise TypeError("X must be a collection of documents, not a string")
        if isinstance(X, pd.Series):
            documents = X.tolist()
        else:
            try:
                documents = list(X)
            except TypeError as exc:
                raise TypeError("X must be an iterable of documents") from exc
        if not documents and not allow_empty:
            raise ValueError("Input documents cannot be empty")
        result = []
        for document in documents:
            if document is None:
                document = ""
            if not isinstance(document, str):
                raise TypeError("documents must contain strings or None")
            result.append(document)
        return result

    def _setup_vectorizer(self):
        """Initialize the vectorizer with default parameters."""
        default_params = {
            "lowercase": True,
            "token_pattern": r"\b[a-zA-Z][a-zA-Z0-9]*\b",
            "min_df": 1,
            "max_df": 0.95,
            "stop_words": "english",
        }
        default_params.update(self.vectorizer_params or {})
        return CountVectorizer(**default_params)

    def _get_vectorized_docs(
        self, X: List[str], vectorizer: Optional[CountVectorizer] = None
    ) -> List[np.ndarray]:
        """Vectorize docs using the fitted vectorizer's own analyzer.

        This ensures tokenization (lowercasing, token pattern, stop words)
        is identical to what CountVectorizer used when building the vocabulary.
        Raw whitespace splitting would silently drop mixed-case words and
        words containing punctuation that the vectorizer would have tokenized
        differently.
        """
        vectorizer = self.vectorizer_ if vectorizer is None else vectorizer
        analyzer = vectorizer.build_analyzer()
        vocab_dict = vectorizer.vocabulary_
        result = []
        for doc in X:
            if doc is None:
                doc = ""
            word_ids = [vocab_dict[w] for w in analyzer(doc) if w in vocab_dict]
            result.append(np.array(word_ids, dtype=np.int32))
        return result

    def fit(self, X: Union[List[str], pd.Series], y=None, verbose: bool = False):
        """Fit the BTM model to documents.

        Parameters
        ----------
        X : array-like of shape (n_documents,)
            Documents to fit the model on. Each element should be a string.
        y : Ignored
            Not used, present for sklearn compatibility.
        verbose : bool, default=False
            Whether to show a progress bar during training.

        Returns
        -------
        self : BTMClassifier
            Returns the instance itself.
        """
        # Re-validate in case params were changed via set_params() after __init__
        self._validate_params()
        effective_n_topics = int(self.n_topics)
        effective_alpha = self.alpha if self.alpha is not None else 50.0 / effective_n_topics

        # Convert input to list of strings
        X = self._validate_documents(X)

        # Vectorize documents using the configured vectorizer
        vectorizer = self._setup_vectorizer()
        doc_term_matrix = vectorizer.fit_transform(X)
        vocabulary = np.array(vectorizer.get_feature_names_out())

        # Prepare documents and biterms using the vectorizer's own analyzer
        # so tokenization (lowercasing, token pattern, stop words) is consistent
        docs_vec = self._get_vectorized_docs(X, vectorizer=vectorizer)
        biterms = get_biterms(docs_vec, win=self.window_size)

        # Adjust coherence window to not exceed vocabulary size
        effective_coherence_window = min(self.coherence_window, len(vocabulary))

        # Initialize and fit BTM model
        model = BTM(
            doc_term_matrix,
            vocabulary,
            T=effective_n_topics,
            M=effective_coherence_window,
            alpha=effective_alpha,
            beta=self.beta,
            seed=self.random_state,
            win=self.window_size,
            has_background=self.has_background,
            epsilon=self.epsilon,
        )

        model.fit(biterms, iterations=self.max_iter, verbose=verbose)
        self.vectorizer_ = vectorizer
        self.vocabulary_ = vocabulary
        self.feature_names_out_ = vocabulary
        self.n_features_in_ = len(vocabulary)
        self.model_ = model
        self.training_docs_vec_ = docs_vec
        self.doc_term_matrix_ = doc_term_matrix

        return self

    def transform(self, X: Union[List[str], pd.Series], infer_type: str = "sum_b") -> np.ndarray:
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
        X = self._validate_documents(X, allow_empty=True)

        # Vectorize documents using the fitted vectorizer's analyzer
        docs_vec = self._get_vectorized_docs(X)

        # Transform using BTM model
        return self.model_.transform(docs_vec, infer_type=infer_type, verbose=False)

    def fit_transform(
        self,
        X: Union[List[str], pd.Series],
        y=None,
        infer_type: str = "sum_b",
        verbose: bool = False,
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
        verbose : bool, default=False
            Whether to show a progress bar during training.

        Returns
        -------
        doc_topic_matrix : np.ndarray of shape (n_documents, n_topics)
            Document-topic probability matrix.
        """
        return self.fit(X, y=y, verbose=verbose).transform(X, infer_type=infer_type)

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
        if isinstance(n_words, bool) or not isinstance(n_words, Integral) or n_words <= 0:
            raise ValueError("n_words must be a positive integer")

        topic_word_matrix = self.model_.matrix_topics_words_

        if topic_id is not None:
            if isinstance(topic_id, bool) or not isinstance(topic_id, Integral):
                raise TypeError("topic_id must be an integer")
            topics_num = self.model_.topics_num_
            if not 0 <= topic_id < topics_num:
                raise ValueError(f"topic_id must be between 0 and {topics_num - 1}")
            word_indices = np.argsort(topic_word_matrix[topic_id])[-n_words:][::-1]
            return self.vocabulary_[word_indices].tolist()
        else:
            result = {}
            for t in range(self.model_.topics_num_):
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
        if not isinstance(threshold, Real) or not np.isfinite(threshold):
            raise TypeError("threshold must be a finite number")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
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
        training_topics = self.model_.transform(self.training_docs_vec_, verbose=False)
        return perplexity(
            self.model_.matrix_topics_words_,
            training_topics,
            self.doc_term_matrix_,
            self.model_.topics_num_,
        )

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
        X = self._validate_documents(X)
        counts = self.vectorizer_.transform(X)
        scores = coherence(
            self.model_.matrix_topics_words_,
            counts,
            M=min(self.coherence_window, self.model_.vocabulary_size_),
        )
        return float(np.mean(scores))
