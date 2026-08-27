__all__ = [
    "get_biterms",
    "get_docs_top_topic",
    "get_top_topic_docs",
    "get_top_topic_words",
    "get_vectorized_docs",
    "get_words_freqs",
]

from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
from pandas import DataFrame, Series, concat
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from ._btm import BTM


def get_words_freqs(
    docs: Union[list[str], np.ndarray, Series], **kwargs: dict
) -> tuple[csr_matrix, np.ndarray, dict]:
    """Extract word frequencies and vocabulary from text documents.

    This function vectorizes a collection of text documents into a sparse matrix
    representation suitable for topic modeling. It uses scikit-learn's CountVectorizer
    to tokenize, count, and filter words, creating a document-term matrix.

    Parameters
    ----------
    docs : list of str, numpy.ndarray, or pandas.Series
        Collection of text documents to vectorize. Each element should be a string
        containing the text content of one document.
    **kwargs : dict
        Additional keyword arguments passed to CountVectorizer. Common options include:

        - min_df : int or float, minimum document frequency
        - max_df : int or float, maximum document frequency
        - stop_words : str or list, stop words to remove
        - lowercase : bool, whether to convert to lowercase
        - token_pattern : str, regex pattern for tokenization

    Returns
    -------
    doc_term_matrix : scipy.sparse.csr_matrix, shape (n_documents, n_features)
        Sparse matrix where element (i,j) represents the count of term j in document i.
    vocabulary : numpy.ndarray, shape (n_features,)
        Array of feature names (words) corresponding to the matrix columns.
    vocab_dict : dict
        Dictionary mapping terms to their column indices in the matrix.

    Examples
    --------
    Basic usage:

    >>> import bitermplus as btm
    >>> texts = ["machine learning is great", "I love natural language processing"]
    >>> X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
    >>> print(f"Matrix shape: {X.shape}")
    >>> print(f"Vocabulary size: {len(vocabulary)}")

    With custom parameters:

    >>> X, vocab, vocab_dict = btm.get_words_freqs(
    ...     texts, min_df=1, stop_words='english', lowercase=True
    ... )

    Notes
    -----
    This function is primarily used internally by BTMClassifier, but can be useful
    for manual preprocessing when using the low-level BTM class directly.

    See Also
    --------
    get_vectorized_docs : Convert documents to word ID representation
    get_biterms : Generate biterms from vectorized documents
    sklearn.feature_extraction.text.CountVectorizer : Underlying vectorization method
    """
    vec = CountVectorizer(**kwargs)
    X = vec.fit_transform(docs)
    words = np.array(vec.get_feature_names_out())
    return X, words, vec.vocabulary_


def get_vectorized_docs(
    docs: Union[list[str], np.ndarray],
    vocab: Union[list[str], np.ndarray],
    analyzer=None,
) -> list[np.ndarray]:
    """Convert text documents to vectorized representation using word IDs.

    This function transforms raw text documents into a numerical representation
    where each word is replaced by its corresponding index in the vocabulary.
    This is a preprocessing step required before biterm generation and BTM training.

    Parameters
    ----------
    docs : list of str or numpy.ndarray
        Collection of text documents. Each document should be a string.
    vocab : list of str or numpy.ndarray
        Vocabulary array containing all unique terms. Typically obtained from
        get_words_freqs() function.
    analyzer : callable, optional
        Tokenizer to split each document. Defaults to lowercase word splitting
        consistent with get_words_freqs().

    Returns
    -------
    vectorized_docs : list of numpy.ndarray
        List of vectorized documents. Each document is represented as a numpy
        array of word IDs (integers) corresponding to vocabulary indices.
        Words not in the vocabulary are filtered out.

    Examples
    --------
    Basic usage:

    >>> import bitermplus as btm
    >>> texts = ["machine learning is great", "I love deep learning"]
    >>> X, vocabulary, _ = btm.get_words_freqs(texts)
    >>> docs_vec = btm.get_vectorized_docs(texts, vocabulary)
    >>> print(f"Original: {texts[0]}")
    >>> print(f"Vectorized: {docs_vec[0]}")

    Complete preprocessing pipeline:

    >>> texts = ["AI and ML are exciting", "Deep learning transforms data"]
    >>> X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
    >>> docs_vectorized = btm.get_vectorized_docs(texts, vocabulary)
    >>> biterms = btm.get_biterms(docs_vectorized)

    Notes
    -----
    - By default, documents use CountVectorizer's standard analyzer
    - Empty strings and None values are handled gracefully
    - This function is automatically called by BTMClassifier but useful for manual preprocessing

    See Also
    --------
    get_words_freqs : Extract vocabulary and document-term matrix
    get_biterms : Generate biterms from vectorized documents
    BTMClassifier : High-level interface that handles preprocessing automatically
    """
    if len(set(vocab)) != len(vocab):
        raise ValueError("vocab must not contain duplicate terms")
    vocab_idx = {word: idx for idx, word in enumerate(vocab)}
    if analyzer is None:
        analyzer = CountVectorizer(vocabulary=vocab_idx).build_analyzer()

    result = []
    for doc in docs:
        if doc is None:
            doc = ""
        if not isinstance(doc, str):
            raise TypeError("documents must contain strings or None")
        words = analyzer(doc)
        word_ids = [vocab_idx[word] for word in words if word in vocab_idx]
        result.append(np.array(word_ids, dtype=np.int32))
    return result


def _doc_biterms(doc: np.ndarray, win: int) -> np.ndarray:
    """Build one document's biterms as an (n, 2) int32 array.

    Pairs are emitted in position-major order -- every pair starting at
    position 0, then position 1, and so on -- which is the order the Gibbs
    sampler assigns its initial random topics in, so it must not change.
    """
    doc_len = doc.shape[0]
    if doc_len < 2:
        return np.empty((0, 2), dtype=np.int32)

    # counts[i] = number of partners position i has inside the window
    starts_idx = np.arange(doc_len - 1)
    counts = np.minimum(win - 1, doc_len - 1 - starts_idx)
    total = int(counts.sum())

    # Ragged arange: repeat each start position, then number the offsets 1..counts[i]
    i_idx = np.repeat(starts_idx, counts)
    group_starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    offsets = np.arange(total) - np.repeat(group_starts, counts) + 1
    j_idx = i_idx + offsets

    left = doc[i_idx]
    right = doc[j_idx]
    return np.stack(
        (np.minimum(left, right), np.maximum(left, right)), axis=1
    ).astype(np.int32, copy=False)


def get_biterms(
    docs: list[np.ndarray], win: int = 15, as_array: bool = False
) -> list[list[list[int]]]:
    """Generate biterms (word pairs) from vectorized documents.

    Biterms are word co-occurrence pairs that capture local word associations
    within a specified window. This is the core data structure used by BTM
    to model topics in short texts. Unlike traditional topic models that work
    with individual documents, BTM aggregates biterms across the entire corpus.

    Parameters
    ----------
    docs : list of numpy.ndarray
        List of vectorized documents where each document is a numpy array
        of word IDs. Typically obtained from get_vectorized_docs() function.
    win : int, default=15
        Window width for biterm extraction, matching the reference BTM. The
        maximum positional offset is ``win - 1``.
    as_array : bool, default=False
        Return each document's biterms as an (n, 2) int32 numpy array instead
        of a nested list. Much cheaper in time and memory for large corpora;
        :meth:`bitermplus.BTM.fit` accepts either form.

    Returns
    -------
    biterms : list of list of list
        Nested list structure where biterms[i] contains all biterms for document i.
        Each biterm is represented as [word_id1, word_id2] where word_id1 <= word_id2.
        With ``as_array=True`` each element is an (n, 2) int32 array instead.

    Raises
    ------
    ValueError
        If no biterms can be generated from the input documents (e.g., all
        documents are too short or vocabulary overlap is insufficient).

    Examples
    --------
    Basic usage:

    >>> import bitermplus as btm
    >>> texts = ["machine learning algorithms", "deep learning networks"]
    >>> X, vocabulary, _ = btm.get_words_freqs(texts)
    >>> docs_vec = btm.get_vectorized_docs(texts, vocabulary)
    >>> biterms = btm.get_biterms(docs_vec)
    >>> print(f"Number of documents: {len(biterms)}")
    >>> print(f"Biterms in first doc: {biterms[0]}")

    With custom window size:

    >>> biterms = btm.get_biterms(docs_vec, win=10)

    Complete preprocessing pipeline:

    >>> texts = ["AI and machine learning", "Natural language processing"]
    >>> X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
    >>> docs_vec = btm.get_vectorized_docs(texts, vocabulary)
    >>> biterms = btm.get_biterms(docs_vec, win=15)
    >>> # Now ready for BTM training
    >>> model = btm.BTM(X, vocabulary, T=2)
    >>> model.fit(biterms)

    Notes
    -----
    - Documents with fewer than 2 words retain an empty biterm list
    - Biterms are ordered such that the smaller word ID comes first
    - The function validates that at least some biterms are generated
    - Window size should be chosen based on document length and desired dependencies

    See Also
    --------
    get_vectorized_docs : Convert documents to word ID representation
    BTM.fit : Fit BTM model using generated biterms
    BTMClassifier : High-level interface that handles biterm generation automatically
    """
    if isinstance(win, bool) or not isinstance(win, (int, np.integer)) or win < 2:
        raise ValueError("win must be an integer of at least 2")

    biterms = []
    total_biterms = 0
    for doc in docs:
        doc = np.asarray(doc)
        if doc.ndim != 1:
            raise ValueError("each document must be one-dimensional")
        if not np.issubdtype(doc.dtype, np.integer):
            raise TypeError("document word IDs must be integers")
        if np.any(doc < 0) or np.any(doc > np.iinfo(np.int32).max):
            raise ValueError("document word IDs must be non-negative int32 values")
        pairs = _doc_biterms(doc, win)
        total_biterms += pairs.shape[0]
        biterms.append(pairs if as_array else pairs.tolist())

    # Check if we have any biterms at all
    if total_biterms == 0:
        raise ValueError(
            "No biterms could be generated from the documents. "
            "Documents may be too short or have insufficient vocabulary overlap."
        )

    return biterms


def get_top_topic_words(
    model: BTM, words_num: int = 20, topics_idx: Optional[Sequence[Any]] = None
) -> DataFrame:
    """Select top topic words from a fitted model.

    Parameters
    ----------
    model : bitermplus._btm.BTM
        Fitted BTM model.
    words_num : int = 20
        The number of words to select.
    topics_idx : Sequence[Any], optional
        Topics indices. Meant to be used to select only stable
        topics.

    Returns
    -------
    DataFrame
        Words with highest probabilities per each selected topic.

    Example
    -------
    >>> stable_topics = [0, 3, 10, 12, 18, 21]
    >>> top_words = btm.get_top_topic_words(
    ...     model,
    ...     words_num=100,
    ...     topics_idx=stable_topics)
    """

    def _select_words(model, topic_id: int):
        probs = model.matrix_topics_words_[topic_id, :]
        idx = np.argsort(probs)[: -words_num - 1 : -1]
        result = Series(model.vocabulary_[idx])
        result.name = f"topic{topic_id}"
        return result

    if isinstance(words_num, bool) or not isinstance(words_num, (int, np.integer)) \
            or words_num <= 0:
        raise ValueError("words_num must be a positive integer")

    topics_num = model.topics_num_
    topics_idx = np.arange(topics_num) if topics_idx is None else list(topics_idx)
    if len(topics_idx) == 0:
        return DataFrame()
    return concat([_select_words(model, x) for x in topics_idx], axis=1)


def get_top_topic_docs(
    docs: Sequence[Any],
    p_zd: np.ndarray,
    docs_num: int = 20,
    topics_idx: Optional[Sequence[Any]] = None,
) -> DataFrame:
    """Select top topic docs from a fitted model.

    Parameters
    ----------
    docs : Sequence[Any]
        Iterable of documents (e.g. list of strings).
    p_zd : np.ndarray
        Documents vs topics probabilities matrix.
    docs_num : int = 20
        The number of documents to select.
    topics_idx : Sequence[Any] = None
        Topics indices. Meant to be used to select only stable
        topics.

    Returns
    -------
    DataFrame
        Documents with highest probabilities in all selected topics.

    Example
    -------
    >>> top_docs = btm.get_top_topic_docs(
    ...     texts,
    ...     p_zd,
    ...     docs_num=100,
    ...     topics_idx=[1,2,3,4])
    """

    def _select_docs(docs, p_zd, topic_id: int):
        probs = p_zd[:, topic_id]
        idx = np.argsort(probs)[: -docs_num - 1 : -1]
        result = Series(np.asarray(docs)[idx])
        result.name = f"topic{topic_id}"
        return result

    if isinstance(docs_num, bool) or not isinstance(docs_num, (int, np.integer)) \
            or docs_num <= 0:
        raise ValueError("docs_num must be a positive integer")

    topics_num = p_zd.shape[1]
    topics_idx = np.arange(topics_num) if topics_idx is None else list(topics_idx)
    if len(topics_idx) == 0:
        return DataFrame()
    return concat([_select_docs(docs, p_zd, x) for x in topics_idx], axis=1)


def get_docs_top_topic(docs: Sequence[Any], p_zd: np.ndarray) -> DataFrame:
    """Select most probable topic for each document.

    Parameters
    ----------
    docs : Sequence[Any]
        Iterable of documents (e.g. list of strings).
    p_zd : np.ndarray
        Documents vs topics probabilities matrix.

    Returns
    -------
    DataFrame
        Documents and the most probable topic for each of them.

    Example
    -------
    >>> import bitermplus as btm
    >>> # Read documents from file
    >>> # texts = ...
    >>> # Build and train a model
    >>> # model = ...
    >>> # model.fit(...)
    >>> btm.get_docs_top_topic(texts, model.matrix_docs_topics_)
    """
    return DataFrame({"documents": docs, "label": p_zd.argmax(axis=1)})
