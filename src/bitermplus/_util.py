__all__ = [
    'get_words_freqs', 'get_vectorized_docs',
    'get_biterms', 'get_top_topic_words',
    'get_top_topic_docs', 'get_docs_top_topic']

from typing import List, Union, Tuple, Dict, Sequence, Any
from scipy.sparse import csr_matrix
from pandas import DataFrame, Series, concat
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from ._btm import BTM


def get_words_freqs(
        docs: Union[List[str], np.ndarray, Series],
        **kwargs: dict) -> Tuple[csr_matrix, np.ndarray, Dict]:
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
        docs: Union[List[str],  np.ndarray],
        vocab: Union[List[str], np.ndarray]) -> List[np.ndarray]:
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
    - Documents are split on whitespace and filtered to include only known vocabulary
    - Empty strings and None values are handled gracefully
    - This function is automatically called by BTMClassifier but useful for manual preprocessing

    See Also
    --------
    get_words_freqs : Extract vocabulary and document-term matrix
    get_biterms : Generate biterms from vectorized documents
    BTMClassifier : High-level interface that handles preprocessing automatically
    """
    vocab_idx = {word: idx for idx, word in enumerate(vocab)}

    def _parse_words(w):
        return vocab_idx.get(w)

    result = []
    for doc in docs:
        # Handle potential None/empty doc and filter out empty strings
        if doc is None:
            doc = ""
        words = [word.strip() for word in doc.split() if word.strip()]
        word_ids = [vocab_idx[word] for word in words if word in vocab_idx]
        result.append(np.array(word_ids, dtype=np.int32))
    return result


def get_biterms(
        docs: List[np.ndarray],
        win: int = 15) -> List[List[int]]:
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
        Window size for biterm extraction. Biterms are created from all word
        pairs within this distance in each document. Larger windows capture
        more long-range dependencies but may introduce noise.

    Returns
    -------
    biterms : list of list of list
        Nested list structure where biterms[i] contains all biterms for document i.
        Each biterm is represented as [word_id1, word_id2] where word_id1 <= word_id2.

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
    - Documents with fewer than 2 words produce no biterms and are skipped
    - Biterms are ordered such that the smaller word ID comes first
    - The function validates that at least some biterms are generated
    - Window size should be chosen based on document length and desired dependencies

    See Also
    --------
    get_vectorized_docs : Convert documents to word ID representation
    BTM.fit : Fit BTM model using generated biterms
    BTMClassifier : High-level interface that handles biterm generation automatically
    """
    biterms = []
    for doc in docs:
        doc_biterms = []
        doc_len = len(doc)
        if doc_len < 2:
            continue
        for i in range(doc_len-1):
            for j in range(i+1, min(i + win, doc_len)):
                wi = min(doc[i], doc[j])
                wj = max(doc[i], doc[j])
                doc_biterms.append([wi, wj])
        biterms.append(doc_biterms)

    # Check if we have any biterms at all
    total_biterms = sum(len(doc_biterms) for doc_biterms in biterms)
    if total_biterms == 0:
        raise ValueError("No biterms could be generated from the documents. "
                        "Documents may be too short or have insufficient vocabulary overlap.")

    return biterms


def get_top_topic_words(
        model: BTM,
        words_num: int = 20,
        topics_idx: Sequence[Any] = None) -> DataFrame:
    """Select top topic words from a fitted model.

    Parameters
    ----------
    model : bitermplus._btm.BTM
        Fitted BTM model.
    words_num : int = 20
        The number of words to select.
    topics_idx : Union[List, numpy.ndarray] = None
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
        idx = np.argsort(probs)[:-words_num-1:-1]
        result = Series(model.vocabulary_[idx])
        result.name = 'topic{}'.format(topic_id)
        return result

    topics_num = model.topics_num_
    topics_idx = np.arange(topics_num) if topics_idx is None else topics_idx
    return concat(
        map(lambda x: _select_words(model, x), topics_idx), axis=1)


def get_top_topic_docs(
        docs: Sequence[Any],
        p_zd: np.ndarray,
        docs_num: int = 20,
        topics_idx: Sequence[Any] = None) -> DataFrame:
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
        idx = np.argsort(probs)[:-docs_num-1:-1]
        result = Series(np.asarray(docs)[idx])
        result.name = 'topic{}'.format(topic_id)
        return result

    topics_num = p_zd.shape[1]
    topics_idx = np.arange(topics_num) if topics_idx is None else topics_idx
    return concat(
        map(lambda x: _select_docs(docs, p_zd, x), topics_idx), axis=1)


def get_docs_top_topic(
        docs: Sequence[Any],
        p_zd: np.ndarray) -> DataFrame:
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
    return DataFrame({'documents': docs, 'label': p_zd.argmax(axis=1)})
