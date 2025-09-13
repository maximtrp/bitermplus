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
    """Compute words vs documents frequency matrix.

    Parameters
    ----------
    docs : Union[List[str], np.ndarray, Series]
        Documents in any format that can be passed to
        :meth:`sklearn.feature_extraction.text.CountVectorizer` method.
    kwargs : dict
        Keyword arguments for
        :meth:`sklearn.feature_extraction.text.CountVectorizer` method.

    Returns
    -------
    Tuple[scipy.sparse.csr_matrix, np.ndarray, Dict]
        Documents vs words matrix in CSR format,
        vocabulary as a numpy.ndarray of terms,
        and vocabulary as a dictionary of {term: id} pairs.

    Example
    -------
    >>> import pandas as pd
    >>> import bitermplus as btm

    >>> # Loading data
    >>> df = pd.read_csv(
    ...     'dataset/SearchSnippets.txt.gz', header=None, names=['texts'])
    >>> texts = df['texts'].str.strip().tolist()

    >>> # Vectorizing documents, obtaining full vocabulary and biterms
    >>> X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
    """
    vec = CountVectorizer(**kwargs)
    X = vec.fit_transform(docs)
    words = np.array(vec.get_feature_names_out())
    return X, words, vec.vocabulary_


def get_vectorized_docs(
        docs: Union[List[str],  np.ndarray],
        vocab: Union[List[str], np.ndarray]) -> List[np.ndarray]:
    """Replace words with their ids in each document.

    Parameters
    ----------
    docs : Union[List[str],  np.ndarray]
        Documents (iterable of strings).
    vocab: Union[List[str], np.ndarray]
        Vocabulary (iterable of terms).

    Returns
    -------
    docs : List[np.ndarray]
        Vectorised documents (list of ``numpy.ndarray``
        objects with terms ids).

    Example
    -------
    >>> import pandas as pd
    >>> import bitermplus as btm

    >>> # Loading data
    >>> df = pd.read_csv(
    ...     'dataset/SearchSnippets.txt.gz', header=None, names=['texts'])
    >>> texts = df['texts'].str.strip().tolist()

    >>> # Vectorizing documents, obtaining full vocabulary and biterms
    >>> X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
    >>> docs_vec = btm.get_vectorized_docs(texts, vocabulary)
    """
    vocab_idx = {word: idx for idx, word in enumerate(vocab)}

    def _parse_words(w):
        return vocab_idx.get(w)

    result = []
    for doc in docs:
        word_ids = [vocab_idx[word] for word in doc.split() if word in vocab_idx]
        result.append(np.array(word_ids, dtype=np.int32))
    return result


def get_biterms(
        docs: List[np.ndarray],
        win: int = 15) -> List[List[int]]:
    """Biterms creation routine.

    Parameters
    ----------
    docs : List[np.ndarray]
        List of numpy.ndarray objects containing word indices.
    win : int = 15
        Biterms generation window.

    Returns
    -------
    List[List[int]]
        List of biterms for each document.

    Example
    -------
    >>> import pandas as pd
    >>> import bitermplus as btm

    >>> # Loading data
    >>> df = pd.read_csv(
    ...     'dataset/SearchSnippets.txt.gz', header=None, names=['texts'])
    >>> texts = df['texts'].str.strip().tolist()

    >>> # Vectorizing documents, obtaining full vocabulary and biterms
    >>> X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
    >>> docs_vec = btm.get_vectorized_docs(texts, vocabulary)
    >>> biterms = btm.get_biterms(docs_vec)
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
