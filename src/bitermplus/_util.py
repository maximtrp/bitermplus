__all__ = [
    'get_words_freqs', 'get_vectorized_docs',
    'get_biterms', 'get_stable_topics',
    'get_closest_topics', 'get_top_topic_words',
    'get_top_topic_docs']

from typing import List, Union, Tuple, Dict
from scipy.sparse import csr
from pandas import DataFrame, Series, concat
from sklearn.feature_extraction.text import CountVectorizer
from bitermplus._btm import BTM
import numpy as np
import scipy.special as ssp
import tqdm


def get_words_freqs(
        docs: Union[List[str], np.ndarray, Series],
        **kwargs: dict) -> Tuple[csr.csr_matrix, np.ndarray, Dict]:
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
    Tuple[csr.csr_matrix, np.ndarray]
        Words vs documents matrix in CSR format and vocabulary.

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
    words = np.array(vec.get_feature_names())
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
    vocab_idx = dict(zip(vocab, range(len(vocab))))

    def _parse_words(w):
        return vocab_idx.get(w)

    return list(
        map(
            lambda doc:
                np.array(
                    list(filter(None, map(_parse_words, doc.split()))),
                    dtype=int),
            docs))


def get_biterms(
        docs: List[np.ndarray],
        win: int = 15) -> List[List[int]]:
    """Biterms creation routine.

    Parameters
    ----------
    n_wd : Union[csr.csr_matrix, np.ndarray]
        Documents vs words frequency matrix. Typically, the output of
        :meth:`bitermplus.util.get_vectorized_docs` function.
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


def get_closest_topics(
        *matrices: List[np.ndarray],
        ref: int = 0,
        method: str = "klb",
        thres: float = 0.9,
        top_words: int = 100,
        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Finding closest topics in models.

    Parameters
    ----------
    *matrices : List[np.ndarray]
        Sequence of topics vs words matrices (T x W).
        This matrix can be accessed using ``matrix_topics_words_``
        model attribute.
    ref : int = 0
        Index of reference matrix (zero-based indexing).
    method : str = "klb"
        Comparison method. Possible variants:
        1) "klb" - Kullback-Leibler divergence. Topics are compared by words
        probabilities distributions.
        2) "jaccard" - Jaccard index. Topics are compared by top words sets.
    thres : float = 0.9
        Threshold for topic filtering.
    top_words : int = 100
        Number of top words in each topic to use in Jaccard index calculation.
    verbose : bool = True
        Verbose output (progress bar).

    Returns
    -------
    closest_topics : np.ndarray
        Closest topics indices in one two-dimensional array.
        Columns correspond to the compared matrices (their indices),
        rows are the closest topics pairs.
    dist : np.ndarray
        Kullback-Leibler (if ``method`` is set to ``klb``) or Jaccard index
        values corresponding to the matrix of the closest topics.

    Example
    -------
    >>> # `models` must be an iterable of fitted BTM models
    >>> closest_topics, kldiv = btm.get_closest_topics(
    ...     *list(map(lambda x: x.matrix_topics_words_, models)))
    """
    matrices_num = len(matrices)
    ref = matrices_num - 1 if ref >= matrices_num else ref
    matrix_ref = matrices[ref]
    topics_num = matrix_ref.shape[0]
    closest_topics = np.zeros(shape=(topics_num, matrices_num), dtype=int)
    closest_topics[:, ref] = np.arange(topics_num)

    def enum_func(x):
        return enumerate(tqdm.tqdm(x)) if verbose else enumerate(x)

    if method == "klb":
        kldiv = np.zeros(shape=(topics_num, matrices_num), dtype=float)

        for mid, matrix in enum_func(matrices):
            if mid == ref:
                continue
            kld_values = np.zeros((topics_num, topics_num))

            for t_ref in range(topics_num):
                for t in range(topics_num):
                    # kld_raw = 0.5 * (
                    #     ssp.kl_div(matrix[t, :], matrix_ref[t_ref, :]) +
                    #     ssp.kl_div(matrix_ref[t_ref, :], matrix[t, :]))
                    kld_raw = ssp.kl_div(matrix_ref[t_ref, :], matrix[t, :])
                    kld_values[t_ref, t] = kld_raw[np.isfinite(kld_raw)].sum()

            closest_topics[:, mid] = np.argmin(kld_values, axis=1)
            kldiv[:, mid] = np.min(kld_values, axis=1)

        return closest_topics, kldiv
    elif method == "jaccard":
        jaccard = np.zeros(shape=(topics_num, matrices_num), dtype=float)

        for mid, matrix in enum_func(matrices):
            if mid == ref:
                continue
            jaccard_values = np.zeros_like(matrix_ref)

            for t_ref in range(topics_num):
                for t in range(topics_num):
                    a = np.argsort(matrix_ref[t_ref, :])[:-top_words-1:-1]
                    b = np.argsort(matrix[t, :])[:-top_words-1:-1]
                    j_num = np.intersect1d(a, b, assume_unique=False).size
                    j_den = np.union1d(a, b).size
                    jaccard_value = j_num / j_den
                    jaccard_values[t_ref, t] = jaccard_value

            closest_topics[:, mid] = np.argmax(jaccard_values, axis=1)
            jaccard[:, mid] = np.max(jaccard_values, axis=1)

        return closest_topics, jaccard
    return None


def get_stable_topics(
        closest_topics: np.ndarray,
        dist: np.ndarray,
        ref: int = 0,
        thres: float = 0.9,
        thres_models: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Finding stable topics in models.

    Parameters
    ----------
    closest_topics : np.ndarray
        Closest topics indices in a two-dimensional array.
        Columns correspond to the compared matrices (their indices),
        rows are the closest topics pairs. Typically, this should be
        the first value returned by :meth:`bitermplus.get_closest_topics`
        function.
    dist : np.ndarray
        Distance values: Kullback-Leibler divergence or Jaccard index values
        corresponding to the matrix of the closest topics.
        Typically, this should be the second value returned by
        :meth:`bitermplus.get_closest_topics` function.
    ref : int = 0
        Index of reference matrix (i.e. reference column index,
        zero-based indexing).
    thres : float = 0.9
        Threshold for distance values filtering.
    thres_models : int = 2
        Minimum topic recurrence frequency across all models.

    Returns
    -------
    stable_topics : np.ndarray
        Filtered matrix of the closest topics indices (i.e. stable topics).
    dist : np.ndarray
        Filtered distance values corresponding to the matrix of
        the closest topics.

    See Also
    --------
    bitermplus.get_closest_topics

    Example
    -------
    >>> closest_topics, kldiv = btm.get_closest_topics(
    ...     *list(map(lambda x: x.matrix_topics_words_, models)))
    >>> stable_topics, stable_kldiv = btm.get_stable_topics(
    ...     closest_topics, kldiv)
    """
    dist_arr = np.asarray(dist)
    dist_norm = 1 - (dist_arr / dist_arr.max())
    mask = (
        np.sum(np.delete(dist_norm, ref, axis=1) >= thres, axis=1)
        >= thres_models)
    return closest_topics[mask], dist_norm[mask]


def get_top_topic_words(
        model: BTM,
        words_num: int = 20,
        topics_idx: Union[List[int], np.ndarray] = None) -> DataFrame:
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
        Words with highest probabilities in all selected topics.

    Example
    -------
    >>> stable_topics = [0, 3, 10, 12, 18, 21]
    >>> top_words = btm.get_top_topic_words(
    ...     model,
    ...     words_num=100,
    ...     topics_idx=stable_topics)
    """
    def _select_words(model, topic_id: int):
        ps = model.matrix_topics_words_[topic_id, :]
        idx = np.argsort(ps)[:-words_num-1:-1]
        result = Series(model.vocabulary_[idx])
        result.name = 'topic{}'.format(topic_id)
        return result

    topics_num = model.topics_num_
    topics_idx = np.arange(topics_num) if topics_idx is None else topics_idx
    return concat(
        map(lambda x: _select_words(model, x), topics_idx), axis=1)


def get_top_topic_docs(
        docs: Union[List[str], np.ndarray],
        p_zd: np.ndarray,
        docs_num: int = 20,
        topics_idx: Union[List[int], np.ndarray] = None) -> DataFrame:
    """Select top topic docs from a fitted model.

    Parameters
    ----------
    docs : Union[List[str], np.ndarray]
        Iterable of documents (e.g. list of strings).
    p_zd : np.ndarray
        Documents vs topics probabilities matrix.
    docs_num : int = 20
        The number of documents to select.
    topics_idx : Union[List, numpy.ndarray] = None
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
        ps = p_zd[:, topic_id]
        idx = np.argsort(ps)[:-docs_num-1:-1]
        result = Series(np.asarray(docs)[idx])
        result.name = 'topic{}'.format(topic_id)
        return result

    topics_num = p_zd.shape[1]
    topics_idx = np.arange(topics_num) if topics_idx is None else topics_idx
    return concat(
        map(lambda x: _select_docs(docs, p_zd, x), topics_idx), axis=1)
