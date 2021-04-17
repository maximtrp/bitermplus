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
                    dtype=np.int32),
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
        method: str = "sklb",
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
    method : str = "sklb"
        Comparison method. Possible variants:
        1) "klb" - Kullback-Leibler divergence.
        2) "sklb" - Symmetric Kullback-Leibler divergence.
        3) "jsd" - Jensen-Shannon divergence.
        4) "jef" - Jeffrey's divergence.
        5) "hel" - Hellinger distance.
        6) "bhat" - Bhattacharyya distance.
        6) "jac" - Jaccard index.
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

    # Distance values
    dist_vals = np.zeros(shape=(topics_num, matrices_num), dtype=float)

    for mid, matrix in enum_func(matrices):
        if mid == ref:
            continue

        # Matrix for distances between all topics
        all_vs_all_dists = np.zeros((topics_num, topics_num))

        for t_ref in range(topics_num):
            for t in range(topics_num):

                if method == "klb":
                    val_raw = ssp.kl_div(matrix_ref[t_ref, :], matrix[t, :])
                    all_vs_all_dists[t_ref, t] = val_raw[np.isfinite(val_raw)].sum()

                elif method == "sklb":
                    val_raw = ssp.kl_div(matrix_ref[t_ref, :], matrix[t, :])\
                        + ssp.kl_div(matrix[t, :], matrix_ref[t_ref, :])
                    all_vs_all_dists[t_ref, t] = val_raw[np.isfinite(val_raw)].sum()

                elif method == "jsd":
                    val_raw = 0.5 * ssp.kl_div(matrix_ref[t_ref, :], matrix[t, :])\
                        + 0.5 * ssp.kl_div(matrix[t, :], matrix_ref[t_ref, :])
                    all_vs_all_dists[t_ref, t] = val_raw[np.isfinite(val_raw)].sum()

                elif method == "jef":
                    p = matrix_ref[t_ref, :]
                    q = matrix[t, :]
                    vals = (p - q) * (np.log(p) - np.log(q))
                    vals[(vals <= 0) | ~np.isfinite(vals)] = 0.
                    all_vs_all_dists[t_ref, t] = vals.sum()

                elif method == "hel":
                    p = matrix_ref[t_ref, :]
                    q = matrix[t, :]
                    p[(p <= 0) | ~np.isfinite(p)] = 1e-64
                    q[(q <= 0) | ~np.isfinite(q)] = 1e-64
                    hel_val = ssp.distance.euclidean(
                        np.sqrt(p), np.sqrt(q)) / np.sqrt(2)
                    all_vs_all_dists[t_ref, t] = hel_val

                elif method == "bhat":
                    p = matrix_ref[t_ref, :]
                    q = matrix[t, :]
                    pq = p * q
                    pq[(pq <= 0) | ~np.isfinite(pq)] = 1e-64
                    dist = -np.log(np.sum(np.sqrt(pq)))
                    all_vs_all_dists[t_ref, t] = dist

                elif method == "jac":
                    a = np.argsort(matrix_ref[t_ref, :])[:-top_words-1:-1]
                    b = np.argsort(matrix[t, :])[:-top_words-1:-1]
                    j_num = np.intersect1d(a, b, assume_unique=False).size
                    j_den = np.union1d(a, b).size
                    jac_val = j_num / j_den
                    all_vs_all_dists[t_ref, t] = jac_val

    if method == "jaccard":
        closest_topics[:, mid] = np.argmax(all_vs_all_dists, axis=1)
        dist_vals[:, mid] = np.max(all_vs_all_dists, axis=1)
    else:
        closest_topics[:, mid] = np.argmin(all_vs_all_dists, axis=1)
        dist_vals[:, mid] = np.min(all_vs_all_dists, axis=1)

    return closest_topics, dist_vals


def get_stable_topics(
        closest_topics: np.ndarray,
        dist: np.ndarray,
        norm: bool = True,
        inverse: bool = True,
        inverse_factor: float = 1.0,
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
    norm : bool = True
        Normalize distance values (passed as ``dist`` argument).
    inverse : bool = True
        Inverse distance values by subtracting them from ``inverse_factor``.
    inverse_factor : float = 1.0
        Subtract distance values from this factor to inverse.
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
    dist_ready = dist_arr / dist_arr.max() if norm else dist_arr.copy()
    dist_ready = inverse_factor - dist_ready if inverse else dist_ready
    mask = (
        np.sum(np.delete(dist_ready, ref, axis=1) >= thres, axis=1)
        >= thres_models)
    return closest_topics[mask], dist_ready[mask]


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
