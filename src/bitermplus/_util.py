__all__ = [
    'get_words_freqs', 'get_vectorized_docs',
    'get_biterms', 'get_stable_topics',
    'get_closest_topics', 'get_top_topic_words',
    'get_top_topic_docs']

from typing import List, Union, Tuple
from scipy.sparse import csr
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy.special as ssp
import tqdm


def get_words_freqs(
        docs: Union[List[str], np.ndarray, Series],
        **kwargs: dict) -> Tuple[csr.csr_matrix, np.ndarray]:
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
    """
    vec = CountVectorizer(**kwargs)
    X = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names())
    return X, vocab


def get_vectorized_docs(
        x: Union[csr.csr_matrix, np.ndarray]) -> np.ndarray:
    """Replace words with their ids in each document.

    Parameters
    ----------
    x : Union[np.ndarray]
        Words vs documents matrix as :meth:`scipy.sparse.csr_matrix`
        or `numpy.ndarray`.

    Returns
    -------
    docs : np.ndarray
        Vectorised documents.
    """
    return list(map(lambda z: z.nonzero()[1].astype(int), x))


def get_biterms(
        n_wd: Union[csr.csr_matrix, np.ndarray],
        win: int = 15) -> List:
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
    List[List]
        List of biterms for each document.
    """
    biterms = []
    for a in n_wd:
        doc_biterms = []
        words = np.nonzero(a)[1]
        for i in range(len(words)-1):
            for j in range(i+1, min(i + win, len(words))):
                wi = min(words[i], words[j])
                wj = max(words[i], words[j])
                doc_biterms.append([wi, wj])
        biterms.append(doc_biterms)
    return biterms


def get_closest_topics(
        *matrices: List[Union[np.ndarray, DataFrame]],
        ref: int = 0,
        method: str = "klb",
        thres: float = 0.9,
        top_words: int = 100,
        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Finding closest topics in models.

    Parameters
    ----------
    *matrices : List[Union[DataFrame, np.ndarray]]
        Sequence of topics vs words matrices (T x W).
        This matrix can be accessed using ``matrix_words_topics_``
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

    Example
    -------
    >>> closest_topics, kldiv = btm.get_closest_topics(
            *list(map(lambda x: x.matrix_words_topics_, models)))
    >>> stable_topics, stable_kldiv = btm.get_stable_topics(
            closest_topics, kldiv)
    """
    dist_arr = np.asarray(dist)
    dist_norm = 1 - (dist_arr / dist_arr.max())
    mask = (
        np.sum(np.delete(dist_norm, ref, axis=1) >= thres, axis=1)
        >= thres_models)
    return closest_topics[mask], dist_norm[mask]


def get_top_topic_words(
        model,
        words_num: int = 20,
        topics_idx: Union[List, np.ndarray] = None) -> DataFrame:
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
    """
    def _select_words(model, topic_id: int):
        ps = model.matrix_topics_words_[topic_id, :]
        idx = np.argsort(ps)[:-words_num-1:-1]
        result = pd.Series(model.vocabulary[idx])
        result.name = 'topic{}'.format(topic_id)
        return result

    topics_num = model.T
    topics_idx = np.arange(topics_num) if not topics_idx else topics_idx
    return pd.concat(
        map(lambda x: _select_words(model, x), topics_idx), axis=1)


def get_top_topic_docs(
        docs: Union[List[str], np.ndarray],
        p_zd: np.ndarray,
        docs_num: int = 20,
        topics_idx: Union[List, np.ndarray] = None) -> DataFrame:
    """Select top topic docs from a fitted model.

    Parameters
    ----------
    docs : Union[List[str], np.ndarray]
        List of documents.
    p_zd : np.ndarray,
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
    """
    def _select_docs(docs, p_zd, topic_id: int):
        ps = p_zd[:, topic_id]
        idx = np.argsort(ps)[:-words_num-1:-1]
        result = pd.Series(docs[idx])
        result.name = 'topic{}'.format(topic_id)
        return result

    topics_num = p_zd.shape[1]
    topics_idx = np.arange(topics_num) if not topics_idx else topics_idx
    return pd.concat(
        map(lambda x: _select_docs(docs, p_zd, x), topics_idx), axis=1)
