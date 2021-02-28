__all__ = ['get_vectorized_docs', 'get_biterms', 'get_stable_topics']

from itertools import combinations_with_replacement
from typing import List, Union, Tuple
from scipy.sparse import csr
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy.special as ssp


def get_vectorized_docs(
        docs: Union[List, np.ndarray, Series],
        **kwargs: dict) -> Tuple[csr.csr_matrix, np.ndarray]:
    """Vectorize documents.

    Parameters
    ----------
    docs : Union[List, np.ndarray, Series]
        Documents in any format that is compatible with `CountVectorizer`
        method from `sklearn.feature_extraction`.
    kwargs : dict
        Keyword arguments for `CountVectorizer` method.

    Returns
    -------
    Tuple[csr.csr_matrix, np.ndarray]
        Words vs documents matrix in CSR format and vocabulary.
    """
    vec = CountVectorizer(**kwargs)
    X = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names())
    return X, vocab


def get_biterms(n_wd: Union[csr.csr_matrix, np.ndarray]) -> List:
    """Biterms creation routine.

    Parameters
    ----------
    n_wd : Union[csr.csr_matrix, np.ndarray]
        Terms vs documents matrix. Typically, the output of
        `get_vectorized_docs` function.

    Returns
    -------
    List[List]
        List of biterms for each document.
    """
    B_d = []
    for a in n_wd:
        b_i = [b for b in combinations_with_replacement(np.nonzero(a)[1], 2)]
        B_d.append(b_i)
    return B_d


def get_stable_topics(
        *matrices: List[Union[np.ndarray, DataFrame]],
        ref: int = 0,
        method: str = "klb",
        thres: float = 0.9,
        top_words: int = 100) -> DataFrame:
    """Finding stable topics in models.

    Parameters
    ----------
    *matrices : List[Union[DataFrame, np.ndarray]]
        Sequence of words vs topics matrices (W x T).
    ref : int = 0
        Index of reference matrix (zero-based indexing).
    method : str = "klb"
        Comparison method. Possible variants:
        1) "klb" - Kullback-Leibler divergence. Topics are compared by words
        probabilities distributions.
        2) "jaccard" - Jaccard index. Topics are compared by top words sets.
    thres : float = 0.1
        Threshold for topic filtering.
    top_words : int = 100
        Number of top words in each topic to use in Jaccard index calculation.

    Returns
    -------
    stable_topics : np.ndarray
        Related topics indices in one two-dimensional array. Columns correspond
        to compared matrices (their indices), rows are related topics pairs.
    kldiv : np.ndarray
        Kullback-Leibler values corresponding to the matrix of stable topics.
    """
    matrices_num = len(matrices)
    ref = matrices_num - 1 if ref >= matrices_num else ref
    matrix_ref = matrices[ref]
    topics_num = matrix_ref.shape[1]
    stable_topics = np.zeros(shape=(topics_num, matrices_num), dtype=int)
    stable_topics[:, ref] = np.arange(topics_num)

    if method == "klb":
        kldiv = np.zeros(shape=(topics_num, matrices_num), dtype=float)

        for mid, matrix in enumerate(matrices):
            if mid == ref:
                continue
            kld_values = np.zeros_like(matrix_ref)

            for t_ref in range(topics_num):
                for t in range(topics_num):
                    kld_raw = 0.5 * (
                        ssp.kl_div(matrix[:, t], matrix_ref[:, t_ref]) +
                        ssp.kl_div(matrix_ref[:, t_ref], matrix[:, t]))
                    kld_values[t_ref, t] = kld_raw[np.isfinite(kld_raw)].sum()

            stable_topics[:, mid] = np.argmin(kld_values, axis=1)
            kldiv[:, mid] = np.min(kld_values, axis=1)

        return stable_topics, kldiv
    elif method == "jaccard":
        jaccard = np.zeros(shape=(topics_num, matrices_num), dtype=float)

        for mid, matrix in enumerate(matrices):
            if mid == ref:
                continue
            jaccard_values = np.zeros_like(matrix_ref)

            for t_ref in range(topics_num):
                for t in range(topics_num):
                    a = np.argsort(matrix_ref[:, t_ref])[:-top_words-1:-1]
                    b = np.argsort(matrix[:, t])[:-top_words-1:-1]
                    jaccard_value = np.intersect1d(a, b, assume_unique=False).size / np.union1d(a, b).size
                    jaccard_values[t_ref, t] = jaccard_value

            stable_topics[:, mid] = np.argmax(jaccard_values, axis=1)
            jaccard[:, mid] = np.max(jaccard_values, axis=1)

        return stable_topics, jaccard
    return None
