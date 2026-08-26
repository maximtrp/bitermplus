# cython: language_level=3, embedsignature=True
__all__ = ['perplexity', 'coherence', 'entropy']

from cython.view cimport array
from libc.math cimport exp, log
from typing import Union
from pandas import DataFrame
from scipy.sparse import csr_matrix
from cython import boundscheck, wraparound, cdivision
import numpy as np


@boundscheck(False)
# @wraparound(False)
cpdef double perplexity(
        double[:, :] p_wz,
        double[:, :] p_zd,
        n_dw,
        long T):
    """Perplexity calculation [1]_.

    Parameters
    ----------
    p_wz : np.ndarray
        Topics vs words probabilities matrix (T x W).

    p_zd : np.ndarray
        Documents vs topics probabilities matrix (D x T).

    n_dw : scipy.sparse.csr_matrix
        Words frequency matrix for all documents (D x W).

    T : int
        Number of topics.

    Returns
    -------
    perplexity : float
        Perplexity estimate.

    References
    ----------
    .. [1] Heinrich, G. (2005). Parameter estimation for text analysis (pp.
        1-32). Technical report.

    Example
    -------
    >>> import bitermplus as btm
    >>> # Preprocessing step
    >>> # ...
    >>> # X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
    >>> # Model fitting step
    >>> # model = ...
    >>> # Inference step
    >>> # p_zd = model.transform(docs_vec_subset)
    >>> # Coherence calculation
    >>> perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, 8)
    """
    cdef long D = p_zd.shape[0]
    p_zd_arr = np.asarray(p_zd)
    p_wz_arr = np.asarray(p_wz)
    if T != p_wz.shape[0] or p_zd.shape[1] != p_wz.shape[0]:
        raise ValueError("topic dimensions of p_wz, p_zd, and T must match")
    if n_dw.ndim != 2 or n_dw.shape[1] != p_wz.shape[1]:
        raise ValueError("n_dw word dimension must match p_wz")
    if D > n_dw.shape[0]:
        raise ValueError("p_zd has more documents than n_dw")
    if not np.all(np.isfinite(p_wz_arr)) or np.any(p_wz_arr < 0):
        raise ValueError("p_wz must contain finite non-negative probabilities")
    if not np.all(np.isfinite(p_zd_arr)) or np.any(p_zd_arr < 0):
        raise ValueError("p_zd must contain finite non-negative probabilities")
    n_dw_subset = n_dw[:D]
    if not np.all(np.isfinite(n_dw_subset.data)) or np.any(n_dw_subset.data < 0):
        raise ValueError("n_dw must contain finite non-negative counts")
    cdef double n_dw_sum = n_dw_subset.sum()
    if n_dw_sum <= 0:
        raise ValueError("n_dw subset must contain at least one token")
    coo = n_dw_subset.tocoo()
    d_idx = np.asarray(coo.row)
    w_idx = np.asarray(coo.col)
    counts = np.asarray(coo.data, dtype=float)
    # Bound the temporary matrix instead of materializing nnz x topics at once.
    probs = np.empty(counts.shape[0], dtype=float)
    for start in range(0, counts.shape[0], 100000):
        stop = min(start + 100000, counts.shape[0])
        probs[start:stop] = (
            p_zd_arr[d_idx[start:stop]]
            * p_wz_arr[:, w_idx[start:stop]].T
        ).sum(axis=1)
    np.clip(probs, 1e-300, None, out=probs)
    return exp(float(-np.dot(counts, np.log(probs)) / n_dw_sum))


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef coherence(
        double[:, :] p_wz,
        n_dw,
        double eps=1.,
        int M=20):
    """Semantic topic coherence calculation [1]_.

    Parameters
    ----------
    p_wz : np.ndarray
        Topics vs words probabilities matrix (T x W).

    n_dw : scipy.sparse.csr_matrix
        Words frequency matrix for all documents (D x W).

    eps : float
        Calculation parameter. It is summed with a word pair
        conditional probability.

    M : int
        Number of top words in a topic to take.

    Returns
    -------
    coherence : np.ndarray
        Semantic coherence estimates for all topics.

    References
    ----------
    .. [1] Mimno, D., Wallach, H., Talley, E., Leenders, M., & McCallum, A.
        (2011, July). Optimizing semantic coherence in topic models. In
        Proceedings of the 2011 conference on empirical methods in natural
        language processing (pp. 262-272).

    Example
    -------
    >>> import bitermplus as btm
    >>> # Preprocessing step
    >>> # ...
    >>> # X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
    >>> # Model fitting step
    >>> # model = ...
    >>> # Coherence calculation
    >>> coherence = btm.coherence(model.matrix_topics_words_, X, M=20)
    """
    cdef long T = p_wz.shape[0]
    cdef long W = p_wz.shape[1]
    if T == 0 or W == 0:
        raise ValueError("p_wz must have non-empty topic and word dimensions")
    if M <= 0 or M > W:
        raise ValueError("M must be between 1 and the vocabulary size")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    p_wz_arr = np.asarray(p_wz)
    if not np.all(np.isfinite(p_wz_arr)) or np.any(p_wz_arr < 0):
        raise ValueError("p_wz must contain finite non-negative probabilities")
    matrix = csr_matrix(n_dw, dtype=float)
    if matrix.shape[1] != W:
        raise ValueError("n_dw word dimension must match p_wz")
    if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise ValueError("n_dw must contain finite non-negative counts")
    matrix.data[:] = 1.
    matrix.eliminate_zeros()
    result = np.zeros(T, dtype=float)
    for t in range(T):
        top_words = np.argsort(p_wz_arr[t])[-M:][::-1]
        topic_matrix = matrix[:, top_words]
        doc_freq = np.asarray(topic_matrix.sum(axis=0)).ravel()
        cooc = (topic_matrix.T @ topic_matrix).toarray()
        for i in range(1, M):
            for j in range(i):
                result[t] += np.log(
                    (cooc[i, j] + eps) / max(doc_freq[j], eps)
                )
    return result


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef entropy(
        double[:, :] p_wz,
        bint max_probs=True):
    """Renyi entropy calculation routine [1]_.

    Renyi entropy can be used to estimate the optimal number of topics: just fit
    several models with a different number of topics and choose the number of
    topics for which the Renyi entropy is the least.

    Parameters
    ----------
    p_wz : np.ndarray
        Topics vs words probabilities matrix (T x W).

    Returns
    -------
    renyi : double
        Renyi entropy value.
    max_probs : bool
        Use maximum probabilities of terms per topics instead of all probability values.

    References
    ----------
    .. [1] Koltcov, S. (2018). Application of Rényi and Tsallis entropies to
           topic modeling optimization. Physica A: Statistical Mechanics and its
           Applications, 512, 1192-1204.

    Example
    -------
    >>> import bitermplus as btm
    >>> # Preprocessing step
    >>> # ...
    >>> # Model fitting step
    >>> # model = ...
    >>> # Entropy calculation
    >>> entropy = btm.entropy(model.matrix_topics_words_)
    """
    cdef int W = p_wz.shape[1]
    cdef int T = p_wz.shape[0]
    if W == 0 or T == 0:
        raise ValueError("p_wz must have non-empty topic and word dimensions")
    cdef double thresh = 1.0 / W
    cdef double word_ratio, sum_prob, shannon, int_energy, free_energy
    p_wz_arr = np.asarray(p_wz)
    if not np.all(np.isfinite(p_wz_arr)) or np.any(p_wz_arr < 0):
        raise ValueError("p_wz must contain finite non-negative probabilities")
    if max_probs:
        mask = p_wz_arr >= thresh
    else:
        mask = np.ones((T, W), dtype=bool)
    sum_prob = float(p_wz_arr[mask].sum())
    word_ratio = float(mask.sum())
    if word_ratio == 0 or sum_prob <= 0:
        raise ValueError("p_wz does not contain usable probabilities")
    shannon = log(word_ratio / (W * T))
    int_energy = -log(sum_prob / T)
    free_energy = int_energy - shannon * T
    if T == 1:
        return free_energy / T
    return free_energy / (T - 1)
