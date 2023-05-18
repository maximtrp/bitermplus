__all__ = ['perplexity', 'coherence', 'entropy']

from cython.view cimport array
from libc.math cimport exp, log
from typing import Union
from pandas import DataFrame
from scipy.sparse import csr
from cython.parallel import prange
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
    cdef double pwz_pzd_sum = 0.
    cdef double exp_num = 0.
    cdef double perplexity = 0.
    cdef double n = 0
    cdef long d, w, t, w_i, w_ri, w_rj
    cdef long D = p_zd.shape[0]
    cdef long W = p_wz.shape[1]
    cdef long[:] n_dw_indptr = n_dw.indptr.astype(int)
    cdef long[:] n_dw_indices = n_dw.indices.astype(int)
    cdef double n_dw_sum = n_dw.sum()
    cdef double[:] n_dw_data = n_dw.data.astype(float)

    # Iterating over all documents
    for d in prange(D, nogil=True):

        w_ri = n_dw_indptr[d]
        # if d + 1 == D:
        #     w_rj = W
        # else:
        w_rj = n_dw_indptr[d+1]

        for w_i in range(w_ri, w_rj):
            w = n_dw_indices[w_i]
            n = n_dw_data[w_i]

            pwz_pzd_sum = 0.
            for t in range(T):
                pwz_pzd_sum = pwz_pzd_sum + p_zd[d, t] * p_wz[t, w]
            if pwz_pzd_sum > 0:
                exp_num += n * log(pwz_pzd_sum)

    perplexity = exp(-exp_num / n_dw_sum)
    return perplexity


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
    cdef int d, i, j, k, t, tw, w_i, w_ri, w_rj, w
    cdef double logSum = 0.
    cdef long T = p_wz.shape[0]
    cdef long W = p_wz.shape[1]
    cdef long D = n_dw.shape[0]
    cdef long n
    cdef long[:] n_dw_indices = n_dw.indices.astype(int)
    cdef long[:] n_dw_indptr = n_dw.indptr.astype(int)
    cdef long n_dw_len = n_dw_indices.shape[0]
    cdef long[:] n_dw_data = n_dw.data.astype(int)
    cdef long[:, :] top_words = np.zeros((M, T), dtype=int)
    cdef double[:] coherence = np.zeros(T, dtype=float)
    cdef int w1 = 0
    cdef int w2 = 0
    cdef double D_ij = 0.
    cdef double D_j = 0.

    for t in range(T):
        words_idx_sorted = np.argsort(p_wz[t, :])[:-M-1:-1]
        for i in range(M):
            top_words[i, t] = words_idx_sorted[i]

    for t in range(T):
        logSum = 0.
        for i in prange(1, M, nogil=True):
            for j in range(0, i):
                D_ij = 0.
                D_j = 0.

                for d in range(D):
                    w1 = 0
                    w2 = 0
                    w_ri = n_dw_indptr[d]
                    if d + 1 == D:
                        w_rj = W
                    else:
                        w_rj = n_dw_indptr[d+1]

                    for w_i in range(w_ri, w_rj):
                        w = n_dw_indices[w_i]
                        n = n_dw_data[w_i]
                        for tw in range(M):
                            if (top_words[i, t] == w and n > 0):
                                w1 = 1
                            elif (top_words[j, t] == w and n > 0):
                                w2 = 1
                    D_ij += float(w1 & w2)
                    D_j += float(w2)
                logSum += log((D_ij + eps) / D_j)
        coherence[t] = logSum

    return np.array(coherence)


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
    # Words number
    cdef int W = p_wz.shape[1]
    # Topics number
    cdef int T = p_wz.shape[0]

    # Initializing variables
    cdef double word_ratio = 0.
    cdef double sum_prob = 0.
    cdef double shannon = 0.
    cdef double energy = 0.
    cdef double int_energy = 0.
    cdef double free_energy = 0.
    cdef double renyi = 0.
    cdef int t = 0
    cdef int w = 0

    # Setting threshold
    cdef double thresh = 1. / W

    for w in range(W):
        for t in range(T):
            if not max_probs or (max_probs and p_wz[t, w] > thresh):
                sum_prob += p_wz[t, w]
                word_ratio += 1

    # Shannon entropy
    shannon = log(word_ratio / (W * T))

    # Internal energy
    int_energy = -log(sum_prob / T)

    # Free energy
    free_energy = int_energy - shannon * T

    # Renyi entropy
    if T == 1:
        renyi = free_energy / T
    else:
        renyi = free_energy / (T-1)

    return renyi
