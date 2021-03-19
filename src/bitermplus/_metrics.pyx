__all__ = ['perplexity', 'coherence']

from libc.math cimport exp, log
from pandas import DataFrame
from scipy.sparse import csr
from typing import Union
import numpy as np
import cython
from cython.parallel import prange


@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef double perplexity(
        double[:, :] p_wz,
        double[:, :] p_zd,
        n_dw,
        long T):
    """Perplexity calculation.

    Parameters
    ----------
    p_wz : np.ndarray
        Topics vs words probabilities matrix (T x W).

    p_zd : np.ndarray
        Documents vs topics probabilities matrix (D x T).

    n_dw : np.ndarray
        Words frequency matrix for all documents (D x W).

    T : int
        Number of topics.

    Returns
    -------
    perplexity : float
        Perplexity estimate.
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

    for d in prange(D, nogil=True):
    #for d in range(D):
        w_ri = n_dw_indptr[d]
        if d + 1 == D:
            w_rj = W
        else:
            w_rj = n_dw_indptr[d+1]

        for w_i in range(w_ri, w_rj):
            w = n_dw_indices[w_i]
            n = n_dw_data[w_i]

            pwz_pzd_sum = 0.
            for t in range(T):
                pwz_pzd_sum += p_zd[d, t] * p_wz[t, w]
            exp_num += n * log(pwz_pzd_sum)

    perplexity = exp(-exp_num / n_dw_sum)
    return perplexity


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef coherence(
        double[:, :] p_wz,
        n_dw,
        int M):
    """Semantic topic coherence calculation.

    Parameters
    ----------
    p_wz : np.ndarray
        Topics vs words probabilities matrix (T x W).

    n_dw : scipy.sparse.csr_matrix
        Words frequency matrix for all documents (D x W).

    M : int
        Number of top words in a topic.

    Returns
    -------
    coherence : np.ndarray
        Semantic coherence estimates for all topics.
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
                logSum += log((D_ij + 1.) / D_j)
        coherence[t] = logSum

    return np.array(coherence)
