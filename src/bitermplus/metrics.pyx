__all__ = ['perplexity', 'coherence']

from libc.math cimport exp, log
from pandas import DataFrame
from scipy.sparse import csr
from typing import Union
import numpy as np
import cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double perplexity(
        double[:, :] phi,
        double[:, :] P_zd,
        n_dw,
        long T):
    """Perplexity calculation.

    Parameters
    ----------
    phi : np.ndarray
        Words vs topics probabilities matrix (W x T).

    P_zd : np.ndarray
        Topics probabilities vs documents matrix (D x T).

    n_dw : np.ndarray
        Matrix of words occurrences in documents (D x W)

    T : int
        Number of topics.

    Returns
    -------
    perplexity : float
        Perplexity estimate.
    """
    cdef double phi_pzd_sum = 0.
    cdef double exp_num = 0.
    cdef double perplexity = 0.
    cdef double n = 0
    cdef long d, w, t, w_i, w_ri, w_rj
    cdef long D = P_zd.shape[0]
    cdef long W = phi.shape[0]
    cdef long[:] n_dw_indptr = n_dw.indptr.astype(int)
    cdef long[:] n_dw_indices = n_dw.indices.astype(int)
    cdef double n_dw_sum = n_dw.sum()
    cdef double[:] n_dw_data = n_dw.data.astype(float)

    for d in prange(D, nogil=True):
        w_ri = n_dw_indptr[d]
        if d + 1 == D:
            w_rj = W
        else:
            w_rj = n_dw_indptr[d+1]

        for w_i in range(w_ri, w_rj):
            w = n_dw_indices[w_i]
            n = n_dw_data[w_i]

            phi_pzd_sum = 0.
            for t in range(T):
                phi_pzd_sum += phi[w, t] * P_zd[d, t]
            exp_num += n * log(phi_pzd_sum)

    perplexity = exp(-exp_num / n_dw_sum)
    return perplexity


def coherence(
        phi_wt: np.ndarray,
        n_dw: Union[np.ndarray, csr.csr_matrix],
        M: int) -> np.ndarray:
    """Semantic coherence calculation.

    Parameters
    ----------
    phi_wt : np.ndarray
        Words vs topics probabilities matrix (W x T).

    n_dw : np.ndarray
        Matrix of words occurrences in documents (D x W).

    M : int
        Number of top words in a topic.

    Returns
    -------
    coherence : np.ndarray
        Semantic coherence estimate for each topic.
    """
    phi = np.asarray(phi_wt)

    logSum = 0.
    T = phi.shape[1]
    coherence = np.zeros(T, dtype=float)

    for t in range(T):
        top_words = np.argsort(phi[:, t])[:-M-1:-1]
        logSum = 0.
        for i in range(1, M):
            for j in range(0, i):
                D_ij = (n_dw[:, top_words[i]].toarray().astype(bool) & n_dw[:, top_words[j]].toarray().astype(bool)).sum()
                D_j = n_dw[:, top_words[j]].toarray().astype(bool).sum()
                logSum += np.log((D_ij + 1) / D_j)
        coherence[t] = logSum

    return np.array(coherence)
