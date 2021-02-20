__all__ = ['perplexity', 'coherence']

from pandas import DataFrame
import numpy as np


def perplexity(
        phi: np.ndarray,
        P_zd: np.ndarray,
        n_wd: np.ndarray,
        int T: int) -> float:
    """Perplexity calculation.

    Parameters
    ----------
    phi : np.ndarray
        Words vs topics probabilities matrix.

    P_zd : np.ndarray
        Topics probabilities vs documents matrix.

    n_wd : np.ndarray
        Matrix of words occurrences in documents.

    T : int
        Number of topics.

    Returns
    -------
    perplexity : float
        Perplexity estimate.
    """
    exp_num = 0

    D = P_zd.shape[0]
    W = phi.shape[0]

    for d in range(D):
        for w in range(W):
            phi_pzd_sum = phi[w, :] @ P_zd[:, d]
            exp_num += n_wd[d, w] * np.log(phi_pzd_sum)

    perplexity = np.exp(-exp_num / n_wd.sum())
    return perplexity


def coherence(
        phi_wt: np.ndarray,
        n_wd: np.ndarray,
        M: int) -> list:
    """Semantic coherence calculation.

    Parameters
    ----------
    phi_wt : np.ndarray
        Words vs topics probabilities matrix.

    n_wd : np.ndarray
        Matrix of words occurrences in documents.

    M : int
        Number of top words in a topic.

    Returns
    -------
    coherence : float
        Semantic coherence estimate.
    """
    if not isinstance(phi_wt, np.ndarray)

    logSum = 0.
    T = phi_wt.shape[1]
    coherence = np.zeros(T, dtype=float)

    for t in range(T):
        top_words = phi_wt.iloc[:, t].nlargest(M).index.tolist()
        logSum = 0.
        for i in range(1, M):
            for j in range(0, i):
                D_ij = (n_wd[top_words[i]].toarray().astype(bool) & n_wd[top_words[j]].toarray().astype(bool)).sum()
                D_j = n_wd[top_words[j]].toarray().astype(bool).sum()
                logSum += log((D_ij + 1) / D_j)
        coherence[t] = logSum

    return np.array(coherence)
