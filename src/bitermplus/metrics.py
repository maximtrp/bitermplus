__all__ = ['perplexity', 'coherence']

from pandas import DataFrame
import numpy as np


def perplexity(
        phi: np.ndarray,
        P_zd: np.ndarray,
        n_wd: np.ndarray,
        T: int) -> float:
    """Perplexity calculation.

    Parameters
    ----------
    phi : np.ndarray
        Words vs topics probabilities matrix (W x T).

    P_zd : np.ndarray
        Topics probabilities vs documents matrix (T x D).

    n_dw : np.ndarray
        Matrix of words occurrences in documents (D x W)

    T : int
        Number of topics.

    Returns
    -------
    perplexity : float
        Perplexity estimate.
    """
    exp_num = 0.

    D = P_zd.shape[0]
    W = phi.shape[0]

    for d in range(D):
        for w in range(W):
            phi_pzd_sum = phi[w, :] @ P_zd[:, d]
            exp_num += n_dw[d, w] * np.log(phi_pzd_sum)

    perplexity = np.exp(-exp_num / n_dw.sum())
    return perplexity


def coherence(
        phi_wt: np.ndarray,
        n_dw: np.ndarray,
        M: int) -> list:
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
    coherence : float
        Semantic coherence estimate.
    """
    phi_wt_df = DataFrame(phi_wt) if isinstance(phi_wt, np.ndarray) else DataFrame(np.array(phi_wt))

    logSum = 0.
    T = phi_wt_df.shape[1]
    coherence = np.zeros(T, dtype=float)

    for t in range(T):
        top_words = phi_wt_df.iloc[:, t].nlargest(M).index.tolist()
        logSum = 0.
        for i in range(1, M):
            for j in range(0, i):
                D_ij = (n_dw[:, top_words[i]].toarray().astype(bool) & n_dw[:, top_words[j]].toarray().astype(bool)).sum()
                D_j = n_dw[:, top_words[j]].toarray().astype(bool).sum()
                logSum += log((D_ij + 1) / D_j)
        coherence[t] = logSum

    return np.array(coherence)
