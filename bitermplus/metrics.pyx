__all__ = ['perplexity']

from libc.math cimport exp, log


cpdef double perplexity(
        double[:, :] phi,
        double[:, :] P_zd,
        double[:, :] n_wd,
        double N_d,
        int Z):
    """Perplexity calculation.

    Parameters
    ----------
    phi : double[:, :]
        Words vs topics probabilities matrix.

    P_zd : double[:, :]
        Topics probabilities vs documents matrix.

    n_wd : double[:, :]
        Matrix of words occurrences in documents.

    N_d : double
        Total number of words.

    K : int
        Number of topics.

    Returns
    -------
    perplexity : double
        Perplexity estimate.
    """
    cdef double exp_num = 0
    cdef double phi_theta_sum = 0
    cdef int i, j, z
    cdef long d, w

    d = P_zd.shape[0]
    w = phi.shape[0]

    for i in range(d):
        for j in range(w):
            for z in range(Z):
                phi_pzd_sum += phi[j, z] * P_zd[z, i]
            exp_num += n_wd[i, j] * log(phi_pzd_sum)

    perplexity = exp(-exp_num / N_d)
    return perplexity
