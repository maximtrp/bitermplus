__all__ = ['BTM']

from libc.stdlib cimport malloc, free, rand, srand
from libc.time cimport time
from numpy import asarray, ndarray
from itertools import chain
import cython
from cython.parallel import prange
from bitermplus.metrics import coherence, perplexity

cdef extern from "stdlib.h":
    cdef double drand48()


@cython.cdivision(True)
cdef long randint(long lower, long upper):
    return rand() % (upper - lower + 1)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef int sample_mult(double[:] p):
    cdef int K = p.shape[0]
    cdef int i
    for i in range(1, K):
        p[i] += p[i - 1]

    cdef double u = drand48()
    cdef int k
    for k in range(0, K):
        if p[k] >= u * p[K - 1]:
            break

    if k == K:
        k -= 1

    return k


cdef long[:] dynamic_long(long N, long value):
    cdef long *arr = <long*>malloc(N * sizeof(long))
    cdef long[:] mv = <long[:N]>arr
    mv[...] = value
    return mv


cdef double[:] dynamic_double(long N, double value):
    cdef double *arr = <double*>malloc(N * sizeof(double))
    cdef double[:] mv = <double[:N]>arr
    mv[...] = value
    return mv


cdef double[:, :] dynamic_double_twodim(long N, long M, double value):
    cdef double *arr = <double*>malloc(N * M * sizeof(double))
    cdef double[:, :] mv = <double[:N, :M]>arr
    mv[...] = value
    return mv


cdef class BTM:
    """Biterm Topic Model.

    Parameters
    ----------
    n_wd : csr.csr_matrix
        Words vs documents frequency matrix. Typically, it should be the output
        of `CountVectorizer` from sklearn package.
    T : int
        Number of topics.
    W : int
        Number of words (vocabulary size).
    M : int = 20
        Number of top words for coherence calculation.
    alpha : float = 1
        Model parameter.
    beta : float = 0.01
        Model parameter.
    L : float = 0.5
        Model parameter.
    """
    cdef:
        n_dw
        int T
        int W
        int M
        double L
        double alpha
        double beta
        double[:] n_bz
        double[:, :] n_wz
        double[:, :] p_zd
        long[:] p_wb
        long[:, :] B

    def __init__(
            self, n_dw, int T, int W, int M=20,
            double alpha=1., double beta=0.01, double L=0.5):
        self.n_dw = n_dw
        self.p_wb = asarray(n_dw.sum(axis=0))[0]
        self.T = T
        self.W = W
        self.M = M
        self.L = L
        self.alpha = alpha  # dynamic_double(self.T, alpha)
        self.theta = dynamic_double(self.T, 0.)
        self.beta = beta  # dynamic_double_twodim(self.W, self.T, beta)
        self.phi = dynamic_double_twodim(self.W, self.T, 0.)

        self.n_bz = dynamic_double(self.T, 0.)
        self.n_wz = dynamic_double_twodim(self.T, self.W, 0.)

    cdef long[:, :] _biterms_to_array(self, list B):
        arr = asarray(list(chain(*B)), dtype=int)
        arr = np.append(arr, np.zeros((arr.shape[0], 1)), axis=1)
        return arr

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double[:, :] _compute_p_wz(self):
        cdef double[:, :] p_wz = dynamic_double(self.T, self.W, 0.)
        for k in range(self.T):
            for w in range(self.W):
                p_wz[k][w] = (self.n_wz[k][w] + self.beta) / (self.n_bz[k] * 2 + self.W * self.beta)
        return p_wz

    cdef double[:] _compute_p_zb(self, long i, double[:] p_z) {
        cdef double pw1k, pw2k, pk, pz_sum
        cdef long w1 = self.B[i, 0]
        cdef long w2 = self.B[i, 1]

        for k in range(self.T):
            if self.has_background and k == 0:
                pw1k = self.p_wb[w1]
                pw2k = self.p_wb[w2]
            else:
                pw1k = (self.n_wz[k][w1] + self.beta) / (2 * self.n_bz[k] + self.W * self.beta)
                pw2k = (self.n_wz[k][w2] + self.beta) / (2 * self.n_bz[k] + 1 + self.W * self.beta)
            pk = (n_bz[k] + self.alpha) / (self.B.shape[0] + self.T * self.alpha)
            p_z[k] = pk * pw1k * pw2k
            p_z_sum += p_z[k]

        for k in range(self.T):
            p_z[k] /= p_z_sum
        return p_z


    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fit(self, list Bs, int iterations):
        """Model fitting method.

        Parameters
        ----------
        B : list
            Biterms list.
        iterations : int
            Iterations number.
        """
        self.B = self._biterms_to_array(Bs)

        cdef:
            int _, i, topic
            long w1, w2
            long B_len = self.B.shape[0]
            double[:] p_z = dynamic_double(self.T, 0.)

        # Randomly assign topics to biterms
        srand(time(NULL))
        for i in range(B_len):
            topic = randint(0, self.T)
            B[i, 2] = topic

        for _ in range(iterations):
            for i in range(B_len):
                w1 = B[i, 0]
                w2 = B[i, 1]
                topic = B[i, 2]

                nb_z[topic] -= 1
                n_wz[topic][w1] -= 1
                n_wz[topic][w2] -= 1

                # Topic reset
                B[i, 2] = -1

                # Topic sample
                p_z = self.compute_p_zb(i, p_z)
                topic = sample_mult(p_z)
                self.B[i, 2] = topic

                self.n_bz[topic] += 1
                self.n_wz[topic][w1] += 1
                self.n_wz[topic][w2] += 1

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef transform(self, list Bs):
        """Return documents vs topics matrix.

        Parameters
        ----------
        B : list
            Biterms list.

        Returns
        -------
        P_zd : np.ndarray
            Documents vs topics matrix.
        """
        self.P_zd = dynamic_double_twodim(len(B), self.T, 0.)
        cdef double[:, :] P_zb
        cdef double[:] P_zbi = dynamic_double(self.T, 0.)
        cdef double[:] P_zb_sum = dynamic_double(self.T, 0.)
        cdef double P_zbi_sum = 0.
        cdef double P_zb_total_sum = 0.
        cdef long i, j, m, t, b0, b1, d_len

        for i, d in enumerate(B):
            d_len = len(d)
            P_zb = dynamic_double_twodim(d_len, self.T, 0.)
            P_zb_sum[...] = 0.
            for j, b in enumerate(d):
                b0 = b[0]
                b1 = b[1]

                for t in range(self.T):
                    P_zbi[t] = self.theta[t] * self.phi[b0, t] * self.phi[b1, t]
                    P_zbi_sum += P_zbi[t]

                for t in range(self.T):
                    P_zb[j, t] = P_zbi[t] / P_zbi_sum

            for m in range(d_len):
                for t in range(self.T):
                    P_zb_sum[t] += P_zb[m, t]
                    P_zb_total_sum += P_zb[m, t]
                for t in range(self.T):
                    P_zb_sum[t] /= P_zb_total_sum

            for t in range(self.T):
                self.P_zd[i, t] = P_zb_sum[t]

        return asarray(self.P_zd)

    cpdef fit_transform(self, list Bs, int iterations):
        """Run model fitting and return documents vs topics matrix.

        Parameters
        ----------
        B : list
            Biterms list.
        iterations : int
            Iterations number.

        Returns
        -------
        P_zd : np.ndarray
            Documents vs topics matrix.
        """
        self.fit(Bs, iterations)
        return self.transform(Bs)

    @property
    def matrix_words_topics_(self) -> ndarray:
        """Words vs topics matrix"""
        return asarray(self.phi)

    @property
    def matrix_topics_docs_(self) -> ndarray:
        """Documents vs topics matrix"""
        return asarray(self.P_zd)

    @property
    def coherence_(self) -> ndarray:
        """Semantic topics coherence"""
        return coherence(self.phi, self.n_dw, self.M)

    @property
    def perplexity_(self) -> float:
        """Perplexity"""
        return perplexity(self.phi, self.P_zd, self.n_dw, self.T)
