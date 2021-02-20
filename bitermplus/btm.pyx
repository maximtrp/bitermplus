__all__ = ['BTM']

from libc.stdlib cimport malloc, free, rand, srand
from libc.time cimport time
from numpy import asarray
from itertools import chain
import cython
cdef extern from "stdlib.h":
    cdef double drand48()


@cython.cdivision(True)
cdef long randint(long lower, long upper):
    return rand() % (upper - lower + 1)

cdef int sample_mult(double[:] p):
    cdef int K = p.shape[0]
    cdef int i
    for i in range(K):
        p[i] += p[i - 1]

    cdef double u = drand48()
    cdef int k = -1
    for _ in range(K):
        k += 1
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
    T : int
        Number of topics.
    W : int
        Number of words (vocabulary size).
    alpha : float
        Model parameter. By default, 1.
    beta : float
        Model parameter. By default, 0.01.
    L : float
        Model parameter. By default, 0.5.
    """

    cdef:
        int T
        int W
        double l
        double[:] alpha
        double[:] theta
        double[:] n_z
        double[:, :] beta
        double[:, :] phi
        double[:, :] n_wz

    def __init__(self, int T, int W, double alpha=1., double beta=0.01, double L=0.5):
        self.T = T
        self.W = W
        self.L = L
        self.alpha = dynamic_double(self.T, alpha)
        self.theta = dynamic_double(self.T, 0.)
        self.n_z = dynamic_double(self.T, 0.)
        self.beta = dynamic_double_twodim(self.W, self.T, beta)
        self.phi = dynamic_double_twodim(self.W, self.T, 0.)
        self.n_wz = dynamic_double_twodim(self.W, self.T, 0.)

    cpdef long[:, :] biterms2array(self, list B):
        return asarray(list(chain(*B)), dtype=int)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void _gibbs(self, unsigned int iterations, long[:, :] B):
        cdef:
            int b_i0, b_i1, Z_iprior, Z_ipost
            int _, i, j, topic
            double P_z_sum
            long b_i[2]
            long B_ax0 = B.shape[0]
            long B_ax1 = B.shape[1]
            long[:] Z = dynamic_long(B_ax0, 0)
            double[:] P_z = dynamic_double(self.T, 0.)
            double[:] P_w0z = dynamic_double(self.T, 0.)
            double[:] P_w1z = dynamic_double(self.T, 0.)
            double[:] beta_sum = dynamic_double(self.T, 0.)

        srand(time(NULL))
        for i in range(B_ax0):
            topic = randint(0, self.T)
            self.n_wz[B[i, 0], topic] += 1.
            self.n_wz[B[i, 1], topic] += 1.
            self.n_z[topic] += 1.
            Z[i] = topic

        for j in range(self.T):
            for i in range(self.W):
                beta_sum[j] += self.beta[i, j]

        for _ in range(iterations):
            for i in range(B_ax0):
                Z_iprior = Z[i]
                b_i0 = B[i, 0]
                b_i1 = B[i, 1]
                self.n_wz[b_i0, Z_iprior] -= 1.
                self.n_wz[b_i1, Z_iprior] -= 1.
                self.n_z[Z_iprior] -= 1.

                for j in range(self.T):
                    P_w0z[j] = (self.n_wz[b_i0, j] + self.beta[b_i0, j]) / (2 * self.n_z[j] + beta_sum[j])
                    P_w1z[j] = (self.n_wz[b_i1, j] + self.beta[b_i1, j]) / (2 * self.n_z[j] + 1 + beta_sum[j])
                    P_z[j] = (self.n_z[j] + self.alpha[j]) * P_w0z[j] * P_w1z[j]
                    P_z_sum += P_z[j]

                for j in range(self.T):
                    P_z[j] = P_z[j] / P_z_sum

                Z_ipost = sample_mult(P_z)
                Z[i] = Z_ipost
                self.n_wz[b_i0, Z_ipost] += 1
                self.n_wz[b_i1, Z_ipost] += 1
                self.n_z[Z_ipost] += 1

    @property
    def phi_(self):
        return asarray(self.phi)

    cpdef fit_transform(self, list B, int iterations):
        cdef long[:, :] B_a = self.biterms2array(B)
        self.fit(B_a, iterations)
        return self.transform(B)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fit(self, long[:, :] B, int iterations):
        cdef int i, j
        cdef double[:] n_wz_beta_colsum = dynamic_double(self.T, 0.)
        cdef double n_z_alpha_sum = 0

        self._gibbs(iterations, B)

        for i in range(self.W):
            for j in range(self.T):
                n_wz_beta_colsum[j] += self.n_wz[i, j] + self.beta[i, j]

        for i in range(self.W):
            for j in range(self.T):
                self.phi[i, j] = (self.n_wz[i, j] + self.beta[i, j]) / n_wz_beta_colsum[j]
                self.beta[i, j] += self.l * self.n_wz[i, j]

        for j in range(self.T):
            n_z_alpha_sum += self.n_z[j] + self.alpha[j]

        for j in range(self.T):
            self.theta[j] = (self.n_z[j] + self.alpha[j]) / n_z_alpha_sum
            self.alpha[j] += self.l * self.n_z[j]

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef transform(self, list B):

        cdef double[:, :] P_zb
        cdef double[:, :] P_zd = dynamic_double_twodim(len(B), self.T, 0.)
        cdef double[:] P_zbi = dynamic_double(self.T, 0.)
        cdef double[:] P_zb_sum = dynamic_double(self.T, 0.)
        cdef double P_zbi_sum = 0.
        cdef double P_zb_total_sum = 0.
        cdef long i, j, m, l, b0, b1

        for i, d in enumerate(B):
            P_zb = dynamic_double_twodim(len(d), self.T, 0.)
            P_zb_sum[...] = 0.
            for j, b in enumerate(d):
                b0 = b[0]
                b1 = b[1]

                for l in range(self.T):
                    P_zbi[l] = self.theta[l] * self.phi[b0, l] * self.phi[b1, l]
                    P_zbi_sum += P_zbi[l]

                for l in range(self.T):
                    P_zb[j, l] = P_zbi[l] / P_zbi_sum

            for m in range(len(d)):
                for l in range(self.T):
                    P_zb_sum[l] += P_zb[m, l]
                    P_zb_total_sum += P_zb[m, l]
                for l in range(self.T):
                    P_zb_sum[l] /= P_zb_total_sum
            P_zd[i] = P_zb_sum

        return P_zd
