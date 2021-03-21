__all__ = ['BTM']

from libc.stdlib cimport malloc, free, rand, srand
from libc.time cimport time
from libc.limits cimport INT_MAX
from itertools import chain
import numpy as np
from cython.view cimport array
import cython
from cython.parallel import prange
from bitermplus._metrics import coherence, perplexity
import tqdm


@cython.cdivision(True)
cdef float drand48():
    return float(rand()) / float(INT_MAX)


@cython.cdivision(True)
cdef long randint(long lower, long upper):
    return rand() % (upper - lower)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef int sample_mult(double[:] p):
    cdef long K = p.shape[0]
    cdef long i, k

    for i in range(1, K):
        p[i] += p[i - 1]

    cdef double u = drand48()
    for k in range(0, K):
        if p[k] >= u * p[K - 1]:
            break

    if k == K:
        k -= 1

    return k


@cython.auto_pickle(False)
cdef class BTM:
    """Biterm Topic Model.

    Parameters
    ----------
    n_dw : csr.csr_matrix
        Documents vs words frequency matrix. Typically, it should be the output
        of `CountVectorizer` from sklearn package.
    vocabulary : list
        Vocabulary (a list of words).
    T : int
        Number of topics.
    W : int
        Vocabulary size (number of words).
    M : int = 20
        Number of top words for coherence calculation.
    alpha : float = 1
        Model parameter.
    beta : float = 0.01
        Model parameter.
    win : int = 15
        Biterms generation window.
    has_background : bool = False
        Use a background topic to accumulate highly frequent words.

    Attributes
    ----------
    vocabulary : Union[list, np.ndarray]
        Vocabulary (list of words).
    T : int
        Number of topics.
    W : int
        Vocabulary size (number of words).
    M : int = 20
        Number of top words for coherence calculation.
    alpha : float = 1
        Model parameter.
    beta : float = 0.01
        Model parameter.
    win : int = 15
        Biterms generation window.
    has_background : bool = False
        Use a background topic to accumulate highly frequent words.
    """
    cdef public:
        vocabulary
        int T
        int W
        int M
        double alpha
        double beta
        int win
        bint has_background

    cdef:
        n_dw
        double[:] n_bz  # T x 1
        double[:] p_z  # T x 1
        double[:, :] p_wz  # T x W
        double[:, :] n_wz  # T x W
        double[:, :] p_zd
        double[:] p_wb
        long[:, :] B

    # cdef dict __dict__

    def __init__(
            self, n_dw, vocabulary, int T, int W, int M=20,
            double alpha=1., double beta=0.01,
            int win=15, bint has_background=False):
        self.n_dw = n_dw
        self.vocabulary = vocabulary
        self.T = T
        self.W = W
        self.M = M
        self.alpha = alpha
        self.beta = beta
        self.win = win
        self.p_wb = np.asarray(n_dw.sum(axis=0) / n_dw.sum())[0]
        self.n_bz = array(
            shape=(self.T, ), itemsize=sizeof(double), format="d",
            allocate_buffer=True)
        self.n_wz = array(
            shape=(self.T, self.W), itemsize=sizeof(double), format="d",
            allocate_buffer=True)
        self.p_zd = array(
            shape=(self.n_dw.shape[0], self.T), itemsize=sizeof(double), format="d",
            allocate_buffer=True)
        self.has_background = has_background

    def __getstate__(self):
        return (
                self.alpha,
                self.beta,
                self.T,
                self.W,
                self.M,
                self.win,
                self.n_dw,
                self.vocabulary,
                self.has_background,
                np.asarray(self.n_bz),
                np.asarray(self.n_wz),
                np.asarray(self.p_zd),
                np.asarray(self.p_wz),
                np.asarray(self.p_wb),
                np.asarray(self.p_z))

    def __setstate__(self, state):
        self.alpha = state[0]
        self.beta = state[1]
        self.T = state[2]
        self.W = state[3]
        self.M = state[4]
        self.win = state[5]
        self.n_dw = state[6]
        self.vocabulary = state[7]
        self.has_background = state[8]
        self.n_bz = state[9]
        self.n_wz = state[10]
        self.p_zd = state[11]
        self.p_wz = state[12]
        self.p_wb = state[13]
        self.p_z = state[14]

    cdef long[:, :] _biterms_to_array(self, list B):
        arr = np.asarray(list(chain(*B)), dtype=int)
        arr = np.append(arr, np.zeros((arr.shape[0], 1), dtype=int), axis=1)
        return arr

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double[:, :] _compute_p_wz(self):
        cdef double[:, :] p_wz = array(
            shape=(self.T, self.W), itemsize=sizeof(double), format="d",
            allocate_buffer=True)
        p_wz[...] = 0.
        cdef long k, w
        for k in range(self.T):
            for w in range(self.W):
                p_wz[k][w] = (self.n_wz[k][w] + self.beta) / (self.n_bz[k] * 2 + self.W * self.beta)
        return p_wz

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double[:] _compute_p_zb(self, long i, double[:] p_z):
        cdef double pw1k, pw2k, pk, p_z_sum
        # cdef double[:] p_z = dynamic_double(self.T, 0.)
        cdef long w1 = self.B[i, 0]
        cdef long w2 = self.B[i, 1]
        cdef long k

        for k in range(self.T):
            if self.has_background and k == 0:
                pw1k = self.p_wb[w1]
                pw2k = self.p_wb[w2]
            else:
                pw1k = (self.n_wz[k][w1] + self.beta) / (2 * self.n_bz[k] + self.W * self.beta)
                pw2k = (self.n_wz[k][w2] + self.beta) / (2 * self.n_bz[k] + 1 + self.W * self.beta)
            pk = (self.n_bz[k] + self.alpha) / (self.B.shape[0] + self.T * self.alpha)
            p_z[k] = pk * pw1k * pw2k
            # p_z_sum += p_z[k]

        # for k in range(self.T):
        #     p_z[k] /= p_z_sum

        return p_z

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double[:] _normalize(self, double[:] p, double smoother=0.0):
        cdef long i, num
        cdef double p_sum = 0.
        num = p.shape[0]
        for i in range(num):
            p_sum += p[i]

        for i in range(num):
            p[i] = (p[i] + smoother) / (p_sum + num * smoother)
        return p

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fit(self, list Bs, int iterations=333, bint verbose=True):
        """Biterm topic model fitting method.

        Parameters
        ----------
        B : list
            Biterms list.
        iterations : int
            Iterations number.
        """
        self.B = self._biterms_to_array(Bs)

        cdef:
            long i, topic, j
            long w1, w2
            long B_len = self.B.shape[0]
            double[:] p_z = array(
                shape=(self.T, ), itemsize=sizeof(double), format="d",
                allocate_buffer=True)
            double[:] p_wz_norm = array(
                shape=(self.W, ), itemsize=sizeof(double), format="d",
                allocate_buffer=True)

        trange = tqdm.trange if verbose else range

        # Randomly assign topics to biterms
        srand(time(NULL))
        for i in range(B_len):
            topic = randint(0, self.T)
            self.B[i, 2] = topic

            w1 = self.B[i, 0]
            w2 = self.B[i, 1]
            self.n_bz[topic] += 1
            self.n_wz[topic][w1] += 1
            self.n_wz[topic][w2] += 1

        for j in trange(iterations):
            for i in range(B_len):
                w1 = self.B[i, 0]
                w2 = self.B[i, 1]
                topic = self.B[i, 2]

                self.n_bz[topic] -= 1
                self.n_wz[topic][w1] -= 1
                self.n_wz[topic][w2] -= 1

                # Topic reset
                self.B[i, 2] = -1

                # Topic sample
                p_z = self._compute_p_zb(i, p_z)
                topic = sample_mult(p_z)
                self.B[i, 2] = topic

                self.n_bz[topic] += 1
                self.n_wz[topic][w1] += 1
                self.n_wz[topic][w2] += 1

        self.p_z = self._normalize(self.n_bz, self.alpha)
        self.p_wz = self._compute_p_wz()

        for topic in range(self.T):
            p_wz_norm = self._normalize(self.p_wz[topic])
            for i in range(self.W):
                self.p_wz[topic, i] = p_wz_norm[i]

    @cython.cdivision(True)
    cdef long _count_biterms(self, long n, long win=15):
        cdef long i, j, btn = 0
        for i in range(n-1):
            for j in range(i+1, min(i + win, n)): #range(i+1, n):
                btn += 1
        return btn

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long[:, :] _generate_biterms(
            self, long[:, :] biterms, long[:] words,
            long combs_num, long win=15):
        cdef long i, j, n = 0
        cdef long words_len = words.shape[0]

        for i in range(words_len-1):
            #for j in range(i+1, words_len):  # min(i + win, words_len)):
            for j in range(i+1, min(i + win, words_len)):
                biterms[n, 0] = words[i]
                biterms[n, 1] = words[j]
                n += 1
        return biterms

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] _infer_doc(self, long[:] doc, str infer_type):
        cdef double[:] p_zd = array(
            shape=(self.T, ), itemsize=sizeof(double), format="d",
            allocate_buffer=True)

        if (infer_type == "sum_b"):
            p_zd = self._infer_doc_sum_b(doc)
        elif (infer_type == "sum_w"):
            p_zd = self._infer_doc_sum_w(doc)
        elif (infer_type == "mix"):
            p_zd = self._infer_doc_mix(doc)
        else:
            return None

        return p_zd

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] _infer_doc_sum_b(self, long[:] doc):
        cdef double[:] p_zd = array(
            shape=(self.T, ), itemsize=sizeof(double), format="d",
            allocate_buffer=True)
        cdef double[:] p_zb = array(
            shape=(self.T, ), itemsize=sizeof(double), format="d",
            allocate_buffer=True)
        p_zd[...] = 0.
        p_zb[...] = 0.
        cdef long doc_len = doc.shape[0]
        cdef long b, w1, w2
        cdef long combs_num
        cdef long[:, :] biterms

        if doc_len == 1:
            for t in range(self.T):
                p_zd[t] = self.n_bz[t] * self.p_wz[t][doc[0]]
        else:
            combs_num = self._count_biterms(doc_len, self.win)
            biterms = array(
                shape=(combs_num, 2), itemsize=sizeof(long), format="l",
                allocate_buffer=True)
            biterms = self._generate_biterms(biterms, doc, combs_num, self.win)

            for b in range(combs_num):
                w1 = biterms[b, 0]
                w2 = biterms[b, 1]

                if w2 >= self.W:
                    continue

                for t in range(self.T):
                    p_zb[t] = self.p_z[t] * self.p_wz[t][w1] * self.p_wz[t][w2]
                p_zb = self._normalize(p_zb)

                for t in range(self.T):
                    p_zd[t] += p_zb[t]

        return self._normalize(p_zd)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] _infer_doc_sum_w(self, long[:] doc):
        cdef int i
        cdef long w
        cdef long doc_len = doc.shape[0]
        cdef double[:] p_zd = array(
            shape=(self.T, ), itemsize=sizeof(double), format="d")
        cdef double[:] p_zw = array(
            shape=(self.T, ), itemsize=sizeof(double), format="d")
        p_zd[...] = 0.
        p_zw[...] = 0.

        for i in range(doc_len):
            w = doc[i]
            if (w >= self.W):
                continue

            for t in range(self.T):
                p_zw[t] = self.p_z[t] * self.p_wz[t][w]

            p_zw = self._normalize(p_zw)

            for t in range(self.T):
                p_zd[t] += p_zw[t]

        return self._normalize(p_zd)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] _infer_doc_mix(self, long[:] doc):
        cdef double[:] p_zd = array(
            shape=(self.T, ), itemsize=sizeof(double), format="d")
        p_zd[...] = 0.
        cdef long doc_len = doc.shape[0]
        cdef long i, w, t

        for t in range(self.T):
            p_zd[t] = self.p_z[t]

        for i in range(doc_len):
            w = doc[i]
            if (w >= self.W):
                continue

            for t in range(self.T):
                p_zd[t] *= (self.p_wz[t][w] * self.W)

        return self._normalize(p_zd)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef transform(
            self, list docs, str infer_type='sum_b', bint verbose=True):
        """Return documents vs topics probability matrix.

        Parameters
        ----------
        docs : list
            Documents list. Each document must be presented as
            a list of words ids. Typically, it can be the output of
            :meth:`bitermplus.get_vectorized_docs`.
        infer_type : str
            Inference type. The following options are available:

            1) ``sum_b`` (default).
            2) ``sum_w``.
            3) ``mix``.
        verbose : bool = True
            Be verbose (show progress bar).

        Returns
        -------
        p_zd : np.ndarray
            Documents vs topics probability matrix (D vs T).
        """
        cdef long d
        cdef long docs_len = len(docs)
        cdef long[:] doc
        trange = tqdm.trange if verbose else range

        for d in trange(docs_len):
            doc = docs[d]
            self.p_zd[d, :] = self._infer_doc(doc, infer_type)

        p_zd = np.asarray(self.p_zd)
        p_zd[np.isnan(p_zd)] = 0.
        return p_zd

    cpdef fit_transform(
            self, docs, list biterms,
            str infer_type='sum_b', int iterations=333, bint verbose=True):
        """Run model fitting and return documents vs topics matrix.

        Parameters
        ----------
        docs : np.ndarray
            Vectorized documents.
        biterms : list
            List of biterms.
        infer_type : str
            Inference type. The following options are available:

            1) ``sum_b`` (default).
            2) ``sum_w``.
            3) ``mix``.
        iterations : int = 333
            Iterations number.
        verbose : bool = True
            Be verbose (show progress bars).

        Returns
        -------
        p_zd : np.ndarray
            Documents vs topics matrix (D x T).
        """
        self.fit(biterms, iterations=iterations, verbose=verbose)
        self.p_zd = self.transform(
            docs, infer_type=infer_type, verbose=verbose)
        return np.asarray(self.p_zd)

    @property
    def matrix_topics_words_(self) -> np.ndarray:
        """Topics vs words probabilities matrix."""
        return np.asarray(self.p_wz)

    @property
    def matrix_docs_topics_(self) -> np.ndarray:
        """Documents vs topics probabilities matrix."""
        p_zd = np.asarray(self.p_zd)
        p_zd[np.isnan(p_zd)] = 0.
        return p_zd

    @property
    def coherence_(self) -> np.ndarray:
        """Semantic topics coherence."""
        return coherence(self.p_wz, self.n_dw, self.M)

    @property
    def perplexity_(self) -> float:
        """Perplexity."""
        return perplexity(self.p_wz, self.p_zd, self.n_dw, self.T)
