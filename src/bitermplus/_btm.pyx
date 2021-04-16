__all__ = ['BTM']

# from cython.parallel import prange
from libc.time cimport time
from itertools import chain
from cython.view cimport array
from cython import cdivision, wraparound, boundscheck, initializedcheck,\
    auto_pickle, nonecheck
from bitermplus._metrics import coherence, perplexity
import numpy as np
import tqdm


@cdivision(True)
@wraparound(False)
@boundscheck(False)
cdef int sample_mult(double[:] p, double random_factor):
    cdef int K = p.shape[0]
    cdef int i, k

    for i in range(1, K):
        p[i] += p[i - 1]

    for k in range(0, K):
        if p[k] >= random_factor * p[K - 1]:
            break

    return k


@auto_pickle(False)
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
    seed : int = 0
        Random state seed. If seed is equal to 0 (default),
        use ``time(NULL)``.
    win : int = 15
        Biterms generation window.
    has_background : bool = False
        Use a background topic to accumulate highly frequent words.
    """
    cdef:
        n_dw
        vocabulary
        int T
        int W
        int M
        double alpha
        double beta
        int win
        bint has_background
        double[:] n_bz  # T x 1
        double[:] p_z  # T x 1
        double[:, :] p_wz  # T x W
        double[:, :] n_wz  # T x W
        double[:, :] p_zd  # D x T
        double[:] p_wb
        int[:, :] B
        int iters
        unsigned int seed

    # cdef dict __dict__

    def __init__(
            self, n_dw, vocabulary, int T, int W, int M=20,
            double alpha=1., double beta=0.01, unsigned int seed=0,
            int win=15, bint has_background=False):
        self.n_dw = n_dw
        self.vocabulary = vocabulary
        self.T = T
        self.W = W
        self.M = M
        self.alpha = alpha
        self.beta = beta
        self.win = win
        self.seed = seed
        self.p_wb = np.asarray(n_dw.sum(axis=0) / n_dw.sum())[0]
        self.p_z = array(
            shape=(self.T, ), itemsize=sizeof(double), format="d",
            allocate_buffer=True)
        self.n_bz = array(
            shape=(self.T, ), itemsize=sizeof(double), format="d",
            allocate_buffer=True)
        self.n_wz = array(
            shape=(self.T, self.W), itemsize=sizeof(double), format="d",
            allocate_buffer=True)
        self.p_wz = array(
            shape=(self.T, self.W), itemsize=sizeof(double), format="d",
            allocate_buffer=True)
        self.p_zd = array(
            shape=(self.n_dw.shape[0], self.T), itemsize=sizeof(double),
            format="d", allocate_buffer=True)
        self.p_wz[...] = 0.
        self.n_wz[...] = 0.
        self.n_bz[...] = 0.
        self.has_background = has_background
        self.iters = 0

    def __getstate__(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'B': np.asarray(self.B),
            'T': self.T,
            'W': self.W,
            'M': self.M,
            'win': self.win,
            'n_dw': self.n_dw,
            'vocabulary': self.vocabulary,
            'has_background': self.has_background,
            'iters': self.iters,
            'alpha': self.alpha,
            'n_bz': np.asarray(self.n_bz),
            'n_wz': np.asarray(self.n_wz),
            'p_zd': np.asarray(self.p_zd),
            'p_wz': np.asarray(self.p_wz),
            'p_wb': np.asarray(self.p_wb),
            'p_z': np.asarray(self.p_z)
        }

    def __setstate__(self, state):
        self.alpha = state.get('alpha')
        self.beta = state.get('beta')
        self.B = state.get('B', np.zeros((0, 0))).astype(np.int32)
        self.T = state.get('T')
        self.W = state.get('W')
        self.M = state.get('M')
        self.win = state.get('win')
        self.n_dw = state.get('n_dw')
        self.vocabulary = state.get('vocabulary')
        self.has_background = state.get('has_background')
        self.iters = state.get('iters', 0)
        self.n_bz = state.get('n_bz')
        self.n_wz = state.get('n_wz')
        self.p_zd = state.get('p_zd')
        self.p_wz = state.get('p_wz')
        self.p_wb = state.get('p_wb')
        self.p_z = state.get('p_z')

    cdef int[:, :] _biterms_to_array(self, list B):
        rng = np.random.default_rng(self.seed if self.seed else time(NULL))
        arr = np.asarray(list(chain(*B)), dtype=np.int32)
        random_topics = rng.integers(
            low=0, high=self.T, size=(arr.shape[0], 1), dtype=np.int32)
        arr = np.append(arr, random_topics, axis=1)
        return arr

    @initializedcheck(False)
    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cpdef void _compute_p_wz(self):
        cdef int k, w
        for k in range(self.T):
            for w in range(self.W):
                self.p_wz[k][w] = (self.n_wz[k][w] + self.beta) / (self.n_bz[k] * 2 + self.W * self.beta)

    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    @initializedcheck(False)
    cdef void _compute_p_zb(self, long i, double[:] p_z):
        cdef double pw1k, pw2k, pk, p_z_sum
        cdef int w1 = self.B[i, 0]
        cdef int w2 = self.B[i, 1]
        cdef int k

        for k in range(self.T):
            if self.has_background is True and k == 0:
                pw1k = self.p_wb[w1]
                pw2k = self.p_wb[w2]
            else:
                pw1k = (self.n_wz[k][w1] + self.beta) / (2 * self.n_bz[k] + self.W * self.beta)
                pw2k = (self.n_wz[k][w2] + self.beta) / (2 * self.n_bz[k] + 1 + self.W * self.beta)
            pk = (self.n_bz[k] + self.alpha) / (self.B.shape[0] + self.T * self.alpha)
            p_z[k] = pk * pw1k * pw2k

        # return p_z  # self._normalize(p_z)

    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    @initializedcheck(False)
    cdef void _normalize(self, double[:] p, double smoother=0.0):
        """Normalize values in place."""
        cdef:
            int i = 0
            int num = p.shape[0]

        cdef double p_sum = 0.
        for i in range(num):
            p_sum += p[i]

        for i in range(num):
            p[i] = (p[i] + smoother) / (p_sum + num * smoother)

    @initializedcheck(False)
    @boundscheck(False)
    @wraparound(False)
    cpdef fit(self, list Bs, int iterations=333, bint verbose=True):
        """Biterm topic model fitting method.

        Parameters
        ----------
        B : list
            Biterms list.
        iterations : int = 333
            Iterations number.
        verbose : bool = True
            Show progress bar.
        """
        self.B = self._biterms_to_array(Bs)
        # rng = np.random.default_rng(self.seed if self.seed else time(NULL))
        # random_factors = rng.random(
        #     low=0, high=self.T, size=(arr.shape[0], 1))

        cdef:
            long i
            int j, w1, w2, topic
            long B_len = self.B.shape[0]
            double[:] p_z = array(
                shape=(self.T, ), itemsize=sizeof(double), format="d",
                allocate_buffer=True)
            double[:] p_wz_norm = array(
                shape=(self.W, ), itemsize=sizeof(double), format="d",
                allocate_buffer=True)
            double[:] rnd_uniform = array(
                shape=(B_len, ), itemsize=sizeof(double), format="d",
                allocate_buffer=True)

        rng = np.random.default_rng(self.seed if self.seed else time(NULL))
        trange = tqdm.trange if verbose else range

        for i in range(B_len):
            w1 = self.B[i, 0]
            w2 = self.B[i, 1]
            topic = self.B[i, 2]
            self.n_bz[topic] += 1
            self.n_wz[topic][w1] += 1
            self.n_wz[topic][w2] += 1

        for j in trange(iterations):
            rnd_uniform = rng.uniform(0, 1, B_len)
            for i in range(B_len):
                w1 = self.B[i, 0]
                w2 = self.B[i, 1]
                topic = self.B[i, 2]

                self.n_bz[topic] -= 1
                self.n_wz[topic][w1] -= 1
                self.n_wz[topic][w2] -= 1

                # Topic reset
                # self.B[i, 2] = -1

                # Topic sample
                self._compute_p_zb(i, p_z)
                topic = sample_mult(p_z, rnd_uniform[i])
                self.B[i, 2] = topic

                self.n_bz[topic] += 1
                self.n_wz[topic][w1] += 1
                self.n_wz[topic][w2] += 1
                self.iters = j+1

        self.p_z[:] = self.n_bz
        self._normalize(self.p_z, self.alpha)
        self._compute_p_wz()

        for topic in range(self.T):
            p_wz_norm[:] = self.p_wz[topic]
            self._normalize(p_wz_norm)
            for i in range(self.W):
                self.p_wz[topic, i] = p_wz_norm[i]

    @cdivision(True)
    cdef long _count_biterms(self, int n, int win=15):
        cdef:
            int i, j
            long btn = 0
        for i in range(n-1):
            for j in range(i+1, min(i + win, n)):  # range(i+1, n):
                btn += 1
        return btn

    @initializedcheck(False)
    @boundscheck(False)
    @wraparound(False)
    cdef int[:, :] _generate_biterms(
            self,
            int[:, :] biterms,
            int[:] words,
            int win=15):
        cdef int i, j, words_len = words.shape[0]
        cdef long n = 0

        for i in range(words_len-1):
            # for j in range(i+1, words_len):  # min(i + win, words_len)):
            for j in range(i+1, min(i + win, words_len)):
                biterms[n, 0] = min(words[i], words[j])
                biterms[n, 1] = max(words[i], words[j])
                n += 1
        return biterms

    @initializedcheck(False)
    @boundscheck(False)
    @wraparound(False)
    cdef double[:] _infer_doc(self, int[:] doc, str infer_type, int doc_len):
        cdef double[:] p_zd = array(
            shape=(self.T, ), itemsize=sizeof(double), format="d",
            allocate_buffer=True)

        if (infer_type == "sum_b"):
            p_zd = self._infer_doc_sum_b(doc, doc_len)
        elif (infer_type == "sum_w"):
            p_zd = self._infer_doc_sum_w(doc, doc_len)
        elif (infer_type == "mix"):
            p_zd = self._infer_doc_mix(doc, doc_len)
        else:
            return None

        return p_zd

    @initializedcheck(False)
    @boundscheck(False)
    @wraparound(False)
    cdef double[:] _infer_doc_sum_b(self, int[:] doc, int doc_len):
        cdef double[:] p_zd = array(
            shape=(self.T, ), itemsize=sizeof(double), format="d",
            allocate_buffer=True)

        cdef double[:] p_zb = array(
            shape=(self.T, ), itemsize=sizeof(double), format="d",
            allocate_buffer=True)

        p_zd[...] = 0.
        p_zb[...] = 0.
        cdef long b, combs_num
        cdef int w1, w2
        cdef int[:, :] biterms

        if doc_len == 1:
            for t in range(self.T):
                p_zd[t] = self.p_z[t] * self.p_wz[t][doc[0]]
        else:
            combs_num = self._count_biterms(doc_len, self.win)
            biterms = array(
                shape=(combs_num, 2), itemsize=sizeof(int), format="i",
                allocate_buffer=True)
            biterms = self._generate_biterms(biterms, doc, self.win)

            for b in range(combs_num):
                w1 = biterms[b, 0]
                w2 = biterms[b, 1]

                if w2 >= self.W:
                    continue

                for t in range(self.T):
                    p_zb[t] = self.p_z[t] * self.p_wz[t][w1] * self.p_wz[t][w2]
                self._normalize(p_zb)

                for t in range(self.T):
                    p_zd[t] += p_zb[t]
        self._normalize(p_zd)
        return p_zd

    @initializedcheck(False)
    @boundscheck(False)
    @wraparound(False)
    cdef double[:] _infer_doc_sum_w(self, int[:] doc, int doc_len):
        cdef int i
        cdef int w
        cdef double[:] p_zd = array(
            shape=(self.T, ), itemsize=sizeof(double), format="d",
            allocate_buffer=True)
        cdef double[:] p_zw = array(
            shape=(self.T, ), itemsize=sizeof(double), format="d",
            allocate_buffer=True)
        p_zd[...] = 0.
        p_zw[...] = 0.

        for i in range(doc_len):
            w = doc[i]
            if (w >= self.W):
                continue

            for t in range(self.T):
                p_zw[t] = self.p_z[t] * self.p_wz[t][w]

            self._normalize(p_zw)

            for t in range(self.T):
                p_zd[t] += p_zw[t]

        self._normalize(p_zd)
        return p_zd

    @initializedcheck(False)
    @boundscheck(False)
    @wraparound(False)
    cdef double[:] _infer_doc_mix(self, int[:] doc, int doc_len):
        cdef double[:] p_zd = array(
            shape=(self.T, ), itemsize=sizeof(double), format="d")
        p_zd[...] = 0.
        cdef int i, w, t

        for t in range(self.T):
            p_zd[t] = self.p_z[t]

        for i in range(doc_len):
            w = doc[i]
            if (w >= self.W):
                continue

            for t in range(self.T):
                p_zd[t] *= (self.p_wz[t][w] * self.W)

        self._normalize(p_zd)
        return p_zd

    @initializedcheck(False)
    @boundscheck(False)
    @wraparound(False)
    @nonecheck(False)
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
        cdef int d
        cdef int doc_len
        cdef int docs_len = len(docs)
        cdef double[:, :] p_zd = array(
            shape=(docs_len, self.T), itemsize=sizeof(double), format="d",
            allocate_buffer=True)
        p_zd[...] = 0.
        cdef int[:] doc

        trange = tqdm.trange if verbose else range

        for d in trange(docs_len):
            doc = docs[d]
            doc_len = doc.shape[0]
            if doc_len > 0:
                p_zd[d, :] = self._infer_doc(doc, infer_type, doc_len)
            else:
                p_zd[d, :] = 0.

        self.p_zd = p_zd
        np_p_zd = np.asarray(self.p_zd)
        np_p_zd[np.isnan(np_p_zd)] = 0.
        return np_p_zd

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
        p_zd = self.transform(
            docs, infer_type=infer_type, verbose=verbose)
        return p_zd

    @property
    def matrix_topics_words_(self) -> np.ndarray:
        """Topics vs words probabilities matrix."""
        return np.asarray(self.p_wz)

    @property
    def matrix_docs_topics_(self) -> np.ndarray:
        """Documents vs topics probabilities matrix."""
        return np.asarray(self.p_zd)

    @property
    def coherence_(self) -> np.ndarray:
        """Semantic topics coherence."""
        return coherence(self.p_wz, self.n_dw, self.M)

    @property
    def perplexity_(self) -> float:
        """Perplexity.

        Run `transform` method before calculating perplexity"""
        return perplexity(self.p_wz, self.p_zd, self.n_dw, self.T)

    @property
    def vocabulary_(self) -> np.ndarray:
        """Vocabulary (list of words)."""
        return np.asarray(self.vocabulary)

    @property
    def alpha_(self) -> float:
        """Model parameter."""
        return self.alpha

    @property
    def beta_(self) -> float:
        """Model parameter."""
        return self.beta

    @property
    def window_(self) -> int:
        """Biterms generation window size."""
        return self.win

    @property
    def has_background_(self) -> bool:
        """Specifies whether the model has a background topic
        to accumulate highly frequent words."""
        return self.has_background

    @property
    def topics_num_(self) -> int:
        """Number of topics."""
        return self.T

    @property
    def vocabulary_size_(self) -> int:
        """Vocabulary size (number of words)."""
        return len(self.vocabulary)

    @property
    def coherence_window_(self) -> int:
        """Number of top words for coherence calculation."""
        return self.M

    @property
    def iterations_(self) -> int:
        """Number of iterations the model fitting process has
        gone through."""
        return self.iters

    @property
    def theta_(self) -> np.ndarray:
        """Topics probabilities vector."""
        return np.array(self.p_z)

    @property
    def biterms_(self) -> np.ndarray:
        """Model biterms. Terms are coded with the corresponding ids."""
        return np.asarray(self.B)
