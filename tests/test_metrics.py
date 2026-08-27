"""Unit tests for bitermplus metric functions: perplexity, coherence, entropy."""

import unittest
import warnings

import numpy as np
import scipy.sparse as sp

try:
    from src import bitermplus as btm
except ImportError:
    import bitermplus as btm


class TestPerplexity(unittest.TestCase):
    """Tests for btm.perplexity().

    Formula: exp(-sum(n_dw[d,w] * log(sum_t p_zd[d,t] * p_wz[t,w])) / N)
    where N = n_dw.sum().
    """

    def setUp(self):
        # 2 topics, 4 words, 3 documents
        self.T = 2
        self.p_wz = np.array(
            [
                [0.4, 0.3, 0.2, 0.1],
                [0.1, 0.2, 0.3, 0.4],
            ],
            dtype=float,
        )
        self.p_zd = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.5, 0.5],
            ],
            dtype=float,
        )
        self.n_dw = sp.csr_matrix(
            np.array(
                [
                    [2, 1, 0, 0],
                    [0, 1, 0, 3],
                    [1, 0, 1, 0],
                ],
                dtype=float,
            )
        )

    def _expected(self, p_wz, p_zd, n_dw):
        """Reference computation using COO structure."""
        coo = n_dw.tocoo()
        counts = coo.data.astype(float)
        d_idx, w_idx = coo.row, coo.col
        probs = (p_zd[d_idx] * p_wz[:, w_idx].T).sum(axis=1)
        probs = np.clip(probs, 1e-300, None)
        return float(np.exp(-np.dot(counts, np.log(probs)) / n_dw.sum()))

    def test_returns_float(self):
        result = btm.perplexity(self.p_wz, self.p_zd, self.n_dw, self.T)
        self.assertIsInstance(result, float)

    def test_positive(self):
        result = btm.perplexity(self.p_wz, self.p_zd, self.n_dw, self.T)
        self.assertGreater(result, 0.0)

    def test_known_value(self):
        expected = self._expected(self.p_wz, self.p_zd, self.n_dw)
        result = btm.perplexity(self.p_wz, self.p_zd, self.n_dw, self.T)
        self.assertAlmostEqual(result, expected, places=10)

    def test_perfect_model_lower_than_uniform(self):
        """A model aligned with the data should have lower perplexity than uniform."""
        # Topic 0 covers words 0-1, topic 1 covers words 2-3 — matches n_dw layout
        p_wz_aligned = np.array(
            [
                [0.6, 0.35, 0.03, 0.02],
                [0.02, 0.03, 0.35, 0.60],
            ],
            dtype=float,
        )
        p_zd_aligned = np.array(
            [
                [0.9, 0.1],  # doc 0: words 0,1 → topic 0
                [0.1, 0.9],  # doc 1: words 1,3 → topic 1
                [0.5, 0.5],
            ],
            dtype=float,
        )
        p_zd_uniform = np.full((3, 2), 0.5)

        good = btm.perplexity(p_wz_aligned, p_zd_aligned, self.n_dw, self.T)
        bad = btm.perplexity(p_wz_aligned, p_zd_uniform, self.n_dw, self.T)
        self.assertLess(good, bad)

    def test_single_document(self):
        n_dw_1d = sp.csr_matrix(np.array([[3, 0, 2, 0]], dtype=float))
        p_zd_1d = np.array([[0.6, 0.4]], dtype=float)
        result = btm.perplexity(self.p_wz, p_zd_1d, n_dw_1d, self.T)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)

    def test_p_zd_subset_of_n_dw(self):
        """p_zd may cover fewer docs than n_dw (transform on a subset).
        Only the first len(p_zd) rows of n_dw should be used."""
        p_zd_sub = self.p_zd[:2]  # 2 docs
        # n_dw has 3 rows — the third must not cause an index error
        result = btm.perplexity(self.p_wz, p_zd_sub, self.n_dw, self.T)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)

    def test_subset_ignores_trailing_document_counts(self):
        p_zd_sub = self.p_zd[:2]
        expected = self._expected(self.p_wz, p_zd_sub, self.n_dw[:2])

        result = btm.perplexity(self.p_wz, p_zd_sub, self.n_dw, self.T)

        self.assertAlmostEqual(result, expected)


class TestCoherence(unittest.TestCase):
    """Tests for btm.coherence().

    Formula: for each topic t, sum over top-M word pairs (i,j):
        log((D(w_i, w_j) + eps) / D(w_j))
    where D(w_i, w_j) = docs containing both words, D(w_j) = docs containing w_j.
    """

    def setUp(self):
        self.T = 2
        self.W = 4
        self.p_wz = np.array(
            [
                [0.4, 0.3, 0.2, 0.1],
                [0.1, 0.2, 0.3, 0.4],
            ],
            dtype=float,
        )
        self.n_dw = sp.csr_matrix(
            np.array(
                [
                    [2, 1, 0, 0],
                    [0, 1, 0, 3],
                    [1, 0, 1, 0],
                ],
                dtype=float,
            )
        )

    def test_returns_ndarray(self):
        result = btm.coherence(self.p_wz, self.n_dw, M=2)
        self.assertIsInstance(result, np.ndarray)

    def test_shape(self):
        result = btm.coherence(self.p_wz, self.n_dw, M=2)
        self.assertEqual(result.shape, (self.T,))

    def test_finite_values(self):
        result = btm.coherence(self.p_wz, self.n_dw, M=2)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_csc_input_matches_csr(self):
        csr_result = btm.coherence(self.p_wz, self.n_dw, M=2)
        csc_result = btm.coherence(self.p_wz, self.n_dw.tocsc(), M=2)

        np.testing.assert_allclose(csc_result, csr_result)

    def test_known_value(self):
        """Hand-computed: T=1, W=2, M=2, 2 docs.

        top words: [0, 1]. Pair (i=1, j=0):
          D_ij=1 (doc 0 has both), D_j=1 (doc 0 has word 0)
          → log((1+1)/1) = log(2)
        """
        p_wz_1t = np.array([[0.6, 0.4]], dtype=float)
        n_dw_2d = sp.csr_matrix(
            np.array(
                [
                    [1, 1],  # doc 0: has word 0 and word 1
                    [0, 1],  # doc 1: has only word 1  (last doc, safely empty range)
                ],
                dtype=float,
            )
        )
        result = btm.coherence(p_wz_1t, n_dw_2d, eps=1.0, M=2)
        self.assertAlmostEqual(float(result[0]), np.log(2.0), places=10)

    def test_last_doc_words_counted(self):
        """Last document's words must be counted (not silently skipped).

        p_wz=[[0.6,0.4]], top words [0,1]. Doc 0 has word 0; doc 1 has words 0,1.
        Pair (i=1,j=0): D_ij=1, D_j=2 → log((1+1)/2) = 0.0
        """
        p_wz_1t = np.array([[0.6, 0.4]], dtype=float)
        n_dw_last = sp.csr_matrix(np.array([[1, 0], [1, 1]], dtype=float))
        result = btm.coherence(p_wz_1t, n_dw_last, eps=1.0, M=2)
        self.assertAlmostEqual(float(result[0]), 0.0, places=10)

    def test_m_equals_1_is_zero(self):
        """With M=1 there are no word pairs, so coherence is 0 for all topics."""
        result = btm.coherence(self.p_wz, self.n_dw, M=1)
        np.testing.assert_array_equal(result, np.zeros(self.T))


class TestEntropy(unittest.TestCase):
    """Tests for btm.entropy().

    Renyi entropy formula:
        thresh    = 1 / W
        word_ratio = count of elements matching mask
        sum_prob   = sum of p_wz values matching mask
        shannon    = log(word_ratio / (W * T))
        int_energy = -log(sum_prob / T)
        free_energy = int_energy - shannon * T
        renyi      = free_energy / (T - 1)   [or / T when T == 1]
    """

    def setUp(self):
        # T=2, W=4, rows sum to 1.0
        self.p_wz = np.array(
            [
                [0.4, 0.3, 0.2, 0.1],
                [0.1, 0.2, 0.3, 0.4],
            ],
            dtype=float,
        )
        self.T = 2
        self.W = 4

    def test_returns_float(self):
        result = btm.entropy(self.p_wz)
        self.assertIsInstance(result, float)

    def test_uniform_distribution_is_finite(self):
        result = btm.entropy(np.full((2, 4), 0.25), max_probs=True)

        self.assertTrue(np.isfinite(result))

    def test_known_value_max_probs(self):
        """Verify max-probability metrics against hand-computed values.

        thresh = 1/4 = 0.25
        mask (>0.25): [[T,T,F,F],[F,F,T,T]]  → word_ratio=4, sum_prob=1.4
        shannon    = log(4 / 8) = log(0.5)
        int_energy = -log(1.4 / 2) = -log(0.7)
        free_energy = int_energy - shannon * 2
        renyi      = free_energy / 1
        """
        W, T = self.W, self.T
        sum_prob, word_ratio = 1.4, 4.0
        shannon = np.log(word_ratio / (W * T))
        int_energy = -np.log(sum_prob / T)
        expected = float((int_energy - shannon * T) / (T - 1))

        result = btm.entropy(self.p_wz, max_probs=True)
        self.assertAlmostEqual(result, expected, places=10)

    def test_matches_paper_formula(self):
        """Validate against an independent implementation of Koltcov (2018).

        The other tests here re-derive the implementation's own arithmetic, so
        they would pass regardless of correctness. This one restates the paper:
            rho = N/(W*T), S = ln(rho); P = (1/T)*sum(phi>1/W), E = -ln(P)
            F = -q*E + S with q = 1/T;  S_q^R = F/(q-1)
        """

        def paper_renyi(phi):
            T, W = phi.shape
            mask = phi >= 1.0 / W
            P = phi[mask].sum() / T
            S = np.log(mask.sum() / (W * T))
            E = -np.log(P)
            q = 1.0 / T
            return (-q * E + S) / (q - 1)

        rng = np.random.default_rng(0)
        for T, W in [(2, 4), (3, 10), (5, 50), (20, 500)]:
            with self.subTest(T=T, W=W):
                phi = rng.dirichlet(np.full(W, 0.1), size=T)
                self.assertAlmostEqual(
                    btm.entropy(phi), paper_renyi(phi), places=10
                )

        self.assertAlmostEqual(btm.entropy(self.p_wz), paper_renyi(self.p_wz), places=10)

    def test_max_probs_false_is_degenerate_and_deprecated(self):
        """max_probs=False yields P = rho = 1, so the result is always zero."""
        with self.assertWarns(DeprecationWarning):
            result = btm.entropy(self.p_wz, max_probs=False)
        self.assertAlmostEqual(result, 0.0, places=10)

        rng = np.random.default_rng(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for T, W in [(3, 10), (5, 50)]:
                phi = rng.dirichlet(np.full(W, 0.1), size=T)
                self.assertAlmostEqual(btm.entropy(phi, max_probs=False), 0.0, places=10)

    def test_single_topic_is_undefined(self):
        """T=1 gives q=1, so F/(q-1) divides by zero."""
        with self.assertRaises(ValueError) as ctx:
            btm.entropy(np.array([[0.4, 0.3, 0.2, 0.1]], dtype=float))
        self.assertIn("undefined", str(ctx.exception))

    def test_fewer_topics_is_not_silently_accepted(self):
        """Guard the T=1 error against being re-broken by a refactor."""
        rng = np.random.default_rng(2)
        two = btm.entropy(rng.dirichlet(np.full(20, 0.1), size=2))
        self.assertTrue(np.isfinite(two))


if __name__ == "__main__":
    unittest.main()
