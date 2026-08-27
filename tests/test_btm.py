import logging
import pickle as pkl
import unittest

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

try:
    from src import bitermplus as btm
except ImportError:
    import bitermplus as btm

# import time
LOGGER = logging.getLogger(__name__)


def tiny_model(**kwargs):
    """Two-word, one-document model used by the input-validation tests."""
    return btm.BTM(sparse.csr_matrix([[1, 1]]), np.array(["first", "second"]), **kwargs)


class TestBTM(unittest.TestCase):
    def test_float_topics_num_is_converted_to_int(self):
        model = tiny_model(T=3.5)

        self.assertEqual(model.topics_num_, 3)
        self.assertEqual(model.matrix_topics_words_.shape, (3, 2))
        model.fit([[[0, 1]]], iterations=1, verbose=False)
        result = model.transform([np.array([0, 1], dtype=np.int32)], verbose=False)
        self.assertEqual(result.shape, (1, 3))

    def test_topics_num_must_be_positive_after_conversion(self):
        for topics_num in (0, -1, 0.5):
            with self.subTest(topics_num=topics_num), \
                    self.assertRaisesRegex(ValueError, "T must be positive"):
                tiny_model(T=topics_num)

    def test_rejects_invalid_word_ids(self):
        model = tiny_model(T=2)

        for biterms in ([[[-1, 0]]], [[[0, 2]]]):
            with self.subTest(biterms=biterms), \
                    self.assertRaisesRegex(ValueError, "within the vocabulary"):
                model.fit(biterms, iterations=1, verbose=False)
        with self.assertRaisesRegex(TypeError, "must be integers"):
            model.fit([[[0.5, 1]]], iterations=1, verbose=False)

    def test_transform_requires_fit_and_valid_word_ids(self):
        model = tiny_model(T=2)
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            model.transform([np.array([0], dtype=np.int32)], verbose=False)

        model.fit([[[0, 1]]], iterations=1, verbose=False)
        for word_id in (-1, 2):
            with self.subTest(word_id=word_id), \
                    self.assertRaisesRegex(ValueError, "within the vocabulary"):
                model.transform([np.array([word_id], dtype=np.int32)], verbose=False)

    def test_repeated_fit_resets_counts(self):
        model = tiny_model(T=2, seed=4)
        biterms = [[[0, 1], [0, 1]]]

        model.fit(biterms, iterations=1, verbose=False)
        model.fit(biterms, iterations=1, verbose=False)

        self.assertEqual(np.asarray(model.__getstate__()["n_bz"]).sum(), 2)

    def test_seed_zero_is_reproducible(self):
        biterms = [[[0, 1], [0, 1]]]
        models = [tiny_model(T=2, seed=0) for _ in range(2)]
        for model in models:
            model.fit(biterms, iterations=3, verbose=False)

        np.testing.assert_array_equal(models[0].biterms_, models[1].biterms_)

    def test_self_biterm_uses_sequential_dirichlet_factor(self):
        n_dw = sparse.csr_matrix([[2, 1]])
        model = btm.BTM(
            n_dw,
            np.array(["first", "second"]),
            T=2,
            alpha=1.0,
            beta=0.01,
            seed=0,
        )

        model.fit([[[0, 0], [0, 1]]], iterations=1, verbose=False)

        # The original C++ code omits the +1 numerator for the second draw of
        # a self-biterm. The corrected collapsed conditional assigns both
        # biterms to topic zero for this deterministic random stream.
        np.testing.assert_array_equal(model.biterms_[:, 2], np.array([0, 0]))

    # Main tests
    @pytest.mark.slow
    def test_btm_class(self):
        # Importing and vectorizing text data
        df = pd.read_csv("dataset/SearchSnippets.txt.gz", header=None, names=["texts"])
        texts = df["texts"].str.strip().tolist()

        # Vectorizing documents, obtaining full vocabulary and biterms
        X, vocabulary, _ = btm.get_words_freqs(texts)
        docs_vec = btm.get_vectorized_docs(texts, vocabulary)
        biterms = btm.get_biterms(docs_vec)

        LOGGER.info("Modeling started")
        topics_num = 8
        model = btm.BTM(
            X,
            vocabulary,
            seed=52214,
            T=topics_num,
            M=20,
            alpha=50 / topics_num,
            beta=0.01,
        )
        # t1 = time.time()
        model.fit(biterms, iterations=20)
        # t2 = time.time()
        # LOGGER.info(t2 - t1)
        # LOGGER.info(model.theta_)
        self.assertIsInstance(model.matrix_topics_words_, np.ndarray)
        self.assertTupleEqual(model.matrix_topics_words_.shape, (topics_num, vocabulary.size))
        LOGGER.info("Modeling finished")

        LOGGER.info('Inference "sum_b" started')
        docs_vec_subset = docs_vec[:1000]
        docs_vec_subset[100] = np.array([], dtype=np.int32)
        p_zd = model.transform(docs_vec_subset)
        self.assertTupleEqual(p_zd.shape, (1000, topics_num))
        # LOGGER.info(p_zd)
        LOGGER.info('Inference "sum_b" finished')

        LOGGER.info("Model saving started")
        with open("model.pickle", "wb") as file:
            pkl.dump(model, file)
        LOGGER.info("Model saving finished")

        LOGGER.info('Inference "sum_w" started')
        p_zd = model.transform(docs_vec_subset, infer_type="sum_w")
        # LOGGER.info(p_zd)
        LOGGER.info('Inference "sum_w" finished')

        LOGGER.info('Inference "mix" started')
        p_zd = model.transform(docs_vec_subset, infer_type="mix")
        # LOGGER.info(p_zd)
        LOGGER.info('Inference "mix" finished')

        LOGGER.info("Perplexity testing started")
        perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, 8)
        self.assertIsInstance(perplexity, float)
        self.assertNotEqual(perplexity, 0.0)
        LOGGER.info("Perplexity value: %s", perplexity)
        LOGGER.info("Perplexity testing finished")

        LOGGER.info("Coherence testing started")
        coherence = btm.coherence(model.matrix_topics_words_, X, M=20)
        self.assertTrue(np.allclose(coherence, model.coherence_))
        self.assertIsInstance(coherence, np.ndarray)
        self.assertGreater(coherence.shape[0], 0)
        LOGGER.info("Coherence value: %s", coherence)
        LOGGER.info("Coherence testing finished")

        LOGGER.info("Entropy testing started")
        entropy = btm.entropy(model.matrix_topics_words_, True)
        self.assertNotEqual(entropy, 0)
        LOGGER.info("Entropy value: %s", entropy)
        LOGGER.info("Entropy testing finished")

        LOGGER.info("Model loading started")
        with open("model.pickle", "rb") as file:
            self.assertIsInstance(pkl.load(file), btm.BTM)
        LOGGER.info("Model loading finished")


if __name__ == "__main__":
    unittest.main()
