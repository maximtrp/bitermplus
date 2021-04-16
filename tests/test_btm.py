import bitermplus as btm
import unittest
import os.path
import sys
import numpy as np
import logging
import pickle as pkl
import pandas as pd
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
LOGGER = logging.getLogger(__name__)


class TestBTM(unittest.TestCase):

    # Plotting tests
    def test_btm_class(self):
        # Importing and vectorizing text data
        df = pd.read_csv(
            'dataset/SearchSnippets.txt.gz', header=None, names=['texts'])
        texts = df['texts'].str.strip().tolist()

        # Vectorizing documents, obtaining full vocabulary and biterms
        X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
        docs_vec = btm.get_vectorized_docs(texts, vocabulary)
        biterms = btm.get_biterms(docs_vec)

        LOGGER.info('Modeling started')
        topics_num = 8
        model = btm.BTM(
            X, vocabulary, seed=12321, T=topics_num, W=vocabulary.size,
            M=20, alpha=50/topics_num, beta=0.01)
        t1 = time.time()
        model.fit(biterms, iterations=20)
        t2 = time.time()
        LOGGER.info(t2 - t1)
        # LOGGER.info(model.theta_)
        self.assertIsInstance(model.matrix_topics_words_, np.ndarray)
        self.assertTupleEqual(
            model.matrix_topics_words_.shape, (topics_num, vocabulary.size))
        LOGGER.info('Modeling finished')

        LOGGER.info('Inference "sum_b" started')
        docs_vec_subset = docs_vec[:1000]
        docs_vec_subset[100] = np.array([], dtype=np.int32)
        p_zd = model.transform(docs_vec_subset)
        self.assertTupleEqual(p_zd.shape, (1000, topics_num))
        # LOGGER.info(p_zd)
        LOGGER.info('Inference "sum_b" finished')

        LOGGER.info('Model saving started')
        with open('model.pickle', 'wb') as file:
            self.assertIsNone(pkl.dump(model, file))
        LOGGER.info('Model saving finished')

        LOGGER.info('Inference "sum_w" started')
        p_zd = model.transform(docs_vec_subset, infer_type='sum_w')
        # LOGGER.info(p_zd)
        LOGGER.info('Inference "sum_w" finished')

        LOGGER.info('Inference "mix" started')
        p_zd = model.transform(docs_vec_subset, infer_type='mix')
        # LOGGER.info(p_zd)
        LOGGER.info('Inference "mix" finished')

        LOGGER.info('Perplexity testing started')
        perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, 8)
        self.assertTrue(perplexity, model.perplexity_)
        self.assertIsInstance(perplexity, float)
        self.assertNotEqual(perplexity, 0.)
        LOGGER.info('Perplexity testing finished')

        LOGGER.info('Coherence testing started')
        coherence = btm.coherence(model.matrix_topics_words_, X, M=20)
        self.assertTrue(np.allclose(coherence, model.coherence_))
        self.assertIsInstance(coherence, np.ndarray)
        self.assertGreater(coherence.shape[0], 0)
        LOGGER.info('Coherence testing finished')

        LOGGER.info('Model loading started')
        with open('model.pickle', 'rb') as file:
            self.assertIsInstance(pkl.load(file), btm._btm.BTM)
        LOGGER.info('Model loading finished')


if __name__ == '__main__':
    unittest.main()
