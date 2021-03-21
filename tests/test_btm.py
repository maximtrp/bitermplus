import bitermplus as btm
import unittest
import os.path
import sys
import numpy as np
import logging
import pickle as pkl
# import time
from gzip import open as gzip_open
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
LOGGER = logging.getLogger(__name__)


class TestBTM(unittest.TestCase):

    # Plotting tests
    def test_btm_class(self):
        with gzip_open('dataset/SearchSnippets.txt.gz', 'rb') as file:
            texts = file.readlines()

        X, vocab = btm.get_words_freqs(texts)
        docs_vec = btm.get_vectorized_docs(X)
        biterms = btm.get_biterms(X)

        LOGGER.info('Modeling started')
        model = btm.BTM(X, vocab, T=8, W=vocab.size, M=20, alpha=50/8, beta=0.01)
        # t1 = time.time()
        model.fit(biterms, iterations=20)
        # t2 = time.time()
        # LOGGER.info(t2 - t1)
        self.assertIsInstance(model.matrix_topics_words_, np.ndarray)
        self.assertTupleEqual(model.matrix_topics_words_.shape, (8, vocab.size))
        LOGGER.info('Modeling finished')

        LOGGER.info('Inference started')
        p_zd = model.transform(docs_vec)
        LOGGER.info('Inference "sum_b" finished')
        p_zd = model.transform(docs_vec, infer_type='sum_w')
        LOGGER.info('Inference "sum_w" finished')
        p_zd = model.transform(docs_vec, infer_type='mix')
        LOGGER.info('Inference "mix" finished')

        LOGGER.info('Perplexity started')
        perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, 8)
        self.assertIsInstance(perplexity, float)
        self.assertNotEqual(perplexity, 0.)
        LOGGER.info('Perplexity finished')

        LOGGER.info('Coherence started')
        coherence = btm.coherence(model.matrix_topics_words_, X, M=20)
        self.assertIsInstance(coherence, np.ndarray)
        self.assertGreater(coherence.shape[0], 0)
        LOGGER.info('Coherence finished')

        LOGGER.info('Model saving/loading started')
        with open('model.pickle', 'wb') as file:
            self.assertIsNone(pkl.dump(model, file))
        with open('model.pickle', 'rb') as file:
            self.assertIsInstance(pkl.load(file), btm._btm.BTM)
        LOGGER.info('Model saving/loading finished')


if __name__ == '__main__':
    unittest.main()
