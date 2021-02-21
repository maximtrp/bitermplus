import bitermplus as btm
import unittest
import os.path
import sys
import numpy as np
from gzip import open as gzip_open
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime as dt
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
LOGGER = logging.getLogger(__name__)


class TestBTM(unittest.TestCase):

    # Plotting tests
    def test_btm_class(self):
        with gzip_open('dataset/SearchSnippets.txt.gz', 'rb') as file:
            texts = file.readlines()

        vec = CountVectorizer(lowercase=False)
        X = vec.fit_transform(texts)

        vocab = np.array(vec.get_feature_names())
        biterms = btm.util.biterms(X)

        LOGGER.info('Modeling started')
        model = btm.BTM(8, vocab.size, alpha=50/8, beta=0.01, L=0.5)
        model.fit(biterms, iterations=10)
        LOGGER.info('Modeling finished')

        self.assertIsInstance(model.phi_, np.ndarray)
        self.assertTupleEqual(model.phi_.shape, (vocab.size, 8))

        P_zd = model.transform(biterms)

        LOGGER.info('Perplexity started')
        perplexity = btm.metrics.perplexity(model.phi_, P_zd, X, 8)
        self.assertIsInstance(perplexity, float)
        self.assertNotEqual(perplexity, 0.)
        LOGGER.info('Perplexity finished')

        LOGGER.info('Coherence started')
        coherence = btm.metrics.coherence(model.phi_, X, M=20)
        self.assertIsInstance(coherence, np.ndarray)
        self.assertGreater(len(coherence), 0.)
        LOGGER.info('Coherence finished')

if __name__ == '__main__':
    unittest.main()
