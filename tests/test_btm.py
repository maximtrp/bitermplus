import bitermplus as btm
import unittest
import os.path
import sys
import numpy as np
from gzip import open as gzip_open
from numpy import ndarray
from sklearn.feature_extraction.text import CountVectorizer
from itertools import combinations

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestBTM(unittest.TestCase):

    # Plotting tests
    def test_btm_class(self):
        with gzip_open('dataset/SearchSnippets.txt.gz', 'rb') as file:
            texts = file.readlines()

        vec = CountVectorizer(lowercase=False)
        X = vec.fit_transform(texts)

        vocab = np.array(vec.get_feature_names())
        biterms = btm.util.biterms(X)

        model = btm.BTM(8, vocab.size, alpha=50/8, beta=0.01, L=0.5)
        model.fit(biterms, iterations=10)

        self.assertIsInstance(model.phi_, ndarray)
        self.assertTupleEqual(model.phi_.shape, (vocab.size, 8))

    # def test_perplexity(self, phi, P_zd, n_dw, T):
    #     perplexity = btm.metrics.perplexity(phi, P_zd, n_dw, T)
    #     self.assertIsInstance(perplexity, float)
    #     self.assertNotEqual(perplexity, 0)

    # def test_coherence(self, phi, n_wd, M=20):
    #     coherence = btm.metrics.coherence(phi, n_dw, M=20)
    #     self.assertIsInstance(coherence, list)
    #     self.assertGreater(len(coherence), 0)


if __name__ == '__main__':
    unittest.main()
