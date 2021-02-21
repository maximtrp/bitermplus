import unittest
import tests.test_btm


def btm_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(tests.test_btm)
    return suite
