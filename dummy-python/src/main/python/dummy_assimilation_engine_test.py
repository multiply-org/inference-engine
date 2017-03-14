import unittest
import dummy_assimilation_engine


class dummy_assimilation_engine_test(unittest.TestCase):

    def test_assimilate(self):
        engine = dummy_assimilation_engine.dummy_assimilation_engine()
        engine.assimilate()
