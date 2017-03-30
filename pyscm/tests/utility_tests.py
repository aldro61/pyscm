import numpy as np

from numpy import infty as inf
from unittest import TestCase

from .._scm_utility import find_max


class TrivialTests(TestCase):
    def setUp(self):
        """
        Called before each test

        """
        self.n_examples = 10
        self.n_features = 100
        self.X = np.random.rand(self.n_examples, self.n_features)
        self.y = np.random.randint(0, 2, self.n_examples)

    def tearDown(self):
        """
        Called after each test

        """
        pass

    def test_1(self):
        """
        margin=0 yields total cost 0
        """
        Xas = np.argsort(self.X, axis=0)
        find_max(self.X, self.y, Xas, np.arange(self.n_examples))
        find_max(self.X, self.y, Xas, np.arange(self.n_examples), np.zeros(self.n_features))