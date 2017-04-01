from __future__ import print_function

import numpy as np
import sys

from numpy import infty as inf
from unittest import TestCase

from .._scm_utility import find_max


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# TODO: Things that must be tested:
# * Verify that the the solver handles equivalent feature values correctly
# * Go crazy, try to break it!

class UtilityTests(TestCase):
    def setUp(self):
        """
        Called before each test

        """
        pass

    def tearDown(self):
        """
        Called after each test

        """
        pass

    def test_1(self):
        """
        Dummy test #1
        """
        X = np.array([[1, 2, 2, 2, 3, 4]], dtype=np.double).reshape(-1, 1).copy()
        y = np.array([0, 1, 0, 1, 1, 1])
        p = 1
        Xas = np.argsort(X, axis=0)
        best_utility, best_feat_idx, \
        best_thresholds, best_kinds = find_max(p, X, y, Xas, np.arange(X.shape[0]), np.ones(1))
        np.testing.assert_almost_equal(actual=best_utility, desired=1.0)
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[0])
        np.testing.assert_almost_equal(actual=best_thresholds, desired=[1])
        np.testing.assert_almost_equal(actual=best_kinds, desired=[0])

    def test_2(self):
        """
        Test that hyperparameter p works
        """
        X = np.array([[1, 2, 2, 2, 3, 4]], dtype=np.double).reshape(-1, 1).copy()
        y = np.array([0, 1, 0, 1, 1, 1])
        Xas = np.argsort(X, axis=0)
        p = 0.5
        best_utility, best_feat_idx, \
        best_thresholds, best_kinds = find_max(p, X, y, Xas, np.arange(X.shape[0]), np.ones(1))

        np.testing.assert_almost_equal(actual=best_utility, desired=1.0)
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[0, 0])
        np.testing.assert_almost_equal(actual=best_thresholds, desired=[1, 2])
        np.testing.assert_almost_equal(actual=best_kinds, desired=[0, 0])

    def test_3(self):
        """
        Test that feature_weights works
        """
        X = np.array([[1, 1],
                      [1, 0]], dtype=np.double)
        y = np.array([0, 1])
        Xas = np.argsort(X, axis=0)
        p = 1.0

        # Equal weights, feat 1 should be the best with utility 1
        best_utility, best_feat_idx, \
        best_thresholds, best_kinds = find_max(p, X, y, Xas, np.arange(X.shape[0]), np.ones(X.shape[1]))
        np.testing.assert_almost_equal(actual=best_utility, desired=1)
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[1])

        # Double weight for feat 1, should be the best with utility 2
        best_utility, best_feat_idx, \
        best_thresholds, best_kinds = find_max(p, X, y, Xas, np.arange(X.shape[0]), np.array([1.0, 2.0]))
        np.testing.assert_almost_equal(actual=best_utility, desired=2)
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[1])

        # 10 times the weight for feat 1, should be the best with utility 10
        best_utility, best_feat_idx, \
        best_thresholds, best_kinds = find_max(p, X, y, Xas, np.arange(X.shape[0]), np.array([1.0, 10.0]))
        np.testing.assert_almost_equal(actual=best_utility, desired=10)
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[1])

    def test_4(self):
        """
        Test that example_idx works
        """
        X = np.array([[1, 1],
                      [0, 0],
                      [1, 0]], dtype=np.double)
        y = np.array([0, 1, 1])
        Xas = np.argsort(X, axis=0)
        p = 1.0

        # If example 3 is included, the best feature is feat1
        best_utility, best_feat_idx, \
        best_thresholds, best_kinds = find_max(p, X, y, Xas, np.arange(X.shape[0]), np.ones(X.shape[1]))
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[1])

        # If example 3 is included, the best feature is feat1
        best_utility, best_feat_idx, \
        best_thresholds, best_kinds = find_max(p, X, y, Xas, np.array([1, 2], dtype=np.int), np.ones(X.shape[1]))
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[0, 1])