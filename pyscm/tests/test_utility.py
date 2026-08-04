from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import sys

from numpy import infty as inf
from unittest import TestCase
from sklearn.utils import estimator_checks

from .._scm_utility import find_max


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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
        Xas = np.argsort(X, axis=0).T.copy()
        (
            best_utility,
            best_feat_idx,
            best_thresholds,
            best_kinds,
            best_N,
            best_P_bar,
        ) = find_max(p, X, y, Xas, np.arange(X.shape[0]), np.ones(1))
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
        Xas = np.argsort(X, axis=0).T.copy()
        p = 0.5
        (
            best_utility,
            best_feat_idx,
            best_thresholds,
            best_kinds,
            best_N,
            best_P_bar,
        ) = find_max(p, X, y, Xas, np.arange(X.shape[0]), np.ones(1))

        np.testing.assert_almost_equal(actual=best_utility, desired=1.0)
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[0, 0])
        np.testing.assert_almost_equal(actual=best_thresholds, desired=[1, 2])
        np.testing.assert_almost_equal(actual=best_kinds, desired=[0, 0])

    def test_3(self):
        """
        Test that feature_weights works
        """
        X = np.array([[1, 1], [1, 0]], dtype=np.double)
        y = np.array([0, 1])
        Xas = np.argsort(X, axis=0).T.copy()
        p = 1.0

        # Equal weights, feat 1 should be the best with utility 1
        (
            best_utility,
            best_feat_idx,
            best_thresholds,
            best_kinds,
            best_N,
            best_P_bar,
        ) = find_max(p, X, y, Xas, np.arange(X.shape[0]), np.ones(X.shape[1]))
        np.testing.assert_almost_equal(actual=best_utility, desired=1)
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[1])

        # Double weight for feat 1, should be the best with utility 2
        (
            best_utility,
            best_feat_idx,
            best_thresholds,
            best_kinds,
            best_N,
            best_P_bar,
        ) = find_max(p, X, y, Xas, np.arange(X.shape[0]), np.array([1.0, 2.0]))
        np.testing.assert_almost_equal(actual=best_utility, desired=2)
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[1])

        # 10 times the weight for feat 1, should be the best with utility 10
        (
            best_utility,
            best_feat_idx,
            best_thresholds,
            best_kinds,
            best_N,
            best_P_bar,
        ) = find_max(p, X, y, Xas, np.arange(X.shape[0]), np.array([1.0, 10.0]))
        np.testing.assert_almost_equal(actual=best_utility, desired=10)
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[1])

    def test_4(self):
        """
        Test that example_idx works
        """
        X = np.array([[1, 1], [0, 0], [1, 0]], dtype=np.double)
        y = np.array([0, 1, 1])
        Xas = np.argsort(X, axis=0).T.copy()
        p = 1.0

        # If example 3 is included, the best feature is feat1
        (
            best_utility,
            best_feat_idx,
            best_thresholds,
            best_kinds,
            best_N,
            best_P_bar,
        ) = find_max(p, X, y, Xas, np.arange(X.shape[0]), np.ones(X.shape[1]))
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[1])

        # If example 3 is included, the best feature is feat1
        (
            best_utility,
            best_feat_idx,
            best_thresholds,
            best_kinds,
            best_N,
            best_P_bar,
        ) = find_max(p, X, y, Xas, np.array([1, 2], dtype=int), np.ones(X.shape[1]))
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[0, 1])

    def test_5(self):
        """
        Test that solver return accurate equivalent rules
        """
        X = np.array(
            [
                [1, 1, 0.5, 1],
                [2, 1, 0.5, 1],
                [2, 1, 0.5, 1],
                [3, 1, 1.7, 0],
                [4, 1, 1.7, 0],
                [5, 1, 1.7, 0],
                [6, 1, 1.7, 0],
                [7, 1, 1.7, 0],
            ],
            dtype=np.double,
        )
        y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        Xas = np.argsort(X, axis=0).T.copy()
        p = 1.0

        (
            best_utility,
            best_feat_idx,
            best_thresholds,
            best_kinds,
            best_N,
            best_P_bar,
        ) = find_max(p, X, y, Xas, np.arange(X.shape[0]), np.ones(X.shape[1]))
        np.testing.assert_almost_equal(actual=best_utility, desired=3.0)
        np.testing.assert_almost_equal(actual=best_feat_idx, desired=[0, 2, 3])
        np.testing.assert_almost_equal(actual=best_thresholds, desired=[2.0, 0.5, 0.0])
        np.testing.assert_almost_equal(actual=best_kinds, desired=[0, 0, 1])

    def test_6(self):
        """
        Test that solver return accurate N and P_bar
        """
        X = np.array(
            [
                [0, 0.5, 0],
                [0, 1.7, 0],
                [1, 0.5, 0],
                [0, 1.7, 0],
                [1, 0.5, 0],
                [1, 1.7, 0],
                [1, 1.7, 0],
                [1, 1.7, 0],
            ],
            dtype=np.double,
        )
        y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        Xas = np.argsort(X, axis=0).T.copy()
        p = 1.0

        (
            best_utility,
            best_feat_idx,
            best_thresholds,
            best_kinds,
            best_N,
            best_P_bar,
        ) = find_max(p, X, y, Xas, np.arange(X.shape[0]), np.ones(X.shape[1]))
        np.testing.assert_almost_equal(actual=best_N, desired=[2, 2])
        np.testing.assert_almost_equal(actual=best_P_bar, desired=[1, 1])

    def test_random_data(self):
        """
        Random testing
        """
        n_tests = 10  # 10000

        # Using rounding generates cases with equal feature values for examples
        for n_decimals in range(3):

            # The more examples, the more likely we are to have equal feature values
            for n_examples in [10, 100, 1000]:

                # Do this a few times for each configuration
                for _ in range(n_tests):
                    p = max(0, np.random.rand() * 100.0)
                    x = (
                        (np.random.rand(n_examples) * 5.0)
                        .round(n_decimals)
                        .reshape(-1, 1)
                        .copy()
                    )
                    xas = np.argsort(x, axis=0).T.copy()
                    y = np.random.randint(0, 2, n_examples)
                    thresholds = np.unique(x)

                    # Use the solver to find the solution
                    (
                        solver_best_utility,
                        solver_best_feat_idx,
                        solver_best_thresholds,
                        solver_best_kinds,
                        solver_best_N,
                        solver_best_P_bar,
                    ) = find_max(p, x, y, xas, np.arange(n_examples))

                    # Less equal rule utilities
                    le_rule_utilities = []
                    for t in thresholds:
                        rule_classifications = (x <= t).reshape(
                            -1,
                        )
                        N = (~rule_classifications[y == 0]).sum()
                        P_bar = (~rule_classifications[y == 1]).sum()
                        le_rule_utilities.append(N - p * P_bar)

                    # Greater rule utilities
                    g_rule_utilities = []
                    for t in thresholds:
                        rule_classifications = (x > t).reshape(
                            -1,
                        )
                        N = 1.0 * (~rule_classifications[y == 0]).sum()
                        P_bar = 1.0 * (~rule_classifications[y == 1]).sum()
                        g_rule_utilities.append(N - p * P_bar)

                    np.testing.assert_almost_equal(
                        actual=solver_best_utility,
                        desired=max(max(le_rule_utilities), max(g_rule_utilities)),
                    )

    def test_variable_feature_weights_scaling(self):
        """
        Test that variable feature weights correctly scale the computed utility.
        """
        X = np.array([[1.0, 2.0], [0.0, 3.0], [2.0, 1.0]], dtype=np.double)
        y = np.array([0, 1, 0])
        Xas = np.argsort(X, axis=0).T.copy()
        p = 1.0

        # Uniform weights
        (util_uniform, idx_uniform, _, _, N_uniform, P_uniform) = find_max(
            p, X, y, Xas, np.arange(X.shape[0]), np.ones(X.shape[1])
        )

        # Double weights for feature 0
        weights_feat0_double = np.array([2.0, 1.0])
        (util_double, idx_double, _, _, N_double, P_double) = find_max(
            p, X, y, Xas, np.arange(X.shape[0]), weights_feat0_double
        )

        # If feature 0 is optimal in both cases, utility should double
        np.testing.assert_almost_equal(idx_uniform, [0])
        np.testing.assert_almost_equal(util_double, util_uniform * 2.0)

    def test_shared_feature_values_threshold_grouping(self):
        """
        Test that the solver correctly groups examples sharing the same feature values
        and computes utility at the correct thresholds.
        """
        X = np.array([
            [1.0, 0.5],
            [1.0, 1.5],
            [1.0, 0.5],
            [2.0, 1.5],
        ], dtype=np.double)
        y = np.array([0, 1, 0, 1])
        Xas = np.argsort(X, axis=0).T.copy()
        p = 1.0
        weights = np.ones(X.shape[1])

        (util, idx, th, kind, N, P) = find_max(p, X, y, Xas, np.arange(X.shape[0]), weights)

        # Feature 0 has thresholds 1.0 and 2.0
        # Threshold 1.0: > 1.0 covers index 3 (y=1) -> N=0, P=1 -> util = -1
        #                 <= 1.0 covers indices 0,1,2 (y=[0,1,0]) -> N=2, P=1 -> util = 1
        np.testing.assert_almost_equal(util, 1.0)

        # Check that returned N and P_bar match the threshold 1.0 <= case for feature 0
        feat0_indices = [i for i, x in enumerate(idx) if x == 0]
        self.assertTrue(len(feat0_indices) > 0)
        for i in feat0_indices:
            np.testing.assert_almost_equal(N[i], 2)
            np.testing.assert_almost_equal(P[i], 1)

    def test_hand_calculated_utility_comparison(self):
        """
        Compare solver output against hand-calculated utility functions for a specific case.
        """
        X = np.array([[0.2, 0.8], [0.6, 0.3], [0.9, 0.9]], dtype=np.double)
        y = np.array([1, 0, 1])
        Xas = np.argsort(X, axis=0).T.copy()
        p = 2.0
        weights = np.array([1.0, 2.0])

        (solver_util, solver_idx, solver_th, solver_kind, solver_N, solver_P) = find_max(
            p, X, y, Xas, np.arange(X.shape[0]), weights
        )

        # Manually enumerate all possible thresholds and kinds to find true max utility
        true_max_util = -np.inf
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                # Kind 0: greater
                mask_gt = X[:, feat] > t
                N_gt = (~mask_gt[y == 0]).sum()
                P_gt = (~mask_gt[y == 1]).sum()
                util_gt = (N_gt - p * P_gt) * weights[feat]
                if util_gt > true_max_util:
                    true_max_util = util_gt

                # Kind 1: less_equal
                mask_le = X[:, feat] <= t
                N_le = (~mask_le[y == 0]).sum()
                P_le = (~mask_le[y == 1]).sum()
                util_le = (N_le - p * P_le) * weights[feat]
                if util_le > true_max_util:
                    true_max_util = util_le

        np.testing.assert_almost_equal(solver_util, true_max_util)
