from __future__ import print_function, division, absolute_import, unicode_literals

import sys
import numpy as np

from unittest import TestCase
from sklearn.utils import estimator_checks
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from ..scm import SetCoveringMachineClassifier


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class SklearnCompatibilityTests(TestCase):
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

    def test_sklearn_gridSearch(self):
        """
        Test compatibility with sklearn GridSearchCV function

        """
        rnd = np.random.RandomState(0)
        X = np.random.rand(100,100)
        y = np.random.randint(2, size=100)

        scm_param_grid = {"p": [0.01, 0.1, 1, 10, 100],
                          "max_rules": [1,2,3,4,5],
                          "model_type": ["conjunction", "disjunction"]}
        gscv = GridSearchCV(SetCoveringMachineClassifier(), scm_param_grid, n_jobs=1, cv=5)
        try:
            gscv.fit(X, y)
        except Exception as e:
            self.fail("GridSearchCV.fit() raised " + e.message + " unexpectedly!")

        gscv = GridSearchCV(SetCoveringMachineClassifier(), scm_param_grid, n_jobs=2, cv=5)
        try:
            gscv.fit(X, y)
        except Exception as e:
            self.fail("GridSearchCV.fit() raised "+e.message+" unexpectedly when running with 2 jobs!")


    def test_sklearn_pipeline(self):
        """
        Test compatibility with sklearn pipelines

        """
        rnd = np.random.RandomState(0)
        X = np.random.rand(10, 10)
        y = np.random.randint(2, size=10)

        scm = SetCoveringMachineClassifier(model_type="conjunction", p=0, max_rules=3)
        pipeline = Pipeline([("Scale", StandardScaler()),
                             ("SCM", scm)])
        try:
            pipeline.fit(X, y)
        except Exception as e:
            self.fail("Pipeline.fit() raised "+e.message+" unexpectedly!")


def _yield_check_sklearn_compatibility():
    """
    Yields Sklearn compatibility tests but most of them fail.
    Uses sklearn test suits.
    If fails, will raise an exception
    pySCM is fully compatible except it does not handle multi-class causing most of these test to fail.
    """
    yield estimator_checks.check_parameters_default_constructible("SetCoveringMachineClassifier",
                                                         SetCoveringMachineClassifier)
    for check in estimator_checks._yield_non_meta_checks("SetCoveringMachineClassifier",
                                                         SetCoveringMachineClassifier):
        yield check

    for check in estimator_checks._yield_classifier_checks("SetCoveringMachineClassifier",
                                                           SetCoveringMachineClassifier):
        yield check

    # Will fail because SCM does not handle multi-class.
    #yield estimator_checks.check_fit2d_predict1d("SetCoveringMachineClassifier",
    #                                                     SetCoveringMachineClassifier)
    yield estimator_checks.check_fit2d_1sample("SetCoveringMachineClassifier",
                                                         SetCoveringMachineClassifier)
    yield estimator_checks.check_fit2d_1feature("SetCoveringMachineClassifier",
                                                         SetCoveringMachineClassifier)
    yield estimator_checks.check_fit1d_1feature("SetCoveringMachineClassifier",
                                                         SetCoveringMachineClassifier)
    yield estimator_checks.check_fit1d_1sample("SetCoveringMachineClassifier",
                                                         SetCoveringMachineClassifier)
    yield estimator_checks.check_get_params_invariance("SetCoveringMachineClassifier",
                                                         SetCoveringMachineClassifier)
    yield estimator_checks.check_dict_unchanged("SetCoveringMachineClassifier",
                                                         SetCoveringMachineClassifier)