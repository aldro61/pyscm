from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from unittest import TestCase

from ..scm import SetCoveringMachineClassifier


class SklearnCompatibilityTests(TestCase):
    def test_sklearn_grid_search(self):
        """
        Test compatibility with sklearn GridSearchCV function

        """
        rnd = np.random.RandomState(0)
        X = np.random.rand(100, 100)
        y = np.random.randint(2, size=100)

        scm_param_grid = {
            "p": [0.01, 0.1, 1, 10, 100],
            "max_rules": [1, 2, 3, 4, 5],
            "model_type": ["conjunction", "disjunction"],
            "random_state": [rnd],
        }
        gscv = GridSearchCV(
            SetCoveringMachineClassifier(), scm_param_grid, n_jobs=1, cv=5
        )
        try:
            gscv.fit(X, y)
        except Exception as e:
            self.fail("GridSearchCV.fit() raised " + e.message + " unexpectedly!")

        gscv = GridSearchCV(
            SetCoveringMachineClassifier(), scm_param_grid, n_jobs=2, cv=5
        )
        try:
            gscv.fit(X, y)
        except Exception as e:
            self.fail(
                "GridSearchCV.fit() raised "
                + e.message
                + " unexpectedly when running with 2 jobs!"
            )

    def test_sklearn_pipeline(self):
        """
        Test compatibility with sklearn pipelines

        """
        rnd = np.random.RandomState(0)
        X = np.random.rand(10, 10)
        y = np.random.randint(2, size=10)

        scm = SetCoveringMachineClassifier(
            model_type="conjunction", p=0, max_rules=3, random_state=rnd
        )
        pipeline = Pipeline([("Scale", StandardScaler()), ("SCM", scm)])
        try:
            pipeline.fit(X, y)
        except Exception as e:
            self.fail("Pipeline.fit() raised " + e.message + " unexpectedly!")
