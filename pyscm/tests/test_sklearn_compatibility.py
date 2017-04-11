from __future__ import print_function, division, absolute_import, unicode_literals

import sys

from unittest import TestCase
from sklearn.utils import estimator_checks

from ..scm import SetCoveringMachineClassifier


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class SklearnCompatibilityTests(TestCase):
    def test_sklearn_compatibility(self):
        """
        Test Sklearn compatibility

        """
        for check in _yield_check_sklearn_compatibility():
            check


def _yield_check_sklearn_compatibility():
    """
    Yields Sklearn compatibility tests.
    Uses sklearn test suits.
    If fails, will raise an exception
    pySCM is fully compatible except it does not handle multi-class.
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