"""
    pyscm -- The Set Covering Machine in Python
    Copyright (C) 2017 Alexandre Drouin

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from __future__ import print_function, division, absolute_import, unicode_literals

from .utils import _class_to_string


class BaseRule(object):
    """
    A rule mixin class

    """

    def __init__(self):
        super(BaseRule, self).__init__()

    def classify(self, X):
        """
        Classifies a set of examples using the rule.

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features), dtype=np.float
            The feature vectors of examples to classify.

        Returns:
        --------
        classifications: array-like, shape=(n_examples,), dtype=bool
            The outcome of the rule (True or False) for each example.

        """
        raise NotImplementedError()

    def inverse(self):
        """
        Creates a rule that is the opposite of the current rule (self).

        Returns:
        --------
        inverse: BaseRule
            A rule that is the inverse of self.

        """
        raise NotImplementedError()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return _class_to_string(self)


class DecisionStump(BaseRule):
    """
    A decision stump is a rule that applies a threshold to the value of some feature

    Parameters:
    -----------
    feature_idx: uint
        The index of the feature
    threshold: float
        The threshold at which the outcome of the rule changes
    kind: str, default="greater"
        The case in which the rule returns 1, either "greater" or "less_equal".

    """

    def __init__(self, feature_idx, threshold, kind="greater"):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.kind = kind
        super(DecisionStump, self).__init__()

    def classify(self, X):
        """
        Classifies a set of examples using the decision stump.

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features), dtype=np.float
            The feature vectors of examples to classify.

        Returns:
        --------
        classifications: array-like, shape=(n_examples,), dtype=bool
            The outcome of the rule (True or False) for each example.

        """
        if self.kind == "greater":
            c = X[:, self.feature_idx] > self.threshold
        else:
            c = X[:, self.feature_idx] <= self.threshold
        return c

    def inverse(self):
        """
        Creates a rule that is the opposite of the current rule (self).

        Returns:
        --------
        inverse: BaseRule
            A rule that is the inverse of self.

        """
        return DecisionStump(
            feature_idx=self.feature_idx,
            threshold=self.threshold,
            kind="greater" if self.kind == "less_equal" else "less_equal",
        )

    def __str__(self):
        return "X[{0:d}] {1!s} {2:.3f}".format(
            self.feature_idx, ">" if self.kind == "greater" else "<=", self.threshold
        )
