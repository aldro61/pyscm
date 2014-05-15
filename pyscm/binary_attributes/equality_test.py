#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    pyscm -- The Set Covering Machine in Python
    Copyright (C) 2014 Alexandre Drouin

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

import numpy as np

from .base import BinaryAttributeMixin
from .base import BinaryAttributeListMixin


class EqualityTest(BinaryAttributeMixin):
    """
    A binary attribute that checks if a feature has a given value.

    Parameters:
    -----------
    feature_idx: int
        The index of the feature in the example vectors.

    value: float
        The value for discriminating positive and negative examples.

    value: bool, default=True
        The outcome of the test if the examples feature at index feature_idx equals to value.

    example_dependencies: array_like, shape=(n_example_dependencies,), default=[]
            A list containing an element of any type for each example on which the attribute depends.
    """

    def __init__(self, feature_idx, value, outcome=True, example_dependencies=[]):
        self.feature_idx = feature_idx
        self.value = value
        self.outcome = outcome

        BinaryAttributeMixin.__init__(self, example_dependencies)

    def classify(self, X):
        """
        Classifies a set of examples using the equality test.

        Parameters:
        -----------
        X: numpy_array, (n_examples, n_features)
            The feature vectors of examples to classify.

        Returns:
        --------
        labels: numpy_array, (n_examples,)
            Labels assigned to each example by the test.
        """
        if self.outcome:
            labels = np.asarray(X[:, self.feature_idx] == self.value, dtype=np.int8)
        else:
            labels = np.asarray(X[:, self.feature_idx] != self.value, dtype=np.int8)

        return labels

    def inverse(self):
        """
        Creates an equality test that is the inverse of the current equality test (self).
        For any example, the label attributed by self must be the opposite of the label attributed
        by the inverse of self.

        Returns:
        --------
        inverse: EqualityTest
            A decision stump that is the inverse of self.
        """
        return EqualityTest(self.feature_idx, self.value, False, self.example_dependencies)

    def __str__(self):
        return "x[" + str(self.feature_idx) + "] " + ("==" if self.outcome == True else "!=") + " " + str(self.value)


class EqualityTestBinaryAttributeList(BinaryAttributeListMixin):
    """
    A equality test binary attribute list.

    Parameters:
    -----------
    feature_idx: numpy_array, shape=(n_indexes,)
        A list of indexes of the feature used to create the equality test in the example vectors and to classify a set
        of examples.

    values: numpy_array, shape=(n_values,)
        A list of values for discriminating positive and negative examples.

    outcomes: numpy_array, shape=(n_outcomes,)
        A list of outcomes of the test if the examples feature at index feature_idx equals to value.

    example_dependencies: array_like, shape=(n_items, n_example_dependencies), default=[]
            A list of lists of elements of any type for each example on which the attribute depends.
    """

    def __init__(self, feature_idx, values, outcomes=True, example_dependencies=[]):
        if len(set(map(len, (feature_idx, values, outcomes, example_dependencies)))) != 1:
            raise ValueError("EqualityTestBinaryAttributeList constructor: The input lists length should be equal.")

        self.feature_idx = np.asarray(feature_idx)
        self.values = np.asarray(values)
        self.outcomes = np.asarray(outcomes)
        self.example_dependencies = np.asarray(example_dependencies)

        BinaryAttributeListMixin.__init__(self)

    def __len__(self):
        return self.feature_idx.shape[0]

    def __getitem__(self, item_idx):
        return EqualityTest(self.feature_idx[item_idx], self.values[item_idx], self.outcomes[item_idx],
                            self.example_dependencies[item_idx])

    def classify(self, X):
        """
        Classifies a set of examples using the elements of equality test.

        Parameters:
        -----------
        X: numpy_array, (n_examples, n_features)
            The feature vectors of examples to classify.

        Returns:
        --------
        attribute_classifications: numpy_array, (n_examples, n_decision_stumps)
            List of labels assigned to each example by the equality test.
        """
        attribute_classifications = np.logical_xor(X[:, self.feature_idx] == self.values, self.outcomes)
        np.logical_not(attribute_classifications, out=attribute_classifications)
        return np.asarray(attribute_classifications, dtype=np.int8)
