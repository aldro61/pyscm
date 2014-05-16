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

from math import ceil
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

    outcome: bool, default=True
        The outcome of the test if the examples feature at index feature_idx equals to value.

    example_dependencies: array_like, shape=(n_example_dependencies,), default=[]
            A list containing an identifier for each training example on which the attribute depends.
    """

    def __init__(self, feature_idx, value, outcome=True, example_dependencies=[]):
        self.feature_idx = feature_idx
        self.value = value
        #TODO: check that outcome is True or False
        self.outcome = outcome

        BinaryAttributeMixin.__init__(self, example_dependencies)

    def classify(self, X):
        """
        Classifies a set of examples using the equality test.

        Parameters:
        -----------
        X: numpy_array, (n_examples, n_features)
            The feature vectors of the examples to classify.

        Returns:
        --------
        labels: numpy_array, (n_examples,)
            Labels assigned to each example by the test.
        """
        if self.outcome:
            labels = np.asarray(X[:, self.feature_idx] == self.value, dtype=np.uint8)
        else:
            labels = np.asarray(X[:, self.feature_idx] != self.value, dtype=np.uint8)

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
        return EqualityTest(feature_idx=self.feature_idx,
                            value=self.value,
                            outcome=False,
                            example_dependencies=self.example_dependencies)

    def __str__(self):
        return "x[" + str(self.feature_idx) + "] " + ("==" if self.outcome == True else "!=") + " " + str(self.value)


class EqualityTestList(BinaryAttributeListMixin):
    """
    A binary attribute list specially designed for equality tests.

    Parameters:
    -----------
    feature_idx: numpy_array, shape=(n_indexes,)
        The feature indexes used to define each equality test.

    values: numpy_array, shape=(n_values,)
        The values used to define each equality test. Equality tests use these values to discriminate positive and
        negative examples.

    outcomes: numpy_array, shape=(n_outcomes,)
        The outcomes of each equality test. The outcome defines whether the test is feature_value == value or
        feature_value != value.

    example_dependencies: list of lists, default=None
            A list of lists of containing an identifier for each examples on which the equality tests depend on.

    Note:
    -----
    This class uses lazy generation of the EqualityTest objects to reduce memory consumption.
    """

    def __init__(self, feature_idx, values, outcomes, example_dependencies=None):
        if example_dependencies is None:
            if len(set(map(len, (feature_idx, values, outcomes)))) != 1:
                raise ValueError("EqualityTestList constructor: The input lists length should be equal.")
        else:
            if len(set(map(len, (feature_idx, values, outcomes, example_dependencies)))) != 1:
                raise ValueError("EqualityTestList constructor: The input lists length should be equal.")

        if not all((value == True or value == False) for value in outcomes):
            raise ValueError('The outcomes list  should\'t contain the values other than {True, False}')

        self.feature_idx = np.asarray(feature_idx)
        self.values = np.asarray(values)
        self.outcomes = np.asarray(outcomes, np.bool)
        self.example_dependencies = example_dependencies

        BinaryAttributeListMixin.__init__(self)

    def __len__(self):
        return self.feature_idx.shape[0]

    def __getitem__(self, item_idx):
        if item_idx > len(self) - 1:
            raise IndexError()

        return EqualityTest(feature_idx=self.feature_idx[item_idx],
                            value=self.values[item_idx],
                            outcome=self.outcomes[item_idx],
                            example_dependencies=[] if self.example_dependencies is None \
                                else self.example_dependencies[item_idx])

    def classify(self, X):
        """
        Classifies a set of examples using the equality tests in the list.

        Parameters:
        -----------
        X: numpy_array, (n_examples, n_features)
            The feature vectors of the examples to classify.

        Returns:
        --------
        attribute_classifications: numpy_array, (n_examples, n_decision_stumps)
            A matrix containing the labels assigned to each example by each equality test individually.
        """
        attribute_classifications = np.zeros((X.shape[0], len(self)), dtype=np.uint8)
        block_size = 1000 #TODO: Make this a parameter or compute an optimal value based on memory usage
        for i in xrange(int(ceil(float(len(self))/block_size))):
            tmp = np.logical_xor(
                X[:, self.feature_idx[i*block_size:(i+1)*block_size]] == self.values[i*block_size:(i+1)*block_size],
                self.outcomes[i*block_size:(i+1)*block_size])
            np.logical_not(tmp, out=attribute_classifications[i*block_size:(i+1)*block_size])
        return attribute_classifications


