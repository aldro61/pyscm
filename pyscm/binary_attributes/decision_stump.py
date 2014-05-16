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


class DecisionStump(BinaryAttributeMixin):
    """
    A decision stump binary attribute.

    Parameters:
    -----------
    feature_idx: int
        The index of the feature used to create the decision stump in the example vectors.

    direction: int, value in {-1, +1}
        The direction of the decision stump. 1 stands for feature_value > threshold, and -1 stands for
        feature_value < threshold.

    threshold: float
        The decision stump's threshold value for discriminating positive and negative examples.

    example_dependencies: array_like, shape=(n_example_dependencies,), default=[]
            A list containing an identifier for each training example on which the decision stump depends.
    """

    def __init__(self, feature_idx, direction, threshold, example_dependencies=[]):
        if direction != 1 and direction != -1:
            raise ValueError("Invalid decision stump direction.")

        #TODO: check that direction is -1 or 1
        self.feature_idx = feature_idx
        self.direction = direction
        self.threshold = threshold

        BinaryAttributeMixin.__init__(self, example_dependencies)

    def classify(self, X):
        """
        Classifies a set of examples using the decision stump.

        Parameters:
        -----------
        X: numpy_array, (n_examples, n_features)
            The feature vectors of the examples to classify.

        Returns:
        --------
        labels: numpy_array, (n_examples,)
            Labels assigned to each example by the decision stump.
        """
        if self.direction == 1:
            labels = np.asarray(X[:, self.feature_idx] > self.threshold, dtype=np.uint8)
        else:
            labels = np.asarray(X[:, self.feature_idx] < self.threshold, dtype=np.uint8)
        return labels

    def inverse(self):
        """
        Creates a decision stump that is the inverse of the current decision stump (self).
        For any example, the label attributed by self must be the opposite of the label attributed
        by the inverse of self.

        Returns:
        --------
        inverse: DecisionStump
            A decision stump that is the inverse of self.
        """
        return DecisionStump(feature_idx=self.feature_idx,
                             direction=self.direction * -1,
                             threshold=self.threshold,
                             example_dependencies=self.example_dependencies)

    def __str__(self):
        return "x[" + str(self.feature_idx) + "] " + (">" if self.direction == 1 else "<") + " " + str(self.threshold)


class DecisionStumpList(BinaryAttributeListMixin):
    """
    A binary attribute list specially designed for decision stumps.

    Parameters:
    -----------
    feature_idx: numpy_array, shape=(n_decision_stumps,)
        The feature indexes used to define each decision stump.

    directions: numpy_array, shape=(n_decision_stumps,)
        The directions used to define each decision stump.
        Possible values {-1, +1}: 1 stands for feature_value > threshold and -1 stands for feature_value < threshold.

    thresholds: numpy_array, shape=(n_decision_stumps,)
        The thresholds used to define each decision stump. Decision stumps use thresholds to discriminate positive and
        negative examples.

    example_dependencies: list of lists, default=None
            A list of lists of containing an identifier for each examples on which the decision stumps depend on.

    Note:
    -----
    This class uses lazy generation of the DecisionStump objects to reduce memory consumption.
    """

    def __init__(self, feature_idx, directions, thresholds, example_dependencies=None):
        if example_dependencies is None:
            if len(set(map(len, (feature_idx, directions, thresholds)))) != 1:
                raise ValueError("DecisionStumpList constructor: The input lists length should be equal.")
        else:
            if len(set(map(len, (feature_idx, directions, thresholds, example_dependencies)))) != 1:
                raise ValueError("DecisionStumpList constructor: The input lists length should be equal.")

        if not all((value == 1 or value == -1) for value in directions):
            raise ValueError('The directions list should\'t contain the values other than {-1, +1}')

        self.feature_idx = np.asarray(feature_idx)
        self.directions = np.asarray(directions, np.int8)
        self.thresholds = np.asarray(thresholds)
        self.example_dependencies = example_dependencies

        BinaryAttributeListMixin.__init__(self)

    def __len__(self):
        return self.feature_idx.shape[0]

    def __getitem__(self, item_idx):
        if item_idx > len(self) - 1:
            raise IndexError()

        return DecisionStump(feature_idx=self.feature_idx[item_idx],
                             direction=self.directions[item_idx],
                             threshold=self.thresholds[item_idx],
                             example_dependencies=[] if self.example_dependencies is None \
                                 else self.example_dependencies[item_idx])

    def classify(self, X):
        """
        Classifies a set of examples using the decision stumps in the list.

        Parameters:
        -----------
        X: numpy_array, (n_examples, n_features)
            The feature vectors of the examples to classify.

        Returns:
        --------
        attribute_classifications: numpy_array, (n_examples, n_decision_stumps)
            A matrix containing the labels assigned to each example by each decision stump individually.
        """
        attribute_classifications = np.zeros((X.shape[0], len(self)), dtype=np.uint8)
        block_size = 1000 #TODO: Make this a parameter or compute an optimal value based on memory usage
        for i in xrange(int(ceil(float(len(self))/block_size))):
            tmp = (X[:, self.feature_idx[i*block_size:(i+1)*block_size]] - self.thresholds[i*block_size:(i+1)*block_size]) * self.directions[i*block_size:(i+1)*block_size]
            attribute_classifications[:, i*block_size:(i+1)*block_size][tmp > 0] = 1
            attribute_classifications[:, i*block_size:(i+1)*block_size][tmp <= 0] = 0
        return attribute_classifications
