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


class DecisionStump(BinaryAttributeMixin):
    """
    A decision stump binary attribute.

    Parameters:
    -----------
    feature_idx: int
        The index of the feature used to create the decision stump in the example vectors.

    direction: int, {-1, +1}
        The direction of the decision stump. 1 stands for feature_value > threshold, whereas -1 stands for
        feature_value < threshold.

    threshold: float
        The decision stump's threshold value for discriminating positive and negative examples.

    example_dependencies: array_like, shape=(n_example_dependencies,), default=[]
            A list containing an element of any type for each example on which the attribute depends.
    """

    def __init__(self, feature_idx, direction, threshold, example_dependencies=[]):
        if direction != 1 and direction != -1:
            raise ValueError("Invalid decision stump direction.")

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
            The feature vectors of examples to classify.

        Returns:
        --------
        labels: numpy_array, (n_examples,)
            Labels assigned to each example by the decision stump.
        """
        if self.direction == 1:
            labels = np.array(X[:, self.feature_idx] > self.threshold, dtype=np.int8)
        else:
            labels = np.array(X[:, self.feature_idx] < self.threshold, dtype=np.int8)
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
        return DecisionStump(self.feature_idx, self.direction * -1, self.threshold, self.example_dependencies)

    def __str__(self):
        return "x[" + str(self.feature_idx) + "] " + (">" if self.direction == 1 else "<=") + " " + str(self.threshold)


class DecisionStumpBinaryAttributeList(BinaryAttributeListMixin):
    """
    A decision stump binary attribute list.

    Parameters:
    -----------
    feature_idx: numpy_array, shape=(n_indexes,)
        A list of indexes of the feature used to create the decision stump in the example vectors and to classify a set
        of examples.

    directions: numpy_array, shape=(n_directions,)
        A list of directions of the decision stump. Possible values {-1, +1}: 1 stands for feature_value > threshold,
        whereas -1 stands for feature_value < threshold.

    thresholds: numpy_array, shape=(n_thresholds,)
        A list of decision stump's threshold values for discriminating positive and negative examples.

    example_dependencies: array_like, shape=(n_items, n_example_dependencies), default=[]
            A list of lists of elements of any type for each example on which the attribute depends.
    """

    def __init__(self, feature_idx, directions, thresholds, example_dependencies=[]):
        if len(set(map(len, (feature_idx, directions, thresholds, example_dependencies)))) != 1:
            raise ValueError("DecisionStumpBinaryAttributeList constructor: The input lists length should be equal.")

        self.feature_idx = np.asarray(feature_idx)
        self.directions = np.asarray(directions)
        self.thresholds = np.asarray(thresholds)
        self.example_dependencies = np.asarray(example_dependencies)

        BinaryAttributeListMixin.__init__(self)

    def __len__(self):
        return self.feature_idx.shape[0]

    def __getitem__(self, item_idx):
        return DecisionStump(self.feature_idx[item_idx], self.directions[item_idx], self.thresholds[item_idx],
                             self.example_dependencies[item_idx])

    def classify(self, X):
        """
        Classifies a set of examples using the elements of decision stump.

        Parameters:
        -----------
        X: numpy_array, (n_examples, n_features)
            The feature vectors of examples to classify.

        Returns:
        --------
        attribute_classifications: numpy_array, (n_examples, n_decision_stumps)
            List of labels assigned to each example by the decision stump.
        """
        attribute_classifications = (X[:, self.feature_idx] - self.thresholds) * self.directions
        attribute_classifications[attribute_classifications > 0] = 1
        attribute_classifications[attribute_classifications <= 0] = 0
        return attribute_classifications
