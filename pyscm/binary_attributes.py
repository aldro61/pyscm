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


class BinaryAttribute:
    """
    A binary attribute template class
    """

    def classify(self, X):
        """
        Classifies a set of examples using the binary attribute.

        Parameters:
        -----------
        X: numpy_array, (n_examples, n_features)
            The feature vectors of examples to classify.

        Returns:
        --------
        labels: numpy_array, (n_examples,)
            Labels assigned to each example by the binary attribute.
        """
        raise NotImplementedError()

    def inverse(self):
        """
        Creates a binary attribute that is the inverse of the current binary attribute (self).
        For any example, the label attributed by self must be the opposite of the label attributed
        by the inverse of self.
        
        Returns:
        --------
        inverse: BinaryAttribute
            A binary attribute that is the inverse of self.
        """
        raise NotImplementedError()


class DecisionStump(BinaryAttribute):
    """
    A decision stump binary attribute.

    Parameters:
    -----------
    feature_idx: int
        The index of the feature used to create the decision stump in the example vectors.

    direction: int, {-1, +1}
        The direction of the decision stump. 1 stands for feature_value > threshold, whereas -1 stands for
        feature_value <= threshold.

    threshold: float
        The decision stump's threshold value for discriminating positive and negative examples.
    """

    def __init__(self, feature_idx, direction, threshold):
        if direction != 1 and direction != -1:
            raise ValueError("Invalid decision stump direction.")

        self.feature_idx = feature_idx
        self.direction = direction
        self.threshold = threshold

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
            labels = np.array(X[:, self.feature_idx] <= self.threshold, dtype=np.int8)
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
        return DecisionStump(self.feature_idx, self.direction*-1, self.threshold)

    def __str__(self):
        return "x[" + str(self.feature_idx) + "] " + (">" if self.direction == 1 else "<=") + " " + str(self.threshold)
