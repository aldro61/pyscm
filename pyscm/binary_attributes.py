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
from .utils import _class_to_string

class BinaryAttribute(object):
    """
    A binary attribute template class

    Parameters:
    -----------
    example_dependencies: array_like, shape=(n_example_dependencies,), default=[]
            A list containing an element of any type for each example on which the attribute depends.
    """
    def __init__(self, example_dependencies=[]):
        self._example_dependencies = example_dependencies

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

    @property
    def example_dependencies(self):
        """
        Returns the example dependencies

        Returns:
        --------
        example_dependencies: array_like, shape=(n_example_dependencies,)
            A list containing an element of any type for each example on which the attribute depends.
        """
        return self._example_dependencies

    @example_dependencies.setter
    def example_dependencies(self, value):
        """
        Sets the example dependencies for a binary attribute. This is especially useful in the compression set setting.

        Parameters:
        -----------
        value: array_like, shape=(n_example_dependencies,)
            A list containing an element of any type for each example on which the attribute depends.
        """
        self._example_dependencies = value

    def __str__(self):
        return _class_to_string(self)



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

    example_dependencies: array_like, shape=(n_example_dependencies,), default=[]
            A list containing an element of any type for each example on which the attribute depends.
    """

    def __init__(self, feature_idx, direction, threshold, example_dependencies=[]):
        if direction != 1 and direction != -1:
            raise ValueError("Invalid decision stump direction.")

        self.feature_idx = feature_idx
        self.direction = direction
        self.threshold = threshold

        BinaryAttribute.__init__(self, example_dependencies)

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
        return DecisionStump(self.feature_idx, self.direction*-1, self.threshold, self.example_dependencies)

    def __str__(self):
        return "x[" + str(self.feature_idx) + "] " + (">" if self.direction == 1 else "<=") + " " + str(self.threshold)


class EqualityTest(BinaryAttribute):
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

        BinaryAttribute.__init__(self, example_dependencies)

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
        if self.outcome == True:
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