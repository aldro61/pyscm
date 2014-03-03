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

from model import ConjunctionModel, DisjunctionModel


class SetCoveringMachine:
    """
    The Set Covering Machine (SCM).

    Marchand, M., & Taylor, J. S. (2003). The set covering machine. Journal of Machine Learning Research, 3, 723–746.

    Parameters:
    -----------
    model_type: string, {"conjunction", "disjunction"}, default="conjunction"
        The type of model to be built.

    max_attributes: int, default=10
        The maximum number of binary attributes to include in the model.

    p: float, default=1.0
        The trade-off parameter for the SCM.
    """

    def __init__(self, model_type="conjunction", max_attributes=10, p=1.0):
        if model_type == "conjunction":
            self.model = ConjunctionModel()
        elif model_type == "disjunction":
            self.model = DisjunctionModel()
        else:
            raise ValueError("Unsupported model type.")
        self.model_type = model_type

        self.max_attributes = max_attributes
        self.p = p

    def fit(self, binary_attributes, X, y):
        """
        Fit a SCM model.

        Parameters:
        -----------
        binary_attributes: binary_attribute_like
            A list of binary attributes to be used to build the model.

        X: numpy_array, shape=(n_examples, n_features)
            The feature vectors associated to the training examples.

        y: numpy_array, shape=(n_examples,)
            The labels associated to the training examples.
        """
        binary_attributes = np.asarray(binary_attributes, dtype=np.object)

        classes, y = np.unique(y, return_inverse=True)
        self._classes = classes

        attribute_classifications = np.zeros((len(binary_attributes), X.shape[0]))
        for i, a in enumerate(binary_attributes):
            attribute_classifications[i] = a.classify(X)

        if self.model_type == "conjunction":
            negative_example_idx = np.where(y == 0)[0]
            positive_example_idx = np.where(y == 1)[0]
        elif self.model_type == "disjunction":
            negative_example_idx = np.where(y == 1)[0]
            positive_example_idx = np.where(y == 0)[0]

        while len(negative_example_idx) > 0 and len(self.model) < self.max_attributes and len(binary_attributes) > 0:
            negative_cover_counts = np.sum(attribute_classifications[:, negative_example_idx], axis=1) * -1 + len(
                negative_example_idx)
            positive_error_counts = np.sum(attribute_classifications[:, positive_example_idx], axis=1) * -1 + len(
                positive_example_idx)
            utilities = negative_cover_counts - self.p * positive_error_counts

            best_attribute_idx = np.argmax(utilities)

            if self.model_type == "conjunction":
                self.model.add(binary_attributes[best_attribute_idx])
            elif self.model_type == "disjunction":
                self.model.add(binary_attributes[best_attribute_idx].inverse())

            # Remove the covered negative examples from the negative example set
            negative_example_idx = negative_example_idx[
                attribute_classifications[best_attribute_idx][negative_example_idx] != 0]

            # Remove the mistaken positive examples from the positive example set
            positive_example_idx = positive_example_idx[
                attribute_classifications[best_attribute_idx][positive_example_idx] != 0]

            # Remove the stumps that were created from the same attribute of the input space
            np.delete(attribute_classifications, best_attribute_idx, axis=0)
            np.delete(binary_attributes, best_attribute_idx)

    def predict(self, X):
        """
        Compute predictions.

        Parameters:
        -----------
        X: numpy_array, shape=(n_examples,)
            The feature vectors associated to some examples.

        Returns:
        --------
        predictions: numpy_array, shape=(n_examples,)
            The predicted class for each example.
        """
        if not self._is_fitted():
            raise RuntimeError("A model must be fitted prior to calling predict.")
        return self._classes.take(self.model.predict(X), dtype=np.int8)

    def _is_fitted(self):
        return len(self.model) > 0