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
from functools import partial
from math import ceil

import numpy as np

from .utils import _conditional_print, _class_to_string
from .model import ConjunctionModel, DisjunctionModel, conjunction, disjunction


class SetCoveringMachine(object):
    """
    The Set Covering Machine (SCM).

    Marchand, M., & Taylor, J. S. (2003). The set covering machine. Journal of Machine Learning Research, 3, 723–746.

    Parameters:
    -----------
    model_type: string, {"conjunction", "disjunction"}, default="conjunction"
        The type of model to be built.

    p: float, default=1.0
        The trade-off parameter for the SCM.

    max_attributes: int, default=10
        The maximum number of binary attributes to include in the model.

    verbose: bool, default=False
        Sets verbose mode on/off.
    """

    def __init__(self, model_type=conjunction, p=1.0, max_attributes=10, verbose=False):
        if model_type == "conjunction":
            self.model = ConjunctionModel()
        elif model_type == "disjunction":
            self.model = DisjunctionModel()
        else:
            raise ValueError("Unsupported model type.")
        self.model_type = model_type

        self.max_attributes = max_attributes
        self.p = p

        self.verbose = verbose
        self._verbose_print = partial(_conditional_print, condition=verbose)


    def fit(self, X, y, binary_attributes, attribute_classifications=None, model_append_callback=None):
        """
        Fit a SCM model.

        Parameters:
        -----------
        X: numpy_array, shape=(n_examples, n_features)
            The feature vectors associated to the training examples.

        y: numpy_array, shape=(n_examples,)
            The labels associated to the training examples.

        binary_attributes: binary_attribute_like
            A list of unique binary attributes to be used to build the model.

        attribute_classifications: numpy_array, shape=(n_binary_attributes, n_examples), default=None
            The binary attribute labels (0 or 1) assigned to the examples in X. This can be used to precompute the long
            classification process. If the value is None, the classifications will be computed.

        model_append_callback: function, arguments: new_attribute=instance_of(BinaryAttribute), default=None
            A function which is called when a new binary attribute is appended to the model.

        Notes:
        ------
        * HDF5: The SCM can learn from a great number of attributes. Storing them in memory can require a large amount
                of memory space. Therefore, great care is taken to allow attribute_classifications to be a HDF5 dataset.
                We try to prevent loading the entire dataset into memory. The user is assumed to be using h5py.
        """

        classes, y = np.unique(y, return_inverse=True)
        self._classes = classes
        self._verbose_print(
            "Example classes are: positive (" + str(self._classes[1]) + "), negative (" + str(self._classes[0]) + ")")

        self._verbose_print("Got " + str(len(binary_attributes)) + " binary attributes.")
        if attribute_classifications is None:
            self._verbose_print("Classifying the examples with the binary attributes")
            attribute_classifications = np.zeros((X.shape[0], len(binary_attributes)), dtype=np.uint8)
            for i, a in enumerate(binary_attributes):
                attribute_classifications[:, i] = a.classify(X)
        else:
            self._verbose_print("Binary attribute classifications were precomputed")
            if attribute_classifications.shape[1] != len(binary_attributes):
                raise ValueError(
                    "The number of attributes must match in attribute_classifications and binary_attributes.")

            if attribute_classifications.shape[0] != X.shape[0]:
                raise ValueError("The number of examples must match in attribute_classifications and X.")

        if self.model_type == conjunction:
            negative_example_idx = np.where(y == 0)[0]
            positive_example_idx = np.where(y == 1)[0]
        elif self.model_type == disjunction:
            negative_example_idx = np.where(y == 1)[0]
            positive_example_idx = np.where(y == 0)[0]
        del X, y

        block_size = 500000
        n_blocks = int(ceil(float(attribute_classifications.shape[1]) / block_size))
        while len(negative_example_idx) > 0 and len(self.model) < self.max_attributes and len(binary_attributes) > 0:

            self._verbose_print("Counting covered negative examples")
            count = np.zeros(attribute_classifications.shape[1])
            for i in xrange(n_blocks):
                count[i * block_size: (i + 1) * block_size] = np.sum(
                    attribute_classifications[negative_example_idx, i * block_size: (i + 1) * block_size], axis=0)
                self._verbose_print("Block " + str(i+1) + " of " + str(n_blocks))
            negative_cover_counts = count * -1 + negative_example_idx.shape[0]
            del count

            self._verbose_print("Couting errors on positive examples")
            count = np.zeros(attribute_classifications.shape[1])
            for i in xrange(n_blocks):
                count[i * block_size: (i + 1) * block_size] = np.sum(
                    attribute_classifications[positive_example_idx, i * block_size: (i + 1) * block_size], axis=0)
                self._verbose_print("Block " + str(i+1) + " of " + str(n_blocks))
            positive_error_counts = count * -1 + positive_example_idx.shape[0]
            del count

            self._verbose_print("Computing attribute utilities")
            utilities = negative_cover_counts - self.p * positive_error_counts
            del negative_cover_counts, positive_error_counts

            best_attribute_idx = np.argmax(utilities)
            best_attribute = binary_attributes[best_attribute_idx]
            if self.model_type == conjunction:
                new_attribute = best_attribute
                self.model.add(new_attribute)
            elif self.model_type == disjunction:
                new_attribute = best_attribute.inverse()
                self.model.add(new_attribute)
            if model_append_callback is not None:
                model_append_callback(new_attribute)
            self._verbose_print("Attribute added to the model (Utility: " + str(utilities[best_attribute_idx]) + \
                                    "): " + str(new_attribute))
            del utilities, new_attribute, best_attribute

            self._verbose_print("Discarding covered negative examples")
            negative_example_idx = negative_example_idx[
                attribute_classifications[negative_example_idx, best_attribute_idx] != 0]
            self._verbose_print("Discarding misclassified positive examples")
            positive_example_idx = positive_example_idx[
                attribute_classifications[positive_example_idx, best_attribute_idx] != 0]
            self._verbose_print("Remaining negative examples:" + str(len(negative_example_idx)))
            self._verbose_print("Remaining positive examples:" + str(len(positive_example_idx)))

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
        return self._classes.take(self.model.predict(X))

    def _is_fitted(self):
        return len(self.model) > 0

    def __str__(self):
        return _class_to_string(self)