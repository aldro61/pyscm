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

from functools import partial
from math import ceil

from .utils import _conditional_print, _class_to_string, _split_into_contiguous
from .model import ConjunctionModel, DisjunctionModel, conjunction, disjunction


def _block_sum_rows(row_idx, array, block_size=1000, verbose=False):
    _verbose_print = partial(_conditional_print, condition=verbose)

    contiguous_rows = _split_into_contiguous(row_idx)
    n_column_blocks = int(ceil(float(array.shape[1]) / block_size))

    sum_res = np.zeros(array.shape[1])
    row_count = 0
    for row_block in contiguous_rows:
        for j in xrange(n_column_blocks):
            sum_res[j * block_size: (j + 1) * block_size] += np.sum(array[min(row_block) : max(row_block)+1,
                                                                          j * block_size: (j + 1) * block_size], axis=0)
        row_count += len(row_block)
        _verbose_print("Processed " + str(row_count) + " of " + str(len(row_idx)) + " rows")

    return sum_res


class SetCoveringMachine(object):
    """
    The Set Covering Machine (SCM).

    Marchand, M., & Taylor, J. S. (2003). The set covering machine. Journal of Machine Learning Research, 3, 723–746.

    Parameters:
    -----------
    model_type: pyscm.model.conjunction or pyscm.model.disjunction, default=pyscm.model.conjunction
        The type of model to be built.

    p: float, default=1.0
        The trade-off parameter for the SCM.

    max_attributes: int, default=10
        The maximum number of binary attributes to include in the model.

    verbose: bool, default=False
        Sets verbose mode on/off.
    """

    def __init__(self, model_type=conjunction, p=1.0, max_attributes=10, verbose=False):
        self.verbose = verbose
        self._verbose_print = partial(_conditional_print, condition=verbose)

        if model_type == conjunction:
            self.model = ConjunctionModel()
            self._add_attribute_to_model = self._append_conjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_conjunction
        elif model_type == disjunction:
            self.model = DisjunctionModel()
            self._add_attribute_to_model = self._append_disjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_disjunction
        else:
            raise ValueError("Unsupported model type.")
        self.model_type = model_type
        self._verbose_print("Model type is: " + model_type)

        self.max_attributes = max_attributes
        self.p = p


    def fit(self, binary_attributes, y, X=None, attribute_classifications=None, model_append_callback=None,
            cover_count_block_size=1000):
        """
        Fit a SCM model.

        Parameters:
        -----------
        binary_attributes: binary_attribute_like
            A list of unique binary attributes to be used to build the model.

        y: numpy_array, shape=(n_examples,)
            The labels associated to the training examples. y must contain 2 unique class identifiers. The smallest
            class identifier is attributed to negative examples.

        X: numpy_array, shape=(n_examples, n_features), default=None
            The feature vectors associated to the training examples. If X is None, then attribute_classifications is
            expected not to be None.

        attribute_classifications: numpy_array, shape=(n_binary_attributes, n_examples), default=None
            The labels (0 or 1) assigned to the examples in X assigned by each binary attribute individually. This can
            be used to precompute the long classification process. If the value is None, the classifications will be
            computed using X. Thus, if attribute_classifications is None, X is expected not to be None.

        model_append_callback: function, arguments: new_attribute=instance_of(BinaryAttribute), default=None
            A function which is called when a new binary attribute is appended to the model.

        cover_count_block_size: int, default=1000
            The maximum number of attributes for which covers are counted at one time. Use this to limit memory usage.

        Notes:
        ------
        * HDF5: The SCM can learn from a great number of attributes. Storing them in memory can require a large amount
                of memory space. Therefore, great care is taken to allow attribute_classifications to be a HDF5 dataset.
                We try to prevent loading the entire dataset into memory.

        """
        if X is None and attribute_classifications is None:
            raise ValueError("X or attribute_classifications must have a value.")

        classes, y = np.unique(y, return_inverse=True)
        if len(classes) < 2 or len(classes) > 2:
            raise ValueError("y must contain two unique classes.")
        self._classes = classes
        self._verbose_print("Example classes are: positive (" + str(self._classes[1]) + "), negative (" + \
                            str(self._classes[0]) + ")")
        del classes

        positive_example_idx, negative_example_idx = self._get_example_idx_by_class(y)

        self._verbose_print("Got " + str(len(binary_attributes)) + " binary attributes.")
        if attribute_classifications is None:
            self._verbose_print("Classifying the examples with the binary attributes")
            attribute_classifications = binary_attributes.classify(X)
        else:
            self._verbose_print("Binary attribute classifications were precomputed")
            if attribute_classifications.shape[1] != len(binary_attributes):
                raise ValueError("The number of attributes must match in attribute_classifications and",
                                 "binary_attributes.")
        del X, y

        while len(negative_example_idx) > 0 and len(self.model) < self.max_attributes:
            self._verbose_print("Counting covered negative examples")
            negative_cover_counts = negative_example_idx.shape[0] - _block_sum_rows(negative_example_idx,
                                                                                    attribute_classifications,
                                                                                    cover_count_block_size,
                                                                                    self.verbose)

            self._verbose_print("Counting errors on positive examples")
            positive_error_counts = positive_example_idx.shape[0] - _block_sum_rows(positive_example_idx,
                                                                                    attribute_classifications,
                                                                                    cover_count_block_size,
                                                                                    self.verbose)

            self._verbose_print("Computing attribute utilities")
            utilities = negative_cover_counts - self.p * positive_error_counts
            del negative_cover_counts, positive_error_counts

            best_attribute_idx = np.argmax(utilities)
            self._verbose_print("Greatest utility is " + str(utilities[best_attribute_idx]))

            if self.verbose:  # Save the computation if verbose is off
                equal_utility_idx = np.where(utilities == utilities[best_attribute_idx])[0]
                self._verbose_print("There are " + str(len(equal_utility_idx) - 1) + \
                                    " attributes with the same utility.")
                if len(equal_utility_idx) > 1:
                    self._verbose_print("These are:")
                    for idx in equal_utility_idx:
                        if idx != best_attribute_idx:
                            self._verbose_print(binary_attributes[idx])

            appended_attribute = self._add_attribute_to_model(binary_attributes[best_attribute_idx])
            if model_append_callback is not None:
                model_append_callback(appended_attribute)
            del utilities, appended_attribute

            self._verbose_print("Discarding covered negative examples")
            # TODO: This is a workaround to issue #425 of h5py (Currently unsolved)
            # https://github.com/h5py/h5py/issues/425
            if len(negative_example_idx) > 1:
                negative_example_idx = negative_example_idx[
                    attribute_classifications[negative_example_idx, best_attribute_idx] != 0]
            else:
                keep = attribute_classifications[negative_example_idx, best_attribute_idx] != 0
                keep = keep.reshape((1,))
                negative_example_idx = negative_example_idx[keep]

            self._verbose_print("Discarding misclassified positive examples")
            # TODO: This is a workaround to issue #425 of h5py (Currently unsolved)
            # https://github.com/h5py/h5py/issues/425
            if len(positive_example_idx) > 1:
                positive_example_idx = positive_example_idx[
                    attribute_classifications[positive_example_idx, best_attribute_idx] != 0]
            else:
                keep = attribute_classifications[positive_example_idx, best_attribute_idx] != 0
                keep = keep.reshape((1,))
                positive_example_idx = positive_example_idx[keep]

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

    def _append_conjunction_model(self, new_attribute):
        self.model.add(new_attribute)
        self._verbose_print("Attribute added to the model: " + str(new_attribute))
        return new_attribute

    def _append_disjunction_model(self, new_attribute):
        new_attribute = new_attribute.inverse()
        self.model.add(new_attribute)
        self._verbose_print("Attribute added to the model: " + str(new_attribute))
        return new_attribute

    def _get_example_idx_by_class_conjunction(self, y):
        positive_example_idx = np.where(y == 1)[0]
        negative_example_idx = np.where(y == 0)[0]
        return positive_example_idx, negative_example_idx

    def _get_example_idx_by_class_disjunction(self, y):
        positive_example_idx = np.where(y == 0)[0]
        negative_example_idx = np.where(y == 1)[0]
        return positive_example_idx, negative_example_idx

    def _is_fitted(self):
        return len(self.model) > 0

    def __str__(self):
        return _class_to_string(self)
