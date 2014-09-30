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

import numpy as np

from ..model import conjunction, disjunction
from ..utils import _conditional_print, _class_to_string


def _n_training_errors(n_remaining_positives, n_remaining_negatives, n_total_positives):
    return n_remaining_negatives + (n_total_positives - n_remaining_positives)


class BaseSetCoveringMachine(object):
    def __init__(self, model_type, max_attributes, verbose):
        super(BaseSetCoveringMachine, self).__init__()

        self.verbose = verbose
        self._verbose_print = partial(_conditional_print, condition=verbose)

        if model_type == conjunction:
            self._add_attribute_to_model = self._append_conjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_conjunction
        elif model_type == disjunction:
            self._add_attribute_to_model = self._append_disjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_disjunction
        else:
            raise ValueError("Unsupported model type.")

        self.max_attributes = max_attributes
        self._flags = {}

    def fit(self, binary_attributes, y, X=None, attribute_classifications=None, model_append_callback=None,
            cover_count_block_size=1000, **kwargs):
        """
        Fit a SCM model.

        Parameters:
        -----------
        binary_attributes: BinaryAttributeListMixin
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

        model_append_callback: function, arguments: {new_attribute: BinaryAttribute, default=None}
            A function which is called when a new binary attribute is appended to the model.

        cover_count_block_size: int, default=1000
            The maximum number of attributes for which covers are counted at one time. Use this to limit memory usage.

        Notes:
        ------
        * HDF5: The SCM can learn from a great number of attributes. Storing them in memory can require a large amount
                of memory space. Therefore, great care is taken to allow attribute_classifications to be a HDF5 dataset.
                We try to prevent loading the entire dataset into memory.
        """
        utility_function_additional_args = {}
        if kwargs != None:
            for key, value in kwargs.iteritems():
                if key[:9] == "utility__":
                    utility_function_additional_args[key[9:]] = value

        if X is None and attribute_classifications is None:
            raise ValueError("X or attribute_classifications must have a value.")

        self._classes, y = np.unique(y, return_inverse=True)
        if len(self._classes) < 2 or len(self._classes) > 2:
            raise ValueError("y must contain two unique classes.")

        self._verbose_print("Example classes are: positive (" + str(self._classes[1]) + "), negative (" + \
                            str(self._classes[0]) + ")")

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

        n_positive_examples_initial = len(positive_example_idx)
        n_trn_errors_prev_iter = len(positive_example_idx) + len(negative_example_idx)

        while len(negative_example_idx) > 0 and len(self.model) < self.max_attributes:

            utilities = self._get_binary_attribute_utilities(attribute_classifications=attribute_classifications,
                                                             positive_example_idx=positive_example_idx,
                                                             negative_example_idx=negative_example_idx,
                                                             cover_count_block_size=cover_count_block_size,
                                                             **utility_function_additional_args)
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
                            # self._verbose_print(binary_attributes[idx])
                            print idx

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
            elif len(positive_example_idx) > 0:
                keep = attribute_classifications[positive_example_idx, best_attribute_idx] != 0
                keep = keep.reshape((1,))
                positive_example_idx = positive_example_idx[keep]

            n_trn_errors = _n_training_errors(len(positive_example_idx),
                                              len(negative_example_idx),
                                              n_positive_examples_initial)

            # If the number of training errors is not reduced by adding the best attribute, stop adding attributes.
            if n_trn_errors < n_trn_errors_prev_iter:
                n_trn_errors_prev_iter = n_trn_errors
            else:
                self._verbose_print("The attribute does not reduce the training risk. It will not be added to the" + \
                                    " model. Stopping here.")
                break

            appended_attribute = self._add_attribute_to_model(binary_attributes[best_attribute_idx])

            if model_append_callback is not None:
                model_append_callback(appended_attribute)

            del utilities, appended_attribute

            self._verbose_print("Remaining negative examples:" + str(len(negative_example_idx)))
            self._verbose_print("Remaining positive examples:" + str(len(positive_example_idx)))

    def predict(self, X):
        """
        Compute binary predictions.

        Parameters:
        -----------
        X: numpy_array, shape=(n_examples,)
            The feature vectors associated to some examples.

        Returns:
        --------
        predictions: numpy_array, shape=(n_examples,)
            The predicted class for each example.
        """
        return self._predict(X)

    def _append_conjunction_model(self, new_attribute):
        self.model.add(new_attribute)
        self._verbose_print("Attribute added to the model: " + str(new_attribute))
        return new_attribute

    def _append_disjunction_model(self, new_attribute):
        new_attribute = new_attribute.inverse()
        self.model.add(new_attribute)
        self._verbose_print("Attribute added to the model: " + str(new_attribute))
        return new_attribute

    def _get_binary_attribute_utilities(self, attribute_classifications, positive_example_idx, negative_example_idx,
                                        cover_count_block_size):
        raise NotImplementedError()

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

    def _predict(self, X):
        if not self._is_fitted():
            raise RuntimeError("A model must be fitted prior to calling predict.")
        return self._classes.take(self.model.predict(X))

    def _predict_proba(self, X):
        """
        Child classes must have the PROBABILISTIC_PREDICTIONS set to True to use this method.
        """
        if not self._is_fitted():
            raise RuntimeError("A model must be fitted prior to calling predict.")

        if not self._flags.get("PROBABILISTIC_PREDICTIONS", False):
            raise RuntimeError("The predictor does not support probabilistic predictions.")

        return self.model.predict_proba(X)

    def __str__(self):
        return _class_to_string(self)