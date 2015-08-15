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
from ..utils import _conditional_print, _class_to_string, _unpack_binary_bytes_from_ints


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

    def fit(self, binary_attributes, y, X=None, attribute_classifications=None, tiebreaker=None,
            iteration_callback=None, **kwargs):
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

        tie_breaker: function, arguments: best_utility_idx: the index of the binary attributes with the highest utility,
                                          attribute_classifications: the classification matrix for the binary attributes
                                          shape=(n_examples, n_attributes), positive_error_count: number of positive
                                          examples that each binary attribute misclassifies, negative_cover_count:
                                          number of negative examples that are correctly classified by the binary
                                          attribute.
            A function which is called when multiple binary attributes have the same utility in an iteration. It should
            return the index of the binary attribute to add to the model.

        iteration_callback: function, arguments: iteration_info (dict)
            A function which is called at the end of each iteration. It contains information on the learning process,
            such as the best attribute and its utility, the attributes that shared the same utility, the remaining
            examples of each class and more.
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
        y = np.zeros(len(y), dtype=np.uint8)
        y[positive_example_idx] = 1
        y[negative_example_idx] = 0

        self._verbose_print("Got " + str(len(binary_attributes)) + " binary attributes.")
        if attribute_classifications is None:
            self._verbose_print("Classifying the examples with the binary attributes")
            attribute_classifications = binary_attributes.classify(X)
        else:
            self._verbose_print("Binary attribute classifications were precomputed")
            if attribute_classifications.shape[1] != len(binary_attributes):
                raise ValueError("The number of attributes must match in attribute_classifications and",
                                 "binary_attributes.")
            if attribute_classifications.shape[0] != len(y):
                raise ValueError("The number of lines in attribute_classifications must match the number of training" +
                                 "examples.")
        del X

        model_attributes_idx = []  # Contains the index of the attributes in the model
        while len(negative_example_idx) > 0 and len(self.model) < self.max_attributes:
            iteration_info = {}

            utilities, \
            positive_error_count, \
            negative_cover_count = self._get_binary_attribute_utilities(
                attribute_classifications=attribute_classifications,
                positive_example_idx=positive_example_idx,
                negative_example_idx=negative_example_idx,
                **utility_function_additional_args)

            # Find all the indexes of all attributes with the best utility
            iteration_info["utility_max"] = np.max(utilities)
            iteration_info["utility_argmax"] = np.where(utilities == iteration_info["utility_max"])[0]
            iteration_info["utility_argmax_positive_error_counts"] = positive_error_count[iteration_info["utility_argmax"]]
            iteration_info["utility_argmax_negative_cover_counts"] = negative_cover_count[iteration_info["utility_argmax"]]

            # Do not select attributes that cover no negative examples and make errors on no positive examples
            best_utility_idx = iteration_info["utility_argmax"][np.logical_or(negative_cover_count[iteration_info["utility_argmax"]] != 0, positive_error_count[iteration_info["utility_argmax"]] != 0)]
            if len(best_utility_idx) == 0:
                self._verbose_print("The attribute of maximal utility does not cover negative examples or make errors" +
                                    " on positive examples. It will not be added to the model. Stopping here.")
                break

            elif len(best_utility_idx) == 1:
                best_attribute_idx = best_utility_idx[0]
                iteration_info["tiebreaker_optimal_idx"] = best_attribute_idx

            elif len(best_utility_idx) > 1:
                if tiebreaker is not None:
                    best_attribute_idx = tiebreaker(best_utility_idx,
                                                    attribute_classifications,
                                                    positive_error_count[best_utility_idx],
                                                    negative_cover_count[best_utility_idx],
                                                    positive_example_idx,
                                                    negative_example_idx)
                else:
                    # Default tie breaker
                    training_risk_decrease = 1.0 * negative_cover_count[best_utility_idx] - positive_error_count[best_utility_idx]
                    best_attribute_idx = best_utility_idx[training_risk_decrease == training_risk_decrease.max()]
                    del training_risk_decrease

                iteration_info["tiebreaker_optimal_idx"] = best_attribute_idx
                best_attribute_idx = best_attribute_idx[0]  # If many are equivalent, just take the first one.
            del best_utility_idx

            iteration_info["selected_attribute_idx"] = best_attribute_idx

            self._verbose_print("Greatest utility is " + str(utilities[best_attribute_idx]))
            # Save the computation if verbose is off
            if self.verbose:
                self._verbose_print("There are " + str(len(iteration_info["utility_argmax"]) - 1) +
                                    " attributes with the same utility.")
            del utilities

            iteration_info["selected_attribute"] = self._add_attribute_to_model(binary_attributes[best_attribute_idx])
            model_attributes_idx.append(best_attribute_idx)

            # Get the best attribute's classification for each example
            best_attribute_classifications = attribute_classifications.get_columns(best_attribute_idx)

            self._verbose_print("Discarding covered negative examples")
            negative_example_idx = negative_example_idx[best_attribute_classifications[negative_example_idx] != 0]

            self._verbose_print("Discarding misclassified positive examples")
            positive_example_idx = positive_example_idx[best_attribute_classifications[positive_example_idx] != 0]

            self._verbose_print("Remaining negative examples:" + str(len(negative_example_idx)))
            self._verbose_print("Remaining positive examples:" + str(len(positive_example_idx)))

            iteration_info["remaining_positive_examples_idx"] = positive_example_idx
            iteration_info["remaining_negative_examples_idx"] = negative_example_idx

            if iteration_callback is not None:
                iteration_callback(iteration_info)

        # Compute the feature importances
        # --------------------------------

        # new method
        n_covered_negative_examples = np.zeros(len(model_attributes_idx), dtype=np.int)
        for i, idx in enumerate(model_attributes_idx):
            n_covered_negative_examples[i] = len(y) - attribute_classifications.get_columns(idx)[y == 0].sum()
        self.attribute_importances = n_covered_negative_examples

        # # Amputated Risk method
        # # Find the full model's empirical risk
        # classifications = np.ones(len(y), dtype=np.uint8)
        # for idx in model_attributes_idx:
        #     classifications = np.logical_and(classifications, attribute_classifications.get_columns(idx))
        # full_model_risk = 1.0 * (y != classifications).sum() / len(y)
        # print "Full model risk:", full_model_risk
        #
        # # Remove each feature from the model and compute the empirical risk of the amputated model
        # amputated_risks = np.zeros(len(model_attributes_idx), dtype=np.float)
        # for i, idx in enumerate(model_attributes_idx):
        #     classifications = np.ones(len(y), dtype=np.uint8)
        #     for other_idx in (o_idx for o_idx in model_attributes_idx if o_idx != idx):
        #         classifications = np.logical_and(classifications, attribute_classifications.get_columns(other_idx))
        #     amputated_risks[i] = 1.0 * (y != classifications).sum() / len(y)
        #     print "Amputated risk:", amputated_risks[i]
        #
        # self.attribute_importances = amputated_risks - full_model_risk


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