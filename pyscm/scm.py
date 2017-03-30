"""
    pyscm -- The Set Covering Machine in Python
    Copyright (C) 2017 Alexandre Drouin

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
import logging
import numpy as np


from pyscm.deprecated.model import conjunction, disjunction
from pyscm.deprecated.utils import _conditional_print, _class_to_string
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


import ._scm_utility  # cpp extensions
from .model import ProbabilisticTreeNode


class BaseSetCoveringMachine(BaseEstimator, ClassifierMixin):
    def __init__(self, p, model_type, max_attributes, random_state):
        if model_type == conjunction:
            self._add_attribute_to_model = self._append_conjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_conjunction
        elif model_type == disjunction:
            self._add_attribute_to_model = self._append_disjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_disjunction
        else:
            raise ValueError("Unsupported model type.")

        self.p = p
        self.model_type = model_type
        self.max_rules = max_attributes
        self.random_state = random_state

    def get_params(self, deep=True):
        return {"p": self.p, "model_type": self.model_type, "max_rules": self.max_rules,
                "random_state": self.random_state}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def fit(self, X, y, iteration_callback=None, **fit_params):
        """
        Fit a SCM model.
        """
        # Functions that are used by the learning algorithm
        def _find_optimal_split(node):
            opti_utility, \
            opti_feat_idx, \
            opti_threshold, \
            opti_kind = self.get_best_utility_rules(X, y, X_argsort_by_feature, node.example_idx,
                                                    **utility_function_additional_args)

            # TODO: Support user specified tiebreaker
            keep_idx = self.random_state.randint(0, len(opti_feat_idx))

            from .rules import DecisionStump
            stump = DecisionStump(feature_idx=opti_feat_idx[keep_idx], threshold=opti_threshold[keep_idx],
                                  kind="greater_equal" if opti_kind[keep_idx] == 1 else "less")

            # Discard examples that are predicted as negative (i.e., send them to a leaf)
            # Keep all examples that are predicted as positive (i.e., send them to a node that will be split later)
            best_rule_classifications = stump.classify(X[node.example_idx])
            child = ProbabilisticTreeNode(parent=node,
                                          depth=node.depth + 1,
                                          example_idx=node.example_idx[best_rule_classifications],
                                          predicted_value=1)
            leaf = ProbabilisticTreeNode(parent=node,
                                         depth=node.depth + 1,
                                         example_idx=node.example_idx[~best_rule_classifications],
                                         predicted_value=0)

            return stump, child, leaf

        # Parse additional fit parameters
        logging.debug("Parsing additional fit parameters")
        utility_function_additional_args = {}
        if fit_params is not None:
            for key, value in fit_params.iteritems():
                if key[:9] == "utility__":
                    utility_function_additional_args[key[9:]] = value

        # Validate the input data
        logging.debug("Validating the input data")
        X, y = check_X_y(X, y)
        self.classes_, y, total_n_ex_by_class = np.unique(y, return_inverse=True, return_counts=True)
        if len(self.classes_) != 2:
            raise ValueError("y must contain two unique classes.")
        logging.debug("The data contains %d examples. Negative class is %s (n: %d) and positive class is %s (n: %d)." %
                      (len(y), self.classes_[0], total_n_ex_by_class[0], self.classes_[1], total_n_ex_by_class[1]))

        # Compute class prior probabilities
        neg_class_prior = 1.0 * total_n_ex_by_class[0] / len(y)
        pos_class_prior = 1.0 * total_n_ex_by_class[1] / len(y)

        # Invert the classes if we are learning a disjunction
        logging.debug("Preprocessing example labels")
        pos_ex_idx, neg_ex_idx = self._get_example_idx_by_class(y)
        y = np.zeros(len(y), dtype=np.uint8)
        y[pos_ex_idx] = 1
        y[neg_ex_idx] = 0

        # Presort all the features
        logging.debug("Presorting all features")
        X_argsort_by_feature = np.argsort(X, axis=0)

        # Create an empty model
        logging.debug("Initializing empty model")
        self.model_ = ProbabilisticTreeNode(class_examples_idx={0: neg_ex_idx, 1: pos_ex_idx}, depth=0,
                                            criterion_value=None,
                                            class_priors=[neg_class_prior, pos_class_prior],
                                            total_n_examples_by_class=[total_n_ex_by_class[0], total_n_ex_by_class[1]])

        logging.debug("Training start")
        current_node = self.model_
        while len(neg_ex_idx) > 0 and len(self.model_) < self.max_rules:
            logging.debug("Finding the optimal split of the current node")
            stump, child, leaf = _find_optimal_split(current_node)

            logging.debug("Splitting the current node")
            current_node.rule = stump
            current_node.left_child = child
            current_node.right_child = leaf

            # TODO: Save rule importances to node

            logging.debug("Moving on to the right child")
            current_node = child

            iteration_callback(self.model_)

        logging.debug("Training completed")

        #     utilities, \
        #     positive_error_count, \
        #     negative_cover_count = self._get_binary_attribute_utilities(
        #         attribute_classifications=attribute_classifications,
        #         positive_example_idx=pos_ex_idx,
        #         negative_example_idx=neg_ex_idx,
        #         **utility_function_additional_args)
        #
        #     # Find all the indexes of all attributes with the best utility
        #     iteration_info["utility_max"] = np.max(utilities)
        #     iteration_info["utility_argmax"] = np.where(utilities == iteration_info["utility_max"])[0]
        #     iteration_info["utility_argmax_positive_error_counts"] = positive_error_count[iteration_info["utility_argmax"]]
        #     iteration_info["utility_argmax_negative_cover_counts"] = negative_cover_count[iteration_info["utility_argmax"]]
        #
        #     # Do not select attributes that cover no negative examples and make errors on no positive examples
        #     best_utility_idx = iteration_info["utility_argmax"][np.logical_or(negative_cover_count[iteration_info["utility_argmax"]] != 0, positive_error_count[iteration_info["utility_argmax"]] != 0)]
        #     if len(best_utility_idx) == 0:
        #         self._verbose_print("The attribute of maximal utility does not cover negative examples or make errors" +
        #                             " on positive examples. It will not be added to the model. Stopping here.")
        #         break
        #
        #     elif len(best_utility_idx) == 1:
        #         best_attribute_idx = best_utility_idx[0]
        #         iteration_info["tiebreaker_optimal_idx"] = best_attribute_idx
        #
        #     elif len(best_utility_idx) > 1:
        #         if tiebreaker is not None:
        #             best_attribute_idx = tiebreaker(best_utility_idx,
        #                                             attribute_classifications,
        #                                             positive_error_count[best_utility_idx],
        #                                             negative_cover_count[best_utility_idx],
        #                                             pos_ex_idx,
        #                                             neg_ex_idx)
        #         else:
        #             # Default tie breaker
        #             training_risk_decrease = 1.0 * negative_cover_count[best_utility_idx] - positive_error_count[best_utility_idx]
        #             best_attribute_idx = best_utility_idx[training_risk_decrease == training_risk_decrease.max()]
        #             del training_risk_decrease
        #
        #         iteration_info["tiebreaker_optimal_idx"] = best_attribute_idx
        #         best_attribute_idx = best_attribute_idx[0]  # If many are equivalent, just take the first one.
        #     del best_utility_idx
        #
        #     iteration_info["selected_attribute_idx"] = best_attribute_idx
        #
        #     self._verbose_print("Greatest utility is " + str(utilities[best_attribute_idx]))
        #     # Save the computation if verbose is off
        #     if self.verbose:
        #         self._verbose_print("There are " + str(len(iteration_info["utility_argmax"]) - 1) +
        #                             " attributes with the same utility.")
        #     del utilities
        #
        #     iteration_info["selected_attribute"] = self._add_attribute_to_model(binary_attributes[best_attribute_idx])
        #     model_attributes_idx.append(best_attribute_idx)
        #
        #     # Get the best attribute's classification for each example
        #     best_attribute_classifications = attribute_classifications.get_columns(best_attribute_idx)
        #
        #     self._verbose_print("Discarding covered negative examples")
        #     neg_ex_idx = neg_ex_idx[best_attribute_classifications[neg_ex_idx] != 0]
        #
        #     self._verbose_print("Discarding misclassified positive examples")
        #     pos_ex_idx = pos_ex_idx[best_attribute_classifications[pos_ex_idx] != 0]
        #
        #     self._verbose_print("Remaining negative examples:" + str(len(neg_ex_idx)))
        #     self._verbose_print("Remaining positive examples:" + str(len(pos_ex_idx)))
        #
        #     iteration_info["remaining_positive_examples_idx"] = pos_ex_idx
        #     iteration_info["remaining_negative_examples_idx"] = neg_ex_idx
        #
        #     if iteration_callback is not None:
        #         iteration_callback(iteration_info)
        #
        # #Compute the attribute importances
        # #TODO: implement this without making multiple calls to get_columns. Use rule_predictions instead.
        # # Could implement transparent sorting and desorting of the indexes in get_columns.
        # self.attribute_importances = np.zeros(len(model_attributes_idx), dtype=np.float)
        # rule_predictions = attribute_classifications.get_columns(sorted(model_attributes_idx)) # Watch out (sorted for hdf5 slicing...)
        # model_predictions = np.prod(rule_predictions, axis=1)
        # for i, idx in enumerate(model_attributes_idx):
        #     model_neg_prediction_idx = np.where(model_predictions == 0)[0]
        #     self.attribute_importances[i] = float(len(model_neg_prediction_idx) -
        #                                           attribute_classifications.get_columns(idx)[model_neg_prediction_idx].sum()) / len(model_neg_prediction_idx)

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
        check_is_fitted(self, ["model_", "rule_importances_", "classes_"])
        X = check_array(X)
        return self._classes[self._predict(X)]

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


class SetCoveringMachineClassifier(BaseSetCoveringMachine):
    def _get_best_utility_rules(self, X, y, X_argsort_by_feature, example_idx):
        return _scm_utility.find_max(X, y, X_argsort_by_feature, example_idx)