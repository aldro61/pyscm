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
from __future__ import print_function, division, absolute_import, unicode_literals
from six import iteritems

import logging
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    check_random_state,
)
from warnings import warn

from ._scm_utility import find_max as find_max_utility  # cpp extensions
from .model import ConjunctionModel, DisjunctionModel
from .rules import DecisionStump
from .utils import _class_to_string


class BaseSetCoveringMachine(BaseEstimator, ClassifierMixin):
    def __init__(
        self, p=1.0, model_type="conjunction", max_rules=10, random_state=None
    ):
        self.p = p
        self.model_type = model_type
        self.max_rules = max_rules
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            "p": self.p,
            "model_type": self.model_type,
            "max_rules": self.max_rules,
            "random_state": self.random_state,
        }

    def set_params(self, **parameters):
        for parameter, value in iteritems(parameters):
            setattr(self, parameter, value)
        return self

    def fit(self, X, y, tiebreaker=None, iteration_callback=None, **fit_params):
        """
        Fit a SCM model.

        Parameters:
        -----------
        X: array-like, shape=[n_examples, n_features]
            The feature of the input examples.
        y : array-like, shape = [n_samples]
            The labels of the input examples.
        tiebreaker: function(model_type, feature_idx, thresholds, rule_type)
            A function that takes in the model type and information about the
            equivalent rules and outputs the index of the rule to use. The lists
            respectively contain the feature indices, thresholds and type
            corresponding of the equivalent rules. If None, the rule that most
            decreases the training error is selected. Note: the model type is
            provided because the rules that are added to disjunction models
            correspond to the inverse of the rules that are handled during
            training. Handle this case with care.
        iteration_callback: function(model)
            A function that is called each time a rule is added to the model.

        Returns:
        --------
        self: object
            Returns self.

        """
        random_state = check_random_state(self.random_state)

        if self.model_type == "conjunction":
            self._add_attribute_to_model = self._append_conjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_conjunction
        elif self.model_type == "disjunction":
            self._add_attribute_to_model = self._append_disjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_disjunction
        else:
            raise ValueError("Unsupported model type.")

        # Initialize callbacks
        if iteration_callback is None:
            iteration_callback = lambda x: None

        # Parse additional fit parameters
        logging.debug("Parsing additional fit parameters")
        utility_function_additional_args = {}
        if fit_params is not None:
            for key, value in iteritems(fit_params):
                if key[:9] == "utility__":
                    utility_function_additional_args[key[9:]] = value

        # Validate the input data
        logging.debug("Validating the input data")
        X, y = check_X_y(X, y)
        X = np.asarray(X, dtype=np.double)
        self.classes_, y, total_n_ex_by_class = np.unique(
            y, return_inverse=True, return_counts=True
        )
        if len(self.classes_) != 2:
            raise ValueError("y must contain two unique classes.")
        logging.debug(
            "The data contains {0:d} examples. Negative class is {1!s} (n: {2:d}) and positive class is {3!s} (n: {4:d}).".format(
                len(y),
                self.classes_[0],
                total_n_ex_by_class[0],
                self.classes_[1],
                total_n_ex_by_class[1],
            )
        )

        # Invert the classes if we are learning a disjunction
        logging.debug("Preprocessing example labels")
        pos_ex_idx, neg_ex_idx = self._get_example_idx_by_class(y)
        y = np.zeros(len(y), dtype=int)
        y[pos_ex_idx] = 1
        y[neg_ex_idx] = 0

        # Presort all the features
        logging.debug("Presorting all features")
        X_argsort_by_feature_T = np.argsort(X, axis=0).T.copy()

        # Create an empty model
        logging.debug("Initializing empty model")
        self.model_ = (
            ConjunctionModel()
            if self.model_type == "conjunction"
            else DisjunctionModel()
        )

        logging.debug("Training start")
        remaining_example_idx = np.arange(len(y))
        remaining_negative_example_idx = neg_ex_idx
        while (
            len(remaining_negative_example_idx) > 0
            and len(self.model_) < self.max_rules
        ):
            logging.debug("Finding the optimal rule to add to the model")
            (
                opti_utility,
                opti_feat_idx,
                opti_threshold,
                opti_kind,
                opti_N,
                opti_P_bar,
            ) = self._get_best_utility_rules(
                X.copy(),
                y.copy(),
                X_argsort_by_feature_T.copy(),
                remaining_example_idx.copy(),
                **utility_function_additional_args
            )

            logging.debug(
                "Tiebreaking. Found {0:d} optimal rules".format(len(opti_feat_idx))
            )
            if len(opti_feat_idx) > 1:
                if tiebreaker is None:
                    training_risk_decrease = 1.0 * opti_N - opti_P_bar
                    keep_idx = np.where(
                        training_risk_decrease == training_risk_decrease.max()
                    )[0][0]
                else:
                    keep_idx = tiebreaker(
                        self.model_type, opti_feat_idx, opti_threshold, opti_kind
                    )
            else:
                keep_idx = 0
            stump = DecisionStump(
                feature_idx=opti_feat_idx[keep_idx],
                threshold=opti_threshold[keep_idx],
                kind="greater" if opti_kind[keep_idx] == 0 else "less_equal",
            )

            logging.debug("The best rule has utility {0:.3f}".format(opti_utility))
            self._add_attribute_to_model(stump)

            logging.debug(
                "Discarding all examples that the rule classifies as negative"
            )
            remaining_example_idx = remaining_example_idx[
                stump.classify(X[remaining_example_idx])
            ]
            remaining_negative_example_idx = remaining_negative_example_idx[
                stump.classify(X[remaining_negative_example_idx])
            ]
            logging.debug(
                "There are {0:d} examples remaining ({1:d} negatives)".format(
                    len(remaining_example_idx), len(remaining_negative_example_idx)
                )
            )

            iteration_callback(self.model_)

        logging.debug("Training completed")

        logging.debug("Calculating rule importances")
        # Definition: how often each rule outputs a value that causes the value of the model to be final
        final_outcome = 0 if self.model_type == "conjunction" else 1
        total_outcome = (self.model_.predict(X) == final_outcome).sum()  # n times the model outputs the final outcome
        self.rule_importances_ = np.array([(r.classify(X) == final_outcome).sum() / total_outcome for r in self.model_.rules])  # contribution of each rule
        logging.debug("Done.")

        return self

    def predict(self, X):
        """
        Predict class

        Parameters:
        -----------
        X: array-like, shape=[n_examples, n_features]
            The feature of the input examples.

        Returns:
        --------
        predictions: numpy_array, shape=[n_examples]
            The predicted class for each example.

        """
        check_is_fitted(self, ["model_", "rule_importances_", "classes_"])
        X = check_array(X)
        return self.classes_[self.model_.predict(X)]

    def predict_proba(self, X):
        """
        Predict class probabilities

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features)
            The feature of the input examples.

        Returns:
        --------
        p : array of shape = [n_examples, 2]
            The class probabilities for each example. Classes are ordered by lexicographic order.

        """
        warn(
            "SetCoveringMachines do not support probabilistic predictions. The returned values will be zero or one.",
            RuntimeWarning,
        )
        check_is_fitted(self, ["model_", "rule_importances_", "classes_"])
        X = check_array(X)
        pos_proba = self.classes_[self.model_.predict(X)]
        neg_proba = 1.0 - pos_proba
        return np.hstack((neg_proba.reshape(-1, 1), pos_proba.reshape(-1, 1)))
    
    @property
    def rule_importances(self):
        """
        A measure of importance for each rule in the classifier based on how much it contributes to the final predictions.
        
        Returns:
        --------
        rule_importances: list of float
            Importances of each rule, defined as defined in https://doi.org/10.1186/s12864-016-2889-6
        """
        check_is_fitted(self, ["model_", "rule_importances_", "classes_"])
        return self.rule_importances_

    def score(self, X, y):
        """
        Predict classes of examples and measure accuracy

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features)
            The feature of the input examples.
        y : array-like, shape = [n_samples]
            The labels of the input examples.

        Returns:
        --------
        accuracy: float
            The proportion of correctly classified examples.

        """
        check_is_fitted(self, ["model_", "rule_importances_", "classes_"])
        X, y = check_X_y(X, y)
        return accuracy_score(y_true=y, y_pred=self.predict(X))

    def _append_conjunction_model(self, new_rule):
        self.model_.add(new_rule)
        logging.debug("Attribute added to the model: " + str(new_rule))
        return new_rule

    def _append_disjunction_model(self, new_rule):
        new_rule = new_rule.inverse()
        self.model_.add(new_rule)
        logging.debug("Attribute added to the model: " + str(new_rule))
        return new_rule

    def _get_example_idx_by_class_conjunction(self, y):
        positive_example_idx = np.where(y == 1)[0]
        negative_example_idx = np.where(y == 0)[0]
        return positive_example_idx, negative_example_idx

    def _get_example_idx_by_class_disjunction(self, y):
        positive_example_idx = np.where(y == 0)[0]
        negative_example_idx = np.where(y == 1)[0]
        return positive_example_idx, negative_example_idx

    def __str__(self):
        return _class_to_string(self)


class SetCoveringMachineClassifier(BaseSetCoveringMachine):
    """
    A Set Covering Machine classifier

    [1]_ Marchand, M., & Shawe-Taylor, J. (2002). The set covering machine.
    Journal of Machine Learning Research, 3(Dec), 723-746.

    Parameters:
    -----------
    p: float
        The trade-off parameter for the utility function (suggestion: use values >= 1).
    model_type: str, default="conjunction"
        The model type (conjunction or disjunction).
    max_rules: int, default=10
        The maximum number of rules in the model.
    random_state: int, np.random.RandomState or None, default=None
        The random state.

    """

    def __init__(
        self, p=1.0, model_type=str("conjunction"), max_rules=10, random_state=None
    ):
        super(SetCoveringMachineClassifier, self).__init__(
            p=p, model_type=model_type, max_rules=max_rules, random_state=random_state
        )

    def _get_best_utility_rules(self, X, y, X_argsort_by_feature_T, example_idx):
        return find_max_utility(self.p, X, y, X_argsort_by_feature_T, example_idx)
