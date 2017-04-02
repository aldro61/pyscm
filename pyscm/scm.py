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

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ._scm_utility import find_max as find_max_utility  # cpp extensions
from .model import ConjunctionModel, DisjunctionModel
from .rules import DecisionStump
from .utils import _class_to_string


class BaseSetCoveringMachine(BaseEstimator, ClassifierMixin):
    def __init__(self, p=1.0, model_type="conjunction", max_rules=10, random_state=None):
        if model_type == "conjunction":
            self._add_attribute_to_model = self._append_conjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_conjunction
        elif model_type == "disjunction":
            self._add_attribute_to_model = self._append_disjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_disjunction
        else:
            raise ValueError("Unsupported model type.")

        self.p = p
        self.model_type = model_type
        self.max_rules = max_rules
        self.random_state = random_state if isinstance(random_state, np.random.RandomState) \
            else (np.random.RandomState(random_state if random_state is not None else 42))

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
        # Initialize callbacks
        if iteration_callback is None:
            iteration_callback = lambda x: None

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

        # Invert the classes if we are learning a disjunction
        logging.debug("Preprocessing example labels")
        pos_ex_idx, neg_ex_idx = self._get_example_idx_by_class(y)
        y = np.zeros(len(y), dtype=np.int)
        y[pos_ex_idx] = 1
        y[neg_ex_idx] = 0

        # Presort all the features
        logging.debug("Presorting all features")
        X_argsort_by_feature = np.argsort(X, axis=0)

        # Create an empty model
        logging.debug("Initializing empty model")
        self.model_ = ConjunctionModel() if self.model_type == "conjunction" else DisjunctionModel()

        logging.debug("Training start")
        remaining_example_idx = np.arange(len(y))
        remaining_negative_example_idx = neg_ex_idx
        while len(remaining_negative_example_idx) > 0 and len(self.model_) < self.max_rules:
            logging.debug("Finding the optimal rule to add to the model")
            opti_utility, \
            opti_feat_idx, \
            opti_threshold, \
            opti_kind = self._get_best_utility_rules(X, y, X_argsort_by_feature, remaining_example_idx.copy(),
                                                     **utility_function_additional_args)

            # TODO: Support user specified tiebreaker
            logging.debug("Tiebreaking. Found %d optimal rules" % len(opti_feat_idx))
            keep_idx = self.random_state.randint(0, len(opti_feat_idx))
            stump = DecisionStump(feature_idx=opti_feat_idx[keep_idx], threshold=opti_threshold[keep_idx],
                                  kind="greater" if opti_kind[keep_idx] == 0 else "less_equal")

            logging.debug("The best rule has utility %.3f" % opti_utility)
            self._add_attribute_to_model(stump)

            logging.debug("Discarding all examples that the rule classifies as negative")
            remaining_example_idx = remaining_example_idx[stump.classify(X[remaining_example_idx])]
            remaining_negative_example_idx = remaining_negative_example_idx[stump.classify(X[remaining_negative_example_idx])]
            logging.debug("There are %d examples remaining (%d negatives)" % (len(remaining_example_idx),
                                                                              len(remaining_negative_example_idx)))

            iteration_callback(self.model_)

        logging.debug("Training completed")

        self.rule_importances_ = []  # TODO: implement rule importances (like its done in Kover)

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
        return self.classes_[self.model_.predict(X)]

    def score(self, X, y):
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
    def _get_best_utility_rules(self, X, y, X_argsort_by_feature, example_idx):
        return find_max_utility(self.p, X, y, X_argsort_by_feature, example_idx, np.ones(X.shape[1]))

def log_lambda(x, ld):
    return (np.log(1 + x) / float(np.log(ld))) + 1

def tanh_lambda(x, ld):
    return np.tanh(ld * x) + 1

def arctan_lambda(x, ld):
    return ((2 / float(np.pi)) * np.arctan((2 / float(np.pi)) * x * ld)) + 1

def abs_lambda(x, ld):
    return (x * ld / float(1 + np.abs(x * ld))) + 1

class GroupSetCoveringMachineClassifier(BaseSetCoveringMachine):
    # Pour resoudre ca, il va falloir utiliser le utility_function_additional_args car best_utility prends juste 4
    # arguments obligatoires et le reste est lu dans ce dictionnaire.
    # Testons d'abord pour passer un np.ones tranquile pour commencer.
    # def fit(self, X, y, features_weights, iteration_callback=None):

    def fit(self, X, y, groups_ids, iteration_callback=None):
        features_weights = np.ones(X.shape[1])
        groups_ids = np.asarray(groups_ids)
        assert groups_ids.shape[0] == X.shape[1], 'Each features must have a group number'

        # clean version, but keep working with the debug when i get back to it
        # def call_back_function(model):
        #     choosen_rules = model.rules
        #     choosen_rules_idx = [el.feature_idx for el in choosen_rules]
        #     choosen_rules_groups_ids = groups_ids[choosen_rules_idx]
        #     for ids in choosen_rules_groups_ids:
        #         features_weights[groups_ids == ids] += 1

        # TODO: Voir comment je peux proceder sans duplication d'attributs! Car si je duplique va falloir changer la
        # creation de stump obligatoirement. Donc impacter sur la structure du fit; Ca peut eventuellement se controler
        # avec mon attribut groups ids. Vu qu'on est en mode inline maintenant plus besoin de stocker tout le vecteur
        # j'ai juste besoin de la rle choisie (attribut) puis je vais incrementer le poids de toutes les regles
        # appartenant aux groupes de la regle si la regle est dans plusieurs groups sinon juste le groupe unique.
        # Donc tout doit etre controler dans mon parametre groups_ids. Sa forme influencant tout
        def call_back_function(model):
            choosen_rules = model.rules
            choosen_rules_idx = [el.feature_idx for el in choosen_rules]
            # c'est ici je dois travailler avec le choosen_rules_idx 
            print 'the model is', model
            print 'the indexes are', choosen_rules_idx
            choosen_rules_groups_ids = groups_ids[choosen_rules_idx]
            print choosen_rules_groups_ids
            for ids in choosen_rules_groups_ids:
                print 'we r here', ids
                features_weights[groups_ids == ids] += 1
                print 'update feature weights in the loop', features_weights
            print 'update feature weights outside the loop', features_weights

        super(GroupSetCoveringMachineClassifier, self).fit(y=y, X=X, iteration_callback=call_back_function,
                                                utility__features_weights=features_weights)

    def _get_best_utility_rules(self, X, y, X_argsort_by_feature, example_idx, features_weights):
        print features_weights
        print type(features_weights)
        return find_max_utility(self.p, X, y, X_argsort_by_feature, example_idx, features_weights)