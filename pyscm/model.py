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
import numpy as np
import warnings; warnings.filterwarnings("ignore")  # Disable all warnings


class BreimanInfo(object):
    def __init__(self, node_n_examples_by_class, class_priors, total_n_examples_by_class):
        # Eq. 2.2 Probability that an example is in class j and falls into node t
        self.p_j_t = [pi_j * N_j_t / N_j for pi_j, N_j_t, N_j in zip(class_priors, node_n_examples_by_class,
                                                                     total_n_examples_by_class)]
        # Eq. 2.3 Probability that any example falls in node t
        self.p_t = sum(self.p_j_t)
        # Eq. 2.4 Probability that an example is in class j given that it falls in node t
        self.p_j_given_t = [p_j_t / self.p_t for p_j_t in self.p_j_t]
        # Def. 2.10 Probability of misclassification given that an example falls into node t
        self.r_t = 1.0 - max(self.p_j_given_t)
        # Contribution of the node to the tree's overall missclassification rate
        self.R_t = self.r_t * self.p_t


class TreeNode(object):
    def __init__(self, depth, class_examples_idx, total_n_examples_by_class, rule=None, parent=None, left_child=None,
                 right_child=None, criterion_value=-1, class_priors=(1, 1)):
        self.rule = rule
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        self.class_examples_idx = class_examples_idx
        self.examples_idx = np.sort(np.hstack((class_examples_idx[0], class_examples_idx[1])))
        self.depth = depth
        self.criterion_value = criterion_value
        self.breiman_info = BreimanInfo([len(class_examples_idx[0]), len(class_examples_idx[1])],
                                        class_priors, total_n_examples_by_class)

    @property
    def class_proportions(self):
        """
        Returns the proportion of examples of each class in the node

        """
        return {0: 1.0 * len(self.class_examples_idx[0]) / self.n_examples,
                1: 1.0 * len(self.class_examples_idx[1]) / self.n_examples}

    @property
    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    @property
    def is_root(self):
        return self.parent is None and self.left_child is not None and self.right_child is not None

    @property
    def leaves(self):
        return _get_tree_leaves(self)

    @property
    def n_examples(self):
        """
        Returns the number of examples in the node

        """
        return sum(len(x) for x in self.class_examples_idx.itervalues())

    @property
    def rules(self):
        return _get_tree_rules(self)

    def to_tikz(self):
        pass

    def __iter__(self):
        for r in _get_tree_rules(self):
            yield r

    def __len__(self):
        """
        Returns the number of rules in the tree
        """
        return len(_get_tree_rules(self))

    def __str__(self):
        return "%s" % (
            "Node(%s, %s, %s)" % (self.rule, self.left_child, self.right_child) if not (self.left_child is None)
            else "Leaf(0: %.2f (%d), 1: %.2f (%d))" % (
                self.class_proportions[0], len(self.class_examples_idx[0]), self.class_proportions[1],
                len(self.class_examples_idx[1])))


class ProbabilisticTreeNode(TreeNode):
    def predict(self, X):
        """
        Binary class predictions using the current node's rule

        """
        # Get probabilistic predictions
        class_probabilities = self.predict_proba(X)

        # Converts the probabilities to classes with a threshold of 0.5
        predictions = np.zeros(X.shape[0], dtype=np.uint8)
        predictions[class_probabilities[1] <= 0.5] = 0
        predictions[class_probabilities[1] > 0.5] = 1
        return predictions

    def predict_proba(self, X):
        """
        Probabilistic class predictions using the current node's rule

        """
        class_probabilities = {0: np.zeros(X.shape[0]),
                               1: np.zeros(X.shape[0])}

        # Push each example down the tree (an example is a row of X)
        for i, x in enumerate(X):
            x = x.reshape(1, -1)
            current_node = self

            # While we are not at a leaf
            while not current_node.is_leaf:
                # If the rule of the current node returns TRUE, branch left
                if current_node.rule.classify(x):
                    current_node = current_node.left_child
                # Otherwise, branch right
                else:
                    current_node = current_node.right_child

            # A leaf has been reached. Use the leaf class proportions as the the class probabilities.
            class_probabilities[0][i] = current_node.breiman_info.p_j_given_t[0]
            class_probabilities[1][i] = current_node.breiman_info.p_j_given_t[1]

        return class_probabilities


def _get_tree_leaves(root):
    def _get_leaves(node):
        leaves = []
        if not node.is_leaf:
            leaves += _get_leaves(node.left_child)
            leaves += _get_leaves(node.right_child)
        else:
            leaves.append(node)
        return leaves
    return _get_leaves(root)


def _get_tree_rules(root):
    def _get_rules(node):
        rules = []
        if node.rule is not None:
            rules.append(node.rule)
            rules += _get_rules(node.left_child)
            rules += _get_rules(node.right_child)
        return rules
    return _get_rules(root)