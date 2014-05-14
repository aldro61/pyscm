import numpy as np

from .base import BinaryAttributeMixin
from .base import BinaryAttributeListMixin

class DecisionStump(BinaryAttributeMixin):
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

        BinaryAttributeMixin.__init__(self, example_dependencies)

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
            labels = np.array(X[:, self.feature_idx] < self.threshold, dtype=np.int8)
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
        return DecisionStump(self.feature_idx, self.direction * -1, self.threshold, self.example_dependencies)

    def __str__(self):
        return "x[" + str(self.feature_idx) + "] " + (">" if self.direction == 1 else "<=") + " " + str(self.threshold)


class DecisionStumpAttributeList(BinaryAttributeListMixin):

    def __init__(self, feature_idx, directions, thresholds, example_dependencies=[]):

        if len(set(map(len, (feature_idx, directions, thresholds, example_dependencies)))) != 1:
            raise ValueError("DecisionStumpAttributeList constructor: The input lists length should be equal.")

        self.feature_idx = np.asarray(feature_idx)
        self.directions= np.asarray(directions)
        self.thresholds = np.asarray(thresholds)
        self.example_dependencies = np.asarray(example_dependencies)

    def __len__(self):
        return self.feature_idx.shape[0]

    def __getitem__(self, item_idx):
        return DecisionStump(self.feature_idx[item_idx], self.directionst[item_idx], self.thresholds[item_idx], self.example_dependencies[item_idx])

    def classify(self, X):
        attribute_classifications = (X[:, self.feature_idx] - self.thresholds) * self.directions
        attribute_classifications[attribute_classifications > 0] = 1
        attribute_classifications[attribute_classifications <= 0] = 0
        return attribute_classifications
