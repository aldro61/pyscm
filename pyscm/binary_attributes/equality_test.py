import numpy as np
from .base import BinaryAttributeMixin

class EqualityTest(BinaryAttributeMixin):
    """
    A binary attribute that checks if a feature has a given value.

    Parameters:
    -----------
    feature_idx: int
        The index of the feature in the example vectors.

    value: float
        The value for discriminating positive and negative examples.

    value: bool, default=True
        The outcome of the test if the examples feature at index feature_idx equals to value.

    example_dependencies: array_like, shape=(n_example_dependencies,), default=[]
            A list containing an element of any type for each example on which the attribute depends.
    """

    def __init__(self, feature_idx, value, outcome=True, example_dependencies=[]):
        self.feature_idx = feature_idx
        self.value = value
        self.outcome = outcome

        BinaryAttributeMixin.__init__(self, example_dependencies)

    def classify(self, X):
        """
        Classifies a set of examples using the equality test.

        Parameters:
        -----------
        X: numpy_array, (n_examples, n_features)
            The feature vectors of examples to classify.

        Returns:
        --------
        labels: numpy_array, (n_examples,)
            Labels assigned to each example by the test.
        """
        if self.outcome == True:
            labels = np.asarray(X[:, self.feature_idx] == self.value, dtype=np.int8)
        else:
            labels = np.asarray(X[:, self.feature_idx] != self.value, dtype=np.int8)

        return labels

    def inverse(self):
        """
        Creates an equality test that is the inverse of the current equality test (self).
        For any example, the label attributed by self must be the opposite of the label attributed
        by the inverse of self.

        Returns:
        --------
        inverse: EqualityTest
            A decision stump that is the inverse of self.
        """
        return EqualityTest(self.feature_idx, self.value, False, self.example_dependencies)

    def __str__(self):
        return "x[" + str(self.feature_idx) + "] " + ("==" if self.outcome == True else "!=") + " " + str(self.value)
