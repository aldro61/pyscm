"""
An example showing how to use a custom tiebreaker function.

"""
import numpy as np


from pyscm.scm import SetCoveringMachineClassifier
from sklearn.datasets import make_classification

n_examples = 200
n_features = 1000

X,y = make_classification(n_samples=n_examples, n_features=n_features, n_classes=2,
                          random_state=np.random.RandomState(42))

def my_tiebreaker(model_type, feature_idx, thresholds, kind):
    print("Hello from the tiebreaker! Got {0:d} equivalent rules for this {1!s} model.".format(len(feature_idx), model_type))
    return 0

clf = SetCoveringMachineClassifier()
clf.fit(X, y, tiebreaker=my_tiebreaker)
