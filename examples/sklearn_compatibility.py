from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from pyscm.scm import SetCoveringMachineClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

n_examples = 200
n_features = 1000

X,y = make_classification(n_samples=n_examples, n_features=n_features, n_classes=2, 
                          random_state=np.random.RandomState(42))

params = {
    "p" : [0.5,1.,2.],
    "max_rules" : [1,2,3,4,5],
    "model_type" : ["conjunction","disjunction"]
}
clf = SetCoveringMachineClassifier(random_state=np.random.RandomState(42))

print("Fitting in GirdSearchCV...")

grid = GridSearchCV(estimator=clf, param_grid=params, cv=3, n_jobs=-1, verbose=True)
grid.fit(X,y)

print("GridSearch passed!")
print("Fitting in pipeline with StandardScaler...")

clf = Pipeline([("scaler",StandardScaler()),("scm",SetCoveringMachineClassifier())])
clf.fit(X,y)

print("Done without error.")
