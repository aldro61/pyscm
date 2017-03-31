import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pyscm.scm import SetCoveringMachineClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from time import time


#logging.basicConfig(level=logging.DEBUG,
#                    format="%(asctime)s.%(msecs)d %(levelname)s %(module)s - %(funcName)s: %(message)s")

np.random.seed(42)

n_features = range(1000, 10000, 1000)
times = []
for nf in n_features:
    X, y = make_classification(n_samples=1000, n_features=nf, n_classes=2)

    clf = SetCoveringMachineClassifier(model_type="conjunction", p=1.0, max_rules=100)
    t = time()
    clf.fit(X, y)
    took = time() - t
    times.append(took)
    print took
    print accuracy_score(y_true=y, y_pred=clf.predict(X))
    print clf.model_

plt.clf()
plt.plot(n_features, times)
plt.legend()
plt.xlabel("n features")
plt.ylabel("time (seconds)")
plt.title("Training time for 1000 examples")
plt.show()
