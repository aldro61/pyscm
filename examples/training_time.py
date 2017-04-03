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

def increase_n_features():
    n_examples = 1000
    n_features = 100000
    times = []
    nfs = []
    for p in np.linspace(0.01, 1.0, 10):
        nf = int(p * n_features)
        X, y = make_classification(n_samples=n_examples, n_features=nf, n_classes=2,
                                   random_state=np.random.RandomState(42))
        clf = SetCoveringMachineClassifier(model_type="conjunction", p=1.0, max_rules=100)
        t = time()
        clf.fit(X, y)
        took = time() - t
        times.append(took)
        nfs.append(nf)
        print nf
        print took
        print

    plt.clf()
    plt.plot(nfs, times)
    plt.legend()
    plt.xlabel("n features")
    plt.ylabel("time (seconds)")
    plt.title("Training time for 1000 <= n <= 100000 features (1000 examples)")
    plt.show()

def increase_n_examples():
    n_examples = 100000
    n_features = 1000
    times = []
    n_exs = []
    for p in np.linspace(0.01, 1.0, 10):
        n_ex = int(p * n_examples)
        X, y = make_classification(n_samples=n_ex, n_features=n_features, n_classes=2,
                                   random_state=np.random.RandomState(42))
        clf = SetCoveringMachineClassifier(model_type="conjunction", p=1.0, max_rules=100)
        t = time()
        clf.fit(X, y)
        took = time() - t
        times.append(took)
        n_exs.append(n_ex)
        print n_ex
        print took
        print

    plt.clf()
    plt.plot(n_exs, times)
    plt.legend()
    plt.xlabel("n examples")
    plt.ylabel("time (seconds)")
    plt.title("Training time for 1000 <= n <= 100000 examples (1000 features)")
    plt.show()

if __name__ == '__main__':
    increase_n_examples()
    increase_n_features()
