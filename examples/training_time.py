from __future__ import print_function, division, absolute_import, unicode_literals

import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pyscm.scm import SetCoveringMachineClassifier
from sklearn.datasets import make_classification
from time import time


def increase_n_features():
    n_repeats = 5
    n_bench_points = 5
    n_examples = 1000
    n_features = 100000
    
    avg_times = np.zeros(n_bench_points)
    nfs = [int(n_features * p) for p in np.linspace(0.01, 1.0, n_bench_points)]
    for _ in range(n_repeats):
        times = []
        for nf in nfs:
            X, y = make_classification(n_samples=n_examples, n_features=nf, n_classes=2,
                                       random_state=np.random.RandomState(42))
            clf = SetCoveringMachineClassifier(model_type="conjunction", p=1.0, max_rules=100)
            t = time()
            clf.fit(X, y)
            times.append(time() - t)
        avg_times += np.array(times)
    avg_times /= 1.0 * n_repeats

    plt.clf()
    plt.plot(nfs, avg_times)
    plt.xlabel("n features")
    plt.ylabel("time (seconds)")
    plt.title("Training time for {0:d} <= n <= {1:d} features ({2:d} examples)".format(min(nfs), max(nfs), n_examples))
    plt.savefig("n_features.png", bbox_inches="tight")


def increase_n_examples():
    n_repeats = 5
    n_bench_points = 5
    n_examples = 10000
    n_features = 1000
    
    avg_times = np.zeros(n_bench_points)
    n_exs = [int(n_examples * p) for p in np.linspace(0.01, 1.0, n_bench_points)]
    for _ in range(n_repeats):
        times = []
        for n_ex in n_exs:
            X, y = make_classification(n_samples=n_ex, n_features=n_features, n_classes=2,
                                       random_state=np.random.RandomState(42))
            clf = SetCoveringMachineClassifier(model_type="conjunction", p=1.0, max_rules=100)
            t = time()
            clf.fit(X, y)
            times.append(time() - t)
        avg_times += np.array(times)
    avg_times /= n_repeats

    plt.clf()
    plt.plot(n_exs, avg_times)
    plt.xlabel("n examples")
    plt.ylabel("time (seconds)")
    plt.title("Training time for {0:d} <= n <= {1:d} examples ({2:d} features)".format(min(n_exs), max(n_exs), n_features))
    plt.savefig("n_examples.png", bbox_inches="tight")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s.%(msecs)d %(levelname)s %(module)s - %(funcName)s: %(message)s")
    increase_n_examples()
    increase_n_features()
