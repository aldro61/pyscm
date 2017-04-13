#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler and Alexandre Drouin
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pyscm import SetCoveringMachineClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import TREE_LEAF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process", "Neural Net", "Naive Bayes", "QDA",
         "Decision Tree", "Random Forest", "AdaBoost", "SCM-Conjunction", "SCM-Disjunction"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    MLPClassifier(alpha=1),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    SetCoveringMachineClassifier(max_rules=4, model_type="conjunction", p=2.0),
    SetCoveringMachineClassifier(max_rules=4, model_type="disjunction", p=1.0)]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]

figure = plt.figure(figsize=(27, 11))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    #cm = plt.cm.PiYG
    #cm_bright = ListedColormap(['#FF0000', '#00FF00'])
    #cm = plt.cm.bwr
    #cm_bright = ListedColormap(['#0000FF', '#FF0000'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Determine the number of rules used by each rule-based model
        if name == "AdaBoost":
            s = 0
            for t in clf.estimators_:
                s += (t.tree_.children_left != TREE_LEAF).sum()
            n_rules = s
        elif name == "Decision Tree":
            n_rules = (clf.tree_.children_left != TREE_LEAF).sum()
        elif name == "Random Forest":
            s = 0
            for t in clf.estimators_:
                s += (t.tree_.children_left != TREE_LEAF).sum()
            n_rules = s
        elif "SCM" in name:
            n_rules = len(clf.model_)
        else:
            n_rules = None

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name.title())
        ax.text(xx.min() + 0.2, yy.min() + 0.2, 'Acc.: {0:.2f}'.format(score).lstrip('0'), size=15,
                horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
        ax.text(xx.min() + 0.2, yy.min() + 0.8, "Rules: {0!s}".format(n_rules) if n_rules is not None else "",
                size=15, horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
        i += 1

plt.tight_layout()
plt.savefig("decision_boundary.pdf", bbox_inches="tight")