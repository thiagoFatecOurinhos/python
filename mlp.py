#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# Pandas
import pandas as pd

# NumPy
import numpy as np
from numpy import array

# Sci-kit Learn
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.metrics import *

# MLXtend (Stacking)
from mlxtend.classifier import StackingClassifier


ds = array(pd.read_csv("../dataset_preprocessado_bin.csv"))

X = ds[:, 0:77] # X == features
y = ds[:, 78]   # y == rótulos



##############################################################################
#                                    MLP                                     #
##############################################################################
#
# > TUNE

parameters = {'solver': ['lbfgs', 'adam'],
              'alpha': 10.0 ** -np.arange(1, 10),
              'hidden_layer_sizes':np.arange(10, 15),
              'random_state':[0,1,2,3,4,5,6,7,8,9]}

clf = GridSearchCV(MLPClassifier(max_iter=10000), parameters, n_jobs=-1, cv=3, verbose = 1)
clf.fit(X, y)
print(clf.best_params_)
print(clf.score(X, y))


# Resultados: {'alpha': 1e-07, 'hidden_layer_sizes': 10, 'solver': 'adam', 'random_state': 4}
