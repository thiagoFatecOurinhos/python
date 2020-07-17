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
#                                    KNN                                     #
##############################################################################
#
# > TUNE
#
grid_params = {
    'n_neighbors': [1,3,5,7,9,11,13,15,17,19,21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean','manhattan'],

    }

gs = GridSearchCV(
    KNeighborsClassifier(),
    grid_params,
    verbose = 1,
    cv = 10,
    n_jobs = -1,
    error_score = 'raise'
    )

gs_results = gs.fit(X, y)
print(gs_results.best_params_)

# Resultados: {'n_neighbors': 1, 'weights': 'uniform', 'metric': 'manhattan'}
