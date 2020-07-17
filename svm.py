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
#                                  SVM RBF                                   #
##############################################################################
#
# > TUNE
#
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, verbose=1)
grid_search.fit(X, y)
print(grid_search.best_params)

