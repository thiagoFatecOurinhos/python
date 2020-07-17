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
#                                    DT                                      #
##############################################################################
#
# > TUNE
#
parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
clf_tree=DecisionTreeClassifier()
clf=GridSearchCV(clf_tree,parameters, cv=10, verbose=1)
clf.fit(X,y)
print(clf.best_params_)
print(clf.score(X, y))



# Resultados: {'min_samples_split': 10, 'max_depth': 17}
