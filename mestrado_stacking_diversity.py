# -*- coding: utf-8 -*-
print("******* Importando variaveis --->", time.asctime())

# Verificação de tempo dos processos
import time

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

print("******* Importando dataset --->", time.asctime())

ds = array(pd.read_csv("G:\Meu Drive\Mestrado\Datasets\MachineLearningCVE\dataset_preprocessado_1perc_5perc_bin.csv"))

X = ds[:, 0:77] # X == features
y = ds[:, 78]   # y == rótulos




##############################################################################
#                                    KNN                                     #
##############################################################################
#
# > TUNE
#
# grid_params = {
#     'n_neighbors': [1,3,5,7,9],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidian','manhattan'],
    
#     }

# gs = GridSearchCV(
#     KNeighborsClassifier(),
#     grid_params,
#     verbose = 1,
#     cv = 10,
#     n_jobs = -1,
#     error_score = 'raise'
#     )

# gs_results = gs.fit(X, y)
# gs_results.best_params_
# Resultados: {'n_neighbors': 1, 'weights': 'uniform', 'metric': 'manhattan'}

print(">> KNN start --->", time.asctime())
clf1 = KNeighborsClassifier(metric = 'manhattan', n_neighbors = 1, weights = 'uniform')
clf1_pred = cross_val_predict(clf1, X, y, cv=10)
conf_clf1 = confusion_matrix(y, clf1_pred)
print(">> KNN stop --->", time.asctime())


print("********* Desempenho KNN **********")
print("Acuracia:", accuracy_score(y, clf1_pred)*100)
print("AUC:", roc_auc_score(y, clf1_pred)*100)
print("Precision:", precision_score(y, clf1_pred)*100)
print("Recall:", recall_score(y, clf1_pred)*100)
print("F1:", f1_score(y, clf1_pred)*100)
print("***********************************************************")
print("***********************************************************")

##############################################################################

##############################################################################
#                                    MLP                                     #
##############################################################################
#
# > TUNE
#
#parameters = {'solver': ['lbfgs', 'adam'], 
#              'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 
#              'alpha': 10.0 ** -np.arange(1, 10), 
#              'hidden_layer_sizes':np.arange(10, 15),
#              'random_state':[0,1,2,3,4,5,6,7,8,9]}
#
#clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1, cv=3, verbose = 1)
#clf.fit(X, y)
#print(clf.score(trainX, trainY))
#print(clf.best_params_)
#
# Resultados: {'alpha': 1e-07, 'hidden_layer_sizes': 10, 'solver': 'adam', 'random_state': 4}
#
#
# Classificador

print(">> MLP start --->", time.asctime())

clf2 = MLPClassifier(max_iter = 10000, alpha = 1e-07, hidden_layer_sizes = 10,
                     solver = 'adam', random_state = 4)
clf2_pred = cross_val_predict(clf2, X, y, cv=10)
conf_clf2 = confusion_matrix(y, clf2_pred)
print(">> MLP stop --->", time.asctime())

print("********* Desempenho MLP **********")
print("Acuracia:", accuracy_score(y, clf2_pred)*100)
print("AUC:", roc_auc_score(y, clf2_pred)*100)
print("Precision:", precision_score(y, clf2_pred)*100)
print("Recall:", recall_score(y, clf2_pred)*100)
print("F1:", f1_score(y, clf2_pred)*100)
print("***********************************************************")
print("***********************************************************")
##############################################################################

##############################################################################
#                                    DT                                      #
##############################################################################
#
# > TUNE
#
#parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
#clf_tree=DecisionTreeClassifier()
#clf=GridSearchCV(clf_tree,parameters, cv=10, verbose=1)
#clf.fit(X,y)
#
# Resultados: {'min_samples_split': 10, 'max_depth': 17}
# > Classificador:
print(">> DT start --->", time.asctime())

clf3 = DecisionTreeClassifier(max_depth = 17, min_samples_split = 10)
clf3_pred = cross_val_predict(clf3, X, y, cv=10)
conf_clf3 = confusion_matrix(y, clf3_pred)
print(">> DT stop --->", time.asctime())


print("********* Desempenho DT **********")
print("Acuracia:", accuracy_score(y, clf3_pred)*100)
print("AUC:", roc_auc_score(y, clf3_pred)*100)
print("Precision:", precision_score(y, clf3_pred)*100)
print("Recall:", recall_score(y, clf3_pred)*100)
print("F1P:", f1_score(y, clf3_pred)*100)
print("***********************************************************")
print("***********************************************************")

#

##############################################################################
#                                  SVM RBF                                   #
##############################################################################
#
# > TUNE
#
#def svc_param_selection(X, y, nfolds):
#    Cs = [0.001, 0.01, 0.1, 1, 10]
#    gammas = [0.001, 0.01, 0.1, 1]
#    param_grid = {'C': Cs, 'gamma' : gammas}
#    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds, verbose=1)
#    grid_search.fit(X, y)
#    grid_search.best_params_
#    return grid_search.best_params_
#
#svc_param_selection(X, y, 10)
# Resultado: {'C': 10, 'gamma': 1}
#
#
# > CLASSIFICADOR
#
print(">> SVM start --->", time.asctime())

clf4 = SVC(kernel = 'rbf', C=10, gamma=1)
clf4_pred = cross_val_predict(clf4, X, y, cv=10)
conf_clf4 = confusion_matrix(y, clf4_pred)

print(">> SVM stop --->", time.asctime())


print("********* Desempenho SVM **********")
print("Acuracia:", accuracy_score(y, clf4_pred)*100)
print("AUC:", roc_auc_score(y, clf4_pred)*100)
print("Precision:", precision_score(y, clf4_pred)*100)
print("Recall:", recall_score(y, clf4_pred)*100)
print("F1P:", f1_score(y, clf4_pred)*100)
print("***********************************************************")
print("***********************************************************")
#
#
##############################################################################
##############################################################################




print("***********************************************************")
print("************        agreementMatrix       *****************")
print("***********************************************************")


def agreementMatrix(clfA, clfB, realLabels):
    a=0
    b=0
    c=0
    d=0

    for i in range(np.size(y)):
        yLabel = realLabels[i]
        aLabel = clfA[i]
        bLabel = clfB[i]
        
    #    print("*******")
    #    print(yLabel)
    #    print(aLabel)
    #    print(bLabel)
    
        if aLabel == bLabel:
            if aLabel == yLabel: #a
                a = a+1
                            
        if aLabel == yLabel:
            if bLabel != yLabel: #b
                b = b+1
                
        if bLabel == yLabel:
            if aLabel != yLabel: #c
                c = c+1
                
        if aLabel != yLabel:
            if bLabel == aLabel: #d
                d = d+1
                
    agreementMatrix = (b+c)/(a+b+c+d)
    print(agreementMatrix)
    
agreementMatrix(clf1_pred, clf1_pred, y)

import itertools
modelos=['clf1_pred','clf2_pred','clf3_pred', 'clf4_pred']
combinacoes = np.array(list(itertools.combinations(modelos, 2)))

for ii in range(0,int(combinacoes.size/2)):
    a=combinacoes[ii,0]
    b=combinacoes[ii,1]
    cmd=str("agreementMatrix(%s, %s, y)" % (a, b))
    print("***********************************************************")

    print("%s,%s\b\b" % (a,b))
    exec(cmd)










##############################################################################
#                               EXAUSTAO                                     #
##############################################################################

print("Exaustao start --->", time.asctime())


print("layer1,layer2,acuracia,auc,precision,recall,f1")

# STACKING clf1+clf2,clf1
stack = StackingClassifier([clf1, clf2], meta_classifier=clf1)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2,clf1,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf3,clf1
stack = StackingClassifier([clf1, clf3], meta_classifier=clf1)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf3,clf1,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf4,clf1
stack = StackingClassifier([clf1, clf4], meta_classifier=clf1)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf4,clf1,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf2+clf3,clf1
stack = StackingClassifier([clf2, clf3], meta_classifier=clf1)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf2+clf3,clf1,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf2+clf4,clf1
stack = StackingClassifier([clf2, clf4], meta_classifier=clf1)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf2+clf4,clf1,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf3+clf4,clf1
stack = StackingClassifier([clf3, clf4], meta_classifier=clf1)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf3+clf4,clf1,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf2,clf2
stack = StackingClassifier([clf1, clf2], meta_classifier=clf2)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2,clf2,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf3,clf2
stack = StackingClassifier([clf1, clf3], meta_classifier=clf2)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf3,clf2,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf4,clf2
stack = StackingClassifier([clf1, clf4], meta_classifier=clf2)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf4,clf2,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf2+clf3,clf2
stack = StackingClassifier([clf2, clf3], meta_classifier=clf2)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf2+clf3,clf2,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf2+clf4,clf2
stack = StackingClassifier([clf2, clf4], meta_classifier=clf2)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf2+clf4,clf2,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf3+clf4,clf2
stack = StackingClassifier([clf3, clf4], meta_classifier=clf2)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf3+clf4,clf2,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf2,clf3
stack = StackingClassifier([clf1, clf2], meta_classifier=clf3)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2,clf3,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf3,clf3
stack = StackingClassifier([clf1, clf3], meta_classifier=clf3)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf3,clf3,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf4,clf3
stack = StackingClassifier([clf1, clf4], meta_classifier=clf3)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf4,clf3,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf2+clf3,clf3
stack = StackingClassifier([clf2, clf3], meta_classifier=clf3)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf2+clf3,clf3,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf2+clf4,clf3
stack = StackingClassifier([clf2, clf4], meta_classifier=clf3)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf2+clf4,clf3,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf3+clf4,clf3
stack = StackingClassifier([clf3, clf4], meta_classifier=clf3)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf3+clf4,clf3,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf2,clf4
stack = StackingClassifier([clf1, clf2], meta_classifier=clf4)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2,clf4,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf3,clf4
stack = StackingClassifier([clf1, clf3], meta_classifier=clf4)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf3,clf4,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf4,clf4
stack = StackingClassifier([clf1, clf4], meta_classifier=clf4)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf4,clf4,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf2+clf3,clf4
stack = StackingClassifier([clf2, clf3], meta_classifier=clf4)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf2+clf3,clf4,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf2+clf4,clf4
stack = StackingClassifier([clf2, clf4], meta_classifier=clf4)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf2+clf4,clf4,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf3+clf4,clf4
stack = StackingClassifier([clf3, clf4], meta_classifier=clf4)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf3+clf4,clf4,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf2+clf3,clf1
stack = StackingClassifier([clf1, clf2, clf3], meta_classifier=clf1)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2+clf3,clf1,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf2+clf4,clf1
stack = StackingClassifier([clf1, clf2, clf4], meta_classifier=clf1)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2+clf4,clf1,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf3+clf4,clf1
stack = StackingClassifier([clf1, clf3, clf4], meta_classifier=clf1)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf3+clf4,clf1,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf2+clf3+clf4,clf1
stack = StackingClassifier([clf2, clf3, clf4], meta_classifier=clf1)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf2+clf3+clf4,clf1,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf2+clf3,clf2
stack = StackingClassifier([clf1, clf2, clf3], meta_classifier=clf2)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2+clf3,clf2,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf2+clf4,clf2
stack = StackingClassifier([clf1, clf2, clf4], meta_classifier=clf2)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2+clf4,clf2,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf3+clf4,clf2
stack = StackingClassifier([clf1, clf3, clf4], meta_classifier=clf2)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf3+clf4,clf2,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf2+clf3+clf4,clf2
stack = StackingClassifier([clf2, clf3, clf4], meta_classifier=clf2)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf2+clf3+clf4,clf2,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf2+clf3,clf3
stack = StackingClassifier([clf1, clf2, clf3], meta_classifier=clf3)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2+clf3,clf3,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf2+clf4,clf3
stack = StackingClassifier([clf1, clf2, clf4], meta_classifier=clf3)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2+clf4,clf3,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf3+clf4,clf3
stack = StackingClassifier([clf1, clf3, clf4], meta_classifier=clf3)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf3+clf4,clf3,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf2+clf3+clf4,clf3
stack = StackingClassifier([clf2, clf3, clf4], meta_classifier=clf3)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf2+clf3+clf4,clf3,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf2+clf3,clf4
stack = StackingClassifier([clf1, clf2, clf3], meta_classifier=clf4)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2+clf3,clf4,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf2+clf4,clf4
stack = StackingClassifier([clf1, clf2, clf4], meta_classifier=clf4)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2+clf4,clf4,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf3+clf4,clf4
stack = StackingClassifier([clf1, clf3, clf4], meta_classifier=clf4)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf3+clf4,clf4,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf2+clf3+clf4,clf4
stack = StackingClassifier([clf2, clf3, clf4], meta_classifier=clf4)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf2+clf3+clf4,clf4,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf2+clf3+clf4,clf1
stack = StackingClassifier([clf1, clf2, clf3, clf4], meta_classifier=clf1)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2+clf3+clf4,clf1,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf2+clf3+clf4,clf2
stack = StackingClassifier([clf1, clf2, clf3, clf4], meta_classifier=clf2)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2+clf3+clf4,clf2,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf2+clf3+clf4,clf3
stack = StackingClassifier([clf1, clf2, clf3, clf4], meta_classifier=clf3)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2+clf3+clf4,clf3,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))

# STACKING clf1+clf2+clf3+clf4,clf4
stack = StackingClassifier([clf1, clf2, clf3, clf4], meta_classifier=clf4)
stack_pred = cross_val_predict(stack, X, y, cv=10)
conf_stack = confusion_matrix(y, stack_pred)

print("clf1+clf2+clf3+clf4,clf4,%f,%f,%f,%f,%f" % (accuracy_score(y, stack_pred),roc_auc_score(y, stack_pred),precision_score(y, stack_pred),recall_score(y, stack_pred),f1_score(y, stack_pred)))



print("Exaustao stop --->", time.asctime())


