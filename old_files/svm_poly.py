# -*- coding: utf-8 -*-
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np
import the_data as dt
import scores

def run_poly():
    print("\n{0} SVM Model using poly kernel {0}\n".format('-'*10))
    
    X_train, X_test, y_train, y_test = dt.return_data()
    print("split")
    dg = [2,5,7]
    
    cls_list = [SVC(kernel='poly', random_state=0, degree=val, C=1.0).fit(X_train, y_train) for val in dg]
    print("modelos probados")
    score = [cross_val_score(st, X_train, y_train, cv=10, n_jobs=-1).mean() for st in cls_list]
    print("scores")
    pos_max_score = np.argmax(score)
    
    best_degree = dg[pos_max_score]
    
    svm_valid = SVC(kernel='poly', random_state=0, degree=best_degree, C=1.0).fit(X_train, y_train)
    print("mejor modelo")
    model = svm_valid.predict(X_test)
    print("prediccion")
    scores.results(model, y_test, 'degree', best_degree)