# -*- coding: utf-8 -*-

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import sklearn.metrics as mtr
import numpy as np
import the_data as dt

def run():
    print("\n{0} SVM Model using RBF kernel {0}\n".format('-'*10))
    X_train, X_test, y_train, y_test = dt.return_data()
    
    gamma = [10**i for i in range(-3, 3)]
    cs = [10**i for i in range(-5,5)]
    pos_max_score = np.array(0)
    max_scores = np.array(0)
    for gm in gamma:
        cls_list=[]
        for c in cs:
            cls = SVC(kernel='rbf', random_state=0, gamma=gm, C=c).fit(X_train, y_train)
            cls_list.append(cls)
        score = [cross_val_score(st, X_train, y_train, cv=30, n_jobs=-1).mean() for st in cls_list]
        # pos of the best C's
        pos_max_score = np.append(pos_max_score, np.argmax(score))
        max_scores = np.append(max_scores, score[np.argmax(score)])
    
    pos_max_score = np.delete(pos_max_score, 0)
    max_scores = np.delete(max_scores, 0)
    
    pos_best_gamma = np.argmax(max_scores)
    best_gamma = gamma[pos_best_gamma]
    best_c = cs[pos_max_score[pos_best_gamma]]
    
    svm_valid = SVC(kernel='rbf', random_state=0, gamma=best_gamma, C=best_c).fit(X_train, y_train)
    
    model = svm_valid.predict(X_test)
    
    f1 = mtr.f1_score(y_test,model)
    recall = mtr.recall_score(y_test,model)
    accuracy = mtr.accuracy_score(y_test,model)
    precision = mtr.precision_score(y_test,model)
    tn, fp, fn, tp = mtr.confusion_matrix(y_test, model).ravel()
    specificity = tn/float(tn+fp)
    
    print("\n\t   CONFUSION MATRIX")
    print("         Negative     Positive")
    print("Negative   {0}           {1}".format(tn,fp))
    print("Positive   {0}            {1}".format(fn,tp))
    print("\nF1-score: {0}\nRecall: {1}\nAccuracy: {2}\nPrecision: {3}\nSpecificity: {4}".format(f1,recall,accuracy,precision,specificity))
    print("The parameters of this model is: \nC = {0} \nGamma = {1}".format(best_c, best_gamma))