# -*- coding: utf-8 -*-

import sklearn.metrics as mtr

def results(model, y_test, param, value):
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
    print("The parameters of this model is: {0} = {1}".format(param, value))