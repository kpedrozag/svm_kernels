# -*- coding: utf-8 -*-
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import the_data as dt
import scores


def run_rbf():
    print("\n{0} SVM Model using RBF kernel {0}\n".format('-'*10))
    X_train, X_test, y_train, y_test = dt.return_data()
    print("datos cargados")
    gamma = 1.0
    svm_valid = SVC(kernel='poly', random_state=0, gamma=gamma, C=1.0).fit(X_train, y_train)
    print("modelo hallado")
    score = cross_val_score(svm_valid, X_train, y_train, cv=3, n_jobs=-1).mean()
    print("Score del modelo con CV: {0}".format(score))
    model = svm_valid.predict(X_test)
    print("predicciones")
    scores.results(model, y_test, 'gamma', gamma)
"""

def run_rbf():
    print("\n{0} SVM Model using RBF kernel {0}\n".format('-'*10))
    X_train, X_test, y_train, y_test = dt.return_data()
    print("datos cargados")
    gamma = [0.01, 1.0, 10.0]
    
    cls_list = [SVC(kernel='rbf', random_state=0, gamma=val, C=1.0).fit(X_train, y_train) for val in gamma]
    print("modelos hallados")
    score = [cross_val_score(st, X_train, y_train, cv=10, n_jobs=-1).mean() for st in cls_list]
    print("scores")
    pos_max_score = np.argmax(score)
    
    best_gamma = gamma[pos_max_score]
    
    svm_valid = SVC(kernel='poly', random_state=0, gamma=best_gamma, C=1.0).fit(X_train, y_train)
    print("modelo best")
    model = svm_valid.predict(X_test)
    print("predicciones")
    scores.results(model, y_test, 'gamma', best_gamma)
"""