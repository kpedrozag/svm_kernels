# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def return_data():

    X, y = load_breast_cancer(return_X_y=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,
                                                        random_state=0, 
                                                        stratify=y)
    scaler = StandardScaler().fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test