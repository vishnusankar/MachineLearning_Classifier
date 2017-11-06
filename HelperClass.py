#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 06:00:25 2017

@author: Vishnusankar
"""

import numpy as np
import matplotlib.pyplot as plt

class HelperClass():    
    def vsLabelEncoder (self, list=[]):
        from sklearn.preprocessing import LabelEncoder
        labelencoder_X_1 = LabelEncoder()
        list = labelencoder_X_1.fit_transform(list)
        print(list)
        return list
    
    def standardScaler (self,array=[]):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        return sc.fit_transform(array)
    
    def logisticRegression (self, X_train = [],y_train = [],X_test = []):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)                
        classifier.fit(X_train, y_train)        
        
        # Predicting the Test set results
        return classifier.predict(X_test)
    
    def k_NearestNeighbours (self, X_train = [],y_train = [],X_test = []):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        return classifier.predict(X_test)        

    def supportVectorMachine (self, X_train = [],y_train = [],X_test = [], kernel = ''):        
        from sklearn.svm import SVC
        classifier = SVC(kernel = kernel, random_state = 0)
        classifier.fit(X_train, y_train)
        
        # Predicting the Test set results
        return classifier.predict(X_test)  
    
    def navieBayes (self, X_train = [],y_train = [],X_test = []):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        
        # Predicting the Test set results
        return classifier.predict(X_test)
    
    def decisionTreeClassifier (self, X_train = [],y_train = [],X_test = [], criterion = ''):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = criterion, random_state = 0)
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        return classifier.predict(X_test)
    
    
    def randomForestClassifier (self, X_train = [],y_train = [],X_test = [], n_estimators = 10, criterion = ''):   
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, random_state = 0)
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        return classifier.predict(X_test)


        
    def confusionMatrix (self, y_test = [], y_pred = []):
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_test, y_pred)        
    
    def plotConfusionMatrix (self, cm):
        from mlxtend.plotting import plot_confusion_matrix
        import matplotlib.pyplot as plt
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        plt.show()







