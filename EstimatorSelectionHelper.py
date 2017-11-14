#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 07:57:27 2017
Source from :
http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/
@author: Panagiotis Katsaroumpas


"""
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

class EstimatorSelectionHelper():
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
    
    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, 
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X,y)
            self.grid_searches[key] = gs    
    
    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score':np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})
                      
        rows = [row(k, gsc.cv_validation_scores, gsc.parameters) 
                     for k in self.keys
                     for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        print("array after concat %s" % df)
#
        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        
        return df[columns]
    
    def predict_on_bestEstimator(self, X_test, key):
        gs = self.grid_searches[key]
#        print(self.grid_searches)
        return gs.predict(X_test)
    
    def confusionMatrix (self, y_test = [], y_pred = []):
        from sklearn.metrics import confusion_matrix
        print(classification_report(y_test, y_pred, target_names=['0','1']))
        return confusion_matrix(y_test, y_pred)        
    
    def plotConfusionMatrix (self, cm):
        from mlxtend.plotting import plot_confusion_matrix
        import matplotlib.pyplot as plt
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        plt.show()
        
        