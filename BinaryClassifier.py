# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from HelperClass import HelperClass
from EstimatorSelectionHelper import EstimatorSelectionHelper

# Importing the dataset
dataset = pd.read_csv('bank-additional.csv', delimiter = ';')
X = dataset.iloc[:, 0:20].values
y = dataset.iloc[:, 20].values


obj = HelperClass()
## Label encoder
#Job Type Categorical
X[:, 1] = obj.vsLabelEncoder(list = X[:,1])
# Marial Status Categorical
X[:, 2] = obj.vsLabelEncoder(list = X[:,2])
# Education Categorical
X[:, 3] = obj.vsLabelEncoder(list = X[:,3])
# default Categorical
X[:, 4] = obj.vsLabelEncoder(list = X[:,4])
# Housing Categorical
X[:, 5] = obj.vsLabelEncoder(list = X[:,5])
# loan Categorical
X[:, 6] = obj.vsLabelEncoder(list = X[:,6])
# contactmode Categorical
X[:, 7] = obj.vsLabelEncoder(list = X[:,7])
# Month Categorical
X[:, 8] = obj.vsLabelEncoder(list = X[:,8])
#day_of_week Categorical
X[:, 9] = obj.vsLabelEncoder(list = X[:,9])

#pdays column - 999 means client was not previously contacted, convert to -1
X[X == 999] = -1

# Poutcome Categorical
obj.vsLabelEncoder(list=X[:,14])
X[:,14] = obj.vsLabelEncoder(list = X[:,14])

# Poutcome Categorical
y = obj.vsLabelEncoder(list = y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
X_train = obj.standardScaler(X_train)
X_test = obj.standardScaler(X_test)

"""Choosing best algorithm with using GridSearchCV"""

from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, 
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


models1 = { 
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'DecisionTreeClassifier' : DecisionTreeClassifier(),
    'GaussianNB' : GaussianNB(),
    'LogisticRegression' : LogisticRegression(),
    'KNeighborsClassifier' : KNeighborsClassifier(),
    'SVC': SVC()
}

params1 = { 
    'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
    'RandomForestClassifier': { 'n_estimators': [16, 32] },
    'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
    'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    'DecisionTreeClassifier' : {'criterion' : ['entropy','gini'], 'random_state' : [0]},
    'GaussianNB' : {},
    'LogisticRegression' : {'random_state' : [0]},
    'KNeighborsClassifier' : {'n_neighbors' : [5], 'metric' : ['minkowski'], 'p' : [2]},
    'SVC': [
        {'kernel': ['linear','poly','sigmoid'], 'C': [1, 10], 'random_state' : [0]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001],'random_state' : [0]},
    ]
}

helper1 = EstimatorSelectionHelper(models1, params1)
helper1.fit(X_train, y_train, scoring='f1', n_jobs=-1,refit=True, verbose = 2)
result = helper1.score_summary(sort_by='min_score')

result.to_csv('test', sep='\t', encoding='utf-8')

from prettytable import PrettyTable
from prettytable import from_csv
fp = open("test", "r")
mytable = from_csv(fp)
print(mytable)
     
for x in params1.keys():
    y_pred = helper1.predict_on_bestEstimator(X_test,x)
    print("%s Predition on Test Data" % x)
    cm = helper1.confusionMatrix(y_test,y_pred)
    

helper1.plotConfusionMatrix(cm)