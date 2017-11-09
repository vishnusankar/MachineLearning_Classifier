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

# Fitting Logistic Regression to the Training set & Predicting the Test set results
y_pred = obj.logisticRegression(X_train = X_train, y_train = y_train, X_test = X_test)
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
    [[893,36]
    [64,37]]"""
    percentage = ((cm[0,0]+cm[1,1])/((cm[0,0]+cm[1,1]+cm[1,0]+cm[0,1])*100))
    #    930 correct prediction 90.2912621359 %
    """When data sample count is 41188
    [[8904,235]
    [667,491]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    9,395 correct prediction 91.065358842381272 %
obj.plotConfusionMatrix(cm)


# Fitting K-Nearest Neighbour Regression to the Training set & Predicting the Test set results
y_pred = obj.k_NearestNeighbours(X_train = X_train, y_train = y_train, X_test = X_test)
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
""""When data sample count is 1119
    [[892,37]
    [71,30]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    922 correct prediction 89.514563106796118 %, 
    """When data sample count is 41188
    [[8814,325]
    [695,463]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    9277 correct prediction 90.094202194814017 %
obj.plotConfusionMatrix(cm)

# Fitting Support Vector Machine to the Training set & Predicting the Test set results
y_pred = obj.supportVectorMachine(X_train = X_train, y_train = y_train, X_test = X_test, kernel = 'rbf')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
    [[893,36]
    [65,36]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    929 correct prediction 90.194174757281559 %
    """When data sample count is 41188
    [[8927,212]
    [712,446]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    9,139 correct prediction 91.026512576478595 %
obj.plotConfusionMatrix(cm)

# Fitting Support Vector Machine to the Training set & Predicting the Test set results
y_pred = obj.supportVectorMachine(X_train = X_train, y_train = y_train, X_test = X_test, kernel = 'linear')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
[[899,30]
    [68,33]]"""
percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    932 correct prediction 90.485436893203882 %
    """When data sample count is 41188
    [[8960,179]
    [810,348]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    9,308 correct prediction 90.395260755559875 %
obj.plotConfusionMatrix(cm)

# Fitting Support Vector Machine to the Training set & Predicting the Test set results
y_pred = obj.supportVectorMachine(X_train = X_train, y_train = y_train, X_test = X_test, kernel = 'poly')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
[[887,42]
    [65,36]]"""
percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    923 correct prediction 89.611650485436883 %
    """When data sample count is 41188
    [[8943,196]
    [733,425]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    9,368 correct prediction 90.977954744100217 %
obj.plotConfusionMatrix(cm)

# Fitting Support Vector Machine to the Training set & Predicting the Test set results
y_pred = obj.supportVectorMachine(X_train = X_train, y_train = y_train, X_test = X_test, kernel = 'sigmoid')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
    [[860,69]
    [63,38]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    898 correct prediction 87.184466019417471 %
    """When data sample count is 41188
    [[8446,693]
    [674,484]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    8,930 correct prediction 86.724288627755655 %
obj.plotConfusionMatrix(cm)

# Fitting Navie Bayes to the Training set & Predicting the Test set results
y_pred = obj.navieBayes(X_train = X_train, y_train = y_train, X_test = X_test)
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
    [[802,127]
    [47,54]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#   856 correct prediction 83.106796116504853 %
    """When data sample count is 41188
    [[8035,1104]
    [441,717]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    8,752 correct prediction 84.995629795085947 %
obj.plotConfusionMatrix(cm)

# Fitting Decision Tree to the Training set & Predicting the Test set results
y_pred = obj.decisionTreeClassifier(X_train = X_train, y_train = y_train, X_test = X_test, criterion = 'entropy')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
    [[826,103]
    [50,51]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    877 correct prediction 85.145631067961176  %
    """When data sample count is 41188
    [[8568,571]
    [543,615]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    9,183 correct prediction 89.181314946100812 %
obj.plotConfusionMatrix(cm)

# Fitting Decision Tree to the Training set & Predicting the Test set results
y_pred = obj.decisionTreeClassifier(X_train = X_train, y_train = y_train, X_test = X_test, criterion = 'gini')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
    [[844,84]
    [51,50]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#  894 correct prediction 86.796116504854368 %
    """When data sample count is 41188
    [[8554,585]
    [551,607]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    9,368 correct prediction 88.96766048363601 %
obj.plotConfusionMatrix(cm)


# Fitting Random Forest to the Training set & Predicting the Test set results
y_pred = obj.randomForestClassifier(X_train = X_train, y_train = y_train, X_test = X_test, n_estimators = 10, criterion = 'entropy')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
"""When data sample count is 1119
    [[890,39]
    [60,41]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    931 correct prediction 90.388349514563103 %
    """When data sample count is 41188
    [[8864,275]
    [645,513]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    9,377 correct prediction 91.0653588424 %
obj.plotConfusionMatrix(cm)




!--------------------------------------------- Deep Learning ANN -----------------------------------------------------------
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'random_uniform', activation = 'relu', input_dim = 20))

# Adding the second hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'random_uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'random_uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'random_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 30, epochs = 100)

# Part 3 - Making predicti
ons and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = y_pred.astype(int)


cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
    [[873,56]
    [42,59]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    932 correct prediction 90.4854368932 %
    """When data sample count is 41188
    [[8828,311]
    [507,651]]"""
    percentage = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+(cm[1,0]+cm[0,1]))*100
#    9,479 correct prediction 92.0559386229 %
obj.plotConfusionMatrix(cm)



!--------------------------------------------- Deep Learning ANN -----------------------------------------------------------
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
helper1.fit(X_train, y_train, scoring='f1', n_jobs=-1)
result = helper1.score_summary(sort_by='min_score')



