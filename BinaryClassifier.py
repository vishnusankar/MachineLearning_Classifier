# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from HelperClass import HelperClass


# Importing the dataset
dataset = pd.read_csv('bank-additional-full.csv', delimiter = ';')
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
    #    930 correct prediction 83.109919571 %
    """When data sample count is 41188
    [[8904,235]
    [667,491]]"""
#    9,395 correct prediction 22.8100417597 %
obj.plotConfusionMatrix(cm)


# Fitting K-Nearest Neighbour Regression to the Training set & Predicting the Test set results
y_pred = obj.k_NearestNeighbours(X_train = X_train, y_train = y_train, X_test = X_test)
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
""""When data sample count is 1119
    [[892,37]
    [71,30]]"""
#    922 correct prediction 82.3949955317247 %, 
    """When data sample count is 41188
    [[8814,325]
    [695,463]]"""
#    9277 correct prediction 22.5235505487
obj.plotConfusionMatrix(cm)

# Fitting Support Vector Machine to the Training set & Predicting the Test set results
y_pred = obj.supportVectorMachine(X_train = X_train, y_train = y_train, X_test = X_test, kernel = 'rbf')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
    [[893,36]
    [65,36]]"""
#    929 correct prediction
    """When data sample count is 41188
    [[8927,212]
    [712,446]]"""
#    9,139 correct prediction 22.1885015053 %
obj.plotConfusionMatrix(cm)

# Fitting Support Vector Machine to the Training set & Predicting the Test set results
y_pred = obj.supportVectorMachine(X_train = X_train, y_train = y_train, X_test = X_test, kernel = 'linear')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
[[899,30]
    [68,33]]"""
#    932 correct prediction
    """When data sample count is 41188
    [[8960,179]
    [810,348]]"""
#    9,308 correct prediction 22.5988151889 %
obj.plotConfusionMatrix(cm)

# Fitting Support Vector Machine to the Training set & Predicting the Test set results
y_pred = obj.supportVectorMachine(X_train = X_train, y_train = y_train, X_test = X_test, kernel = 'poly')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
[[887,42]
    [65,36]]"""
#    923 correct prediction
    """When data sample count is 41188
    [[8943,196]
    [733,425]]"""
#    9,368 correct prediction 22.744488686 %
obj.plotConfusionMatrix(cm)

# Fitting Support Vector Machine to the Training set & Predicting the Test set results
y_pred = obj.supportVectorMachine(X_train = X_train, y_train = y_train, X_test = X_test, kernel = 'sigmoid')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
    [[860,69]
    [63,38]]"""
#    898 correct prediction
    """When data sample count is 41188
    [[8446,693]
    [674,484]]"""
#    8,930 correct prediction 21.6810721569 %
obj.plotConfusionMatrix(cm)

# Fitting Navie Bayes to the Training set & Predicting the Test set results
y_pred = obj.navieBayes(X_train = X_train, y_train = y_train, X_test = X_test)
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
    [[802,127]
    [47,54]]"""
#   856 correct prediction
    """When data sample count is 41188
    [[8035,1104]
    [441,717]]"""
#    8,752 correct prediction 21.2489074488 %
obj.plotConfusionMatrix(cm)

# Fitting Decision Tree to the Training set & Predicting the Test set results
y_pred = obj.decisionTreeClassifier(X_train = X_train, y_train = y_train, X_test = X_test, criterion = 'entropy')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
    [[826,103]
    [50,51]]"""
#    877 correct prediction
    """When data sample count is 41188
    [[8568,571]
    [543,615]]"""
#    9,183 correct prediction 22.2953287365%
obj.plotConfusionMatrix(cm)

# Fitting Decision Tree to the Training set & Predicting the Test set results
y_pred = obj.decisionTreeClassifier(X_train = X_train, y_train = y_train, X_test = X_test, criterion = 'gini')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
    [[844,84]
    [51,50]"""
#  894 correct prediction
    """When data sample count is 41188
    [[8554,585]
    [551,607]]"""
#    9,368 correct prediction 22.2419151209 %
obj.plotConfusionMatrix(cm)


# Fitting Random Forest to the Training set & Predicting the Test set results
y_pred = obj.randomForestClassifier(X_train = X_train, y_train = y_train, X_test = X_test, n_estimators = 10, criterion = 'entropy')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
    [[890,39]
    [60,41]]"""
#    931 correct prediction
    """When data sample count is 41188
    [[8864,275]
    [645,513]]"""
    sdfsdf
#    9,377 correct prediction 22.7663397106 %
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

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = y_pred.astype(int)


cm = obj.confusionMatrix(y_test, y_pred)
"""When data sample count is 1119
    [[873,56]
    [42,59]"""
#    932 correct prediction
    """When data sample count is 41188
    [[8828,311]
    [507,651]]"""
#    9,479 correct prediction 23.0139846557 %
obj.plotConfusionMatrix(cm)

!--------------------------------------------- Deep Learning ANN -----------------------------------------------------------




