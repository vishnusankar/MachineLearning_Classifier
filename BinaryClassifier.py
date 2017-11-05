# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from HelperClass import HelperClass


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
"""[[893,36]
    [64,37]]"""
#    930 correct prediction
obj.plotConfusionMatrix(cm)

# Fitting K-Nearest Neighbour Regression to the Training set & Predicting the Test set results
y_pred = obj.k_NearestNeighbours(X_train = X_train, y_train = y_train, X_test = X_test)
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""[[892,37]
    [71,30]]"""
#    922 correct prediction
obj.plotConfusionMatrix(cm)

# Fitting Support Vector Machine to the Training set & Predicting the Test set results
y_pred = obj.supportVectorMachine(X_train = X_train, y_train = y_train, X_test = X_test, kernel = 'rbf')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""[[893,36]
    [65,36]]"""
#    929 correct prediction
obj.plotConfusionMatrix(cm)

# Fitting Support Vector Machine to the Training set & Predicting the Test set results
y_pred = obj.supportVectorMachine(X_train = X_train, y_train = y_train, X_test = X_test, kernel = 'linear')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""[[899,30]
    [68,33]]"""
#    932 correct prediction
obj.plotConfusionMatrix(cm)

# Fitting Support Vector Machine to the Training set & Predicting the Test set results
y_pred = obj.supportVectorMachine(X_train = X_train, y_train = y_train, X_test = X_test, kernel = 'poly')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""[[887,42]
    [65,36]]"""
#    923 correct prediction
obj.plotConfusionMatrix(cm)

# Fitting Support Vector Machine to the Training set & Predicting the Test set results
y_pred = obj.supportVectorMachine(X_train = X_train, y_train = y_train, X_test = X_test, kernel = 'sigmoid')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""[[860,69]
    [63,38]]"""
#    898 correct prediction
obj.plotConfusionMatrix(cm)

# Fitting Navie Bayes to the Training set & Predicting the Test set results
y_pred = obj.navieBayes(X_train = X_train, y_train = y_train, X_test = X_test)
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""[[802,127]
    [47,54]]"""
#   856 correct prediction
obj.plotConfusionMatrix(cm)

# Fitting Decision Tree to the Training set & Predicting the Test set results
y_pred = obj.decisionTreeClassifier(X_train = X_train, y_train = y_train, X_test = X_test, criterion = 'entropy')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""[[826,103]
    [50,51]]"""
#    877 correct prediction
obj.plotConfusionMatrix(cm)

# Fitting Decision Tree to the Training set & Predicting the Test set results
y_pred = obj.decisionTreeClassifier(X_train = X_train, y_train = y_train, X_test = X_test, criterion = 'gini')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""[[844,84]
    [51,50]"""
#  894 correct prediction
obj.plotConfusionMatrix(cm)


# Fitting Random Forest to the Training set & Predicting the Test set results
y_pred = obj.randomForestClassifier(X_train = X_train, y_train = y_train, X_test = X_test, n_estimators = 10, criterion = 'entropy')
# Making the Confusion Matrix
cm = obj.confusionMatrix(y_test, y_pred)
"""[[890,39]
    [60,41]]"""
#    931 correct prediction
obj.plotConfusionMatrix(cm)




!--------------------------------------------- Deep Learning ANN -----------------------------------------------------------
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'random_uniform', activation = 'relu', input_dim = 20))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'random_uniform', activation = 'relu'))

# Adding the second hidden layer
#classifier.add(Dense(units = 3, kernel_initializer = 'random_uniform', activation = 'relu'))

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
"""[[873,56]
    [42,59]"""
#    932 correct prediction
obj.plotConfusionMatrix(cm)



















from matplotlib.colors import ListedColormap
labels = ['business', 'health']
cm = confusion_matrix(y_test, y_pred)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
pl.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
pl.xlabel('Predicted')
pl.ylabel('True')
pl.show()


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


