# MachineLearning_Classifier
I tried these binary classification algorithms to find out which one is best.   
#### I need some suggestion from experts, is there way to improve any of these algorithms performance for this dataset.

## Algorithms & Percentage of Correct Prediction on TestData

| Alogrithms | Small Dataset (4119) Prediction (%)  | Large Dataset (41188) Prediction (%)  |
| :---:         |     :---:           |          :---: |
| **Logistic Regression** |       90.2912621359         |  :+1: 91.065358842381272        |
|K-Nearest Neighbors (K-NN) | 89.514563106796118| 90.094202194814017|
|Support Vector Machine (RBF) (SVM) | 90.194174757281559 |91.026512576478595|
|**Linear (SVM)**                 | :+1::+1: **90.485436893203882**|90.395260755559875|
|Poly (SVM)                   | 89.611650485436883|90.977954744100217|
|Sigmoid (SVM)                | 87.184466019417471|86.724288627755655|
|Naive Bayes                  | 83.106796116504853|84.995629795085947|
|Decision Tree Classification (entropy) | 85.145631067961176|89.181314946100812|
|Decision Tree Classification (gini) | 86.796116504854368|88.96766048363601|
|**Random Forest Classification** | 90.388349514563103|:+1:91.0653588424|
| **Deep Learning (ANN)**     |:+1::+1: **90.4854368932**|:+1::+1: **92.0559386229**|



|    |         estimator          |   min_score    |   mean_score   |   max_score    |     std_score     | C  | criterion | gamma  |  kernel | learning_rate |   metric  | n_estimators | n_neighbors | p | random_state |
|   :---: |         :---:          |   :---:    |   :---:   |   :---:    |     :---:     | :---:  | :---: | :---:  |  :---: | :---: |   :---:  | :---: | :---: | :---: | :---: |
| 0  |    ExtraTreesClassifier    | 0.373626373626 | 0.414866080211 | 0.447058823529 |  0.0306536222216  |    |           |        |         |               |           |      16      |             |   |              |
| 1  |    ExtraTreesClassifier    | 0.413407821229 | 0.418602492339 | 0.425287356322 |  0.0049629523878  |    |           |        |         |               |           |      32      |             |   |              |
| 2  |   RandomForestClassifier   | 0.470588235294 | 0.488332057472 | 0.522613065327 |  0.0242453379376  |    |           |        |         |               |           |      16      |             |   |              |
| 3  |   RandomForestClassifier   | 0.471204188482 | 0.513239586163 | 0.580952380952 |  0.0483436516297  |    |           |        |         |               |           |      32      |             |   |              |
| 4  |     AdaBoostClassifier     |     0.5625     | 0.570569562017 | 0.584615384615 |  0.0099685949219  |    |           |        |         |               |           |      16      |             |   |              |
| 5  |     AdaBoostClassifier     | 0.547169811321 | 0.570006638746 | 0.587628865979 |  0.0169239045404  |    |           |        |         |               |           |      32      |             |   |              |
| 6  | GradientBoostingClassifier | 0.566210045662 | 0.566986532812 | 0.567441860465 | 0.000551784676399 |    |           |        |         |      0.8      |           |      16      |             |   |              |
| 7  | GradientBoostingClassifier | 0.541062801932 | 0.558040908155 | 0.578947368421 |  0.0157137746075  |    |           |        |         |      0.8      |           |      32      |             |   |              |
| 8  | GradientBoostingClassifier | 0.543778801843 | 0.56412594522  | 0.579710144928 |  0.0150505720435  |    |           |        |         |      1.0      |           |      16      |             |   |              |
| 9  | GradientBoostingClassifier | 0.512077294686 | 0.544965217599 | 0.579831932773 |  0.0276960822703  |    |           |        |         |      1.0      |           |      32      |             |   |              |
| 10 |   DecisionTreeClassifier   | 0.465517241379 | 0.52001361551  | 0.56652360515  |  0.0416205729055  |    |  entropy  |        |         |               |           |              |             |   |      0       |
| 11 |   DecisionTreeClassifier   | 0.544642857143 | 0.560393772894 |     0.575      |  0.0124196555154  |    |    gini   |        |         |               |           |              |             |   |      0       |
| 12 |         GaussianNB         | 0.489932885906 | 0.493361224905 | 0.496644295302 |  0.00274184624478 |    |           |        |         |               |           |              |             |   |              |
| 13 |     LogisticRegression     | 0.538461538462 | 0.559148627877 | 0.592592592593 |  0.0238690337173  |    |           |        |         |               |           |              |             |   |      0       |
| 14 |    KNeighborsClassifier    | 0.37125748503  | 0.397717146024 | 0.415300546448 |  0.0190444885473  |    |           |        |         |               | minkowski |              |      5      | 2 |              |
| 15 |            SVC             | 0.511363636364 | 0.521197574649 | 0.538860103627 |  0.0125160976921  | 1  |           |        |  linear |               |           |              |             |   |      0       |
| 16 |            SVC             | 0.450261780105 | 0.482812928956 | 0.513966480447 |  0.0260261044621  | 1  |           |        |   poly  |               |           |              |             |   |      0       |
| 17 |            SVC             | 0.434782608696 | 0.454954803814 | 0.478468899522 |  0.0179907240208  | 1  |           |        | sigmoid |               |           |              |             |   |      0       |
| 18 |            SVC             | 0.511363636364 | 0.52291408617  | 0.538860103627 |  0.0116477376694  | 10 |           |        |  linear |               |           |              |             |   |      0       |
| 19 |            SVC             | 0.534562211982 | 0.556919597726 | 0.568888888889 |  0.0158222326417  | 10 |           |        |   poly  |               |           |              |             |   |      0       |
| 20 |            SVC             | 0.438356164384 | 0.486711928776 | 0.515555555556 |  0.0344042549212  | 10 |           |        | sigmoid |               |           |              |             |   |      0       |
| 21 |            SVC             | 0.310810810811 | 0.322824382824 | 0.337662337662 |  0.0111425304522  | 1  |           | 0.001  |   rbf   |               |           |              |             |   |      0       |
| 22 |            SVC             |      0.0       |      0.0       |      0.0       |        0.0        | 1  |           | 0.0001 |   rbf   |               |           |              |             |   |      0       |
| 23 |            SVC             | 0.448275862069 | 0.462276205921 | 0.475138121547 |  0.0109959768456  | 10 |           | 0.001  |   rbf   |               |           |              |             |   |      0       |
| 24 |            SVC             | 0.315068493151 | 0.325695141734 | 0.339869281046 |  0.0104309887004  | 10 |           | 0.0001 |   rbf   |               |           |              |             |   |      0       |

I am using UCI Machine Learning Repository details are given bellow:
http://archive.ics.uci.edu/ml/datasets/Bank+Marketing

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

Here is a BiBTeX citation as well:

@misc{Lichman:2013 ,
author = "M. Lichman",
year = "2013",
title = "{UCI} Machine Learning Repository",
url = "http://archive.ics.uci.edu/ml",
institution = "University of California, Irvine, School of Information and Computer Sciences" }
