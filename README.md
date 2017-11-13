# MachineLearning_Classifier
I tried these binary classification algorithms to find out which one is best.   
#### I need some suggestion from experts, is there way to improve any of these algorithms performance for this dataset.

#Models' Scores#
|    |         estimator          |   min_score    |   mean_score   |   max_score    |    std_score     | C  | criterion | gamma  |  kernel | learning_rate |   metric  | n_estimators | n_neighbors | p | random_state |
|  :---:   |         :---:           |   :---:     |   :---:    |   :---:     |    :---:      | :---:   | :---:  | :---:   |  :---:  | :---:  |   :---:   | :---:  | :---:  | :---:  | :---:  |
| 4  |     AdaBoostClassifier     |     0.5625     | 0.570569562017 | 0.584615384615 | 0.0099685949219  |    |           |        |         |               |           |      16      |             |   |              |
| 5  |     AdaBoostClassifier     | 0.547169811321 | 0.570006638746 | 0.587628865979 | 0.0169239045404  |    |           |        |         |               |           |      32      |             |   |              |
| 11 |   DecisionTreeClassifier   | 0.544642857143 | 0.560393772894 |     0.575      | 0.0124196555154  |    |    gini   |        |         |               |           |              |             |   |      0       |
| 8  | GradientBoostingClassifier | 0.543778801843 | 0.569085225282 | 0.593301435407 | 0.0202322223621  |    |           |        |         |      1.0      |           |      16      |             |   |              |
| 7  | GradientBoostingClassifier | 0.540540540541 | 0.553442248909 | 0.578723404255 | 0.0178777478253  |    |           |        |         |      0.8      |           |      32      |             |   |              |
| 13 |     LogisticRegression     | 0.538461538462 | 0.559148627877 | 0.592592592593 | 0.0238690337173  |    |           |        |         |               |           |              |             |   |      0       |
| 6  | GradientBoostingClassifier | 0.536363636364 | 0.560038531257 | 0.58064516129  | 0.0182075817431  |    |           |        |         |      0.8      |           |      16      |             |   |              |
| 19 |            SVC             | 0.534562211982 | 0.556919597726 | 0.568888888889 | 0.0158222326417  | 10 |           |        |   poly  |               |           |              |             |   |      0       |
| 9  | GradientBoostingClassifier | 0.526315789474 | 0.552837944357 | 0.589211618257 | 0.0266052706533  |    |           |        |         |      1.0      |           |      32      |             |   |              |
| 15 |            SVC             | 0.511363636364 | 0.521197574649 | 0.538860103627 | 0.0125160976921  | 1  |           |        |  linear |               |           |              |             |   |      0       |
| 18 |            SVC             | 0.511363636364 | 0.52291408617  | 0.538860103627 | 0.0116477376694  | 10 |           |        |  linear |               |           |              |             |   |      0       |
| 12 |         GaussianNB         | 0.489932885906 | 0.493361224905 | 0.496644295302 | 0.00274184624478 |    |           |        |         |               |           |              |             |   |              |
| 2  |   RandomForestClassifier   | 0.47311827957  | 0.496185531146 | 0.512820512821 | 0.0168344369935  |    |           |        |         |               |           |      16      |             |   |              |
| 10 |   DecisionTreeClassifier   | 0.465517241379 | 0.52001361551  | 0.56652360515  | 0.0416205729055  |    |  entropy  |        |         |               |           |              |             |   |      0       |
| 3  |   RandomForestClassifier   | 0.463917525773 | 0.498990249021 | 0.542857142857 | 0.0328214245436  |    |           |        |         |               |           |      32      |             |   |              |
| 16 |            SVC             | 0.450261780105 | 0.482812928956 | 0.513966480447 | 0.0260261044621  | 1  |           |        |   poly  |               |           |              |             |   |      0       |
| 23 |            SVC             | 0.448275862069 | 0.462276205921 | 0.475138121547 | 0.0109959768456  | 10 |           | 0.001  |   rbf   |               |           |              |             |   |      0       |
| 20 |            SVC             | 0.438356164384 | 0.486711928776 | 0.515555555556 | 0.0344042549212  | 10 |           |        | sigmoid |               |           |              |             |   |      0       |
| 17 |            SVC             | 0.434782608696 | 0.454954803814 | 0.478468899522 | 0.0179907240208  | 1  |           |        | sigmoid |               |           |              |             |   |      0       |
| 0  |    ExtraTreesClassifier    | 0.434285714286 | 0.450349752549 | 0.478260869565 | 0.0198110832059  |    |           |        |         |               |           |      16      |             |   |              |
| 1  |    ExtraTreesClassifier    | 0.41935483871  | 0.426868948377 | 0.434285714286 | 0.00609589305583 |    |           |        |         |               |           |      32      |             |   |              |
| 14 |    KNeighborsClassifier    | 0.37125748503  | 0.397717146024 | 0.415300546448 | 0.0190444885473  |    |           |        |         |               | minkowski |              |      5      | 2 |              |
| 24 |            SVC             | 0.315068493151 | 0.325695141734 | 0.339869281046 | 0.0104309887004  | 10 |           | 0.0001 |   rbf   |               |           |              |             |   |      0       |
| 21 |            SVC             | 0.310810810811 | 0.322824382824 | 0.337662337662 | 0.0111425304522  | 1  |           | 0.001  |   rbf   |               |           |              |             |   |      0       |
| 22 |            SVC             |      0.0       |      0.0       |      0.0       |       0.0        | 1  |           | 0.0001 |   rbf   |               |           |              |             |   |      0       |


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
