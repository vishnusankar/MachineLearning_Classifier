# MachineLearning_Classifier
I tried these binary classification algorithms to find out which one is best.   
#### I need some suggestion from experts, is there way to improve any of these algorithms performance for this dataset.

## Algorithms & Confusion Matrix's Correct Prediction

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
