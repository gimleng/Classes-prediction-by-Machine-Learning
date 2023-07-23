import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

df = pd.read_csv('project_datas/meter_replace.csv', encoding='utf-8')

X = df[['meter_age', 'prswtusg', 'prev_change1', 'prev_change2', 'prev_change3', 'prev_change4',       'prev_change5', 'prev_change6', 'prev_change7', 'prev_change8', 'prev_change9']].values.tolist()
y = df['check_result'].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)


# Model 1: Linear SVC version
# Create a pipeline
clf_linSVC = Pipeline([
    ("linear_svc", LinearSVC(C=200, loss="hinge", max_iter=10000))
])

#clf_linSVC.fit(X_train, y_train)

#print("Train set accuracy = " + str(clf_linSVC.score(X_train, y_train)))
#print("Test set accuracy = " + str(clf_linSVC.score(X_test, y_test)))
# Hight range of result 

#------------------------------------------#


# Model 2: Gaussian RBF Kernel version
# Create a pipeline
clf_SVC = Pipeline([
    ("linear_svc", SVC(kernel="sigmoid", gamma=2, C=300,random_state=30, max_iter=10000))
])

clf_SVC.fit(X_train, y_train)

print("Train set accuracy = " + str(clf_SVC.score(X_train, y_train)))
print("Test set accuracy = " + str(clf_SVC.score(X_test, y_test)))