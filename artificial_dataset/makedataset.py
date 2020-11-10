import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import pandas as pd
eps = np.finfo(float).eps
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
import timeit

start = timeit.default_timer()
X, y = make_classification(n_samples = 5000000, n_features=200, n_redundant=0, n_informative=20, random_state = 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
stop = timeit.default_timer()

print(stop-start)

f = open("output.csv", "a") 

f.write("Accuracy = "+str(metrics.accuracy_score(y_test, y_pred))+", \n")
f.write("Total Time = "+str(stop-start)+", \n")

f.close()