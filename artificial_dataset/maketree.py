import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import pandas as pd
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
import timeit
import sys

d = float(sys.argv[1])
start = timeit.default_timer()
X, y = make_classification(n_samples = 5000000, n_features=200, n_redundant=0, n_informative=20, random_state = 0)
t3 = timeit.default_timer()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=d, random_state=1) # 70% training and 30% test

clf = DecisionTreeClassifier()

t1 = timeit.default_timer()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
t2 = timeit.default_timer()

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
stop = timeit.default_timer()

#print(stop-start)

f = open("testsize.csv", "a") 

f.write("Max Depth = "+str(clf.get_depth())+ ", \n")
f.write("Test size = "+str(d)+ ", \n")
f.write("Accuracy = "+str(metrics.accuracy_score(y_test, y_pred))+", \n")
f.write("Dataset Time = "+str(t3-start)+ ", \n")
f.write("Training Time = "+str(t2-t1)+", \n")
f.write("Total Time = "+str(stop-start)+", \n")


f.close()