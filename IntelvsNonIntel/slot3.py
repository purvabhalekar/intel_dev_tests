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

d = int(sys.argv[1])
#b = int(sys.argv[2])

start = timeit.default_timer()
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
t3 = timeit.default_timer()

y_train = train_data.iloc[:,-1:]
y_train = y_train[:d]
X_train = train_data[train_data.columns[:-1]]
X_train = X_train[:d]
y_test = test_data.iloc[:,-1:]
X_test = test_data[test_data.columns[:-1]]
 
 
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

f = open("samples_nonintel.txt", "a") 

f.write("Depth = "+str(clf.get_depth())+ ", \n")
f.write("Test size = "+str(0.25)+ ", \n")
f.write("Max Features = "+str(20)+ ", \n")
f.write("Samples = "+str(d)+ ",\n")
f.write("Accuracy = "+str(metrics.accuracy_score(y_test, y_pred))+", \n")
f.write("Dataset Time = "+str(t3-start)+ ", \n")
f.write("Training Time = "+str(t2-t1)+", \n")
f.write("Total Time = "+str(stop-start)+", \n")


f.close()