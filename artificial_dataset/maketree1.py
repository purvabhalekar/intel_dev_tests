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

t_size = float(sys.argv[1])

start = timeit.default_timer()

data = pd.read_csv('synth_data.csv', index_col=0,low_memory=False)

X = data.iloc[:,:-1]
y = data.iloc[:,-1:]
#print(X)
t3 =  timeit.default_timer()
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=1) # 70% training and 30% test

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
f.write("Test size = "+str(t_size)+ ", \n")
f.write("Accuracy = "+str(metrics.accuracy_score(y_test, y_pred))+", \n")
f.write("Dataset Time = "+str(t3-start)+ ", \n")
f.write("Training Time = "+str(t2-t1)+", \n")
f.write("Total Time = "+str(stop-start)+", \n")

f.close()