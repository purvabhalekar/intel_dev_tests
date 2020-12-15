import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import pandas as pd
import pickle
from pandas import DataFrame
import timeit

start = timeit.default_timer()
X, y = make_classification(n_samples = 500, n_features=200, n_redundant=0, n_informative=20, random_state = 0)
stop = timeit.default_timer()

#print(X)
#print(y)
#pickle.dump(model, open("synth_data.csv", 'wb'))

df = pd.DataFrame(data = np.c_[X, y])
print(df)

df.to_csv(r'synth_data.csv')

print(stop-start)

