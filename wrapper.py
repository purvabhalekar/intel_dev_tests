import pandas as pd
import os
import sys

for i in range(1,11):
    os.system("python3 makedataset.py")



df = pd.read_csv("output.csv")
print(df)