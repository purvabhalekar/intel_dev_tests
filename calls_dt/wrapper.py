import pandas as pd
import os
import sys

s = int(sys.argv[1])
e = int(sys.argv[2])

for i in range(s,e):
    os.system("python3 test.py %s" %(i))



df = pd.read_csv("output.csv")
print(df)