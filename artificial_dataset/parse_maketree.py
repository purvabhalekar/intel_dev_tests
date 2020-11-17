import re
import pandas as pd
import os

for i in range(1,10):
    os.system("python3 maketree.py %s" %(i/10))


with open('testsize.csv') as fd:
    data = fd.read()

val_to_pattern = {
    'Max Depth': r'Max Depth = (\d+)',
    'Test size': r'Test size = (\-?\d+\.\d+)',
    'Accuracy': r'Accuracy = (\-?\d+\.\d+)',
    'Dataset Time': r'Dataset Time = (\-?\d+\.\d+)',
    'Training Time': r'Training Time = (\-?\d+\.\d+)',
    'Total Time': r'Total Time = (\-?\d+\.\d+)',
}
    

val_dict = {}
for key, patt in val_to_pattern.items():
    val_dict[key] = re.findall(patt, data)

df = pd.DataFrame.from_dict(val_dict)
print(df)