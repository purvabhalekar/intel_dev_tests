import re
import pandas as pd
import os

for i in range(10):
    os.system("python3 makedataset.py")


with open('output.csv') as fd:
    data = fd.read()

val_to_pattern = {
    'Accuracy': r'Accuracy = (\-?\d+\.\d+)',
    'Total Time': r'Total Time = (\-?\d+\.\d+)',
}

val_dict = {}
for key, patt in val_to_pattern.items():
    val_dict[key] = re.findall(patt, data)

df = pd.DataFrame.from_dict(val_dict)
print(df)