import re
import pandas as pd
import os

print('aefhoqifhqoih')

with open('output.csv') as fd:
    data = fd.read()

val_to_pattern = {
    'Max Depth': r'Max Depth = (\d+)',
    'Test size': r'Test size = (\-?\d+\.\d+)',
    'Accuracy': r'Accuracy = (\-?\d+\.\d+)',
    'Dataset Time': r'Dataset Time = (\-?\d+\.\d+)',
    'Training Time': r'Training Time = (\-?\d+\.\d+)',
    'Total Time': r'Total Time = (\-?\d+\.\d+)',
}

pd.set_option("display.max_rows", None, "display.max_columns", None)

val_dict = {}
for key, patt in val_to_pattern.items():
    val_dict[key] = re.findall(patt, data)

df = pd.DataFrame.from_dict(val_dict)
print(df.to_string())