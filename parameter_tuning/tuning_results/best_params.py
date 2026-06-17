import os
import re
import json

kan_pattern = r'KAN\.json$'
mlp_pattern = r'MLP\.json$'

files = os.listdir()
kan_files = [file for file in files if re.search(kan_pattern, file)]
mlp_files = [file for file in files if re.search(mlp_pattern, file)]

print('KAN results:')
for file in kan_files:
    with open(file,'r') as f:
        d = json.load(f)
        filtered_data = {k: v for k, v in d.items() if re.search(r'_mean$', k)}
        best_key = max(filtered_data,key=filtered_data.get)
        best_val = filtered_data[best_key]
        print(file.replace('_tuning_KAN.json',""),'\n',best_key.replace('_mean',""), '\t',best_val)
        print(filtered_data)

print('\nMLP results:')
for file in mlp_files:
    with open(file,'r') as f:
        d = json.load(f)
        filtered_data = {k: v for k, v in d.items() if re.search(r'_mean$', k)}
        best_key = max(filtered_data,key=filtered_data.get)
        best_val = filtered_data[best_key]
        print(file.replace('_tuning_MLP.json',""),'\n',best_key.replace('_mean',""),'\t',best_val)
        print(filtered_data)

