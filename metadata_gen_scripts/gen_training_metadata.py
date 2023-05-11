import json
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from structures import instrument_dict as all_instruments_counter, instrument_list

project_path = os.path.dirname(os.getcwd())
training_dataset_path = os.path.join(project_path, "IRMAS_Training_Data")
columns = ['path']
columns.extend(instrument_list)
data_raw = []
data_percentages = []

os.chdir(training_dataset_path)

for folder in instrument_list:

    for file in os.listdir(folder):
        all_instruments_counter[str(folder)] += 1

if not os.path.exists(os.path.join(project_path, 'metadata/training_metadata/Plots')):
    os.makedirs('Plots')
os.chdir(os.path.join(project_path, 'metadata/training_metadata/Plots'))

num_of_instruments = np.sum(list(all_instruments_counter.values()))
temp_raw = ['all']
temp_percentage = ['all']

for key in all_instruments_counter.keys():
    temp_raw.append(all_instruments_counter[key])
    all_instruments_counter[key] = np.round(all_instruments_counter[key] / num_of_instruments * 100, decimals=4)
    temp_percentage.append(all_instruments_counter[key])

plt.bar(all_instruments_counter.keys(), all_instruments_counter.values(), width=0.9)
plt.savefig('training_instruments_count.png')
plt.clf()

data_raw.append(temp_raw)
data_percentages.append(temp_percentage)

pd.DataFrame(data_raw, columns=columns).to_csv('training_metadata_raw.csv')
pd.DataFrame(data_percentages, columns=columns).to_csv('training_metadata_percentages.csv')

print(f'Saved to: {os.getcwd()}')
