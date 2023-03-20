import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

working_dir = os.getcwd()
training_dataset_path = r'.../IRMAS_Training_Data/'
os.chdir(training_dataset_path)

folders = ['tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']
columns = ['path', 'tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']
data_raw = []
data_percentages = []
all_instruments_counter = {
    "tru": 0,
    "gac": 0,
    "sax": 0,
    "cel": 0,
    "flu": 0,
    "gel": 0,
    "vio": 0,
    "cla": 0,
    "pia": 0,
    "org": 0,
    "voi": 0
}

for folder in folders:

    for file in os.listdir(folder):
        all_instruments_counter[str(folder)] +=1
        

os.chdir(os.path.join(working_dir, 'IRMAS_Training_Data'))
if not os.path.exists(os.path.join(working_dir, 'IRMAS_Training_Data/Plots')):
    os.makedirs('Plots')
os.chdir(os.path.join(working_dir, 'IRMAS_Training_Data/Plots'))

sum = np.sum(list(all_instruments_counter.values()))
temp_raw = ['all']
temp_percentage = ['all']

for key in all_instruments_counter.keys():
    temp_raw.append(all_instruments_counter[key])
    all_instruments_counter[key] = np.round(all_instruments_counter[key] / sum * 100, decimals=4)
    temp_percentage.append(all_instruments_counter[key])

plt.bar(all_instruments_counter.keys(), all_instruments_counter.values(), width = 0.9)
plt.savefig('training_instruments_count.png')
plt.clf()

data_raw.append(temp_raw)
data_percentages.append(temp_percentage)

pd.DataFrame(data_raw, columns=columns).to_csv('training_metadata_raw.csv')
pd.DataFrame(data_percentages, columns=columns).to_csv('training_metadata_percentages.csv')

print(f'Saved to: {os.getcwd()}')