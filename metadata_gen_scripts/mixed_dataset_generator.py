import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

data_raw = []
data_percentages = []
columns = ['type', 'tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']

def create_plot(dictionary, instrument_name):
    sum = np.sum(list(dictionary.values()))
    temp_raw = [instrument_name]
    temp_percentage = [instrument_name]

    for key in dictionary.keys():
        temp_raw.append(dictionary[key])
        dictionary[key] = dictionary[key] / sum
        temp_percentage.append(dictionary[key])

    plt.bar(dictionary.keys(), dictionary.values(), width = 0.9)
    plt.savefig('validation_' + instrument_name + '_instruments_count.png')
    plt.clf()

    data_raw.append(temp_raw)
    data_percentages.append(temp_percentage)


percentages_dict = {
    "all": [],
    "tru": [],
    "gac": [],
    "sax": [],
    "cel": [],
    "flu": [],
    "gel": [],
    "vio": [],
    "cla": [],
    "pia": [],
    "org": [],
    "voi": []
}

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

specific_instrument_counter = {
    "tru": all_instruments_counter.copy(),
    "gac": all_instruments_counter.copy(),
    "sax": all_instruments_counter.copy(),
    "cel": all_instruments_counter.copy(),
    "flu": all_instruments_counter.copy(),
    "gel": all_instruments_counter.copy(),
    "vio": all_instruments_counter.copy(),
    "cla": all_instruments_counter.copy(),
    "pia": all_instruments_counter.copy(),
    "org": all_instruments_counter.copy(),
    "voi": all_instruments_counter.copy()
}

instruments = ['tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']
instrument_num_pd = [0.43006263, 0.44467641, 0.11064718, 0.01356994, 0.00104384]

num_of_data = 10_000
data = pd.read_csv('IRMAS_Validation_Data/Plots/validation_metadata_percentage.csv')
for i, key in enumerate(percentages_dict):
    percentages_dict[key] = np.array(data.iloc[i].to_numpy()[2:], dtype=float)

for i in range(num_of_data):
    instrument = instruments[np.random.choice(np.arange(11), p=percentages_dict["all"])]
    all_instruments_counter[instrument] += 1
    num_of_instruments_to_combine = np.random.choice(np.arange(len(instrument_num_pd)), p=instrument_num_pd)
    
    if num_of_instruments_to_combine == 0:
        specific_instrument_counter[instrument][instrument] += 1
        continue
    
    array = np.random.choice(11, num_of_instruments_to_combine, replace=False, p=percentages_dict[instrument])

    for instrument_index in array:
        instrument_name = instruments[instrument_index]
        all_instruments_counter[instrument_name]+=1
        for nested_instrument_index in array:
            nested_instrument_name = instruments[nested_instrument_index]
            if nested_instrument_index == instrument_index:
                continue
            specific_instrument_counter[instrument_name][nested_instrument_name] += 1


os.chdir(r'/home/dominik/Desktop/Plots/')
create_plot(all_instruments_counter, 'all')
for key in specific_instrument_counter.keys():
    create_plot(specific_instrument_counter[key], key)

pd.DataFrame(data_raw, columns = columns).to_csv('validation_metadata_raw.csv')
pd.DataFrame(data_percentages, columns = columns).to_csv('validation_metadata_percentage.csv')