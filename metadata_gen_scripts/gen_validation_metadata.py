import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

working_dir = os.getcwd()
validation_dataset_path = r'/home/dominik/Work/Lumen Datascience/Dataset/IRMAS_Validation_Data/'
os.chdir(validation_dataset_path)

columns = ['type', 'tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']
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


files = os.listdir(os.getcwd())
files.sort()

array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(0, len(files), 2):
    txt_file = files[i]
    wav_file = files[i+1]

    if not(wav_file.__contains__(txt_file[: len(txt_file) - 4])):
        print(wav_file)
        print(txt_file)
        print("Error: Adjecent files not the same!")
        break

    with open(txt_file) as f:
        instruments = f.readlines()
        array[len(instruments)-1] += 1
        for instrument in instruments:
            instrument_name = str(instrument[0:3])
            all_instruments_counter[instrument_name] += 1

            for nested_instrument in instruments:
                nested_instrument_name = nested_instrument[0:3]
                if len(instruments) == 1:
                    specific_instrument_counter[instrument_name][nested_instrument_name] += 1
                else:
                    if instrument == nested_instrument:
                        continue
                    specific_instrument_counter[instrument_name][nested_instrument_name] += 1
    

print(array / np.sum(array))

os.chdir(os.path.join(working_dir, 'IRMAS_Validation_Data'))
if not os.path.exists(os.path.join(working_dir, 'IRMAS_Validation_Data/Plots')):
    os.makedirs('Plots')
os.chdir(os.path.join(working_dir, 'IRMAS_Validation_Data/Plots'))

create_plot(all_instruments_counter, 'all')

for key in specific_instrument_counter.keys():
    create_plot(specific_instrument_counter[key], key)


pd.DataFrame(data_raw, columns = columns).to_csv('validation_metadata_raw.csv')
pd.DataFrame(data_percentages, columns = columns).to_csv('validation_metadata_percentage.csv')

print(f'Saved to: {os.getcwd()}')