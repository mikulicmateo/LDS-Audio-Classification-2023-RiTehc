import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

working_dir = os.getcwd()
validation_dataset_path = r'.../IRMAS_Validation_Data/'
os.chdir(validation_dataset_path)

columns = ['type', 'tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']
data_raw = []
data_percentages = []


class_dict = {
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
    "tru": class_dict.copy(),
    "gac": class_dict.copy(),
    "sax": class_dict.copy(),
    "cel": class_dict.copy(),
    "flu": class_dict.copy(),
    "gel": class_dict.copy(),
    "vio": class_dict.copy(),
    "cla": class_dict.copy(),
    "pia": class_dict.copy(),
    "org": class_dict.copy(),
    "voi": class_dict.copy()
}

def create_plot(dictionary, instrument_name):
    sum = np.sum(list(dictionary.values()))
    temp_raw = [instrument_name]
    temp_percentage = [instrument_name]

    for key in dictionary.keys():
        temp_raw.append(dictionary[key])
        dictionary[key] = np.round(dictionary[key] / sum * 100, decimals=4)
        temp_percentage.append(dictionary[key])

    if dictionary.__contains__(instrument_name):
            dictionary.__delitem__(instrument_name)

    plt.bar(dictionary.keys(), dictionary.values(), width = 0.9)
    plt.savefig('validation_' + instrument_name + '_instruments_count.png')
    plt.clf()

    data_raw.append(temp_raw)
    data_percentages.append(temp_percentage)


files = os.listdir(os.getcwd())
files.sort()

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
        for instrument in instruments:
            all_instruments_counter.update({instrument[0:3]: (all_instruments_counter.get(instrument[0:3]) + 1)})

            for nested_instrument in instruments:
                if instrument == nested_instrument:
                    continue
                specific_instrument_counter[instrument[0:3]][nested_instrument[0:3]] += 1
    


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