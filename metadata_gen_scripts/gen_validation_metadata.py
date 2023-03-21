import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

working_dir = os.path.dirname(os.getcwd())
validation_dataset_path = r'../IRMAS_Validation_Data/'
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

instrument_mix_counter = {
    "tru": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "gac": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "sax": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cel": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "flu": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "gel": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "vio": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cla": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "pia": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "org": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "voi": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

solo_instrument_counter = {
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

two_instrument_counter = {
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

three_instrument_counter = {
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


four_instrument_counter = {
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

five_instrument_counter = {
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

for i in range(0, len(files)-1, 2):
    txt_file = files[i]
    wav_file = files[i+1]

    if not(wav_file.__contains__(txt_file[: len(txt_file) - 4])):
        print(wav_file)
        print(txt_file)
        print("Error: Adjacent files not the same!")
        break

    with open(txt_file) as f:
        instruments = f.readlines()
        num_of_instruments = len(instruments)

        array[num_of_instruments - 1] += 1
        for instrument in instruments:
            instrument_name = instrument.strip()

            instrument_mix_counter[instrument_name][num_of_instruments - 1] += 1
            all_instruments_counter[instrument_name] += 1

            for companion_instrument in instruments:
                companion_instrument_name = companion_instrument.strip()
                if len(instruments) == 1:
                    solo_instrument_counter[instrument_name] += 1
                    specific_instrument_counter[instrument_name][companion_instrument_name] += 1
                else:
                    if instrument == companion_instrument:
                        continue

                    if num_of_instruments == 2:
                        two_instrument_counter[instrument_name][companion_instrument_name] += 1
                    elif num_of_instruments == 3:
                        three_instrument_counter[instrument_name][companion_instrument_name] += 1
                    elif num_of_instruments == 4:
                        four_instrument_counter[instrument_name][companion_instrument_name] += 1
                    elif num_of_instruments == 5:
                        five_instrument_counter[instrument_name][companion_instrument_name] += 1

                    specific_instrument_counter[instrument_name][companion_instrument_name] += 1
    
# print(four_instrument_counter['tru']/)
print(instrument_mix_counter['tru']/np.sum(instrument_mix_counter['tru']))

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