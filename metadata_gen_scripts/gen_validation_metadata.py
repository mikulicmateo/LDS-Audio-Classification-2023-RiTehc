import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

working_dir = os.getcwd()
validation_dataset_path = r'/home/dominik/Work/Lumen Datascience/Dataset/IRMAS_Validation_Data/'
os.chdir(validation_dataset_path)

columns = ['type', 'num_of_instruments', 'tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']
columns_instrument_num = ['type', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

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

specific_instrument_counter = {
    "tru": [0, all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy()],
    "gac": [0, all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy()],
    "sax": [0, all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy()],
    "cel": [0, all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy()],
    "flu": [0, all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy()],
    "gel": [0, all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy()],
    "vio": [0, all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy()],
    "cla": [0, all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy()],
    "pia": [0, all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy()],
    "org": [0, all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy()],
    "voi": [0, all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy(), all_instruments_counter.copy()]
}



def create_distribution_plot(dictionary, instrument_name, data_raw, data_percentages, num_of_instruments = 0):
    sum = np.sum(list(dictionary.values()))
    temp_raw = [instrument_name, num_of_instruments]
    temp_percentage = [instrument_name, num_of_instruments]

    for key in dictionary.keys():
        temp_raw.append(dictionary[key])
        if dictionary[key] != 0:
            dictionary[key] = dictionary[key] / sum
        temp_percentage.append(dictionary[key])

    plt.bar(dictionary.keys(), dictionary.values(), width = 0.9)
    if num_of_instruments != 0:
        plt.savefig('validation_' + instrument_name + '_' + str(num_of_instruments) + '_instruments_count.png')
    else:
        plt.savefig('validation_' + instrument_name + '_instruments_count.png')
    plt.clf()

    data_raw.append(temp_raw)
    data_percentages.append(temp_percentage)

    return (data_raw, data_percentages)



files = os.listdir(os.getcwd())
files.sort()

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

        for instrument in instruments:
            instrument_name = instrument.strip()

            instrument_mix_counter[instrument_name][num_of_instruments - 1] += 1
            all_instruments_counter[instrument_name] += 1

            for companion_instrument in instruments:
                companion_instrument_name = companion_instrument.strip()
                if len(instruments) == 1:
                    specific_instrument_counter[instrument_name][num_of_instruments - 1] +=1
                else:
                    if instrument == companion_instrument:
                        continue
                    specific_instrument_counter[instrument_name][num_of_instruments - 1][companion_instrument_name] += 1



os.chdir(os.path.join(working_dir, 'IRMAS_Validation_Data'))
if not os.path.exists(os.path.join(working_dir, 'IRMAS_Validation_Data/Plots')):
    os.makedirs('Plots')
os.chdir(os.path.join(working_dir, 'IRMAS_Validation_Data/Plots'))



#Creating instrument mix ratios metadata csv file
data_raw = []
data_percentages = []

for instrument in instrument_mix_counter.keys():
    sum = np.sum(instrument_mix_counter[instrument])
    temp_raw = [instrument]
    temp_percentages = [instrument]

    for i in range(len(instrument_mix_counter[instrument])):
        temp_raw.append(instrument_mix_counter[instrument][i])
        temp_percentages.append(instrument_mix_counter[instrument][i] / sum)
    
    data_raw.append(temp_raw)
    data_percentages.append(temp_percentages)

pd.DataFrame(data_raw, columns = columns_instrument_num).to_csv('validation_instrument_mix_ratios_metadata_raw.csv')
pd.DataFrame(data_percentages, columns = columns_instrument_num).to_csv('validation_instrument_mix_ratios_metadata_percentage.csv')



#Creating all instrument metadata csv file
data_raw = []
data_percentages = []

data_raw, data_percentages = create_distribution_plot(all_instruments_counter, 'all', data_raw, data_percentages)

pd.DataFrame(data_raw, columns = columns).to_csv('validation_metadata_raw.csv')
pd.DataFrame(data_percentages, columns = columns).to_csv('validation_metadata_percentage.csv')



#Creating metadata csv file for specific instruments
for instrument in specific_instrument_counter.keys():

    os.chdir(os.path.join(working_dir, 'IRMAS_Validation_Data/Plots'))
    if not os.path.exists(os.path.join(working_dir, 'IRMAS_Validation_Data/Plots/' + instrument)):
        os.makedirs(instrument)
    os.chdir(os.path.join(working_dir, 'IRMAS_Validation_Data/Plots/' + instrument))

    data_raw = []
    data_percentages = []

    for i in range(len(specific_instrument_counter[instrument])):
        if i == 0:
            continue

        data_raw, data_percentages = create_distribution_plot(specific_instrument_counter[instrument][i],
                                                              instrument,
                                                              data_raw,
                                                              data_percentages,
                                                              i+1)

    pd.DataFrame(data_raw, columns = columns).to_csv(f'validation_{instrument}_metadata_raw.csv')
    pd.DataFrame(data_percentages, columns = columns).to_csv(f'validation_{instrument}_metadata_percentage.csv')



print(f'Saved to: {os.getcwd()}')