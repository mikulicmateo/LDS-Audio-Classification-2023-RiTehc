import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

working_dir = os.getcwd()
instruments = ['tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']

all_instruments_percentage = []

num_of_companion_instruments_percentage = {
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

instrument_percentages_dict = {
    "tru": [[], [], [], []],
    "gac": [[], [], [], []],
    "sax": [[], [], [], []],
    "cel": [[], [], [], []],
    "flu": [[], [], [], []],
    "gel": [[], [], [], []],
    "vio": [[], [], [], []],
    "cla": [[], [], [], []],
    "pia": [[], [], [], []],
    "org": [[], [], [], []],
    "voi": [[], [], [], []]
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

r_specific_instrument_counter = {
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

def create_distribution_plot(dictionary, instrument_name, num_of_instruments = 0):
    sum = np.sum(list(dictionary.values()))

    for key in dictionary.keys():
        if dictionary[key] != 0:
            dictionary[key] = dictionary[key] / sum

    plt.bar(dictionary.keys(), dictionary.values(), width = 0.9)
    if num_of_instruments != 0:
        plt.savefig('generated_' + instrument_name + '_' + str(num_of_instruments) + '_instruments_count.png')
    else:
        plt.savefig('generated_' + instrument_name + '_instruments_count.png')
    plt.clf()


def save_plots(save_folder_path):
    os.chdir(save_folder_path)
    plt.bar(num_of_companion_instruments_percentage.keys(), r_generated_instrument_counter / np.sum(r_generated_instrument_counter), width = 0.9)
    plt.savefig('generated_all_instruments_count.png')
    plt.clf()

    for instrument in r_specific_instrument_counter.keys():

        os.chdir(save_folder_path)
        if not os.path.exists(os.path.join(save_folder_path, instrument)):
            os.makedirs(instrument)
        os.chdir(os.path.join(save_folder_path, instrument))

        for i in range(len(r_specific_instrument_counter[instrument])):
            if i == 0:
                continue

            create_distribution_plot(r_specific_instrument_counter[instrument][i],
                                                                  instrument,
                                                                  num_of_instruments = i+1)


#--------------
# Data Loading
#--------------
data = pd.read_csv('IRMAS_Validation_Data/Plots/validation_metadata_percentage.csv')
all_instruments_percentage = np.array(data.iloc[0].to_numpy()[3:], dtype=float)

data = pd.read_csv('IRMAS_Validation_Data/Plots/validation_instrument_mix_ratios_metadata_percentage.csv')
for i, instrument in enumerate(num_of_companion_instruments_percentage.keys()):
    num_of_companion_instruments_percentage[instrument] = np.array(data.iloc[i].to_numpy()[2:], dtype=float)

for instrument in instruments:
    data = pd.read_csv(f'IRMAS_Validation_Data/Plots/{instrument}/validation_{instrument}_metadata_percentage.csv')
    for i in range(len(instrument_percentages_dict[instrument])):
        instrument_percentages_dict[instrument][i] = np.array(data.iloc[i].to_numpy()[3:], dtype=float)


#--------------
# Logic
#--------------
samples_amount = 100_000
total_instrument_count = []

for instrument_percentage in all_instruments_percentage:
    #add 1 to samples because of rounding error
    total_instrument_count.append(int(np.round((samples_amount+1) * instrument_percentage)))

r_original_count = total_instrument_count.copy()
r_generated_instrument_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
while np.sum(total_instrument_count) > 0:
    
    instrument = np.random.choice(np.arange(11), p=all_instruments_percentage)
    instrument = instruments[instrument]
    total_instrument_count[instruments.index(instrument)] -= 1
    r_generated_instrument_counter[instruments.index(instrument)] += 1
    num_of_companion_instruments = np.random.choice(np.arange(11), p=num_of_companion_instruments_percentage[instrument])

    if num_of_companion_instruments == 0:
        r_specific_instrument_counter[instrument][num_of_companion_instruments] += 1
        continue
    
    if np.sum(instrument_percentages_dict[instrument][num_of_companion_instruments-1]) == 0.0:
        print(f"{instrument}, {num_of_companion_instruments} | Suma = {np.sum(instrument_percentages_dict[instrument][num_of_companion_instruments])} Nekako prolazi")

    companion_instruments_indices = np.random.choice(11, 
                                             num_of_companion_instruments, 
                                             replace=False, 
                                             p=instrument_percentages_dict[instrument][num_of_companion_instruments-1])
    
    companion_instruments = []
    for companion_instrument_index in companion_instruments_indices:
        r_specific_instrument_counter[instrument][num_of_companion_instruments][instruments[companion_instrument_index]] += 1
        total_instrument_count[companion_instrument_index] -= 1
        r_generated_instrument_counter[companion_instrument_index] += 1
        companion_instruments.append(instruments[companion_instrument_index])



#--------------
# Results
#--------------
print(f'generated instruments: {r_generated_instrument_counter}')
print(f'validation dataset count: {r_original_count}')
print(f'total samples generated: {np.sum(r_generated_instrument_counter)}')

#Uncomment if you want to save plots
#save_folder_path = r'/home/dominik/Desktop/Plots/'
#save_plots(save_folder_path)