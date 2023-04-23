import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import json
import copy

from DataBag import DataBag

working_dir = os.path.dirname(os.getcwd())
os.chdir(working_dir)
DEFAULT_TRAIN_ANNOTATION_FILE = r'IRMAS_Training_Data/training_annotation_file.csv'
instruments = ['tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']

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

r_generated_instrument_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

r_specific_instrument_counter = {
    "tru": [0, copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter),
            copy.deepcopy(all_instruments_counter)],
    "gac": [0, copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter),
            copy.deepcopy(all_instruments_counter)],
    "sax": [0, copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter),
            copy.deepcopy(all_instruments_counter)],
    "cel": [0, copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter),
            copy.deepcopy(all_instruments_counter)],
    "flu": [0, copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter),
            copy.deepcopy(all_instruments_counter)],
    "gel": [0, copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter),
            copy.deepcopy(all_instruments_counter)],
    "vio": [0, copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter),
            copy.deepcopy(all_instruments_counter)],
    "cla": [0, copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter),
            copy.deepcopy(all_instruments_counter)],
    "pia": [0, copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter),
            copy.deepcopy(all_instruments_counter)],
    "org": [0, copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter),
            copy.deepcopy(all_instruments_counter)],
    "voi": [0, copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter), copy.deepcopy(all_instruments_counter),
            copy.deepcopy(all_instruments_counter)]
}


def generate_probabilities(insturment_index):
    probs = []
    for i in range(11):
        if i != insturment_index:
            probs.append(1./10.)
        else:
            probs.append(0.)
    return probs

def generate_uniform(n):
    return [1/n for i in range(n)]

def create_distribution_plot(dictionary, instrument_name, num_of_instruments=0):
    sum = np.sum(list(dictionary.values()))

    for key in dictionary.keys():
        if dictionary[key] != 0:
            dictionary[key] = dictionary[key] / sum

    plt.bar(dictionary.keys(), dictionary.values(), width=0.9)
    if num_of_instruments != 0:
        plt.savefig('generated_' + instrument_name + '_' + str(num_of_instruments) + '_instruments_count.png')
    else:
        plt.savefig('generated_' + instrument_name + '_instruments_count.png')
    plt.clf()


def save_plots(save_folder_path):
    os.chdir(save_folder_path)
    plt.bar(num_of_companion_instruments_percentage.keys(),
            r_generated_instrument_counter / np.sum(r_generated_instrument_counter), width=0.9)
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
                                     num_of_instruments=i + 1)


# --------------
# metadata Loading
# --------------
data = pd.read_csv('IRMAS_Validation_Data/Plots/validation_metadata_percentage.csv')
all_instruments_percentage = np.array(data.iloc[0].to_numpy()[3:], dtype=float)

data = pd.read_csv('IRMAS_Validation_Data/Plots/validation_instrument_mix_ratios_metadata_percentage.csv')
for i, instrument in enumerate(num_of_companion_instruments_percentage.keys()):
    num_of_companion_instruments_percentage[instrument] = np.array(data.iloc[i].to_numpy()[2:], dtype=float)

for instrument in instruments:
    data = pd.read_csv(f'IRMAS_Validation_Data/Plots/{instrument}/validation_{instrument}_metadata_percentage.csv')
    for i in range(len(instrument_percentages_dict[instrument])):
        instrument_percentages_dict[instrument][i] = np.array(data.iloc[i].to_numpy()[3:], dtype=float)

# --------------
# Logic
# --------------
samples_amount = 250_000
num_of_val_files = 2874
num_of_val_instruments = 4917
mixed_dataset_instrument_amount = num_of_val_instruments / num_of_val_files * samples_amount
total_instrument_count = []

data_bag = DataBag(DEFAULT_TRAIN_ANNOTATION_FILE)

# for instrument_percentage in all_instruments_percentage:
# add 1 to samples because of rounding error
# total_instrument_count.append(int(np.round((mixed_dataset_instrument_amount+1) * instrument_percentage)))

# r_original_count = total_instrument_count.copy()
# r_generated_instrument_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

samples_count = 0
mixed_dataset_samples = {
    'samples': []
}

# while np.sum(total_instrument_count) >= 0:
while samples_count < samples_amount:
    dataset_sample = {
        'label': copy.deepcopy(all_instruments_counter),
        'paths': []
    }
    samples_count += 1

    instrument_index = np.random.choice(np.arange(11), p=generate_uniform(11))
    instrument = instruments[instrument_index]

    dataset_sample['paths'].append(data_bag.get_bag_item(instrument))
    dataset_sample['label'][instrument] = 1

    # total_instrument_count[instruments.index(instrument)] -= 1
    r_generated_instrument_counter[instruments.index(instrument)] += 1
    num_of_companion_instruments = np.random.choice(np.arange(11),
                                                    p=generate_uniform(11))

    if num_of_companion_instruments == 0:
        mixed_dataset_samples['samples'].append(dataset_sample)
        #r_specific_instrument_counter[instrument][num_of_companion_instruments] += 1
        continue

    #if np.sum(instrument_percentages_dict[instrument][num_of_companion_instruments - 1]) == 0.0:
    #    print(
    #        f"{instrument}, {num_of_companion_instruments} | Suma = {np.sum(instrument_percentages_dict[instrument][num_of_companion_instruments])} Nekako prolazi")

    companion_instruments_indices = np.random.choice(11,
                                                     num_of_companion_instruments,
                                                     replace=False,
                                                     p=generate_probabilities(instrument_index))

    # companion_instruments = []
    for companion_instrument_index in companion_instruments_indices:
        companion_instrument = instruments[companion_instrument_index]

        dataset_sample['paths'].append(data_bag.get_bag_item(companion_instrument))
        dataset_sample['label'][companion_instrument] = 1

        # total_instrument_count[companion_instrument_index] -= 1
        r_generated_instrument_counter[companion_instrument_index] += 1

        #r_specific_instrument_counter[instrument][num_of_companion_instruments][companion_instrument] += 1
        # companion_instruments.append(companion_instrument)

    mixed_dataset_samples['samples'].append(dataset_sample)

# Serializing json
dataset_json = json.dumps(mixed_dataset_samples, indent=4)

# Writing to sample.json
with open("MIXED_Uniform_Training_Data/generated_dataset.json", "w") as outfile:
    outfile.write(dataset_json)

# --------------
# Results
# --------------
# print(f'generated instruments: {r_generated_instrument_counter}')
# print(f'validation dataset count: {r_original_count}')
# print(f'total instruments generated: {np.sum(r_generated_instrument_counter)}')
print(f'total samples generated: {samples_count}')

# Uncomment if you want to save plots
# save_folder_path = r'/home/dominik/Desktop/Plots/'
# save_plots(save_folder_path)