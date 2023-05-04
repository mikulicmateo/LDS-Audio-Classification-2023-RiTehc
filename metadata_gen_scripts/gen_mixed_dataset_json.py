import copy
import json
import os

import numpy as np
from matplotlib import pyplot as plt

from structures import instrument_dict as all_instruments_counter
from structures import instrument_list, genre_list, specific_instrument_counter
from structures import num_of_companion_instruments_percentage, instrument_percentages_dict

project_path = os.path.dirname(os.getcwd())
os.chdir(project_path)
validation_metadata_path = os.path.join(project_path, "metadata/validation_metadata/Plots")
default_train_annotation_file = os.path.join(project_path, "metadata/training_metadata/training_annotation_file.csv")
config_file_path = os.path.join(project_path, "config.json")
generated_instrument_counter = [0 for _ in range(11)]
all_instruments_percentage = []

with open(config_file_path, "r") as config_file:
    config_dict = json.load(config_file)


def gen_uniform_companion_instruments(instrument_index):
    probability = []
    for i in range(11):
        if i != instrument_index:
            probability.append(1. / 10.)
        else:
            probability.append(0.)
    return probability


def gen_uniform(length):
    return [1 / length for _ in range(length)]


def gen_uniform_companion_num(companion_instruments):
    probability = []
    for i in range(companion_instruments + 1):
        probability.append(1. / companion_instruments)

    for i in range(companion_instruments + 1, len(instrument_list)):
        probability.append(0.)

    return probability


def create_distribution_plot(dictionary, instrument_name, num_of_instruments=0):
    sum_for_percentages = np.sum(list(dictionary.values()))

    for key in dictionary.keys():
        if dictionary[key] != 0:
            dictionary[key] = dictionary[key] / sum_for_percentages

    plt.bar(dictionary.keys(), dictionary.values(), width=0.9)
    if num_of_instruments != 0:
        plt.savefig('generated_' + instrument_name + '_' + str(num_of_instruments) + '_instruments_count.png')
    else:
        plt.savefig('generated_' + instrument_name + '_instruments_count.png')
    plt.clf()


def save_plots(save_folder_path, max_companion_instruments):
    os.chdir(save_folder_path)
    plt.bar(num_of_companion_instruments_percentage.keys(),
            generated_instrument_counter / np.sum(generated_instrument_counter), width=0.9)
    plt.savefig('generated_all_instruments_count.png')
    plt.clf()

    for instrument in specific_instrument_counter.keys():

        os.chdir(save_folder_path)
        if not os.path.exists(os.path.join(save_folder_path, instrument)):
            os.makedirs(instrument)
        os.chdir(os.path.join(save_folder_path, instrument))

        for i in range(max_companion_instruments + 1):
            if i == 0:
                continue

            create_distribution_plot(specific_instrument_counter[instrument][i],
                                     instrument,
                                     num_of_instruments=i + 1)


def load_dataset_gen_metadata_into_dicts():
    import pandas as pd
    val_max_companion_instruments = 4
    global all_instruments_percentage
    data = pd.read_csv(os.path.join(validation_metadata_path, 'validation_metadata_percentage.csv'))
    all_instruments_percentage = np.array(data.iloc[0].to_numpy()[3:], dtype=float)

    data = pd.read_csv(
        os.path.join(validation_metadata_path, 'validation_instrument_mix_ratios_metadata_percentage.csv')
    )
    for i, instrument in enumerate(num_of_companion_instruments_percentage.keys()):
        num_of_companion_instruments_percentage[instrument] = np.array(data.iloc[i].to_numpy()[2:], dtype=float)

    for instrument in instrument_list:
        data = pd.read_csv(
            os.path.join(validation_metadata_path, f'{instrument}/validation_{instrument}_metadata_percentage.csv')
        )
        for i in range(val_max_companion_instruments):
            instrument_percentages_dict[instrument][i] = np.array(data.iloc[i].to_numpy()[3:], dtype=float)

    return val_max_companion_instruments

def load_dataset_uniform_gen_into_dicts(max_companion_instruments):
    global all_instruments_percentage
    all_instruments_percentage = gen_uniform(len(instrument_list))

    for instrument in instrument_list:
        num_of_companion_instruments_percentage[instrument] = gen_uniform_companion_num(
            max_companion_instruments)

    for instrument_index, instrument in enumerate(instrument_list):
        for i in range(max_companion_instruments):
            instrument_percentages_dict[instrument][i] = gen_uniform_companion_instruments(instrument_index)

    return max_companion_instruments

def generate_dataset(mixed_dataset_samples, data_bag, samples_amount):
    samples_count = 0

    while samples_count < samples_amount:
        dataset_sample = {
            'label': all_instruments_counter.copy(),
            'paths': []
        }
        samples_count += 1

        instrument_index = np.random.choice(np.arange(len(instrument_list)), p=all_instruments_percentage)
        instrument = instrument_list[instrument_index]

        generated_instrument_counter[instrument_index] += 1

        dataset_sample['paths'].append(data_bag.get_bag_item(instrument))
        dataset_sample['label'][instrument] = 1

        num_of_companion_instruments = np.random.choice(np.arange(len(instrument_list)),
                                                        p=num_of_companion_instruments_percentage[instrument])

        if num_of_companion_instruments == 0:
            mixed_dataset_samples['samples'].append(dataset_sample)
            continue

        companion_instruments_indices = np.random.choice(np.arange(len(instrument_list)),
                                                         num_of_companion_instruments,
                                                         replace=False,
                                                         p=instrument_percentages_dict[instrument][
                                                             num_of_companion_instruments - 1])

        for companion_instrument_index in companion_instruments_indices:
            companion_instrument = instrument_list[companion_instrument_index]

            dataset_sample['paths'].append(data_bag.get_bag_item(companion_instrument))
            dataset_sample['label'][companion_instrument] = 1

            specific_instrument_counter[instrument][num_of_companion_instruments][companion_instrument] += 1

        mixed_dataset_samples['samples'].append(dataset_sample)

    return mixed_dataset_samples


def generate_genre_valid_dataset(mixed_dataset_samples, data_bag, samples_amount):
    no_genre_flag = False
    samples_count = 0

    while samples_count < samples_amount:
        if no_genre_flag:
            samples_count -= 1
            no_genre_flag = False

        dataset_sample = {
            'genre': "",
            'label': copy.deepcopy(all_instruments_counter),
            'paths': []
        }
        samples_count += 1

        instrument_index = np.random.choice(np.arange(11), p=all_instruments_percentage)
        instrument = instrument_list[instrument_index]

        genre_index = np.random.choice(np.arange(len(genre_list)),
                                       p=gen_uniform(len(genre_list)))
        genre = genre_list[genre_index]

        while True:
            path = data_bag.get_bag_item(instrument, genre)
            if path is None:
                genre_index = np.random.choice(np.arange(len(genre_list)),
                                               p=gen_uniform(len(genre_list)))
                genre = genre_list[genre_index]
            else:
                break

        dataset_sample['genre'] = genre
        dataset_sample['paths'].append(path)
        dataset_sample['label'][instrument] = 1

        generated_instrument_counter[instrument_list.index(instrument)] += 1
        num_of_companion_instruments = np.random.choice(np.arange(11),
                                                        p=num_of_companion_instruments_percentage[instrument])

        if num_of_companion_instruments == 0:
            mixed_dataset_samples['samples'].append(dataset_sample)
            continue

        companion_instruments_indices = np.random.choice(11,
                                                         num_of_companion_instruments,
                                                         replace=False,
                                                         p=instrument_percentages_dict[instrument][
                                                             num_of_companion_instruments - 1])

        for companion_instrument_index in companion_instruments_indices:
            companion_instrument = instrument_list[companion_instrument_index]
            loop_counter = 0
            while True:
                path = data_bag.get_bag_item(companion_instrument, genre)
                if path is None:
                    loop_counter += 1
                    companion_instrument_index = np.random.choice(11, replace=False,
                                                                  p=instrument_percentages_dict[instrument][
                                                                      num_of_companion_instruments - 1])
                    companion_instrument = instrument_list[companion_instrument_index]
                else:
                    break

                # if this genre does not have any other instruments
                # generate new instrument
                if loop_counter == 11:
                    no_genre_flag = True
                    break

            if no_genre_flag:
                break

            dataset_sample['paths'].append(path)
            dataset_sample['label'][companion_instrument] = 1

            generated_instrument_counter[companion_instrument_index] += 1
            specific_instrument_counter[instrument][num_of_companion_instruments][companion_instrument] += 1

        if no_genre_flag:
            continue

        mixed_dataset_samples['samples'].append(dataset_sample)

    return mixed_dataset_samples


def main():
    samples_amount = config_dict["MIXED_DATA_SAMPLES"]
    mixed_dataset_samples = {
        'samples': []
    }

    if not config_dict["GENERATE_UNIFORM_DATASET"]:
        max_companion_instruments = load_dataset_gen_metadata_into_dicts()
    else:
        max_companion_instruments = load_dataset_uniform_gen_into_dicts(config_dict["MAX_COMPANION_INSTRUMENTS"])

    if config_dict["GENERATE_DATASET_WITH_GENRES"]:
        from DataBagGenre import DataBagGenre
        data_bag = DataBagGenre(default_train_annotation_file)
        mixed_dataset_samples = generate_genre_valid_dataset(mixed_dataset_samples, data_bag, samples_amount)
    else:
        from DataBag import DataBag
        data_bag = DataBag(default_train_annotation_file)
        mixed_dataset_samples = generate_dataset(mixed_dataset_samples, data_bag, samples_amount)

    # Serializing json
    dataset_json = json.dumps(mixed_dataset_samples, indent=4)

    # Writing to sample.json
    with open("MIXED_Training_Data/generated_dataset.json", "w") as outfile:
        outfile.write(dataset_json)

    # --------------
    # Results
    # --------------
    # print(f'generated instruments: {r_generated_instrument_counter}')
    # print(f'validation dataset count: {r_original_count}')
    # print(f'total instruments generated: {np.sum(r_generated_instrument_counter)}')
    print(f'total samples generated: {samples_amount}')

    # Uncomment if you want to save plots
    save_folder_path = os.path.join(project_path, "metadata/mixed_dataset_metadata")
    save_plots(save_folder_path, max_companion_instruments)


if __name__ == '__main__':
    main()
