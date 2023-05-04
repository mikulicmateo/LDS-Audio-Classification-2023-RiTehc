import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from structures import instrument_dict as all_instruments_counter
from structures import instrument_mix_counter, specific_instrument_counter, instrument_list


def create_distribution_plot(dictionary, instrument_name, data_raw, data_percentages, num_of_instruments=0):
    sum_for_percentages = np.sum(list(dictionary.values()))
    temp_raw = [instrument_name, num_of_instruments]
    temp_percentage = [instrument_name, num_of_instruments]

    for key in dictionary.keys():
        temp_raw.append(dictionary[key])
        if dictionary[key] != 0:
            dictionary[key] = dictionary[key] / sum_for_percentages
        temp_percentage.append(dictionary[key])

    plt.bar(dictionary.keys(), dictionary.values(), width=0.9)
    if num_of_instruments != 0:
        plt.savefig('validation_' + instrument_name + '_' + str(num_of_instruments) + '_instruments_count.png')
    else:
        plt.savefig('validation_' + instrument_name + '_instruments_count.png')
    plt.clf()

    data_raw.append(temp_raw)
    data_percentages.append(temp_percentage)

    return data_raw, data_percentages


def load_val_data_to_dicts():
    files = os.listdir(os.getcwd())
    files.sort()

    for i in range(0, len(files) - 1, 2):
        txt_file = files[i]
        wav_file = files[i + 1]

        if not (wav_file.__contains__(txt_file[: len(txt_file) - 4])):
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
                        specific_instrument_counter[instrument_name][num_of_instruments - 1] += 1
                    else:
                        if instrument == companion_instrument:
                            continue
                        specific_instrument_counter[instrument_name][num_of_instruments - 1][
                            companion_instrument_name] += 1


def gen_instrument_mix_ratios_csv():
    columns_instrument_num = ['type', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    data_raw = []
    data_percentages = []

    for instrument in instrument_mix_counter.keys():
        instrument_all_companion_instruments_count = np.sum(instrument_mix_counter[instrument])
        temp_raw = [instrument]
        temp_percentages = [instrument]

        for i in range(len(instrument_mix_counter[instrument])):
            temp_raw.append(instrument_mix_counter[instrument][i])
            temp_percentages.append(instrument_mix_counter[instrument][i] / instrument_all_companion_instruments_count)

        data_raw.append(temp_raw)
        data_percentages.append(temp_percentages)

    pd.DataFrame(data_raw, columns=columns_instrument_num).to_csv(
        'validation_instrument_mix_ratios_metadata_raw.csv')
    pd.DataFrame(data_percentages, columns=columns_instrument_num).to_csv(
        'validation_instrument_mix_ratios_metadata_percentage.csv')


def gen_all_instrument_csv(columns):
    data_raw = []
    data_percentages = []

    data_raw, data_percentages = create_distribution_plot(all_instruments_counter, 'all', data_raw, data_percentages)

    pd.DataFrame(data_raw, columns=columns).to_csv('validation_metadata_raw.csv')
    pd.DataFrame(data_percentages, columns=columns).to_csv('validation_metadata_percentage.csv')


def gen_specific_instrument_csv(columns, save_path):
    for instrument in specific_instrument_counter.keys():
        os.chdir(save_path)
        if not os.path.exists(os.path.join(save_path, instrument)):
            os.makedirs(instrument)
        os.chdir(os.path.join(os.path.join(save_path, instrument)))

        data_raw = []
        data_percentages = []

        for i in range(len(specific_instrument_counter[instrument])):
            if i == 0:
                continue

            data_raw, data_percentages = create_distribution_plot(specific_instrument_counter[instrument][i],
                                                                  instrument,
                                                                  data_raw,
                                                                  data_percentages,
                                                                  i + 1)

        pd.DataFrame(data_raw, columns=columns).to_csv(f'validation_{instrument}_metadata_raw.csv')
        pd.DataFrame(data_percentages, columns=columns).to_csv(f'validation_{instrument}_metadata_percentage.csv')


def main():
    project_path = os.path.dirname(os.getcwd())
    validation_dataset_path = os.path.join(project_path, "IRMAS_Validation_Data")
    validation_metadata_path = os.path.join(project_path, "metadata/validation_metadata")
    columns = ['type', 'num_of_instruments']
    columns.extend(instrument_list)

    os.chdir(validation_dataset_path)
    load_val_data_to_dicts()

    os.chdir(validation_metadata_path)
    if not os.path.exists(os.path.join(validation_metadata_path, 'Plots')):
        os.makedirs('Plots')
    os.chdir(os.path.join(validation_metadata_path, 'Plots'))

    # Creating instrument mix ratios metadata csv file
    gen_instrument_mix_ratios_csv()

    # Creating all instrument metadata csv file
    gen_all_instrument_csv(columns)

    # Creating metadata csv file for specific instruments
    gen_specific_instrument_csv(columns, os.getcwd())

    print(f'Saved to: {validation_metadata_path}')


if __name__ == '__main__':
    main()
