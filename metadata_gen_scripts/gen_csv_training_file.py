import os
import pandas as pd
import json
from structures import instrument_dict, instrument_list, genre_list

project_path = os.path.dirname(os.getcwd())
training_dataset_path = os.path.join(project_path, "IRMAS_Training_Data")
config_file_path = os.path.join(project_path, "config.json")
columns = ['path']
columns.extend(instrument_list)
columns.append('genre')

with open(config_file_path, "r") as config_file:
    config_dict = json.load(config_file)

training_files = []
os.chdir(training_dataset_path)

for folder in instrument_list:
    instrument_dict.update({str(folder): 1})

    for file in os.listdir(folder):
        file_genre = ''
        for genre in genre_list:
            if genre in file:
                file_genre = genre

        assert file_genre != ''

        training_files.append([os.path.join('IRMAS_Training_Data', folder, file),
                               instrument_dict.get("tru"),
                               instrument_dict.get("gac"),
                               instrument_dict.get("sax"),
                               instrument_dict.get("cel"),
                               instrument_dict.get("flu"),
                               instrument_dict.get("gel"),
                               instrument_dict.get("vio"),
                               instrument_dict.get("cla"),
                               instrument_dict.get("pia"),
                               instrument_dict.get("org"),
                               instrument_dict.get("voi"),
                               file_genre])

    instrument_dict.update({str(folder): 0})

save_path = os.path.join(project_path, "metadata/training_metadata")
pd.DataFrame(training_files, columns=columns).to_csv(
    os.path.join(save_path, "training_annotation_file.csv")
)
