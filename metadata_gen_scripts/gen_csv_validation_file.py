import os
import pandas as pd
from structures import instrument_dict, instrument_list

project_path = os.path.dirname(os.getcwd())
validation_dataset_path = os.path.join(project_path, "IRMAS_Validation_Data")

columns = ['path']
columns.extend(instrument_list)

os.chdir(validation_dataset_path)
files = os.listdir(os.getcwd())
files.sort()

data = []

for i in range(0, len(files), 2):
    txt_file = files[i]
    wav_file = files[i+1]

    if not(wav_file.__contains__(txt_file[: len(txt_file) - 4])):
        print("Error: Adjecent files not the same!")
        break

    with open(txt_file) as f:
        lines = f.readlines()

    for line in lines:
        instrument_dict.update({line[0:3]: 1})

    data.append([os.path.join('IRMAS_Validation_Data', wav_file),
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
                 instrument_dict.get("voi")])
    
    for line in lines:
        instrument_dict.update({line[0:3]: 0})


save_path = os.path.join(project_path, "metadata/validation_metadata")
pd.DataFrame(data, columns=columns).to_csv(
    os.path.join(save_path, 'validation_annotation_file.csv')
)
