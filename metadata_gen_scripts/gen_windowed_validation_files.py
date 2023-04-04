import pandas as pd
import os
import librosa
import soundfile as sf
import json

project_dir = '/home/dominik/Work/Lumen Datascience/LDS-Audio-Classification-2023-RiTehc'
annotations = pd.read_csv('/home/dominik/Work/Lumen Datascience/LDS-Audio-Classification-2023-RiTehc/IRMAS_Validation_Data/validation_annotation_file.csv')
sample_rate = 44_100
window_duration_sec = 3
file_names = []

for i in range(len(annotations)):
    file_names.append(os.path.join(project_dir, annotations.iloc[i, 1]))

windowed_validation_data_path = "/home/dominik/Work/Lumen Datascience/LDS-Audio-Classification-2023-RiTehc/WINDOWED_Validation_Data/"

for i, file_path in enumerate(file_names):
    os.chdir(windowed_validation_data_path)
    os.mkdir(str(i + 1))
    os.chdir(os.path.join(windowed_validation_data_path, str(i + 1)))

    label = annotations.iloc[i][2:].to_dict()
    name = annotations.iloc[i][1][len('IRMAS_Validation_Data/'):]
    description = {'label': label, 'name': name}

    stream = librosa.stream(file_path,
                            block_length=1,
                            frame_length=window_duration_sec * sample_rate,
                            hop_length=sample_rate,
                            fill_value=0,
                            mono=True)

    for j, block in enumerate(stream):
        sf.write(f"W{j + 1}.wav", block, sample_rate)

    with open(str(i) + ".json", "w") as outfile:
        outfile.write(json.dumps(description, indent=4))
