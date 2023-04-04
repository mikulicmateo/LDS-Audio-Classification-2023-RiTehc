import pandas as pd
import os
import librosa
import soundfile as sf
import json

project_dir = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc'
annotations_file = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/IRMAS_Validation_Data/validation_annotation_file.csv'
annotations = pd.read_csv(annotations_file)
sample_rate = 44_100
window_duration_sec = 3
file_names = []
data = []

for i in range(len(annotations)):
    file_names.append(os.path.join(project_dir, annotations.iloc[i, 1]))

windowed_validation_data_path = os.path.join(project_dir, 'WINDOWED_Validation_Data')
os.chdir(project_dir)
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
    max_j = 0
    for j, block in enumerate(stream):
        sf.write(f"W{j + 1}.wav", block, sample_rate)
        max_j = j
    data.append([str(i+1), str(max_j+1)])




    with open(str(i+1) + ".json", "w") as outfile:
        outfile.write(json.dumps(description, indent=4))

os.chdir(windowed_validation_data_path)
columns = ['folder', 'num_windows']
pd.DataFrame(data, columns=columns).to_csv('folder_file_mapping.csv')