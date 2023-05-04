import pandas as pd
import os
import librosa


project_path = os.path.dirname(os.getcwd())
validation_metadata_path = os.path.join(project_path, "metadata/validation_metadata")
windowed_validation_data_path = os.path.join(project_path, 'WINDOWED_Validation_Data')
annotations_file = os.path.join(validation_metadata_path, "validation_annotation_file.csv")
annotations = pd.read_csv(annotations_file)
SAMPLE_RATE = 44_100
WINDOW_DURATION_SEC = 3
file_names = []
data = []

for i in range(len(annotations)):
    file_names.append(os.path.join(project_path, annotations.iloc[i, 1]))

os.chdir(project_path)
for i, file_path in enumerate(file_names):
    os.chdir(windowed_validation_data_path)
    os.chdir(str(i + 1))
    os.rename(f"{i}.json", f"{i+1}.json")

    label = annotations.iloc[i][2:].to_dict()
    name = annotations.iloc[i][1][len('IRMAS_Validation_Data/'):]
    description = {'label': label, 'name': name}

    stream = librosa.stream(file_path,
                            block_length=1,
                            frame_length=WINDOW_DURATION_SEC * SAMPLE_RATE,
                            hop_length=SAMPLE_RATE,
                            fill_value=0,
                            mono=True)
    max_j = 0
    for j, block in enumerate(stream):
        max_j = j
    data.append([str(i+1), str(max_j+1)])


os.chdir(windowed_validation_data_path)
columns = ['folder', 'num_windows']
pd.DataFrame(data, columns=columns).to_csv('folder_file_mapping.csv')
