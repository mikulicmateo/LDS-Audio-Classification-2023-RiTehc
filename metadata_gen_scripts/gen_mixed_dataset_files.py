import json
import os

import librosa as lr
import soundfile as sf

project_path = os.path.dirname(os.getcwd())
mixed_dataset_path = os.path.join(project_path, "MIXED_Training_Data")
dataset_description_file = os.path.join(mixed_dataset_path, "generated_dataset.json")


def _get_mixed_audios(paths):
    k = 0
    y, s = lr.load(os.path.join(project_path, paths[0]), sr=None)
    for audio_sample_path in paths:
        if k == 0:
            k += 1
            continue
        sig, _ = lr.load(os.path.join(project_path, audio_sample_path), sr=None)
        y = y + sig
    return y, s


def save_signal(file_name, audio):
    sf.write(file_name + ".wav", audio[0], audio[1])


def save_description(file_name, description):
    with open(file_name + ".json", "w") as outfile:
        outfile.write(json.dumps(description, indent=4))


with open(dataset_description_file, 'r') as openfile:
    json_object = json.load(openfile)

num_of_sample = 1
for sample in json_object['samples']:
    signal, sr = _get_mixed_audios(sample['paths'])

    dir_name = os.path.join(mixed_dataset_path, str(num_of_sample))
    os.mkdir(dir_name)
    file_path_name = os.path.join(dir_name, str(num_of_sample))

    save_signal(file_path_name, (signal, sr))
    save_description(file_path_name, sample)

    num_of_sample += 1
