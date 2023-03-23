import os
import json
import librosa as lr
import soundfile as sf

WORKING_DIR = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc'
DATASET_DESCRIPTION_FILE_JSON = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/metadata_gen_scripts/sample.json'
DATASET_DESTINATION_DIR = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/data'


def _get_mixed_audios(paths):
    k = 0
    y, s = lr.load(os.path.join(WORKING_DIR, paths[0]), sr=None)
    for audio_sample_path in paths:
        if k == 0:
            k += 1
            continue
        sig, _ = lr.load(os.path.join(WORKING_DIR, audio_sample_path), sr=None)
        y = y + sig
    return y, s


def save_signal(file_name, audio):
    sf.write(file_name + ".wav", audio[0], audio[1])


def save_description(file_name, description):
    with open(file_name + ".json", "w") as outfile:
        outfile.write(json.dumps(description, indent=4))


with open(DATASET_DESCRIPTION_FILE_JSON, 'r') as openfile:
    json_object = json.load(openfile)

num_of_sample = 1
for sample in json_object['samples']:
    signal, sr = _get_mixed_audios(sample['paths'])

    dir_name = os.path.join(DATASET_DESTINATION_DIR, str(num_of_sample))
    os.mkdir(dir_name)
    file_path_name = os.path.join(dir_name, str(num_of_sample))

    save_signal(file_path_name, (signal, sr))
    save_description(file_path_name, sample)

    num_of_sample += 1
