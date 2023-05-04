import glob
import json
import os
import sys

import pandas as pd
import torchaudio
from torch.utils.data import Dataset

sys.path.insert(0, '../utils/')
from utils.AudioUtil import AudioUtil


class WINDOWEDValidationDataset(Dataset):

    def __init__(self, absolute_path_data_folder, folder_file_mapping_path, new_samplerate, new_channels,
                 max_num_samples, n_mels, n_fft, db_max, hop_len=None, min_val=-100.0, max_val=48.75732421875):
        self.data_folder = absolute_path_data_folder
        self.folder_file_mapping = pd.read_csv(folder_file_mapping_path)
        self.new_samplerate = new_samplerate
        self.new_channels = new_channels
        self.max_num_samples = max_num_samples
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.top_db = db_max
        self.max_val = max_val
        self.min_val = min_val

    def __len__(self):
        folder = glob.glob(os.path.join(self.data_folder, "**/*.json"), recursive=True)
        return len(folder)

    def __getitem__(self, index):
        folder, num_windows = self._get_window_folder_and_num_windows(index)
        label = self._get_audio_common_label(folder)
        full_folder_path = os.path.join(self.data_folder, str(folder))
        window_list = self._get_audio_windows(full_folder_path, num_windows)
        return window_list, label

    def _get_audio_windows(self, full_folder_path, num_windows):
        window_list = []
        for i in range(num_windows):
            path = os.path.join(full_folder_path, f'W{i + 1}.wav')
            audio = torchaudio.load(path, normalize=True)
            window_list.append(self._transform_audio(audio))
        return window_list

    def _transform_audio(self, audio):
        resampled = AudioUtil.resample(audio, self.new_samplerate)
        rechanneled = AudioUtil.rechannel(resampled, self.new_channels)
        # resized = AudioUtil.pad_trunc(rechanneled, self.max_num_samples)
        spectrogram = AudioUtil.generate_spectrogram(rechanneled, self.n_mels, self.n_fft, self.top_db, self.hop_len)
        # spectrogram = AudioUtil.standardize(spectrogram, self.min_val, self.max_val)
        return spectrogram

    def _get_window_folder_and_num_windows(self, index):
        folder = self.folder_file_mapping.iloc[index][1]
        num_windows = self.folder_file_mapping.iloc[index][2]
        return folder, num_windows

    def _get_audio_common_label(self, folder_index):
        file = os.path.join(self.data_folder, str(folder_index), str(folder_index) + ".json")
        with open(file, 'r') as openfile:
            description = json.load(openfile)
        return list(description['label'].values())
