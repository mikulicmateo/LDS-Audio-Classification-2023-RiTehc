import os
import random
import sys

import pandas as pd
import torchaudio
from torch.utils.data import Dataset

from utils.AudioUtil import AudioUtil

sys.path.insert(0, '../utils/')


class IRMASDataset(Dataset):

    def __init__(self, annotations_file, project_dir, new_samplerate, new_channels, max_num_samples, shift_limit,
                 n_mels,
                 n_fft, mask_percent, n_freq_masks, n_time_masks, max_mixes, db_max, hop_len=None):
        self.annotations = pd.read_csv(annotations_file)
        self.project_dir = project_dir
        self.new_samplerate = new_samplerate
        self.new_channels = new_channels
        self.max_num_samples = max_num_samples
        self.shift_limit = shift_limit
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.mask_percent = mask_percent
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.max_mixes = max_mixes
        self.top_db = db_max

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        samples_to_mix = int((self.max_mixes + 1) * random.random())
        max_samples = len(self.annotations)
        print(samples_to_mix)
        audio_sample_paths = [self._get_audio_sample_path(index)]
        print(audio_sample_paths)
        label = self._get_audio_sample_label(index)
        print(1)
        indexes = [index]
        for i in range(samples_to_mix - 1):
            mix_index = int((max_samples + 1) * random.random())
            audio_sample_paths.append(self._get_audio_sample_path(mix_index))
            label = label + self._get_audio_sample_label(mix_index)
            indexes.append(mix_index)
        labels = [(lambda x: 1 if x > 0 else x)(x) for x in label]
        audio = self._get_mixed_audios(audio_sample_paths)
        resampled = AudioUtil.resample(audio, self.new_samplerate)
        rechanneled = AudioUtil.rechannel(resampled, self.new_channels)
        resized = AudioUtil.pad_trunc(rechanneled, self.max_num_samples)
        reshifted = AudioUtil.time_shift(resized, self.shift_limit)
        spectrogram = AudioUtil.generate_spectrogram(reshifted, self.n_mels, self.n_fft, self.top_db, self.hop_len)
        augmented_spectrogram = AudioUtil.spectrogram_augment(spectrogram, self.mask_percent, self.n_freq_masks,
                                                              self.n_time_masks)
        return augmented_spectrogram, labels

    def _get_audio_sample_path(self, index):
        return os.path.join(self.project_dir, self.annotations.iloc[index, 1])

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 2:13]

    def _get_mixed_audios(self, audio_sample_paths):
        k = 0
        audio = torchaudio.load(audio_sample_paths[0], normalize=True)
        for audio_sample_path in audio_sample_paths:
            if k == 0:
                k += 1
                continue
            signal, sr = torchaudio.load(audio_sample_path, normalize=True)
            audio = (audio[0] + signal, audio[1])
        return audio
