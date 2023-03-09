from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
import sys

from utils.AudioUtil import AudioUtil

sys.path.insert(0, '../utils/')


class IRMASDataset(Dataset):

    def __init__(self, annotations_file, project_dir, new_samplerate, new_channels, new_length, shift_limit, n_mels,
                 n_fft, mask_percent, n_freq_masks, n_time_masks, hop_len=None):
        self.annotations = pd.read_csv(annotations_file)
        self.project_dir = project_dir
        self.new_samplerate = new_samplerate
        self.new_channels = new_channels
        self.new_length = new_length
        self.shift_limit = shift_limit
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.mask_percent = mask_percent
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        audio = torchaudio.load(audio_sample_path)
        resampled = AudioUtil.resample(audio, self.new_samplerate)
        rechanneled = AudioUtil.rechannel(resampled, self.new_channels)
        resized = AudioUtil.pad_trunc(rechanneled, self.new_length)
        reshifted = AudioUtil.time_shift(resized, self.shift_limit)
        spectrogram = AudioUtil.generate_spectrogram(reshifted, self.n_mels, self.n_fft, self.hop_len)
        augmented_spectrogram = AudioUtil.spectrogram_augment(spectrogram, self.mask_percent, self.n_freq_masks,
                                                              self.n_time_masks)
        return augmented_spectrogram, label

    def _get_audio_sample_path(self, index):
        return os.path.join(self.project_dir, self.annotations.iloc[index, 1])

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 3]


# Check if data is loaded properly.
if __name__ == "__main__":
    ANNOTATIONS_FILE = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/IRMAS_Training_Data/training_annotation_file.csv'
    PROJECT_DIR = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc'
    NEW_SAMPLERATE = 16000
    NEW_CHANNELS = 1
    NEW_LENGTH_MS = 3000
    SHIFT_PERCENT = 0.1
    N_MELS = 64
    N_FFT = 1024
    MAX_MASK_PERCENT = 0.1
    N_FREQ_MASKS = 2
    N_TIME_MASKS = 2
    HOP_LEN = None

    ds = IRMASDataset(
        ANNOTATIONS_FILE,
        PROJECT_DIR,
        NEW_SAMPLERATE,
        NEW_CHANNELS,
        NEW_LENGTH_MS,
        SHIFT_PERCENT,
        N_MELS,
        N_FFT,
        MAX_MASK_PERCENT,
        N_FREQ_MASKS,
        N_TIME_MASKS,
        HOP_LEN
    )

    print(f'There are {len(ds)} samples')
    # signal, label = ds[0]

    # import matplotlib.pyplot as plt

    # plt.imshow(signal[0])
    # plt.show()

