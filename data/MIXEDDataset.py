from torch.utils.data import Dataset
import torchaudio
import os
import glob
import json
import sys

sys.path.insert(0, '../utils/')
from utils.AudioUtil import AudioUtil


class MIXEDDataset(Dataset):

    def __init__(self, absolute_path_data_folder, new_samplerate, new_channels, max_num_samples, shift_limit, n_mels,
                 n_fft, mask_percent, n_freq_masks, n_time_masks, max_mixes, db_max, hop_len=None, min_val=-100.0, max_val=48.75732421875):
        self.data_folder = absolute_path_data_folder
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
        self.max_val = max_val
        self.min_val = min_val

    def __len__(self):
        folder = glob.glob(os.path.join(self.data_folder, "**/*.wav"), recursive=True)
        return len(folder)

    def __getitem__(self, index):
        index = index + 1
        path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        audio = torchaudio.load(path, normalize=True)
        resampled = AudioUtil.resample(audio, self.new_samplerate)
        rechanneled = AudioUtil.rechannel(audio, self.new_channels)
        # resized = AudioUtil.pad_trunc(rechanneled, self.max_num_samples)
        # reshifted = AudioUtil.time_shift(resized, self.shift_limit)
        spectrogram = AudioUtil.generate_spectrogram(rechanneled, self.n_mels, self.n_fft, self.top_db, self.hop_len)
        spectrogram = AudioUtil.standardize(spectrogram, self.min_val, self.max_val)
        #augmented_spectrogram = AudioUtil.spectrogram_augment(spectrogram, self.mask_percent, self.n_freq_masks, self.n_time_masks)
        return spectrogram, label

    def _get_audio_sample_path(self, index):
        return os.path.join(self.data_folder, str(index), str(index) + ".wav")

    def _get_audio_sample_label(self, index):
        file = os.path.join(self.data_folder, str(index), str(index) + ".json")
        with open(file, 'r') as openfile:
            description = json.load(openfile)
        return list(description['label'].values())


if __name__ == "__main__":
    ABSOLUTE_PATH_DATA_FOLDER = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/MIXED_Training_Data'
    NEW_SAMPLERATE = 22050  # TODO
    NEW_CHANNELS = 1
    MAX_NUM_SAMPLES = 66150  # TODO
    SHIFT_PERCENT = 0.1
    N_MELS = 64  # height of spec
    N_FFT = 1024
    MAX_MASK_PERCENT = 0.1
    N_FREQ_MASKS = 2
    N_TIME_MASKS = 2
    MAX_MIXES = 5
    MAX_DECIBEL = 105
    HOP_LEN = None # width of spec = Total number of samples / hop_length

    ds = MIXEDDataset(
        ABSOLUTE_PATH_DATA_FOLDER,
        NEW_SAMPLERATE,
        NEW_CHANNELS,
        MAX_NUM_SAMPLES,
        SHIFT_PERCENT,
        N_MELS,
        N_FFT,
        MAX_MASK_PERCENT,
        N_FREQ_MASKS,
        N_TIME_MASKS,
        MAX_MIXES,
        MAX_DECIBEL,
        HOP_LEN
    )

    print(f'There are {len(ds)} samples')
    signal, label = ds[0]
    class_names = ['tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']

    import matplotlib.pyplot as plt
    import numpy as np

    indexes = np.where(np.array(label) == 1)[0]
    title = [class_names[i] for i in indexes]
    print(signal.shape)
    print(title)

    #plt.imsave('dada.png', signal[0])
    print(min(signal[0]))
    print(max(signal[0]))

    plt.imshow(signal[0])
    plt.title(title)
    plt.show()

    # a = 1