from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
import sys

from utils.AudioUtil import AudioUtil

sys.path.insert(0, '../utils/')


class IRMASValidationDataset(Dataset):

    def __init__(self, annotations_file, project_dir, window_length=3, window_hop_length=1, new_channels=1, new_samplerate=22050,
                 max_num_samples=66150, n_mels=64, n_fft=1024, max_db=105):
        self.annotations = pd.read_csv(annotations_file)
        self.project_dir = project_dir
        self.window_length = window_length
        self.hop_length = window_hop_length
        self.new_channels = new_channels
        self.new_samplerate = new_samplerate
        self.max_num_samples = max_num_samples
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.top_db = max_db

        self.file_names = []
        for i in range(len(self.annotations)):
            self.file_names.append(os.path.join(self.project_dir, self.annotations.iloc[i, 1]))


    def __len__(self):
        total_windows = sum([(torchaudio.info(file).num_frames // (self.hop_length * 44100)) - (
                    self.window_length // self.hop_length) for file in self.file_names])
        return total_windows


    def __getitem__(self, index):
        file_idx, window_idx = self.get_file_and_window_indices(index)

        waveform, sample_rate = torchaudio.load(self.file_names[file_idx], normalize=True)
        start = window_idx * self.hop_length * sample_rate
        end = start + self.window_length * sample_rate
        windowed_waveform = waveform[:, start:end]
        windowed_waveform = AudioUtil.rechannel((windowed_waveform, sample_rate), self.new_channels)
        windowed_waveform = AudioUtil.resample(windowed_waveform, self.new_samplerate)
        spectrogram = AudioUtil.generate_spectrogram(windowed_waveform, self.n_mels, self.n_fft, self.top_db)

        return spectrogram#, self.file_names[file_idx]


    def get_file_and_window_indices(self, index):
        for file_idx, f in enumerate(self.file_names):
            length = torchaudio.info(f).num_frames
            num_windows = (length // (self.hop_length * 44100)) - (self.window_length // self.hop_length)
            if index < num_windows:
                return file_idx, index
            index -= num_windows


if __name__ == "__main__":
    ANNOTATIONS_FILE = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/IRMAS_Validation_Data/validation_annotation_file.csv'
    PROJECT_DIR = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc'

    ds = IRMASValidationDataset(
        ANNOTATIONS_FILE,
        PROJECT_DIR,
    )

    print(f'There are {len(ds)} samples')
    signals = []
    for i in range(35):
        signals.append(ds[i])
    class_names = ['tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']
    print(signals)
    import matplotlib.pyplot as plt

    s = 0
    for sig in signals:
        plt.imshow(sig[0][0])
        if sig[1] == '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/IRMAS_Validation_Data/(02) dont kill the whale-1.wav':
            s += 1
        plt.title(sig[1][-20:])
        plt.show()

    print(s)
    xx = 'x'