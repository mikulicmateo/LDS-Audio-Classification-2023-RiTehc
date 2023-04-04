import torch
import sys

sys.path.insert(0, '../utils/')
from utils.AudioUtil import AudioUtil

class ModelService:
    def __init__(self, window_length=3, window_hop_length=1, new_channels=1, new_samplerate=22050, max_num_samples=66150,
                 n_mels=64, n_fft=1024, max_db=105):
        #load models
        self.encoder = torch.load('models/Encoder.pt')
        self.encoder.eval()
        self.window_length = window_length
        self.hop_length = window_hop_length
        self.new_channels = new_channels
        self.new_samplerate = new_samplerate
        self.max_num_samples = max_num_samples
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.top_db = max_db


    def preprocess_audio(self, audio):
        resampled = AudioUtil.resample(audio, self.new_samplerate)
        # rechanneled = AudioUtil.rechannel(audio, self.new_channels)
        # resized = AudioUtil.pad_trunc(rechanneled, self.max_num_samples)
        # reshifted = AudioUtil.time_shift(resized, self.shift_limit)
        spectrogram = AudioUtil.generate_spectrogram(resampled, self.n_mels, self.n_fft, self.top_db, self.hop_len)
        #augmented_spectrogram = AudioUtil.spectrogram_augment(spectrogram, self.mask_percent, self.n_freq_masks, self.n_time_masks)
        pass

    def classify_audio(self, file):
        spectrogram = self.preprocess_audio(file)
        #encode
        #classify
        #return classification
        pass