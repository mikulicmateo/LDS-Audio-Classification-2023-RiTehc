import io
import soundfile as sf
from AudioUtil import AudioUtil
import matplotlib.pyplot as plt
import torchaudio


class AudioService:

    def __init__(self, spec_window_hop=296, spec_n_mels=224, spec_n_fft=1024, spec_max_db=80, new_channels=1,
                 new_samplerate=22050, max_num_samples=66150):
        self.spec_hop_len = spec_window_hop
        self.new_channels = new_channels
        self.new_samplerate = new_samplerate
        self.max_num_samples = max_num_samples
        self.spec_n_mels = spec_n_mels
        self.spec_n_fft = spec_n_fft
        self.spec_top_db = spec_max_db

    def create_temporary_wav_block(self, block, sample_rate):
        temp_wav = io.BytesIO()
        sf.write(temp_wav, block, sample_rate, format='WAV')
        temp_wav.seek(0)
        return temp_wav

    def preprocess_audio(self, block, sample_rate):
        wav_block = self.create_temporary_wav_block(block, sample_rate)
        audio = torchaudio.load(wav_block, normalize=True)

        resampled = AudioUtil.resample(audio, self.new_samplerate)
        rechanneled = AudioUtil.rechannel(resampled, self.new_channels)
        spectrogram = AudioUtil.generate_spectrogram(rechanneled, self.spec_n_mels, self.spec_n_fft, self.spec_top_db,
                                                     self.spec_hop_len)
        return spectrogram

    def get_spectrograms_from_stream(self, stream, sample_rate):
        spectrograms = []
        for block in stream:
            spec_file = self.preprocess_audio(block, sample_rate)
            spectrograms.append(self.get_spectrogram_image(spec_file))
        return spectrograms

    def get_spectrogram_image(self, spectrogram):
        temp_png = io.BytesIO()
        plt.imsave(temp_png, spectrogram[0], format='png')
        return temp_png
