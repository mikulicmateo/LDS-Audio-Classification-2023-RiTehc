import torch
import torchaudio
import random
from torchaudio import transforms


class AudioUtil:

    @staticmethod
    def rechannel(audio, new_channel):
        signal, sr = audio

        if signal.shape[0] == new_channel:
            return audio

        if new_channel == 1:
            resignal = torch.mean(signal, dim=0, keepdim=True)  # signal[:1, :]
        else:
            resignal = torch.cat([signal, signal])

        return resignal, sr

    @staticmethod
    def resample(audio, newsr):
        signal, sr = audio

        if sr == newsr:
            return audio

        num_channels = signal.shape[0]
        resignal = torchaudio.transforms.Resample(sr, newsr)(signal[:1, :])

        if num_channels > 1:
            retwo = torchaudio.transforms.Resample(sr, newsr)(signal[1:, :])
            resignal = torch.cat([resignal, retwo])

        return resignal, newsr

    @staticmethod
    def pad_trunc(audio, max_len):
        signal, sr = audio
        num_rows, signal_len = signal.shape

        if signal_len > max_len:
            signal = signal[:, :max_len]

        elif signal_len < max_len:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - signal_len)
            pad_end_len = max_len - signal_len - pad_begin_len
            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            signal = torch.cat((pad_begin, signal, pad_end), 1)

        return signal, sr

    @staticmethod
    def time_shift(audio, shift_limit):
        signal, sr = audio
        _, signal_len = signal.shape
        shift_amt = int(random.random() * shift_limit * signal_len)
        return signal.roll(shift_amt), sr

    @staticmethod
    def generate_spectrogram(audio, n_mels=64, n_fft=1024, top_db=105, hop_len=None):
        signal, sr = audio

        make_spectrogram = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
        spec = make_spectrogram(signal)

        convert_to_db = transforms.AmplitudeToDB(top_db=top_db)
        spec = convert_to_db(spec)

        return spec

    @staticmethod
    def spectrogram_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
        freq_mask_param = max_mask_pct * n_mels

        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec
