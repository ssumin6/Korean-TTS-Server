import torch
import numpy as np
from librosa.filters import mel as librosa_mel_fn
from audio_processing import (
    dynamic_range_compression,
    dynamic_range_decompression,
    mel_normalize,
    mel_denormalize,
)
from stft import STFT
import hparams as hp


class TacotronSTFT(torch.nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = hp.num_mels
        self.sampling_rate = hp.sample_rate
        self.stft_fn = STFT(n_fft, hop_length, win_length)
        self.max_abs_mel_value = hp.max_abs_value
        mel_basis = librosa_mel_fn(
            hp.sample_rate, n_fft, hp.num_mels, hp.mel_fmin, hp.mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def transform2(self, y):
        result = torch.stft(y, n_fft=1024, hop_length=256, win_length=1024)
        real = result[:, :, :, 0]
        imag = result[:, :, :, 1]
        magnitude = torch.sqrt(real**2 + imag**2)
        phase = torch.autograd.Variable(torch.atan2(imag.data, real.data))
        return magnitude, phase

    def cepstrum_from_mel(self, mel, ref_level_db=20, magnitude_power=1.5):
        assert torch.min(mel.data) >= -self.max_abs_mel_value
        assert torch.max(mel.data) <= self.max_abs_mel_value

        spec = mel_denormalize(mel, self.max_abs_mel_value)
        magnitudes = self.spectral_de_normalize(spec + ref_level_db) ** (
            1 / magnitude_power
        )
        pow_spec = (magnitudes**2) / 1024 
        db_pow_spec = torch.log(torch.clamp(pow_spec, min=1e-5)) * 20  
        mcc = dct(db_pow_spec, "ortho")
        return mcc

    def cepstrum_from_audio(self, y):
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1
        magnitudes, phases = self.stft_fn.transform(y)
        pow_spec = (1 / 1024) * magnitudes**2
        mel_spectrogram = (
            torch.matmul(self.mel_basis, pow_spec).squeeze(0).transpose(0, 1)
        )
        
        db_mel_spectrogram = (
            torch.log10(torch.clamp(pow_spec, min=1e-5)) * 20
        )
        mcc = dct(db_mel_spectrogram, "ortho")
        return mcc

    def mel_spectrogram(self, y, ref_level_db=20, magnitude_power=1.5):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data

        mel_output = torch.matmul(
            self.mel_basis, torch.abs(magnitudes) ** magnitude_power
        )
        mel_output = self.spectral_normalize(mel_output) - ref_level_db
        mel_output = mel_normalize(mel_output, self.max_abs_mel_value)
        return mel_output
