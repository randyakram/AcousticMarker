"""
Spectrogram feature extraction and the PyTorch Dataset that wraps it. Shared so Stage 3's
training and Stage 5's confirmation-set inference use byte-for-byte identical features -- if
these ever drifted between stages, the model would silently see slightly different input at
inference time than it was trained on.
"""

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

from . import config
from .audio import load_and_resample, pad_or_trim


def calculate_features(signal, sr=config.SAMPLE_RATE):
    """Band-limited, log-scaled, per-recording z-normalised STFT spectrogram."""
    signal = signal.astype(np.float32)
    signal = signal - signal.mean()
    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak

    win_length = int(config.WINDOW_LENGTH_S * sr)
    hop_length = int(config.WINDOW_STEP_S * sr)

    stft = librosa.stft(signal, n_fft=win_length, hop_length=hop_length,
                         win_length=win_length, window="hann", center=False)
    power_spec = np.abs(stft) ** 2

    n_freq_bins = int(np.ceil(config.FREQUENCY_HIGH * config.WINDOW_LENGTH_S))
    power_spec = power_spec[:n_freq_bins]

    log_spec = np.log(power_spec + 1e-10)
    log_spec = (log_spec - log_spec.mean(axis=-1, keepdims=True)) / (
        log_spec.std(axis=-1, keepdims=True) + 1e-8
    )
    return log_spec.astype(np.float32)


def n_freq_bins():
    return int(np.ceil(config.FREQUENCY_HIGH * config.WINDOW_LENGTH_S))


class PCGDataset(Dataset):
    """Wraps a metadata DataFrame (needs `filepath`, `label` columns) and turns each row into
    a (features, label) pair on the fly."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signal = pad_or_trim(load_and_resample(row["filepath"], config.SAMPLE_RATE),
                             config.SAMPLE_RATE, config.CLIP_SECONDS)
        features = torch.from_numpy(calculate_features(signal)).float()
        label = torch.tensor(row["label"], dtype=torch.long)
        return features, label
