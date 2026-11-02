"""
Shared audio loading utilities. Kept minimal and separate from feature extraction (that's
Stage 3's concern) since Stage 2 only needs raw waveform samples, not spectrograms.
"""

import numpy as np
import librosa


def load_and_resample(filepath, sr):
    signal, _ = librosa.load(filepath, sr=sr, mono=True)
    return signal


def pad_or_trim(signal, sr, seconds):
    target_len = int(sr * seconds)
    if len(signal) >= target_len:
        return signal[:target_len]
    return np.pad(signal, (0, target_len - len(signal)))
