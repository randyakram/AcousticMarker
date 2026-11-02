"""
The CRNN classifier: two Conv2D blocks (learn local time-frequency patterns, and give Grad-CAM
genuine spatial feature maps to explain from) followed by a bidirectional GRU (models how those
patterns evolve over the recording), mean-pooled into a single MI/Normal decision.

Kept as the single model type for this discovery pipeline -- the earlier classifier notebook's
RNN-baseline / attention-pooling variants were for a separate architecture-comparison exercise,
not part of the marker-discovery pipeline.
"""

import torch
import torch.nn as nn


class MIClassifier(nn.Module):
    def __init__(self, n_freq_bins, rnn_hidden=64, rnn_layers=2, dropout=0.3, n_classes=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # halves both freq and time resolution
        )
        self.conv2 = nn.Sequential(          # <- Grad-CAM reads this layer's output
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.freq_pool = nn.MaxPool2d((2, 1))  # halves freq only, keeps time resolution for the GRU

        conv_out_freq = max(n_freq_bins // 2 // 2, 1)
        gru_input_size = 32 * conv_out_freq

        self.rnn = nn.GRU(
            input_size=gru_input_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: [B, freq_bins, time]
        x = x.unsqueeze(1)              # [B, 1, freq, time]
        x = self.conv1(x)               # [B, 16, freq/2, time/2]
        x = self.conv2(x)               # [B, 32, freq/2, time/2]
        x = self.freq_pool(x)           # [B, 32, freq/4, time/2]
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)   # [B, time', channels*freq'] for the GRU
        out, _ = self.rnn(x)
        pooled = out.mean(dim=1)          # mean-pool over time -> [B, rnn_hidden*2]
        return self.head(pooled)          # [B, n_classes]


def make_model(n_freq_bins, device, rnn_hidden=64, rnn_layers=2, dropout=0.3, n_classes=2):
    return MIClassifier(n_freq_bins=n_freq_bins, rnn_hidden=rnn_hidden, rnn_layers=rnn_layers,
                        dropout=dropout, n_classes=n_classes).to(device)
