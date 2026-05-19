# mer/audio_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 64,
        proj_dim: int = 128,
        conv_channels: int = 32,
        gru_hidden: int = 128,
    ):
        super().__init__()

        # Input: [B, 1, n_mels, T]
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # halves both dims
        )

        self.gru_hidden = gru_hidden
        self.gru_input_dim = conv_channels * (n_mels // 2)

        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )

        self.proj = nn.Linear(gru_hidden * 2, proj_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [B, 1, n_mels, T]
        x = self.conv(x)              # [B, C, n_mels2, T2]
        B, C, M, T = x.shape
        x = x.permute(0, 3, 1, 2)     # [B, T, C, M]
        x = x.reshape(B, T, C * M)    # [B, T, C*M]

        out, _ = self.gru(x)          # [B, T, 2H]
        last = out[:, -1, :]          # [B, 2H]

        z = self.proj(last)           # [B, proj_dim]
        z = self.act(z)
        return z
