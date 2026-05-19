# mer/dataset_audio_meld.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa


class MELDAudioDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        label2id: dict,
        split_filter=None,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        max_duration: float = 3.0,
    ):
        """
        csv_path: data/processed/meld_text_audio.csv
        expects columns: ['utterance_id', 'text', 'label', 'split', 'Dialogue_ID',
                          'Utterance_ID', 'audio_path']
        """
        self.df = pd.read_csv(csv_path)
        if split_filter is not None:
            self.df = self.df[self.df["split"].isin(split_filter)].reset_index(drop=True)

        self.label2id = label2id
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_samples = int(max_duration * sample_rate)

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path):
        wav, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        if len(wav) < self.max_samples:
            pad = self.max_samples - len(wav)
            wav = np.pad(wav, (0, pad), mode="constant")
        else:
            wav = wav[: self.max_samples]
        return wav

    def _wav_to_logmel(self, wav):
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        logmel = librosa.power_to_db(mel + 1e-6)
        return logmel  # [n_mels, T]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        label_str = row["label"]

        wav = self._load_audio(audio_path)
        logmel = self._wav_to_logmel(wav)

        logmel_tensor = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, T]
        label_id = self.label2id[label_str]

        return {
            "logmel": logmel_tensor,
            "labels": torch.tensor(label_id, dtype=torch.long),
        }
