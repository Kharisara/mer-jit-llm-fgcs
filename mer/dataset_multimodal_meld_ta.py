# mer/dataset_multimodal_meld_ta.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa


class MELDMultimodalTADataset(Dataset):
    """
    Multimodal (Text + Audio) dataset for MELD.
    Uses meld_text_audio.csv with columns:
      ['utterance_id', 'text', 'label', 'split',
       'Dialogue_ID', 'Utterance_ID', 'audio_path']
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        label2id: dict,
        split_filter=None,
        text_col: str = "text",
        label_col: str = "label",
        max_length: int = 128,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        max_duration: float = 3.0,
    ):
        self.df = pd.read_csv(csv_path)
        if split_filter is not None:
            self.df = self.df[self.df["split"].isin(split_filter)].reset_index(drop=True)

        self.tokenizer = tokenizer
        self.label2id = label2id
        self.text_col = text_col
        self.label_col = label_col

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_samples = int(max_duration * sample_rate)

        # audio paths
        if "audio_path" not in self.df.columns:
            raise ValueError("Expected 'audio_path' column in CSV")

        self.audio_paths = self.df["audio_path"].tolist()

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

        text = str(row[self.text_col])
        label_str = row[self.label_col]
        audio_path = row["audio_path"]

        # --- text ---
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)       # [L]
        attention_mask = enc["attention_mask"].squeeze(0)

        # --- audio ---
        wav = self._load_audio(audio_path)
        logmel = self._wav_to_logmel(wav)            # [n_mels, T]
        logmel_tensor = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, T]

        label_id = self.label2id[label_str]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "logmel": logmel_tensor,
            "labels": torch.tensor(label_id, dtype=torch.long),
        }
