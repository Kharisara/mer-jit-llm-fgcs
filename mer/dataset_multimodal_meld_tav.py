# mer/dataset_multimodal_meld_tav.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa


class MELDMultimodalTAVDataset(Dataset):
    """
    Text + Audio + Video dataset for MELD.
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
        video_dim: int = 512,
    ):
        self.df = pd.read_csv(csv_path)

        if split_filter is not None:
            self.df = self.df[self.df["split"].isin(split_filter)].reset_index(drop=True)

        self.tokenizer = tokenizer
        self.label2id = label2id
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_samples = int(max_duration * sample_rate)

        self.video_dim = video_dim

        # sanity checks
        if "audio_path" not in self.df.columns:
            raise ValueError("Expected 'audio_path' column in CSV")

        if "video_embedding_path" not in self.df.columns:
            raise ValueError("Expected 'video_embedding_path' column in CSV")

    def __len__(self):
        return len(self.df)

    # --------------- AUDIO -----------------

    def _load_audio(self, path):
        try:
            wav, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"[WARN] Failed to load audio: {path} - {e}")
            return np.zeros(self.max_samples, dtype=np.float32)

        if len(wav) < self.max_samples:
            wav = np.pad(wav, (0, self.max_samples - len(wav)))
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
        return librosa.power_to_db(mel + 1e-6)

    # --------------- VIDEO -----------------

    def _load_video_embedding(self, path):
        video_dim = self.video_dim
        zero = torch.zeros(video_dim, dtype=torch.float32)

        if not isinstance(path, str) or not os.path.exists(path):
            print(f"[WARN] Missing video embedding: {path}")
            return zero

        try:
            if path.endswith(".npy"):
                arr = np.load(path)
                if arr.ndim > 1:
                    arr = arr.reshape(-1)
                emb = torch.tensor(arr, dtype=torch.float32)
            else:
                obj = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(obj, dict):
                    for k in ["embedding", "feat", "video_emb"]:
                        if k in obj:
                            obj = obj[k]
                            break
                emb = obj.float()

            if emb.ndim != 1 or emb.shape[0] != video_dim:
                print(f"[WARN] Wrong video dim for {path}, using zeros.")
                return zero

            return emb
        except Exception as e:
            print(f"[WARN] Failed to load video emb: {path} - {e}")
            return zero

    # --------------- MAIN -----------------

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # TEXT
        enc = self.tokenizer(
            str(row[self.text_col]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # AUDIO
        wav = self._load_audio(row["audio_path"])
        logmel = self._wav_to_logmel(wav)
        logmel_tensor = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0)

        # VIDEO
        video_emb_tensor = self._load_video_embedding(row["video_embedding_path"])

        # LABEL
        label_id = self.label2id[row[self.label_col]]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "logmel": logmel_tensor,
            "video_emb": video_emb_tensor,
            "labels": torch.tensor(label_id, dtype=torch.long),
        }
