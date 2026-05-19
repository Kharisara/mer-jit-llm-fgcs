# mer/train_audio_only.py

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import librosa

from .model_audio_only import AudioEmotionClassifier


# -------------------------------------------------
# Utils
# -------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------
# Audio-only MELD dataset
# -------------------------------------------------
class MELDAudioOnlyDataset(Dataset):
    """
    Audio-only MELD dataset.

    Expects CSV with at least:
      - 'split'  (train/dev/test)
      - 'label'
      - 'audio_path'
    We will use: data/processed/meld_text_audio_video_arcface.csv
    """

    def __init__(
        self,
        csv_path: str,
        label2id: dict,
        split: str = "train",
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        max_duration: float = 3.0,
    ):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if "audio_path" not in self.df.columns:
            raise ValueError("Expected 'audio_path' column in CSV")

        self.label2id = label2id

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_samples = int(max_duration * sample_rate)

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path: str):
        try:
            wav, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"[WARN] Failed to load audio {path}: {e}")
            return np.zeros(self.max_samples, dtype=np.float32)

        if len(wav) < self.max_samples:
            wav = np.pad(wav, (0, self.max_samples - len(wav)))
        else:
            wav = wav[: self.max_samples]
        return wav

    def _wav_to_logmel(self, wav: np.ndarray):
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

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        label_str = row["label"]
        label_id = self.label2id[label_str]

        wav = self._load_audio(audio_path)
        logmel = self._wav_to_logmel(wav)  # [n_mels, T]
        logmel_tensor = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, T]

        return {
            "logmel": logmel_tensor,
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


# -------------------------------------------------
# Training script
# -------------------------------------------------
def main():
    set_seed(42)

    this_dir = os.path.dirname(os.path.abspath(__file__))   # .../mer
    project_root = os.path.dirname(this_dir)                # .../mer-jit-llm

    csv_path = os.path.join(
        project_root,
        "data",
        "processed",
        "meld_text_audio_video_arcface.csv",
    )
    ckpt_dir = os.path.join(project_root, "checkpoints_audio")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "best_audio_only.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("CSV:", csv_path)

    # -------------------------
    # Labels
    # -------------------------
    df = pd.read_csv(csv_path)
    # Use train+dev to define label set
    df_train_dev = df[df["split"].isin(["train", "dev"])]
    unique_labels = sorted(df_train_dev["label"].unique())
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(label2id)
    print("Labels:", label2id)

    # -------------------------
    # Datasets & loaders
    # -------------------------
    train_dataset = MELDAudioOnlyDataset(
        csv_path=csv_path,
        label2id=label2id,
        split="train",
    )
    dev_dataset = MELDAudioOnlyDataset(
        csv_path=csv_path,
        label2id=label2id,
        split="dev",
    )
    test_dataset = MELDAudioOnlyDataset(
        csv_path=csv_path,
        label2id=label2id,
        split="test",
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # -------------------------
    # Model & optimizer
    # -------------------------
    model = AudioEmotionClassifier(num_labels=num_labels, proj_dim=128)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 3
    best_dev_f1 = 0.0

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, num_epochs + 1):
        print(f"\n===== EPOCH {epoch}/{num_epochs} =====")
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            logmel = batch["logmel"].to(device)    # [B, 1, n_mels, T]
            labels = batch["labels"].to(device)    # [B]

            optimizer.zero_grad()
            out = model(logmel, labels=labels)
            loss = out["loss"]  # model already computes CE
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"Train loss: {avg_loss:.4f}")

        # ----- Dev evaluation -----
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dev_loader:
                logmel = batch["logmel"].to(device)
                labels = batch["labels"].to(device)

                logits = model(logmel)["logits"]
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        dev_f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"Dev macro F1: {dev_f1:.4f}")

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                },
                best_ckpt_path,
            )
            print(f"New best audio-only model saved → {best_ckpt_path}")

    print("\nBest Dev macro F1 (audio-only):", best_dev_f1)

    # -------------------------
    # Final TEST evaluation
    # -------------------------
    print("\nRunning final TEST evaluation with best checkpoint...")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            logmel = batch["logmel"].to(device)
            labels = batch["labels"].to(device)

            logits = model(logmel)["logits"]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    test_f1 = f1_score(all_labels, all_preds, average="macro")
    print("Test macro F1 (audio-only):", test_f1)


if __name__ == "__main__":
    main()
