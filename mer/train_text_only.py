import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score

from .dataset_multimodal_meld_tav import MELDMultimodalTAVDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TextOnlyClassifier(torch.nn.Module):
    def __init__(self, num_labels, text_model="distilbert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(text_model)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # CLS token
        cls = self.dropout(cls)
        logits = self.fc(cls)
        return logits


def strip_video_audio(ds):
    """Disable audio + video branches."""
    ds._load_audio = lambda x: np.zeros(16000 * 3, dtype=np.float32)
    ds._wav_to_logmel = lambda x: np.zeros((64, 300))
    ds._load_video_embedding = lambda x: torch.zeros(512)
    return ds


def main():
    set_seed(42)

    # -------------------------
    # Paths
    # -------------------------
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)
    csv_path = os.path.join(project_root, "data", "processed", "meld_text_audio_video_arcface.csv")

    # -------------------------
    # Label mapping
    # -------------------------
    df = pd.read_csv(csv_path)
    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    num_labels = len(label2id)

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # -------------------------
    # Datasets
    # -------------------------
    train_ds = MELDMultimodalTAVDataset(csv_path, tokenizer, label2id, split_filter=["train"])
    dev_ds = MELDMultimodalTAVDataset(csv_path, tokenizer, label2id, split_filter=["dev"])
    test_ds = MELDMultimodalTAVDataset(csv_path, tokenizer, label2id, split_filter=["test"])

    # Remove audio + video
    train_ds = strip_video_audio(train_ds)
    dev_ds = strip_video_audio(dev_ds)
    test_ds = strip_video_audio(test_ds)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    # -------------------------
    # Model
    # -------------------------
    model = TextOnlyClassifier(num_labels=num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # -------------------------
    # Checkpoint dir
    # -------------------------
    ckpt_dir = os.path.join(project_root, "checkpoints_text")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_f1 = 0.0

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(3):
        print(f"\n===== EPOCH {epoch+1}/3 =====")

        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = torch.nn.CrossEntropyLoss()(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -------------------------
        # Dev evaluation
        # -------------------------
        model.eval()
        preds, gold = [], []

        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                pred = logits.argmax(-1)

                preds.extend(pred.tolist())
                gold.extend(labels.tolist())

        f1 = f1_score(gold, preds, average="macro")
        print("Dev macro F1:", f1)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))

    print("\nBest Dev F1:", best_f1)

    # -------------------------
    # Test evaluation
    # -------------------------
    print("\nRunning final TEST evaluation...")
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt")))
    model.eval()

    preds, gold = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            preds.extend(logits.argmax(-1).tolist())
            gold.extend(labels.tolist())

    test_f1 = f1_score(gold, preds, average="macro")
    print("\nTest macro F1:", test_f1)


if __name__ == "__main__":
    main()
