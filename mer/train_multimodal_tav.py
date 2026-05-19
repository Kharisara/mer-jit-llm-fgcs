# mer/train_multimodal_tav.py

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

from .dataset_multimodal_meld_tav import MELDMultimodalTAVDataset
from .model_multimodal_tav import MultimodalTAVClassifier


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)

    # -------------------------
    # Paths
    # -------------------------
    this_dir = os.path.dirname(os.path.abspath(__file__))      # /mer
    project_root = os.path.dirname(this_dir)                   # /mer-jit-llm

    csv_path = os.path.join(
        project_root, "data", "processed", "meld_text_audio_video_arcface.csv"
    )
    output_dir = os.path.join(project_root, "checkpoints_multimodal_tav")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Hyperparameters
    # -------------------------
    text_model_name = "distilbert-base-uncased"
    max_length = 128
    batch_size = 16
    num_epochs = 8             # longer training
    lr = 3e-5
    n_mels = 64
    video_dim = 512            # ArcFace embedding dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # Load CSV + labels
    # -------------------------
    df = pd.read_csv(csv_path)
    unique_labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(label2id)

    print("Labels:", label2id)

    # -------------------------
    # Class weights (balanced CE)
    # -------------------------
    from collections import Counter
    counts = Counter(df["label"])
    print("Label counts:", counts)

    class_weights = torch.zeros(num_labels, dtype=torch.float32)
    total = sum(counts.values())
    for label, idx in label2id.items():
        # simple inverse-frequency style
        class_weights[idx] = total / (len(counts) * counts[label])

    class_weights = class_weights.to(device)
    print("Class weights:", class_weights)

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # -------------------------
    # Datasets: proper MELD split
    # -------------------------
    train_dataset = MELDMultimodalTAVDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        label2id=label2id,
        split_filter=["train"],
        max_length=max_length,
        n_mels=n_mels,
        video_dim=video_dim,
    )

    dev_dataset = MELDMultimodalTAVDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        label2id=label2id,
        split_filter=["dev"],
        max_length=max_length,
        n_mels=n_mels,
        video_dim=video_dim,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------
    # Model
    # -------------------------
    model = MultimodalTAVClassifier(
        num_labels=num_labels,
        text_model_name=text_model_name,
        text_proj_dim=256,
        audio_proj_dim=128,
        video_input_dim=video_dim,   # 512
        video_proj_dim=128,
        hidden_dim=256,
        class_weights=class_weights,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_f1 = 0.0
    best_model_path = os.path.join(output_dir, "best_multimodal_tav.pt")

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logmel = batch["logmel"].to(device)
            video_emb = batch["video_emb"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logmel=logmel,
                video_emb=video_emb,
                labels=labels,
            )
            loss = out["loss"]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(train_loader)
        print(f"Train loss: {avg_loss:.4f}")

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        preds, truths = [], []

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logmel = batch["logmel"].to(device)
                video_emb = batch["video_emb"].to(device)
                labels = batch["labels"].to(device)

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logmel=logmel,
                    video_emb=video_emb,
                )["logits"]

                pred = torch.argmax(logits, dim=-1)
                preds.extend(pred.cpu().numpy())
                truths.extend(labels.cpu().numpy())

        macro_f1 = f1_score(truths, preds, average="macro")
        print(f"Dev macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                },
                best_model_path,
            )
            print(f"New best model saved → {best_model_path}")

    print("\nTraining complete.")
    print("Best Dev Macro F1:", best_f1)


if __name__ == "__main__":
    main()
