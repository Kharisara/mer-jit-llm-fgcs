# mer/train_multimodal_tav_context.py

import os
import random
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

from .dataset_multimodal_meld_tav_context import MELDDialogueTAVDataset
from .model_multimodal_tav_context import ContextualMultimodalTAVClassifier


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------
# Training script
# -------------------------------------------------
def main():
    set_seed(42)

    # -------------------------
    # Paths
    # -------------------------
    this_dir = os.path.dirname(os.path.abspath(__file__))  # .../mer
    project_root = os.path.dirname(this_dir)               # .../mer-jit-llm

    csv_path = os.path.join(
        project_root,
        "data",
        "processed",
        "meld_text_audio_video_arcface.csv",  # ArcFace CSV
    )
    output_dir = os.path.join(project_root, "checkpoints_multimodal_tav_context")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Hyperparameters
    # -------------------------
    text_model_name = "distilbert-base-uncased"
    max_length = 128
    batch_size = 1            # 1 dialogue per batch
    num_epochs = 8
    lr = 3e-5
    n_mels = 64
    video_dim = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("CSV:", csv_path)

    # -------------------------
    # Load CSV, labels, class weights
    # -------------------------
    df = pd.read_csv(csv_path)

    # Use train+dev for label set
    df_train_dev = df[df["split"].isin(["train", "dev"])]

    unique_labels = sorted(df_train_dev["label"].unique())
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(label2id)
    print("Labels:", label2id)

    # class weights from train split
    train_df = df[df["split"] == "train"]
    counts = Counter(train_df["label"])
    print("Train label counts:", counts)

    class_weights = torch.zeros(num_labels, dtype=torch.float32)
    total = sum(counts.values())
    for label, idx in label2id.items():
        if counts[label] > 0:
            class_weights[idx] = total / (len(counts) * counts[label])
        else:
            class_weights[idx] = 1.0

    class_weights = class_weights.to(device)
    print("Class weights:", class_weights)

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # -------------------------
    # Dialogue-level datasets (train/dev)
    # -------------------------
    train_dialogue_dataset = MELDDialogueTAVDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        label2id=label2id,
        split="train",
        max_length=max_length,
        n_mels=n_mels,
        video_dim=video_dim,
    )

    dev_dialogue_dataset = MELDDialogueTAVDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        label2id=label2id,
        split="dev",
        max_length=max_length,
        n_mels=n_mels,
        video_dim=video_dim,
    )

    print("Num train dialogues:", len(train_dialogue_dataset))
    print("Num dev dialogues:", len(dev_dialogue_dataset))

    train_loader = DataLoader(
        train_dialogue_dataset,
        batch_size=batch_size,  # 1 dialogue
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dialogue_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # -------------------------
    # Model
    # -------------------------
    model = ContextualMultimodalTAVClassifier(
        num_labels=num_labels,
        text_model_name=text_model_name,
        text_proj_dim=256,
        audio_proj_dim=128,
        video_input_dim=video_dim,
        video_proj_dim=128,
        hidden_dim=256,
        bidirectional=True,
        class_weights=class_weights,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_f1 = 0.0
    best_model_path = os.path.join(output_dir, "best_multimodal_tav_context.pt")

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # batch_size = 1 → shapes [1, T, ...]
            input_ids = batch["input_ids"].to(device)        # [B, T, L]
            attention_mask = batch["attention_mask"].to(device)
            logmel = batch["logmel"].to(device)              # [B, T, 1, n_mels, Tspec]
            video_emb = batch["video_emb"].to(device)        # [B, T, video_dim]
            labels = batch["labels"].to(device)              # [B, T]

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

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"Train loss: {avg_loss:.4f}")

        # -------------------------
        # Dev evaluation
        # -------------------------
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logmel = batch["logmel"].to(device)
                video_emb = batch["video_emb"].to(device)
                labels = batch["labels"].to(device)  # [B, T]

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logmel=logmel,
                    video_emb=video_emb,
                )["logits"]  # [B, T, num_labels]

                preds = torch.argmax(logits, dim=-1)  # [B, T]

                all_preds.extend(preds.view(-1).cpu().numpy().tolist())
                all_labels.extend(labels.view(-1).cpu().numpy().tolist())

        macro_f1 = f1_score(all_labels, all_preds, average="macro")
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
            print(f"New best contextual model saved → {best_model_path}")

    print("\nTraining complete.")
    print("Best Dev Macro F1:", best_f1)


if __name__ == "__main__":
    main()
