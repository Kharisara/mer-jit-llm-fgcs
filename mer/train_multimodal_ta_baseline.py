# mer/train_multimodal_ta_baseline.py

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer

from dataset_multimodal_meld_ta import MELDMultimodalTADataset
from model_multimodal_ta import MultimodalTAClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)

    # ---- paths relative to project root ----
    this_dir = os.path.dirname(os.path.abspath(__file__))      # D:\mer-jit-llm\mer
    project_root = os.path.dirname(this_dir)                   # D:\mer-jit-llm

    csv_path = os.path.join(project_root, "data", "processed", "meld_text_audio.csv")
    output_dir = os.path.join(project_root, "checkpoints_multimodal_ta")
    os.makedirs(output_dir, exist_ok=True)

    text_model_name = "distilbert-base-uncased"
    max_length = 128
    batch_size = 8           # start smaller; can increase later
    num_epochs = 3
    lr = 2e-5
    n_mels = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- load CSV & label mapping ----
    df = pd.read_csv(csv_path)
    unique_labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(label2id)
    print("Labels:", label2id)

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # ---- dataset (train+dev) ----
    base_dataset = MELDMultimodalTADataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        label2id=label2id,
        split_filter=["train", "dev"],
        max_length=max_length,
        n_mels=n_mels,
    )

    indices = list(range(len(base_dataset)))
    split_idx = int(0.9 * len(indices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = Subset(base_dataset, train_indices)
    val_dataset = Subset(base_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---- model ----
    model = MultimodalTAClassifier(
        num_labels=num_labels,
        text_model_name=text_model_name,
        text_proj_dim=256,
        audio_proj_dim=128,
        lmf_rank=8,
        fused_dim=128,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_f1 = 0.0
    best_model_path = os.path.join(output_dir, "multimodal_ta_best.pt")

    # ---- training loop ----
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logmel = batch["logmel"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logmel=logmel,
                labels=labels,
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch} - Avg train loss: {avg_train_loss:.4f}")

        # ---- validation ----
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logmel = batch["logmel"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logmel=logmel,
                )
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"Validation macro F1: {macro_f1:.4f}")

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "config": {
                        "text_model_name": text_model_name,
                        "max_length": max_length,
                        "num_labels": num_labels,
                        "n_mels": n_mels,
                    },
                },
                best_model_path,
            )
            print(f"New best multimodal TA model saved at {best_model_path}")

    print("\nBest validation macro F1:", best_val_f1)
    labels_idx = list(range(num_labels))
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=labels_idx,
            target_names=[id2label[i] for i in labels_idx],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
