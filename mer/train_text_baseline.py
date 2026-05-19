# mer/train_text_baseline.py

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from .dataset_text import TextEmotionDataset
from .model_text_only import TextEmotionClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)

    # ---------- CONFIG ----------
    csv_path = "data/processed/meld_text_only.csv"  # change if needed
    text_col = "text"
    label_col = "label"
    model_name = "distilbert-base-uncased"
    max_length = 128
    batch_size = 16
    num_epochs = 3
    lr = 2e-5
    output_dir = "checkpoints_text_only"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- LOAD DATA ----------
    df = pd.read_csv(csv_path)

    # Build label mapping
    unique_labels = sorted(df[label_col].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(label2id)
    print("Labels:", label2id)

    # ---------- TRAIN/VAL SPLIT ----------
    # ---------- TRAIN/VAL SPLIT USING MELD SPLITS ----------
    # Use train+dev for training/validation, keep test for later if needed
    train_dev_df = df[df["split"].isin(["train", "dev"])].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)  # not used in this script yet

    # Stratified split on train+dev
    train_df, val_df = train_test_split(
        train_dev_df,
        test_size=0.2,
        stratify=train_dev_df[label_col],
        random_state=42,
    )


    # ---------- TOKENIZER ----------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ---------- DATASETS & LOADERS ----------
    train_dataset = TextEmotionDataset(
        df=train_df,
        tokenizer=tokenizer,
        label2id=label2id,
        text_col=text_col,
        label_col=label_col,
        max_length=max_length,
    )

    val_dataset = TextEmotionDataset(
        df=val_df,
        tokenizer=tokenizer,
        label2id=label2id,
        text_col=text_col,
        label_col=label_col,
        max_length=max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---------- MODEL ----------
    model = TextEmotionClassifier(num_labels=num_labels, proj_dim=256)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_f1 = 0.0
    best_model_path = os.path.join(output_dir, "text_only_best.pt")

    # ---------- TRAIN LOOP ----------
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch} - Avg train loss: {avg_train_loss:.4f}")

        # ---------- VALIDATION ----------
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"Validation macro F1: {macro_f1:.4f}")

        # Save best model
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "config": {
                        "model_name": model_name,
                        "max_length": max_length,
                        "num_labels": num_labels,
                    },
                },
                best_model_path,
            )
            print(f"New best model saved at {best_model_path}")

    # ---------- FINAL REPORT ----------
    print("\nBest validation macro F1:", best_val_f1)
    print("\nDetailed classification report on validation set:")

    labels = list(range(num_labels))  # [0, 1, 2, 3] for angry/happy/neutral/sad

    print(
        classification_report(
           all_labels,
           all_preds,
           labels=labels,
           target_names=[id2label[i] for i in labels],
           zero_division=0,  # avoid warnings when a class has no samples
        )
    )



if __name__ == "__main__":
    main()
