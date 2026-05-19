# mer/train_audio_baseline.py

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, f1_score

from dataset_audio_meld import MELDAudioDataset
from model_audio_only import AudioEmotionClassifier



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)

    # figure out paths relative to the project root
    this_dir = os.path.dirname(os.path.abspath(__file__))      # D:\mer-jit-llm\mer
    project_root = os.path.dirname(this_dir)                   # D:\mer-jit-llm

    csv_path = os.path.join(project_root, "data", "processed", "meld_text_audio.csv")
    batch_size = 16
    num_epochs = 3
    lr = 1e-4
    n_mels = 64
    output_dir = os.path.join(project_root, "checkpoints_audio_only")
    os.makedirs(output_dir, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(csv_path)
    unique_labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(label2id)
    print("Labels:", label2id)

    # dataset using train+dev splits
    base_dataset = MELDAudioDataset(
        csv_path=csv_path,
        label2id=label2id,
        split_filter=["train", "dev"],
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

    model = AudioEmotionClassifier(num_labels=num_labels, proj_dim=128)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_f1 = 0.0
    best_model_path = os.path.join(output_dir, "audio_only_best.pt")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            logmel = batch["logmel"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(logmel=logmel, labels=labels)
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
                logmel = batch["logmel"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(logmel=logmel)
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
                },
                best_model_path,
            )
            print(f"New best audio model saved at {best_model_path}")

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
