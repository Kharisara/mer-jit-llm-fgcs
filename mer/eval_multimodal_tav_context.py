# mer/eval_multimodal_tav_context.py

import os
import torch
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset_multimodal_meld_tav_context import MELDDialogueTAVDataset
from .model_multimodal_tav_context import ContextualMultimodalTAVClassifier


def main():
    # -------------------------
    # Paths
    # -------------------------
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)

    csv_path = os.path.join(
        project_root,
        "data",
        "processed",
        "meld_text_audio_video_arcface.csv",
    )
    ckpt_path = os.path.join(
        project_root,
        "checkpoints_multimodal_tav_context",
        "best_multimodal_tav_context.pt",
    )

    print("CSV:", csv_path)
    print("Checkpoint:", ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Eval device:", device)

    # -------------------------
    # Labels (from CSV)
    # -------------------------
    df = pd.read_csv(csv_path)
    unique_labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(label2id)
    print("Labels:", label2id)

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # -------------------------
    # Dialogue-level test dataset
    # -------------------------
    test_dataset = MELDDialogueTAVDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        label2id=label2id,
        split="test",          # <-- use test split
        max_length=128,
        n_mels=64,
        video_dim=512,
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # -------------------------
    # Model
    # -------------------------
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = ContextualMultimodalTAVClassifier(
        num_labels=num_labels,
        text_model_name="distilbert-base-uncased",
        text_proj_dim=256,
        audio_proj_dim=128,
        video_input_dim=512,
        video_proj_dim=128,
        hidden_dim=256,
        bidirectional=True,
        class_weights=None,   # no weights at eval
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # -------------------------
    # Evaluate
    # -------------------------
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # batch_size = 1 → shapes [1, T, ...]
            input_ids = batch["input_ids"].to(device)       # [1, T, L]
            attention_mask = batch["attention_mask"].to(device)
            logmel = batch["logmel"].to(device)             # [1, T, 1, n_mels, Tspec]
            video_emb = batch["video_emb"].to(device)       # [1, T, 512]
            labels = batch["labels"].to(device)             # [1, T]

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logmel=logmel,
                video_emb=video_emb,
            )["logits"]                                     # [1, T, num_labels]

            preds = torch.argmax(logits, dim=-1)            # [1, T]

            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    # -------------------------
    # Report
    # -------------------------
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print("\nTest Macro F1 (context model):", macro_f1)
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=[id2label[i] for i in range(num_labels)],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
