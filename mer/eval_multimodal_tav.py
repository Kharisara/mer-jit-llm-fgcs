# mer/eval_multimodal_tav.py

import os
import torch
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset_multimodal_meld_tav import MELDMultimodalTAVDataset
from .model_multimodal_tav import MultimodalTAVClassifier


def main():
    # -------------------------
    # Paths
    # -------------------------
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)

    # USE THE ARCFACE CSV
    csv_path = os.path.join(
        project_root,
        "data",
        "processed",
        "meld_text_audio_video_arcface.csv",
    )
    ckpt_path = os.path.join(
        project_root,
        "checkpoints_multimodal_tav",
        "best_multimodal_tav.pt",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Eval using device:", device)
    print("CSV path:", csv_path)
    print("Checkpoint path:", ckpt_path)

    # -------------------------
    # Labels
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
    # Test Dataset
    # -------------------------
    test_dataset = MELDMultimodalTAVDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        label2id=label2id,
        split_filter=["test"],
        max_length=128,
        n_mels=64,
        video_dim=512,
    )

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # -------------------------
    # Model
    # -------------------------
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = MultimodalTAVClassifier(
        num_labels=num_labels,
        text_model_name="distilbert-base-uncased",
        text_proj_dim=256,     # <-- FIXED (was text_project_dim)
        audio_proj_dim=128,
        video_input_dim=512,
        video_proj_dim=128,
        hidden_dim=256,
        class_weights=None,    # no weights for eval
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # -------------------------
    # Run Evaluation
    # -------------------------
    preds, truths = [], []

    with torch.no_grad():
        for batch in test_loader:
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

    # -------------------------
    # Report
    # -------------------------
    macro_f1 = f1_score(truths, preds, average="macro")
    print("Test Macro F1:", macro_f1)
    print(
        classification_report(
            truths,
            preds,
            target_names=[id2label[i] for i in range(num_labels)],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
