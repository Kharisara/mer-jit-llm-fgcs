# mer/eval_multimodal_ta.py

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer

from .dataset_multimodal_meld_ta import MELDMultimodalTADataset
from .model_multimodal_ta import MultimodalTAClassifier


def main():
    # -------------------------
    # Paths
    # -------------------------
    this_dir = os.path.dirname(os.path.abspath(__file__))   # .../mer
    project_root = os.path.dirname(this_dir)                # .../mer-jit-llm

    csv_path = os.path.join(project_root, "data", "processed", "meld_text_audio.csv")
    ckpt_path = os.path.join(project_root, "checkpoints_multimodal_ta", "multimodal_ta_best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Eval device:", device)
    print("CSV:", csv_path)
    print("Checkpoint:", ckpt_path)

    # -------------------------
    # Load checkpoint
    # -------------------------
    checkpoint = torch.load(ckpt_path, map_location=device)
    label2id = checkpoint["label2id"]
    id2label = checkpoint["id2label"]
    config = checkpoint.get("config", {})
    num_labels = len(label2id)

    text_model_name = config.get("text_model_name", "distilbert-base-uncased")
    max_length = config.get("max_length", 128)
    n_mels = config.get("n_mels", 64)

    print("Labels:", label2id)

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # -------------------------
    # Dataset
    # -------------------------
    test_dataset = MELDMultimodalTADataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        label2id=label2id,
        split_filter=["test"],
        max_length=max_length,
        n_mels=n_mels,
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # -------------------------
    # Model
    # -------------------------
    model = MultimodalTAClassifier(
        num_labels=num_labels,
        text_model_name=text_model_name,
        text_proj_dim=256,
        audio_proj_dim=128,
        lmf_rank=8,
        fused_dim=128,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # -------------------------
    # Evaluation
    # -------------------------
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logmel = batch["logmel"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logmel=logmel,
            )["logits"]

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print("\nTest Macro F1 (TA):", macro_f1)

    print(classification_report(
        all_labels,
        all_preds,
        labels=list(range(num_labels)),
        target_names=[id2label[i] for i in range(num_labels)],
        zero_division=0,
    ))


if __name__ == "__main__":
    main()
