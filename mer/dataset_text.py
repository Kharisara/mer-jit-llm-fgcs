# mer/dataset_text.py

import torch
from torch.utils.data import Dataset

class TextEmotionDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        label2id,
        text_col="text",
        label_col="label",
        max_length=128,
    ):
        """
        df: pandas DataFrame with at least text_col and label_col
        tokenizer: HuggingFace tokenizer
        label2id: dict, e.g. {"happy": 0, "sad": 1, ...}
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        label_str = row[self.label_col]

        # Map label string → int
        label_id = self.label2id[label_str]

        # Tokenize
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }
        return item
