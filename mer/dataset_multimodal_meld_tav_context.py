# mer/dataset_multimodal_meld_tav_context.py

import torch
from torch.utils.data import Dataset
from .dataset_multimodal_meld_tav import MELDMultimodalTAVDataset


class MELDDialogueTAVDataset(Dataset):
    """
    Dialogue-level Text+Audio+Video dataset for MELD.

    Returns:
        input_ids:   [T, L]
        attention_mask: [T, L]
        logmel:      [T, 1, n_mels, Tspec]
        video_emb:   [T, video_dim]
        labels:      [T]
        length:      int
        dialogue_id: int
        utterance_ids: list of length T   (string / int from CSV)
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        label2id: dict,
        split: str = "train",
        max_length: int = 128,
        n_mels: int = 64,
        video_dim: int = 512,
    ):
        # underlying utterance-level dataset
        self.utter_dataset = MELDMultimodalTAVDataset(
            csv_path=csv_path,
            tokenizer=tokenizer,
            label2id=label2id,
            split_filter=[split],
            max_length=max_length,
            n_mels=n_mels,
            video_dim=video_dim,
        )

        df = self.utter_dataset.df

        if "Dialogue_ID" not in df.columns or "Utterance_ID" not in df.columns:
            raise ValueError("Expected 'Dialogue_ID' and 'Utterance_ID' in CSV")
        if "utterance_id" not in df.columns:
            raise ValueError("Expected 'utterance_id' column in CSV")

        self.dialogue_ids = sorted(df["Dialogue_ID"].unique().tolist())
        self.dialogues = []  # list of lists of row indices

        for d_id in self.dialogue_ids:
            df_d = df[df["Dialogue_ID"] == d_id].sort_values("Utterance_ID")
            utt_indices = df_d.index.tolist()
            self.dialogues.append(utt_indices)

        self.df = df  # keep for accessing utterance_id

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        utt_indices = self.dialogues[idx]
        T = len(utt_indices)

        input_ids_list = []
        attention_mask_list = []
        logmel_list = []
        video_emb_list = []
        labels_list = []
        utterance_ids = []

        for u_idx in utt_indices:
            item = self.utter_dataset[u_idx]
            input_ids_list.append(item["input_ids"])
            attention_mask_list.append(item["attention_mask"])
            logmel_list.append(item["logmel"])
            video_emb_list.append(item["video_emb"])
            labels_list.append(item["labels"])

            # get utterance_id from df
            utterance_ids.append(self.df.loc[u_idx, "utterance_id"])

        input_ids = torch.stack(input_ids_list, dim=0)          # [T, L]
        attention_mask = torch.stack(attention_mask_list, dim=0)
        logmel = torch.stack(logmel_list, dim=0)                # [T, 1, n_mels, Tspec]
        video_emb = torch.stack(video_emb_list, dim=0)          # [T, video_dim]
        labels = torch.stack(labels_list, dim=0)                # [T]

        dialogue_id = self.dialogue_ids[idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "logmel": logmel,
            "video_emb": video_emb,
            "labels": labels,
            "length": T,
            "dialogue_id": dialogue_id,
            "utterance_ids": utterance_ids,
        }
