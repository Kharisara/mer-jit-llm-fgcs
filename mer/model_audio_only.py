# mer/model_audio_only.py

import torch
import torch.nn as nn

from .audio_encoder import AudioEncoder  # <-- IMPORTANT: relative import


class AudioEmotionClassifier(nn.Module):
    def __init__(self, num_labels: int, proj_dim: int = 128):
        super().__init__()
        self.encoder = AudioEncoder(proj_dim=proj_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(proj_dim, num_labels)

    def forward(self, logmel, labels=None):
        """
        logmel: [B, 1, n_mels, T]
        labels: [B] (optional)
        """
        z = self.encoder(logmel)           # [B, proj_dim]
        z = self.dropout(z)
        logits = self.classifier(z)        # [B, num_labels]

        output = {"logits": logits}
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            output["loss"] = loss
        return output
