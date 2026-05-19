# mer/model_multimodal_tav.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import TextEncoder
from .audio_encoder import AudioEncoder


class MultimodalTAVClassifier(nn.Module):
    """
    Text + Audio + Video multimodal classifier.
    - TextEncoder -> text_proj_dim
    - AudioEncoder -> audio_proj_dim
    - Video embedding (precomputed) -> projected to video_proj_dim
    Then concatenated and passed through an MLP classifier.
    """

    def __init__(
        self,
        num_labels: int,
        text_model_name: str = "distilbert-base-uncased",
        text_proj_dim: int = 256,
        audio_proj_dim: int = 128,
        video_input_dim: int = 512,   # dim of saved video embeddings
        video_proj_dim: int = 128,
        hidden_dim: int = 512,        # 🔼 increased from 256
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()

        # Encoders for text and audio
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            proj_dim=text_proj_dim,
        )
        self.audio_encoder = AudioEncoder(
            proj_dim=audio_proj_dim,
        )

        # Video projection + heavier dropout
        self.video_proj = nn.Linear(video_input_dim, video_proj_dim)
        self.video_dropout = nn.Dropout(0.3)   # 🔥 stronger dropout just on video

        fused_dim = text_proj_dim + audio_proj_dim + video_proj_dim

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels),
        )

        self.class_weights = class_weights

    def forward(
        self,
        input_ids,
        attention_mask,
        logmel,
        video_emb,
        labels=None,
    ):
        """
        input_ids: [B, L]
        attention_mask: [B, L]
        logmel: [B, 1, n_mels, T]
        video_emb: [B, video_input_dim]
        labels: [B]
        """

        # Text [B, text_proj_dim]
        h_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Audio [B, audio_proj_dim]
        h_audio = self.audio_encoder(logmel)

        # Video [B, video_proj_dim] + dropout
        h_video = self.video_proj(video_emb)
        h_video = self.video_dropout(h_video)

        # Concatenate
        h_fused = torch.cat([h_text, h_audio, h_video], dim=-1)
        h_fused = self.dropout(h_fused)

        logits = self.classifier(h_fused)

        output = {"logits": logits}
        if labels is not None:
            if self.class_weights is not None:
                loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            output["loss"] = loss

        return output
