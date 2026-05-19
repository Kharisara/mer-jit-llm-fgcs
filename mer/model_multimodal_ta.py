# mer/model_multimodal_ta.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import TextEncoder
from .audio_encoder import AudioEncoder


class LMFTextAudioFusion(nn.Module):
    """
    Low-Rank Multimodal Fusion for Text + Audio
    h_text:  dim_text
    h_audio: dim_audio
    rank r: small (e.g., 4 or 8)
    fused_dim: dimension of fused representation (e.g., 128)
    """

    def __init__(self, dim_text: int, dim_audio: int, rank: int = 8, fused_dim: int = 128):
        super().__init__()
        self.rank = rank
        self.dim_text = dim_text
        self.dim_audio = dim_audio
        self.fused_dim = fused_dim

        # Linear projections to rank-d space (no bias)
        self.U_text = nn.Linear(dim_text, rank, bias=False)
        self.U_audio = nn.Linear(dim_audio, rank, bias=False)

        # Map elementwise product to fused_dim
        self.fuse_out = nn.Linear(rank, fused_dim)
        self.act = nn.ReLU()

    def forward(self, h_text, h_audio):
        # h_text: [B, dim_text]
        # h_audio: [B, dim_audio]
        f_text = self.U_text(h_text)    # [B, r]
        f_audio = self.U_audio(h_audio) # [B, r]

        # multiplicative fusion
        u = f_text * f_audio            # [B, r]
        h = self.fuse_out(u)            # [B, fused_dim]
        h = self.act(h)
        return h


class MultimodalTAClassifier(nn.Module):
    """
    Text + Audio multimodal classifier with LMF fusion.
    """

    def __init__(
        self,
        num_labels: int,
        text_model_name: str = "distilbert-base-uncased",
        text_proj_dim: int = 256,
        audio_proj_dim: int = 128,
        lmf_rank: int = 8,
        fused_dim: int = 128,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(model_name=text_model_name, proj_dim=text_proj_dim)
        self.audio_encoder = AudioEncoder(proj_dim=audio_proj_dim)

        self.fusion = LMFTextAudioFusion(
            dim_text=text_proj_dim,
            dim_audio=audio_proj_dim,
            rank=lmf_rank,
            fused_dim=fused_dim,
        )

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(fused_dim, num_labels)

    def forward(self, input_ids, attention_mask, logmel, labels=None):
        # TextEncoder should accept input_ids, attention_mask and return [B, text_proj_dim]
        h_text = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # AudioEncoder takes [B, 1, n_mels, T] and returns [B, audio_proj_dim]
        h_audio = self.audio_encoder(logmel)

        h_fused = self.fusion(h_text, h_audio)
        h_fused = self.dropout(h_fused)
        logits = self.classifier(h_fused)

        output = {"logits": logits}
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            output["loss"] = loss
        return output
