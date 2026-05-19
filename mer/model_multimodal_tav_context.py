# mer/model_multimodal_tav_context.py

import torch
import torch.nn as nn

from .text_encoder import TextEncoder
from .audio_encoder import AudioEncoder


class ContextualMultimodalTAVClassifier(nn.Module):
    """
    Text + Audio + Video with dialogue-level context (LSTM).

    Inputs (batch of dialogues):
        input_ids:      [B, T, L]
        attention_mask: [B, T, L]
        logmel:         [B, T, 1, n_mels, Tspec]
        video_emb:      [B, T, video_input_dim]
        labels:         [B, T] (optional)

    Outputs:
        {
            "logits": [B, T, num_labels],
            "loss":   scalar (if labels provided),
            "states": [B, T, ctx_out_dim] (if return_states=True)
        }
    """

    def __init__(
        self,
        num_labels: int,
        text_model_name: str = "distilbert-base-uncased",
        text_proj_dim: int = 256,
        audio_proj_dim: int = 128,
        video_input_dim: int = 512,
        video_proj_dim: int = 128,
        hidden_dim: int = 256,
        bidirectional: bool = True,
        class_weights=None,
    ):
        super().__init__()

        # --- unimodal encoders ---
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            proj_dim=text_proj_dim,
        )
        self.audio_encoder = AudioEncoder(
            proj_dim=audio_proj_dim,
        )
        self.video_proj = nn.Linear(video_input_dim, video_proj_dim)

        # fused_dim = concat(text, audio, video)
        fused_dim = text_proj_dim + audio_proj_dim + video_proj_dim  # 256+128+128=512

        # --- dialogue-level LSTM ---
        # IMPORTANT: name and shapes must match the checkpoint:
        #   context_rnn.weight_ih_l0: [1024, 512] => LSTM, hidden_size=256
        self.context_rnn = nn.LSTM(
            input_size=fused_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )

        ctx_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(ctx_out_dim, num_labels)

        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        logmel,
        video_emb,
        labels=None,
        return_states: bool = False,
    ):
        """
        input_ids:      [B, T, L]
        attention_mask: [B, T, L]
        logmel:         [B, T, 1, n_mels, Tspec]
        video_emb:      [B, T, video_input_dim]
        labels:         [B, T] (optional)
        return_states:  if True, also return contextual embeddings under out["states"]
        """
        B, T, L = input_ids.shape

        # flatten (B, T) -> (B*T) for encoders
        input_ids_flat = input_ids.view(B * T, L)
        attention_mask_flat = attention_mask.view(B * T, L)
        logmel_flat = logmel.view(B * T, *logmel.shape[2:])
        video_emb_flat = video_emb.view(B * T, -1)

        # text: [B*T, text_proj_dim]
        h_text = self.text_encoder(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat,
        )

        # audio: [B*T, audio_proj_dim]
        h_audio = self.audio_encoder(logmel_flat)

        # video: [B*T, video_proj_dim]
        h_video = self.video_proj(video_emb_flat)

        # fuse: [B*T, fused_dim] -> [B, T, fused_dim]
        fused = torch.cat([h_text, h_audio, h_video], dim=-1)  # [B*T, fused_dim]
        fused_seq = fused.view(B, T, -1)                       # [B, T, fused_dim]

        # LSTM: [B, T, fused_dim] -> [B, T, ctx_out_dim]
        ctx_out, _ = self.context_rnn(fused_seq)

        ctx_out = self.dropout(ctx_out)
        logits = self.classifier(ctx_out)                      # [B, T, num_labels]

        out = {"logits": logits}

        if labels is not None:
            loss = self.loss_fn(
                logits.view(B * T, -1),
                labels.view(B * T),
            )
            out["loss"] = loss

        if return_states:
            # contextual states that we will save for JITAI
            out["states"] = ctx_out

        return out
