# mer/text_encoder.py

import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        proj_dim: int = 256,
        use_mean_pool: bool = False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for DistilBERT
        self.proj = nn.Linear(hidden_size, proj_dim)
        self.activation = nn.ReLU()
        self.use_mean_pool = use_mean_pool

    def forward(self, input_ids, attention_mask):
        # outputs.last_hidden_state: [batch, seq_len, hidden]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        if self.use_mean_pool:
            # mean over non-masked tokens
            mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            masked_hidden = last_hidden * mask
            sum_hidden = masked_hidden.sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = sum_hidden / lengths
        else:
            # CLS token → for DistilBERT this is index 0
            pooled = last_hidden[:, 0, :]  # [batch, hidden]

        x = self.proj(pooled)       # [batch, proj_dim]
        x = self.activation(x)
        return x  # [batch, proj_dim]
