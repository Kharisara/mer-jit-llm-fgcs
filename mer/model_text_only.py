# mer/model_text_only.py

import torch.nn as nn
from .text_encoder import TextEncoder

class TextEmotionClassifier(nn.Module):
    def __init__(self, num_labels: int, proj_dim: int = 256):
        super().__init__()
        self.encoder = TextEncoder(proj_dim=proj_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(proj_dim, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(x)
        logits = self.classifier(x)

        output = {"logits": logits}
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            output["loss"] = loss

        return output
