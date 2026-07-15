from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.net(x)


model_path = Path("checkpoints/jitai_policy_bc.pt")
input_csv = Path("paper_outputs/replay_input_clean.csv")

df = pd.read_csv(input_csv)

model = BCPolicy()
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

preds = []

with torch.no_grad():
    for _, row in df.head(20).iterrows():
        state_path = Path(row["state_path"])
        x = np.load(state_path, allow_pickle=True).astype("float32").reshape(-1)

        if x.shape[0] != 512:
            raise ValueError(f"Expected 512 features, got {x.shape[0]} from {state_path}")

        xt = torch.from_numpy(x).unsqueeze(0)
        logits = model(xt)
        action = int(torch.argmax(logits, dim=1).item())
        preds.append(action)

print("First 20 live BC actions:", preds)
print("Intervention rate in first 20:", sum(preds) / len(preds))