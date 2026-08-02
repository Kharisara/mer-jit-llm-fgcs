from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from run_fgcs_extended_benchmark import resolve_existing_path


class BCPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_bc_live_checkpoint_smoke() -> None:
    """Load the archived checkpoint and execute a few portable state paths."""
    model_path = Path("checkpoints/jitai_policy_bc.pt")
    input_csv = Path("paper_outputs/replay_input_clean.csv")
    frame = pd.read_csv(input_csv)

    model = BCPolicy()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    predictions: list[int] = []
    with torch.no_grad():
        for _, row in frame.head(3).iterrows():
            state_path = resolve_existing_path(
                row["state_path"], input_csv=str(input_csv), state_root=None
            )
            if state_path is None:
                raise FileNotFoundError(f"State file not found: {row['state_path']!r}")
            values = np.load(state_path, allow_pickle=True).astype("float32").reshape(-1)
            if values.shape[0] != 512:
                raise ValueError(
                    f"Expected 512 features, got {values.shape[0]} from {state_path}"
                )
            logits = model(torch.from_numpy(values).unsqueeze(0))
            predictions.append(int(torch.argmax(logits, dim=1).item()))

    assert len(predictions) == 3
    assert set(predictions).issubset({0, 1})
