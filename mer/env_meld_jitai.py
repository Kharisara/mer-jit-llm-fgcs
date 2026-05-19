# mer/env_meld_jitai.py

import os
from typing import Dict, List, Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------
# Helper: safe .npy/.pt state loader
# ------------------------------------------------------------
def load_state_np(state_path: str, expected_dim: int = 512, device: str = "cpu") -> torch.Tensor:
    if not isinstance(state_path, str) or len(state_path.strip()) == 0:
        return torch.zeros(expected_dim, dtype=torch.float32, device=device)

    try:
        if state_path.endswith(".npy") or state_path.endswith(".npz"):
            arr = np.load(state_path)
        else:
            try:
                arr = np.load(state_path, allow_pickle=True)
            except Exception:
                obj = torch.load(state_path, map_location="cpu")
                if isinstance(obj, np.ndarray):
                    arr = obj
                elif hasattr(obj, "numpy"):
                    arr = obj.numpy()
                elif isinstance(obj, dict):
                    for k in ("state", "embedding", "features", "z"):
                        if k in obj:
                            v = obj[k]
                            arr = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
                            break
                    else:
                        arr = np.asarray(obj)
                else:
                    arr = np.asarray(obj)

        arr = np.asarray(arr).ravel().astype(np.float32)

    except Exception:
        return torch.zeros(expected_dim, dtype=torch.float32, device=device)

    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.size < expected_dim:
        padded = np.zeros(expected_dim, dtype=np.float32)
        padded[:arr.size] = arr
        arr = padded
    elif arr.size > expected_dim:
        arr = arr[:expected_dim]

    return torch.tensor(arr, dtype=torch.float32, device=device)


# ------------------------------------------------------------
# Emotion → valence mapping (MELD)
# ------------------------------------------------------------
LABEL2VALENCE = {
    "angry": -1.0,
    "sad": -1.0,
    "neutral": 0.0,
    "happy": 1.0,
}


def label_to_valence(label: str) -> float:
    return LABEL2VALENCE.get(label, 0.0)


# ------------------------------------------------------------
# Offline JITAI dataset
# ------------------------------------------------------------
class MELDJITAITransitionDataset(Dataset):
    """
    Offline JITAI / contextual bandit dataset.

    reward_mode:
      - "original": r = valence_{t+1} - valence_t
      - "shaped":   small penalty for unnecessary intervention

    negative_prob:
      Probability of intervention when valence <= 0.
      1.0 = deterministic logging (original behaviour)
      <1.0 = stochastic logging (wider action support)
    """

    def __init__(
        self,
        csv_path: str,
        splits: Iterable[str] = ("train",),
        device: str = "cpu",
        reward_mode: str = "original",
        negative_prob: float = 1.0,   # ← sweep parameter
    ):
        super().__init__()

        self.csv_path = csv_path
        self.splits = tuple(splits)
        self.device = torch.device(device)
        self.reward_mode = reward_mode
        self.negative_prob = float(negative_prob)

        df = pd.read_csv(csv_path)

        if "split" in df.columns:
            df = df[df["split"].isin(self.splits)].reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        if "state_path" not in df.columns:
            raise ValueError("CSV must contain 'state_path' column.")

        df = df[df["state_path"].notna()].reset_index(drop=True)

        for col in ["Dialogue_ID", "Utterance_ID", "label"]:
            if col not in df.columns:
                raise ValueError(f"CSV must contain '{col}' column.")

        self.df = df
        self.transitions: List[Dict[str, Any]] = []
        self._build_transitions()

        if len(self.transitions) > 0:
            sample = load_state_np(self.transitions[0]["state_path_t"], device="cpu")
            self.state_dim = int(sample.numel())
        else:
            self.state_dim = 0

        print(
            f"[MELDJITAITransitionDataset] "
            f"Splits={self.splits}  "
            f"Transitions={len(self.transitions)}  "
            f"State dim={self.state_dim}  "
            f"Reward mode={self.reward_mode}  "
            f"negative_prob={self.negative_prob}"
        )

    # --------------------------------------------------------
    def _build_transitions(self):
        for _, df_d in self.df.groupby("Dialogue_ID"):
            df_d = df_d.sort_values("Utterance_ID")
            idxs = df_d.index.tolist()

            for i in range(len(idxs) - 1):
                r_t = self.df.loc[idxs[i]]
                r_n = self.df.loc[idxs[i + 1]]

                if not isinstance(r_t["state_path"], str) or not isinstance(r_n["state_path"], str):
                    continue

                self.transitions.append({
                    "dialogue_id": int(r_t["Dialogue_ID"]),
                    "utt_id_t": int(r_t["Utterance_ID"]),
                    "label_t": str(r_t["label"]),
                    "label_next": str(r_n["label"]),
                    "state_path_t": r_t["state_path"],
                    "state_path_next": r_n["state_path"],
                })

    # --------------------------------------------------------
    # Logging policy (sweep-enabled)
    # --------------------------------------------------------
    def _logging_policy(self, val_t: float) -> int:
        if val_t <= 0.0:
            # stochastic intervention under negative valence
            return 1 if np.random.rand() < self.negative_prob else 0
        else:
            return 0

    # --------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tr = self.transitions[idx]

        val_t = label_to_valence(tr["label_t"])
        val_next = label_to_valence(tr["label_next"])

        reward = float(val_next - val_t)
        action = self._logging_policy(val_t)

        if self.reward_mode == "shaped":
            if val_t >= 0.0 and action == 1:
                reward -= 0.05

        state = load_state_np(tr["state_path_t"], device=str(self.device))
        next_state = load_state_np(tr["state_path_next"], device=str(self.device))

        return {
            "state": state.view(-1),
            "action": torch.tensor(action, dtype=torch.long),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "next_state": next_state.view(-1),
            "dialogue_id": torch.tensor(tr["dialogue_id"], dtype=torch.long),
            "t_index": torch.tensor(tr["utt_id_t"], dtype=torch.long),
            "label_t": tr["label_t"],
            "label_next": tr["label_next"],
        }

    def __len__(self):
        return len(self.transitions)