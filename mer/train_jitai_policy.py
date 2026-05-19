"""
mer/train_jitai_policy.py

Train a small JITAI policy on offline transitions using 1-step REINFORCE.

Now supports logging-policy sweep via --negative_prob.

Example:
python -m mer.train_jitai_policy \
  --csv data/processed/meld_text_audio_video_arcface_states.csv \
  --out_model models/jitai_p1.pt \
  --epochs 8 \
  --batch_size 128 \
  --lr 1e-4 \
  --device cpu \
  --negative_prob 1.0
"""

import os
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mer.env_meld_jitai import MELDJITAITransitionDataset


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------
# Policy Network
# ------------------------------------------------------------
class MLPPolicy(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

    def act_probs(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


# ------------------------------------------------------------
# Collate function
# ------------------------------------------------------------
def collate_batch(batch):
    states = torch.stack([b["state"].float() for b in batch], dim=0)
    actions = torch.tensor(
        [int(b["action"].item()) if hasattr(b["action"], "item") else int(b["action"]) for b in batch],
        dtype=torch.long,
    )
    rewards = torch.tensor(
        [float(b["reward"].item()) if hasattr(b["reward"], "item") else float(b["reward"]) for b in batch],
        dtype=torch.float32,
    )
    return states, actions, rewards


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train(args):

    ds = MELDJITAITransitionDataset(
        csv_path=args.csv,
        splits=("train",),
        device=args.device,
        negative_prob=args.negative_prob,   # ← SWEEP PARAMETER
    )

    print("Dataset transitions:", len(ds), "state_dim:", ds.state_dim)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        drop_last=False,
        num_workers=0,
    )

    device = torch.device(args.device)

    policy = MLPPolicy(state_dim=ds.state_dim, hidden=args.hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    baseline = 0.0
    baseline_deque = deque(maxlen=1000)

    for epoch in range(1, args.epochs + 1):

        total_loss = 0.0
        total_reward = 0.0
        total_samples = 0
        action_counts = torch.zeros(2, dtype=torch.long)

        policy.train()

        for states, actions, rewards in loader:

            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)

            probs = policy.act_probs(states)
            dist = torch.distributions.Categorical(probs=probs)

            # Moving baseline
            batch_mean_reward = rewards.mean().item()
            baseline_deque.append(batch_mean_reward)
            baseline = float(sum(baseline_deque) / len(baseline_deque))

            logp = dist.log_prob(actions)
            advantage = rewards - baseline

            loss = -(logp * advantage).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * states.size(0)
            total_reward += float(rewards.sum().item())
            total_samples += states.size(0)

            with torch.no_grad():
                greedy_actions = probs.argmax(dim=-1)
                for a in greedy_actions.cpu().tolist():
                    action_counts[a] += 1

        avg_loss = total_loss / (total_samples + 1e-9)
        avg_reward = total_reward / (total_samples + 1e-9)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"avg_loss={avg_loss:.6f} "
            f"avg_reward={avg_reward:.6f} "
            f"baseline={baseline:.4f}"
        )
        print("Greedy action counts:", action_counts.tolist())

        os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
        torch.save(policy.state_dict(), args.out_model)

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------
    policy.eval()
    all_actions = []

    with torch.no_grad():
        loader_eval = DataLoader(ds, batch_size=512, collate_fn=collate_batch)
        for states, _, _ in loader_eval:
            states = states.to(device)
            probs = policy.act_probs(states)
            greedy = probs.argmax(dim=-1).cpu()
            all_actions.extend(greedy.tolist())

    all_actions = np.array(all_actions)

    print("\nFinal greedy action distribution:")
    print({0: int((all_actions == 0).sum()), 1: int((all_actions == 1).sum())})
    print("Saved policy ->", args.out_model)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        default="data/processed/meld_text_audio_video_arcface_states.csv",
    )

    parser.add_argument(
        "--out_model",
        type=str,
        default="models/jitai_policy.pt",
    )

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")

    # Logging-policy sweep parameter
    parser.add_argument(
        "--negative_prob",
        type=float,
        default=1.0,
        help="Probability of intervention when valence <= 0. "
             "1.0 = deterministic logging (original).",
    )

    args = parser.parse_args()

    train(args)