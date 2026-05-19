"""
mer/train_jitai_imitation.py

Behavioral cloning (supervised imitation) for JITAI:
- Trains an MLP classifier: state -> action (0/1)
- Uses MELDJITAITransitionDataset to load transitions (uses logged synthetic actions)
- Saves model to models/jitai_policy_bc.pt

Run (from project root):
python -m mer.train_jitai_imitation --csv data/processed/meld_text_audio_video_arcface_states.csv --out models/jitai_policy_bc.pt --epochs 8 --batch_size 128 --lr 1e-4 --device cpu
"""
import os
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from mer.env_meld_jitai import MELDJITAITransitionDataset

class BCPolicy(nn.Module):
    """
    Simple feedforward classifier for imitation learning.
    Takes a 512-dim state vector and predicts action {0,1}.
    """

    def __init__(self, input_dim=512, hidden_dim=256, num_actions=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        """
        x: [batch, input_dim]
        returns logits: [batch, num_actions]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits

def collate_batch(batch):
    states = torch.stack([b["state"].float() for b in batch], dim=0)
    # dataset returns action as torch.tensor already
    actions = torch.stack([b["action"].long() if hasattr(b["action"], "long") else torch.tensor(int(b["action"]), dtype=torch.long) for b in batch]).squeeze()
    return states, actions

def accuracy(preds, labels):
    return (preds.argmax(dim=-1) == labels).float().mean().item()

def train(args):
    ds = MELDJITAITransitionDataset(csv_path=args.csv, splits=("train",), device=args.device)
    n = len(ds)
    if n == 0:
        raise RuntimeError("Dataset has zero transitions.")
    # split train/val
    val_size = min(int(0.1 * n), 1000)
    train_size = n - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

    device = torch.device(args.device)
    model = BCPolicy(input_dim=ds.state_dim, hidden_dim=args.hidden).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    # compute class weights manually
    # penalize action=1 dominance
    class_weights = torch.tensor([1.0, 10.0])  # [no_intervene, intervene]
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))


    print("Train size:", train_size, "Val size:", val_size, "State dim:", ds.state_dim)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for states, actions in train_loader:
            states = states.to(device)
            actions = actions.to(device)

            logits = model(states)
            loss = loss_fn(logits, actions)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += float(loss.item())
            train_batches += 1

        avg_train_loss = train_loss / max(1, train_batches)

        # eval
        model.eval()
        val_acc = 0.0
        val_batches = 0
        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(device)
                actions = actions.to(device)
                logits = model(states)
                val_acc += accuracy(logits.cpu(), actions.cpu())
                val_batches += 1
        val_acc = val_acc / max(1, val_batches)

        # quick train accuracy sample
        train_acc = 0.0
        train_batches = 0
        with torch.no_grad():
            for i, (states, actions) in enumerate(train_loader):
                if i >= 5:
                    break
                states = states.to(device)
                actions = actions.to(device)
                logits = model(states)
                train_acc += accuracy(logits.cpu(), actions.cpu())
                train_batches += 1
        train_acc = train_acc / max(1, train_batches)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={avg_train_loss:.6f}  train_acc~{train_acc:.3f}  val_acc={val_acc:.3f}")

        # save checkpoint
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save(model.state_dict(), args.out)

    # final distribution under model's greedy policy on full dataset
    model.eval()
    all_preds = []
    with torch.no_grad():
        loader_full = DataLoader(ds, batch_size=512, collate_fn=collate_batch)
        for states, actions in loader_full:
            states = states.to(device)
            logits = model(states)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
    dist = Counter(all_preds)
    print("Final greedy action distribution (BC model):", dict(dist))
    print("Saved BC model ->", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/processed/meld_text_audio_video_arcface_states.csv")
    parser.add_argument("--out", type=str, default="models/jitai_policy_bc.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    train(args)
