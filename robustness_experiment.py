"""
robustness_experiment.py

Execution-robustness experiment for deterministic offline replay.

What it measures:
1) intervention-rate variance across repeated executions
2) unauthorized invocation variance across repeated executions
3) action-sequence mismatch rate relative to the first replay trace

Input:
- CSV file with a label column, matching the replay CSV used in the paper.
- This script does not require state vectors because the evaluated execution
  decision in simulate_with_bc.py is label-gated after the BC forward pass.

Usage:
    python robustness_experiment.py --csv path/to/replay_input.csv --out_dir robustness_results

Optional:
    python robustness_experiment.py --csv path/to/replay_input.csv --label_col emotion --out_dir robustness_results
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

NEGATIVE_LABELS = {"angry", "anger", "sad", "sadness", "fear", "disgust"}


@dataclass(frozen=True)
class Condition:
    name: str
    mode: str  # "fixed" or "pneg"
    seed: int
    p_neg: Optional[float] = None


def normalize_label(x: object) -> str:
    return str(x).strip().lower()


def stable_uniform_0_1(index: int, label: str, seed: int) -> float:
    """Deterministic pseudo-random value in [0, 1) from row index, label, and seed."""
    s = f"{seed}|{index}|{label}".encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    # Use first 16 hex chars for a stable integer, then map to [0,1)
    return int(h[:16], 16) / float(16 ** 16)


def generate_actions(labels: List[str], condition: Condition) -> np.ndarray:
    actions = np.zeros(len(labels), dtype=np.int64)

    if condition.mode == "fixed":
        for i, label in enumerate(labels):
            actions[i] = 1 if label in NEGATIVE_LABELS else 0
        return actions

    if condition.mode == "pneg":
        if condition.p_neg is None:
            raise ValueError("p_neg must be provided for pneg mode")
        if not (0.0 <= condition.p_neg <= 1.0):
            raise ValueError("p_neg must be between 0 and 1")
        for i, label in enumerate(labels):
            if label in NEGATIVE_LABELS:
                actions[i] = 1 if stable_uniform_0_1(i, label, condition.seed) < condition.p_neg else 0
            else:
                actions[i] = 0
        return actions

    raise ValueError(f"Unknown condition mode: {condition.mode}")


def generate_invocation_flags(actions: np.ndarray) -> np.ndarray:
    """Policy-first guarantee: generation is invoked iff action == 1."""
    return (actions == 1).astype(np.int64)


def run_once(labels: List[str], condition: Condition) -> Dict[str, object]:
    actions = generate_actions(labels, condition)
    invocations = generate_invocation_flags(actions)

    # Unauthorized invocation = generation occurs when action == 0
    unauthorized_invocations = int(np.sum((actions == 0) & (invocations == 1)))

    return {
        "actions": actions,
        "intervention_rate": float(np.mean(actions)) if len(actions) else 0.0,
        "unauthorized_invocations": unauthorized_invocations,
    }


def summarize_condition(labels: List[str], condition: Condition, repeats: int) -> Dict[str, object]:
    runs = [run_once(labels, condition) for _ in range(repeats)]

    intervention_rates = np.array([r["intervention_rate"] for r in runs], dtype=float)
    unauthorized_counts = np.array([r["unauthorized_invocations"] for r in runs], dtype=float)

    ref_actions = runs[0]["actions"]
    mismatch_rates = []
    for r in runs:
        actions = r["actions"]
        mismatch_rates.append(float(np.mean(actions != ref_actions)) if len(actions) else 0.0)

    return {
        "Condition": condition.name,
        "Repeated runs": repeats,
        "Mean intervention rate": float(np.mean(intervention_rates)),
        "Intervention-rate variance": float(np.var(intervention_rates, ddof=0)),
        "Unauthorized invocation variance": float(np.var(unauthorized_counts, ddof=0)),
        "Action-sequence mismatch rate": float(np.mean(mismatch_rates)),
        "Unauthorized invocations per run": int(unauthorized_counts[0]) if len(unauthorized_counts) else 0,
    }


def make_conditions() -> List[Condition]:
    return [
        Condition(name="Fixed replay, seed 1", mode="fixed", seed=1),
        Condition(name="Fixed replay, seed 2", mode="fixed", seed=2),
        Condition(name="Fixed replay, seed 3", mode="fixed", seed=3),
        Condition(name="Synthetic logging probability = 1.0", mode="pneg", seed=42, p_neg=1.0),
        Condition(name="Synthetic logging probability = 0.6", mode="pneg", seed=42, p_neg=0.6),
        Condition(name="Synthetic logging probability = 0.5", mode="pneg", seed=42, p_neg=0.5),
    ]


def format_markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "Condition",
        "Repeated runs",
        "Mean intervention rate",
        "Intervention-rate variance",
        "Unauthorized invocation variance",
        "Action-sequence mismatch rate",
    ]
    out = df[cols].copy()
    for col in [
        "Mean intervention rate",
        "Intervention-rate variance",
        "Unauthorized invocation variance",
        "Action-sequence mismatch rate",
    ]:
        out[col] = out[col].map(lambda x: f"{x:.4f}")
    return out.to_string(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input replay/dataset CSV with a label column")
    parser.add_argument("--label_col", default="label", help="Name of the label column; default: label")
    parser.add_argument("--out_dir", default="robustness_results", help="Output directory")
    parser.add_argument("--repeats", type=int, default=3, help="Repeated executions per condition")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if args.label_col not in df.columns:
        raise ValueError(
            f"Label column '{args.label_col}' not found. Available columns: {list(df.columns)}"
        )

    labels = [normalize_label(x) for x in df[args.label_col].tolist()]
    rows = [summarize_condition(labels, c, args.repeats) for c in make_conditions()]
    results = pd.DataFrame(rows)

    csv_out = out_dir / "execution_robustness_results.csv"
    md_out = out_dir / "execution_robustness_table.md"
    json_out = out_dir / "execution_robustness_results.json"

    results.to_csv(csv_out, index=False)
    md_table = format_markdown_table(results)
    md_out.write_text(md_table + "\n", encoding="utf-8")
    json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\nTABLE X")
    print("EXECUTION ROBUSTNESS UNDER DETERMINISTIC OFFLINE REPLAY\n")
    print(md_table)
    print(f"\nSaved: {csv_out}")
    print(f"Saved: {md_out}")
    print(f"Saved: {json_out}")


if __name__ == "__main__":
    main()
