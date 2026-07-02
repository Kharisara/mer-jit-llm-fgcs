#!/usr/bin/env python3
"""
Post-processing trace-corruption experiment for FGCS RQ7.

This script takes clean replay trace files from:
    paper_outputs/fgcs_extended_benchmark/

and deliberately corrupts them by:
    1. flipping 1% of logged action values,
    2. dropping 1% of trace rows,
    3. duplicating 1% of trace rows.

It then recomputes SHA-256 action-trace hashes and compares them against
the clean reference hash recorded by the main benchmark.

Outputs:
    paper_outputs/fgcs_fault_trace_corruption/corrupted_traces/*.csv
    paper_outputs/fgcs_tables_figures/fgcs_table_fault_trace_corruption_per_run.csv
    paper_outputs/fgcs_tables_figures/fgcs_table_fault_trace_corruption_detection_summary.csv
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


CLEAN_DIR = Path("paper_outputs/fgcs_extended_benchmark")
OUT_DIR = Path("paper_outputs/fgcs_fault_trace_corruption")
CORRUPTED_TRACE_DIR = OUT_DIR / "corrupted_traces"
TABLE_DIR = Path("paper_outputs/fgcs_tables_figures")

POLICIES = ["risk_proxy", "random", "never"]
SEEDS = [1, 2, 3]
WORKERS = [1, 4]
FULL_FRACTION = 1.0
CORRUPTION_RATE = 0.01


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def trace_hash(actions) -> str:
    """Same action-sequence hash convention used by the benchmark runner."""
    s = ",".join(str(int(a)) for a in actions)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def deterministic_indices(n: int, rate: float, salt: str) -> np.ndarray:
    """Deterministically select at least one row index for corruption."""
    k = max(1, int(round(n * rate)))
    seed_int = int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_int)
    return np.sort(rng.choice(np.arange(n), size=min(k, n), replace=False))


def clean_trace_path(workload_name: str, policy: str, seed: int, workers: int) -> Path:
    return CLEAN_DIR / f"trace_{workload_name}_policy_{policy}_seed_{seed}_workers_{workers}.csv"


def compute_trace_properties(df: pd.DataFrame) -> Dict[str, object]:
    if "action" not in df.columns:
        raise KeyError("Trace file has no 'action' column.")

    out = df.copy()

    # The clean runner sorts by row_index before saving, but we sort again defensively.
    if "row_index" in out.columns:
        out = out.sort_values("row_index").reset_index(drop=True)

    return {
        "row_count": int(len(out)),
        "trace_hash": trace_hash(out["action"].astype(int).tolist()),
    }


def corrupt_action_values(df: pd.DataFrame, salt: str) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    idx = deterministic_indices(len(out), CORRUPTION_RATE, salt)
    out.loc[idx, "action"] = 1 - out.loc[idx, "action"].astype(int)
    return out, int(len(idx))


def corrupt_drop_rows(df: pd.DataFrame, salt: str) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    idx = deterministic_indices(len(out), CORRUPTION_RATE, salt)
    out = out.drop(index=idx).reset_index(drop=True)
    return out, int(len(idx))


def corrupt_duplicate_rows(df: pd.DataFrame, salt: str) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    idx = deterministic_indices(len(out), CORRUPTION_RATE, salt)
    duplicated = out.loc[idx].copy()
    out = pd.concat([out, duplicated], ignore_index=True)

    # Keep deterministic order. Duplicated row_index values are intentional:
    # they simulate repeated logging of the same replay decision.
    if "row_index" in out.columns:
        out = out.sort_values("row_index").reset_index(drop=True)

    return out, int(len(idx))


def main() -> None:
    ensure_dir(OUT_DIR)
    ensure_dir(CORRUPTED_TRACE_DIR)
    ensure_dir(TABLE_DIR)

    det_path = CLEAN_DIR / "determinism_hash_results.csv"
    scaling_path = CLEAN_DIR / "scaling_and_runtime_results.csv"

    det = read_required(det_path)
    scaling = read_required(scaling_path)

    det["dataset_fraction"] = det["dataset_fraction"].astype(float).round(6)
    scaling["dataset_fraction"] = scaling["dataset_fraction"].astype(float).round(6)

    det_sub = det[
        (det["dataset_fraction"] == FULL_FRACTION)
        & (det["policy_mode"].isin(POLICIES))
        & (det["seed"].isin(SEEDS))
        & (det["workers"].isin(WORKERS))
    ].copy()

    expected_runs = len(POLICIES) * len(SEEDS) * len(WORKERS)
    if len(det_sub) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} clean reference rows, found {len(det_sub)}. "
            "Check policy list, seed list, workers, and dataset_fraction."
        )

    per_run_rows: List[Dict[str, object]] = []
    clean_false_positives = 0

    for _, ref in det_sub.iterrows():
        workload_name = str(ref["workload_name"])
        policy = str(ref["policy_mode"])
        seed = int(ref["seed"])
        workers = int(ref["workers"])
        clean_reference_hash = str(ref["trace_hash"])

        path = clean_trace_path(workload_name, policy, seed, workers)
        clean_df = read_required(path)

        clean_props = compute_trace_properties(clean_df)
        clean_row_count = int(clean_props["row_count"])
        clean_computed_hash = str(clean_props["trace_hash"])

        clean_hash_mismatch = int(clean_computed_hash != clean_reference_hash)
        clean_false_positives += clean_hash_mismatch

        common = {
            "dataset_fraction": FULL_FRACTION,
            "workload_name": workload_name,
            "policy_mode": policy,
            "seed": seed,
            "workers": workers,
            "clean_row_count": clean_row_count,
            "clean_reference_hash": clean_reference_hash,
            "clean_computed_hash": clean_computed_hash,
            "clean_hash_mismatch": clean_hash_mismatch,
        }

        corruption_specs = [
            ("trace_action_corruption_1_percent", corrupt_action_values),
            ("drop_trace_rows_1_percent", corrupt_drop_rows),
            ("duplicate_trace_rows_1_percent", corrupt_duplicate_rows),
        ]

        for fault_mode, fn in corruption_specs:
            salt = f"{fault_mode}|{policy}|{seed}|{workers}|{workload_name}"
            corrupted_df, corruptions = fn(clean_df, salt=salt)
            corrupted_props = compute_trace_properties(corrupted_df)

            corrupted_row_count = int(corrupted_props["row_count"])
            corrupted_hash = str(corrupted_props["trace_hash"])

            row_count_mismatch = int(corrupted_row_count != clean_row_count)
            hash_mismatch = int(corrupted_hash != clean_reference_hash)

            if fault_mode == "trace_action_corruption_1_percent":
                detection_channel = "SHA-256 action-trace mismatch"
            else:
                detection_channel = "row-count mismatch + SHA-256 action-trace mismatch"

            detected = int((row_count_mismatch == 1) or (hash_mismatch == 1))

            corrupted_path = (
                CORRUPTED_TRACE_DIR
                / f"corrupted_{fault_mode}_{workload_name}_policy_{policy}_seed_{seed}_workers_{workers}.csv"
            )
            corrupted_df.to_csv(corrupted_path, index=False)

            per_run_rows.append(
                {
                    **common,
                    "fault_mode": fault_mode,
                    "corruptions_injected": int(corruptions),
                    "corrupted_row_count": corrupted_row_count,
                    "corrupted_hash": corrupted_hash,
                    "row_count_mismatch": row_count_mismatch,
                    "hash_mismatch": hash_mismatch,
                    "detected": detected,
                    "detection_channel": detection_channel,
                    "corrupted_trace_path": str(corrupted_path),
                }
            )

    per_run = pd.DataFrame(per_run_rows)

    summary_rows: List[Dict[str, object]] = [
        {
            "fault_mode": "clean_replay",
            "runs": expected_runs,
            "corruptions_injected_total": 0,
            "detected_runs": 0,
            "detection_rate": "",
            "false_positive_runs": int(clean_false_positives),
            "detection_channel": "none",
            "interpretation": "Clean trace files are used as false-positive controls.",
        }
    ]

    for fault_mode, group in per_run.groupby("fault_mode"):
        summary_rows.append(
            {
                "fault_mode": fault_mode,
                "runs": int(len(group)),
                "corruptions_injected_total": int(group["corruptions_injected"].sum()),
                "detected_runs": int(group["detected"].sum()),
                "detection_rate": float(group["detected"].mean()),
                "false_positive_runs": 0,
                "detection_channel": str(group["detection_channel"].iloc[0]),
                "interpretation": (
                    "Post-hoc trace corruption is detected through action-trace hashing "
                    "and row-count invariants."
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    per_run_path = TABLE_DIR / "fgcs_table_fault_trace_corruption_per_run.csv"
    summary_path = TABLE_DIR / "fgcs_table_fault_trace_corruption_detection_summary.csv"

    per_run.to_csv(per_run_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("[DONE] Trace-corruption fault summary generated.")
    print(f"[OUT] {per_run_path}")
    print(f"[OUT] {summary_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()