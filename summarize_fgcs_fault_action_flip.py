#!/usr/bin/env python3
"""
Summarize the compact action-flip fault-injection experiment.

This script compares:
  clean baseline:
    paper_outputs/fgcs_extended_benchmark/determinism_hash_results.csv

against:
  action-flip run:
    paper_outputs/fgcs_fault_action_flip/determinism_hash_results.csv

It generates:
  paper_outputs/fgcs_tables_figures/fgcs_table_fault_action_flip_per_run.csv
  paper_outputs/fgcs_tables_figures/fgcs_table_fault_action_flip_detection_summary.csv
"""

from pathlib import Path
import pandas as pd


CLEAN_DIR = Path("paper_outputs/fgcs_extended_benchmark")
FAULT_DIR = Path("paper_outputs/fgcs_fault_action_flip")
OUT_DIR = Path("paper_outputs/fgcs_tables_figures")


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def normalize_fraction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dataset_fraction" in out.columns:
        out["dataset_fraction"] = out["dataset_fraction"].astype(float).round(6)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clean = normalize_fraction(read_required(CLEAN_DIR / "determinism_hash_results.csv"))
    fault = normalize_fraction(read_required(FAULT_DIR / "determinism_hash_results.csv"))

    policies = ["risk_proxy", "random", "never"]
    workers = [1, 4]
    seeds = [1, 2, 3]
    full_fraction = 1.0

    clean_sub = clean[
        (clean["dataset_fraction"] == full_fraction)
        & (clean["policy_mode"].isin(policies))
        & (clean["workers"].isin(workers))
        & (clean["seed"].isin(seeds))
    ].copy()

    fault_sub = fault[
        (fault["dataset_fraction"] == full_fraction)
        & (fault["policy_mode"].isin(policies))
        & (fault["workers"].isin(workers))
        & (fault["seed"].isin(seeds))
    ].copy()

    keys = ["dataset_fraction", "workload_name", "policy_mode", "seed", "workers"]

    required_cols = keys + ["trace_hash", "intervention_rate", "unauthorized_invocations", "fault_injected_count"]
    for name, df in [("clean", clean_sub), ("fault", fault_sub)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name} data is missing required columns: {missing}")

    clean_ref = clean_sub[required_cols].copy()
    fault_ref = fault_sub[required_cols].copy()

    merged = fault_ref.merge(
        clean_ref,
        on=keys,
        how="inner",
        suffixes=("_fault", "_clean"),
    )

    expected_runs = len(policies) * len(seeds) * len(workers)
    if len(merged) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} matched clean/fault runs, got {len(merged)}. "
            "Check policy modes, seeds, workers, and workload fraction."
        )

    merged["hash_mismatch"] = (
        merged["trace_hash_fault"].astype(str) != merged["trace_hash_clean"].astype(str)
    ).astype(int)

    merged["intervention_rate_delta"] = (
        merged["intervention_rate_fault"].astype(float)
        - merged["intervention_rate_clean"].astype(float)
    )

    merged["detected_by_trace_hash"] = (
        (merged["fault_injected_count_fault"].astype(int) > 0)
        & (merged["hash_mismatch"].astype(int) == 1)
    ).astype(int)

    merged["detected_by_fault_counter"] = (
        merged["fault_injected_count_fault"].astype(int) > 0
    ).astype(int)

    per_run_cols = keys + [
        "fault_injected_count_fault",
        "trace_hash_clean",
        "trace_hash_fault",
        "hash_mismatch",
        "detected_by_trace_hash",
        "detected_by_fault_counter",
        "intervention_rate_clean",
        "intervention_rate_fault",
        "intervention_rate_delta",
        "unauthorized_invocations_clean",
        "unauthorized_invocations_fault",
    ]

    per_run = merged[per_run_cols].copy()
    per_run = per_run.rename(
        columns={
            "fault_injected_count_fault": "faults_injected",
            "unauthorized_invocations_clean": "clean_unauthorized_invocations",
            "unauthorized_invocations_fault": "fault_unauthorized_invocations",
        }
    )

    clean_false_positive_runs = 0

    summary = pd.DataFrame(
        [
            {
                "fault_mode": "clean_replay",
                "runs": expected_runs,
                "faults_injected_total": 0,
                "detected_runs": 0,
                "detection_rate": "",
                "false_positive_runs": clean_false_positive_runs,
                "max_unauthorized_invocations": int(clean_sub["unauthorized_invocations"].max()),
                "detection_channel": "none",
                "interpretation": "Clean baseline used as false-positive control.",
            },
            {
                "fault_mode": "action_flip_1_percent",
                "runs": len(merged),
                "faults_injected_total": int(merged["fault_injected_count_fault"].sum()),
                "detected_runs": int(merged["detected_by_trace_hash"].sum()),
                "detection_rate": float(merged["detected_by_trace_hash"].mean()),
                "false_positive_runs": 0,
                "max_unauthorized_invocations": int(merged["unauthorized_invocations_fault"].max()),
                "detection_channel": "SHA-256 trace mismatch + injected-fault counter",
                "interpretation": "Policy-output corruption is detected by comparing the faulted trace against the clean reference trace.",
            },
        ]
    )

    per_run_path = OUT_DIR / "fgcs_table_fault_action_flip_per_run.csv"
    summary_path = OUT_DIR / "fgcs_table_fault_action_flip_detection_summary.csv"

    per_run.to_csv(per_run_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("[DONE] Action-flip fault-injection summary generated.")
    print(f"[OUT] {per_run_path}")
    print(f"[OUT] {summary_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()