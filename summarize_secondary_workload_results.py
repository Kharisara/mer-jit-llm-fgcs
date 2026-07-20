#!/usr/bin/env python3
"""Summarize the compact MetroPT-3 secondary-workload validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

EXPECTED_CLEAN_RUNS = 72
EXPECTED_FAULT_RUNS_PER_CLASS = 18
DETERMINISTIC_POLICIES = {"rule_gate", "always", "never"}


def require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="paper_outputs/secondary_metropt3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    clean_dir = root / "clean_benchmark"
    tables_dir = root / "tables_figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    det = require(clean_dir / "determinism_hash_results.csv")
    scaling = require(clean_dir / "scaling_and_runtime_results.csv")
    if len(det) != EXPECTED_CLEAN_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_CLEAN_RUNS} clean conditions, found {len(det)}")
    if len(scaling) != EXPECTED_CLEAN_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_CLEAN_RUNS} scaling rows, found {len(scaling)}")

    det["hash_match"] = pd.to_numeric(det["hash_match"], errors="coerce").fillna(0).astype(int)
    det["unauthorized_invocations"] = pd.to_numeric(
        det["unauthorized_invocations"], errors="coerce"
    ).fillna(0).astype(int)

    rows: List[Dict[str, object]] = []
    for policy, group in det.groupby("policy_mode"):
        full = scaling[(scaling["policy_mode"] == policy) & (scaling["dataset_fraction"].astype(float) == 1.0)]
        rows.append(
            {
                "policy_mode": policy,
                "conditions": int(len(group)),
                "unique_hashes": int(group["trace_hash"].astype(str).nunique()),
                "all_worker_matches": bool((group["hash_match"] == 1).all()),
                "max_unauthorized_invocations": int(group["unauthorized_invocations"].max()),
                "full_workload_mean_intervention_rate": float(full["intervention_rate"].astype(float).mean()),
            }
        )

    policy_summary = pd.DataFrame(rows).sort_values("policy_mode").reset_index(drop=True)
    policy_summary.to_csv(tables_dir / "secondary_policy_determinism_summary.csv", index=False)

    for _, row in policy_summary.iterrows():
        policy = str(row["policy_mode"])
        expected_hashes = 3 if policy in DETERMINISTIC_POLICIES else 9
        if int(row["unique_hashes"]) != expected_hashes:
            raise RuntimeError(
                f"Unexpected unique hash count for {policy}: expected {expected_hashes}, "
                f"found {int(row['unique_hashes'])}"
            )
        if not bool(row["all_worker_matches"]):
            raise RuntimeError(f"Worker-level hash mismatch detected for {policy}")
        if int(row["max_unauthorized_invocations"]) != 0:
            raise RuntimeError(f"Clean authorization-execution contradiction detected for {policy}")

    combined_path = tables_dir / "fgcs_table_rq7_fault_detection_combined.csv"
    combined = require(combined_path)
    fault_only = combined[combined["fault_mode"].astype(str) != "clean_replay"].copy()
    if len(fault_only) != 5:
        raise RuntimeError(f"Expected five fault classes, found {len(fault_only)}")
    for _, row in fault_only.iterrows():
        if int(row["runs"]) != EXPECTED_FAULT_RUNS_PER_CLASS:
            raise RuntimeError(
                f"Fault class {row['fault_mode']} expected {EXPECTED_FAULT_RUNS_PER_CLASS} runs, "
                f"found {int(row['runs'])}"
            )
        if int(row["detected_runs"]) != EXPECTED_FAULT_RUNS_PER_CLASS:
            raise RuntimeError(
                f"Fault class {row['fault_mode']} was not detected in every run: "
                f"{int(row['detected_runs'])}/{EXPECTED_FAULT_RUNS_PER_CLASS}"
            )

    report = {
        "secondary_workload": "MetroPT-3",
        "clean_conditions": int(len(det)),
        "policies": policy_summary.to_dict(orient="records"),
        "clean_max_unauthorized_invocations": int(det["unauthorized_invocations"].max()),
        "fault_classes": fault_only[
            ["fault_mode", "runs", "faults_or_corruptions_injected_total", "detected_runs", "false_positive_runs"]
        ].to_dict(orient="records"),
        "all_five_fault_classes_fully_detected": True,
    }
    (tables_dir / "secondary_validation_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(policy_summary.to_string(index=False))
    print(f"[OUT] {tables_dir / 'secondary_policy_determinism_summary.csv'}")
    print(f"[OUT] {tables_dir / 'secondary_validation_summary.json'}")


if __name__ == "__main__":
    main()
