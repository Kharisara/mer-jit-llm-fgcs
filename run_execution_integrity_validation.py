#!/usr/bin/env python3
"""Validate independent downstream receipts and record-bound trace integrity.

This targeted experiment is separate from the frozen 528-run timing study. It
executes a compact 18-condition matrix for each cross-log receipt control and
applies five post-execution integrity corruptions to the 18 clean references.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from replaybench.integrity import (
    config_bound_trace_hash,
    record_bound_trace_hash,
    sha256_json,
)
from run_fgcs_extended_benchmark import (
    ensure_dir,
    load_config,
    normalize_label,
    run_replay,
    validate_config,
)

RECEIPT_MODES = [
    "clean",
    "unlogged_downstream_call",
    "false_execution_log",
    "duplicate_downstream_call",
    "mismatched_correlation_id",
]
RECORD_CORRUPTIONS = [
    "row_reordering",
    "replay_id_action_reassignment",
    "authorization_field_corruption",
    "execution_field_corruption",
    "configuration_label_corruption",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ReplayBench-PG execution-integrity validation."
    )
    parser.add_argument(
        "--config",
        default="configs/execution_integrity_validation.yaml",
    )
    return parser.parse_args()


def _condition_token(policy: str, seed: int, workers: int) -> str:
    return f"policy_{policy}_seed_{seed}_workers_{workers}"


def _first_two_indices(frame: pd.DataFrame) -> tuple[int, int]:
    if len(frame) < 2:
        raise ValueError("At least two trace rows are required for corruption tests")
    return 0, 1


def _corrupt_record_trace(
    trace: pd.DataFrame,
    corruption: str,
) -> pd.DataFrame:
    corrupted = trace.copy(deep=True).reset_index(drop=True)
    first, second = _first_two_indices(corrupted)

    if corruption == "row_reordering":
        order = list(range(len(corrupted)))
        order[first], order[second] = order[second], order[first]
        return corrupted.iloc[order].reset_index(drop=True)

    if corruption == "replay_id_action_reassignment":
        first_id = corrupted.loc[first, "replay_point_id"]
        first_correlation = corrupted.loc[first, "correlation_id"]
        corrupted.loc[first, "replay_point_id"] = corrupted.loc[
            second, "replay_point_id"
        ]
        corrupted.loc[first, "correlation_id"] = corrupted.loc[
            second, "correlation_id"
        ]
        corrupted.loc[second, "replay_point_id"] = first_id
        corrupted.loc[second, "correlation_id"] = first_correlation
        return corrupted

    if corruption == "authorization_field_corruption":
        corrupted.loc[first, "authorized_to_generate"] = 1 - int(
            corrupted.loc[first, "authorized_to_generate"]
        )
        return corrupted

    if corruption == "execution_field_corruption":
        corrupted.loc[first, "generation_invoked"] = 1 - int(
            corrupted.loc[first, "generation_invoked"]
        )
        return corrupted

    raise ValueError(f"Unsupported record corruption: {corruption}")


def _detected_receipt_fault(summary: dict[str, Any], mode: str) -> int:
    if mode == "clean":
        return int(
            int(summary["receipt_validation_passed"]) == 1
            and int(summary["receipt_fault_injected_count"]) == 0
        )

    metric_by_mode = {
        "unlogged_downstream_call": "unlogged_downstream_calls",
        "false_execution_log": "missing_receipts",
        "duplicate_downstream_call": "duplicate_downstream_calls",
        "mismatched_correlation_id": "mismatched_correlation_ids",
    }
    metric = metric_by_mode[mode]
    return int(
        int(summary["receipt_fault_injected_count"]) > 0
        and int(summary[metric]) > 0
        and int(summary["receipt_validation_passed"]) == 0
    )


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    validate_config(base_cfg)

    dataset_cfg = base_cfg["dataset"]
    benchmark_cfg = base_cfg["benchmark"]
    policy_cfg = base_cfg.get("policy", {})
    output_dir = Path(
        base_cfg.get("logging", {}).get(
            "output_dir", "paper_outputs/execution_integrity_validation"
        )
    )
    ensure_dir(output_dir)

    full = pd.read_csv(dataset_cfg["input_csv"]).reset_index(drop=True)
    fraction = float(dataset_cfg.get("fractions", [1.0])[0])
    count = max(1, int(len(full) * fraction))
    frame = full.iloc[:count].copy().reset_index(drop=True)

    policies = [str(value) for value in benchmark_cfg["policy_modes"]]
    seeds = [int(value) for value in benchmark_cfg["seeds"]]
    workers_values = [int(value) for value in benchmark_cfg["workers"]]
    negative_labels = {
        normalize_label(value) for value in policy_cfg.get("negative_labels", [])
    }

    expected_conditions = len(policies) * len(seeds) * len(workers_values)
    receipt_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    clean_runs: list[tuple[str, int, int, pd.DataFrame, dict[str, Any]]] = []

    for mode in RECEIPT_MODES:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("execution_receipts", {})["enabled"] = True
        fault_cfg = cfg.setdefault("fault_injection", {})
        fault_cfg["enabled"] = False
        fault_cfg["action_flip_probability"] = 0.0
        fault_cfg["unauthorized_invoke_probability"] = 0.0
        fault_cfg["receipt_fault_mode"] = mode
        fault_cfg["receipt_fault_probability"] = (
            0.0 if mode == "clean" else float(
                cfg.get("execution_integrity", {}).get(
                    "receipt_fault_probability", 0.01
                )
            )
        )
        fault_cfg["receipt_fault_allowed_policy_modes"] = policies

        for policy in policies:
            for seed in seeds:
                for workers in workers_values:
                    token = _condition_token(policy, seed, workers)
                    print(f"[RUN] receipt_mode={mode}, {token}")
                    trace, summary, _ = run_replay(
                        df=frame,
                        cfg=cfg,
                        policy_mode=policy,
                        negative_labels=negative_labels,
                        seed=seed,
                        workers=workers,
                        bc_actions=None,
                    )
                    receipts = trace.attrs.get("execution_receipts")
                    reconciliation = trace.attrs.get("receipt_reconciliation")
                    run_identity = trace.attrs.get("run_identity", {})
                    if not isinstance(receipts, pd.DataFrame):
                        raise RuntimeError("Execution receipts were not produced")
                    if not isinstance(reconciliation, pd.DataFrame):
                        raise RuntimeError("Receipt reconciliation was not produced")

                    mode_dir = output_dir / mode
                    ensure_dir(mode_dir)
                    trace.to_csv(mode_dir / f"trace_{token}.csv", index=False)
                    receipts.to_csv(
                        mode_dir / f"execution_receipts_{token}.csv", index=False
                    )
                    reconciliation.to_csv(
                        mode_dir / f"receipt_reconciliation_{token}.csv",
                        index=False,
                    )
                    with (mode_dir / f"trace_manifest_{token}.json").open(
                        "w", encoding="utf-8"
                    ) as handle:
                        json.dump(
                            {
                                **dict(run_identity),
                                "action_hash": summary["trace_hash"],
                                "record_trace_hash": summary["record_trace_hash"],
                                "config_bound_trace_hash": summary[
                                    "config_bound_trace_hash"
                                ],
                            },
                            handle,
                            indent=2,
                        )

                    receipt_rows.append(
                        {
                            "fault_mode": mode,
                            "policy_mode": policy,
                            "seed": seed,
                            "workers": workers,
                            "decision_points": int(summary["decision_points"]),
                            "faults_injected": int(
                                summary["receipt_fault_injected_count"]
                            ),
                            "receipt_rows": int(summary["receipt_rows"]),
                            "missing_receipts": int(summary["missing_receipts"]),
                            "unlogged_downstream_calls": int(
                                summary["unlogged_downstream_calls"]
                            ),
                            "duplicate_downstream_calls": int(
                                summary["duplicate_downstream_calls"]
                            ),
                            "mismatched_correlation_ids": int(
                                summary["mismatched_correlation_ids"]
                            ),
                            "unauthorized_downstream_calls": int(
                                summary["unauthorized_invocations"]
                            ),
                            "receipt_validation_passed": int(
                                summary["receipt_validation_passed"]
                            ),
                            "detected_as_expected": _detected_receipt_fault(
                                summary, mode
                            ),
                            "action_hash": summary["trace_hash"],
                            "record_trace_hash": summary["record_trace_hash"],
                            "config_manifest_hash": summary[
                                "config_manifest_hash"
                            ],
                            "config_bound_trace_hash": summary[
                                "config_bound_trace_hash"
                            ],
                        }
                    )

                    if mode == "clean":
                        clean_runs.append(
                            (policy, seed, workers, trace, dict(run_identity))
                        )

    for policy, seed, workers, trace, run_identity in clean_runs:
        reconciliation = trace.attrs["receipt_reconciliation"]
        clean_record_hash = record_bound_trace_hash(trace, reconciliation)
        clean_config_hash = str(run_identity["config_manifest_hash"])
        clean_config_bound = config_bound_trace_hash(
            clean_record_hash, clean_config_hash
        )

        for corruption in RECORD_CORRUPTIONS:
            if corruption == "configuration_label_corruption":
                corrupt_manifest = copy.deepcopy(run_identity["config_manifest"])
                corrupt_manifest["policy_mode"] = (
                    f"corrupted::{corrupt_manifest['policy_mode']}"
                )
                corrupt_config_hash = sha256_json(corrupt_manifest)
                corrupted_record_hash = clean_record_hash
                corrupted_config_bound = config_bound_trace_hash(
                    corrupted_record_hash, corrupt_config_hash
                )
                detected = int(corrupted_config_bound != clean_config_bound)
            else:
                corrupted_trace = _corrupt_record_trace(trace, corruption)
                corrupted_record_hash = record_bound_trace_hash(
                    corrupted_trace, reconciliation
                )
                corrupted_config_bound = config_bound_trace_hash(
                    corrupted_record_hash, clean_config_hash
                )
                detected = int(corrupted_record_hash != clean_record_hash)

            record_rows.append(
                {
                    "corruption_mode": corruption,
                    "policy_mode": policy,
                    "seed": seed,
                    "workers": workers,
                    "detected": detected,
                    "clean_record_trace_hash": clean_record_hash,
                    "corrupted_record_trace_hash": corrupted_record_hash,
                    "clean_config_bound_trace_hash": clean_config_bound,
                    "corrupted_config_bound_trace_hash": corrupted_config_bound,
                }
            )

    receipt_df = pd.DataFrame(receipt_rows)
    record_df = pd.DataFrame(record_rows)
    receipt_df.to_csv(
        output_dir / "execution_receipt_validation_per_run.csv", index=False
    )
    record_df.to_csv(
        output_dir / "record_bound_corruption_validation_per_run.csv", index=False
    )

    receipt_summary = (
        receipt_df.groupby("fault_mode", as_index=False)
        .agg(
            runs=("detected_as_expected", "size"),
            detected_runs=("detected_as_expected", "sum"),
            injected_events=("faults_injected", "sum"),
            receipt_validation_passes=("receipt_validation_passed", "sum"),
        )
    )
    record_summary = (
        record_df.groupby("corruption_mode", as_index=False)
        .agg(runs=("detected", "size"), detected_runs=("detected", "sum"))
    )
    receipt_summary.to_csv(
        output_dir / "execution_receipt_validation_summary.csv", index=False
    )
    record_summary.to_csv(
        output_dir / "record_bound_corruption_validation_summary.csv", index=False
    )

    for row in receipt_summary.to_dict(orient="records"):
        if int(row["runs"]) != expected_conditions:
            raise RuntimeError(
                f"{row['fault_mode']} has {row['runs']} runs; "
                f"expected {expected_conditions}"
            )
        if int(row["detected_runs"]) != expected_conditions:
            raise RuntimeError(
                f"{row['fault_mode']} detected {row['detected_runs']}/"
                f"{expected_conditions}"
            )

    for row in record_summary.to_dict(orient="records"):
        if int(row["runs"]) != expected_conditions:
            raise RuntimeError(
                f"{row['corruption_mode']} has {row['runs']} runs; "
                f"expected {expected_conditions}"
            )
        if int(row["detected_runs"]) != expected_conditions:
            raise RuntimeError(
                f"{row['corruption_mode']} detected {row['detected_runs']}/"
                f"{expected_conditions}"
            )

    manifest = {
        "experiment": "independent_execution_receipts_and_record_bound_trace",
        "input_csv": str(dataset_cfg["input_csv"]),
        "decision_points": int(len(frame)),
        "policies": policies,
        "seeds": seeds,
        "workers": workers_values,
        "conditions_per_mode": expected_conditions,
        "receipt_modes": RECEIPT_MODES,
        "record_corruptions": RECORD_CORRUPTIONS,
        "receipt_fault_probability": float(
            base_cfg.get("execution_integrity", {}).get(
                "receipt_fault_probability", 0.01
            )
        ),
        "receipt_validation_all_expected": bool(
            receipt_summary["detected_runs"].eq(
                receipt_summary["runs"]
            ).all()
        ),
        "record_corruption_all_detected": bool(
            record_summary["detected_runs"].eq(record_summary["runs"]).all()
        ),
    }
    with (output_dir / "execution_integrity_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("[DONE] Independent execution-integrity validation passed")
    print(receipt_summary.to_string(index=False))
    print(record_summary.to_string(index=False))


if __name__ == "__main__":
    main()
