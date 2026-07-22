#!/usr/bin/env python3
"""Run the targeted 15-repetition ReplayBench-PG timing study.

The script executes the two timing arms used in the manuscript:

1. workload scaling: every configured fraction and policy with one worker;
2. worker scaling: full workload, every policy, and every configured worker.

The overlapping full-workload/one-worker configurations are executed once.
Each active configuration receives untimed warm-up execution(s). Non-full
workload-scaling configurations retain seven measured repetitions, while the
24 full-workload worker-scaling configurations are extended to fifteen.
``--resume`` retains the original seven-repetition rows and executes only the
192 missing full-workload measurements.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd

from run_fgcs_extended_benchmark import (
    ensure_dir,
    load_bc_reference_actions,
    load_config,
    normalize_label,
    run_replay,
    summarize_stage_latency,
    validate_config,
)


@dataclass(frozen=True, order=True)
class TimingCondition:
    dataset_fraction: float
    policy_mode: str
    workers: int


def get_git_commit() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def build_target_conditions(
    fractions: Iterable[float],
    policies: Iterable[str],
    workers: Iterable[int],
    full_fraction: float = 1.0,
) -> list[TimingCondition]:
    """Build the union of workload- and worker-scaling configurations."""
    conditions = {
        TimingCondition(float(fraction), str(policy), 1)
        for fraction in fractions
        for policy in policies
    }
    conditions.update(
        TimingCondition(float(full_fraction), str(policy), int(worker))
        for policy in policies
        for worker in workers
    )
    return sorted(conditions)


def condition_key(condition: TimingCondition) -> tuple[float, str, int]:
    return (
        round(float(condition.dataset_fraction), 12),
        condition.policy_mode,
        int(condition.workers),
    )


def normalize_existing(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    frame = frame.copy()

    if "policy_mode" not in frame.columns and "policy" in frame.columns:
        frame["policy_mode"] = frame["policy"]
    if "policy" not in frame.columns and "policy_mode" in frame.columns:
        frame["policy"] = frame["policy_mode"]

    if (
        "dataset_fraction" not in frame.columns
        and "workload_fraction" in frame.columns
    ):
        frame["dataset_fraction"] = frame["workload_fraction"]
    if (
        "workload_fraction" not in frame.columns
        and "dataset_fraction" in frame.columns
    ):
        frame["workload_fraction"] = frame["dataset_fraction"]

    if (
        "total_runtime_seconds" not in frame.columns
        and "runtime_seconds" in frame.columns
    ):
        frame["total_runtime_seconds"] = frame["runtime_seconds"]
    if (
        "runtime_seconds" not in frame.columns
        and "total_runtime_seconds" in frame.columns
    ):
        frame["runtime_seconds"] = frame["total_runtime_seconds"]

    required = {
        "dataset_fraction",
        "policy_mode",
        "workers",
        "repetition",
        "total_runtime_seconds",
        "trace_hash",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Existing timing CSV cannot be resumed; missing columns: "
            f"{missing}"
        )

    frame["dataset_fraction"] = pd.to_numeric(
        frame["dataset_fraction"], errors="raise"
    ).astype(float)
    frame["policy_mode"] = frame["policy_mode"].astype(str)
    frame["workers"] = pd.to_numeric(
        frame["workers"], errors="raise"
    ).astype(int)
    frame["repetition"] = pd.to_numeric(
        frame["repetition"], errors="raise"
    ).astype(int)
    frame["total_runtime_seconds"] = pd.to_numeric(
        frame["total_runtime_seconds"], errors="raise"
    ).astype(float)
    frame["policy"] = frame["policy_mode"]
    frame["runtime_seconds"] = frame["total_runtime_seconds"]
    frame["workload_fraction"] = frame["dataset_fraction"]

    duplicate_mask = frame.duplicated(
        ["dataset_fraction", "policy_mode", "workers", "repetition"],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError("Existing timing CSV contains duplicate repetitions")
    if (frame["total_runtime_seconds"] <= 0).any():
        raise ValueError("Existing timing CSV contains nonpositive runtimes")

    checks = {
        "authorization_execution_consistent": 1,
        "row_count_match": 1,
        "validation_passed": 1,
        "fault_injected_count": 0,
        "unauthorized_invocations": 0,
        "hash_match": 1,
    }
    for column, expected in checks.items():
        if column not in frame.columns:
            continue

        values = pd.to_numeric(frame[column], errors="coerce")
        recorded_values = values.dropna()

        if not recorded_values.eq(expected).all():
            invalid_rows = frame.loc[values.notna() & ~values.eq(expected)]
            raise ValueError(
                f"Existing timing CSV fails functional check: {column}\n"
                f"{invalid_rows.to_string(index=False)}"
            )

    return frame.reset_index(drop=True)


def target_repetitions_for_condition(
    condition: TimingCondition,
    *,
    full_fraction: float,
    workload_repetitions: int,
    worker_repetitions: int,
) -> int:
    return (
        worker_repetitions
        if abs(condition.dataset_fraction - full_fraction) < 1e-12
        else workload_repetitions
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the mixed 7/15-repetition ReplayBench-PG timing study."
    )
    parser.add_argument(
        "--config",
        default="configs/fgcs_extended_benchmark.yaml",
        help="Base ReplayBench-PG benchmark YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_outputs/replaybench_timing_study",
    )
    parser.add_argument("--workload-repetitions", type=int, default=7)
    parser.add_argument("--worker-repetitions", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--policy-seed", type=int, default=1)
    parser.add_argument("--order-seed", type=int, default=20260714)
    parser.add_argument("--full-fraction", type=float, default=1.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep valid existing rows and run only missing repetitions. "
            "With the original seven-repetition file, this adds only the "
            "eight missing repetitions to the 24 full-workload worker "
            "configurations (192 executions)."
        ),
    )
    args = parser.parse_args()

    if args.workload_repetitions < 2:
        raise ValueError("--workload-repetitions must be at least 2")
    if args.worker_repetitions < args.workload_repetitions:
        raise ValueError(
            "--worker-repetitions must be at least --workload-repetitions"
        )
    if args.warmups < 0:
        raise ValueError("--warmups cannot be negative")
    if not 0.0 < args.full_fraction <= 1.0:
        raise ValueError("--full-fraction must be in (0, 1]")

    cfg = load_config(args.config)
    validate_config(cfg)

    dataset_cfg = cfg["dataset"]
    benchmark_cfg = cfg["benchmark"]
    policy_cfg = cfg["policy"]

    fractions = [float(value) for value in dataset_cfg["fractions"]]
    policies = [str(value) for value in benchmark_cfg["policy_modes"]]
    workers = sorted({int(value) for value in benchmark_cfg["workers"]})

    if 1 not in workers:
        raise ValueError("Timing design requires workers=1")
    if not any(abs(value - args.full_fraction) < 1e-12 for value in fractions):
        raise ValueError(
            f"Full fraction {args.full_fraction} is not present in config"
        )

    conditions = build_target_conditions(
        fractions,
        policies,
        workers,
        full_fraction=args.full_fraction,
    )

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    raw_path = output_dir / "timing_repetitions_raw.csv"
    manifest_path = output_dir / "timing_study_manifest.json"
    environment_path = output_dir / "timing_environment_manifest.json"

    if args.resume and raw_path.exists():
        existing = normalize_existing(pd.read_csv(raw_path))
    else:
        existing = pd.DataFrame()

    target_keys = {condition_key(condition) for condition in conditions}
    condition_by_key = {
        condition_key(condition): condition for condition in conditions
    }

    existing_ids: dict[tuple[float, str, int], set[int]] = {
        key: set() for key in target_keys
    }
    if not existing.empty:
        existing_keys = {
            (
                round(float(row.dataset_fraction), 12),
                str(row.policy_mode),
                int(row.workers),
            )
            for row in existing.itertuples(index=False)
        }
        unexpected = sorted(existing_keys - target_keys)
        if unexpected:
            raise ValueError(
                "Existing timing CSV contains configurations outside the "
                f"target design: {unexpected}"
            )

        for row in existing.itertuples(index=False):
            key = (
                round(float(row.dataset_fraction), 12),
                str(row.policy_mode),
                int(row.workers),
            )
            condition = condition_by_key[key]
            target = target_repetitions_for_condition(
                condition,
                full_fraction=args.full_fraction,
                workload_repetitions=args.workload_repetitions,
                worker_repetitions=args.worker_repetitions,
            )
            repetition = int(row.repetition)
            if repetition > target:
                raise ValueError(
                    "Existing timing CSV contains repetition identifiers above "
                    f"the target for {key}: repetition={repetition}, target={target}"
                )
            existing_ids[key].add(repetition)

    input_csv = Path(str(dataset_cfg["input_csv"]))
    df_full = pd.read_csv(input_csv).reset_index(drop=True)
    if df_full.empty:
        raise ValueError(f"Input CSV has no rows: {input_csv}")

    negative_labels = {
        normalize_label(value)
        for value in policy_cfg.get("negative_labels", [])
    }

    bc_actions: Optional[Dict[int, int]] = None
    if "bc" in policies:
        bc_actions = load_bc_reference_actions(
            bc_action_csv=policy_cfg.get(
                "bc_action_csv",
                "paper_outputs/policy_first_outputs_bc.csv",
            ),
            df_full=df_full,
            action_column=str(policy_cfg.get("bc_action_column", "action")),
            key_column=str(policy_cfg.get("bc_key_column", "utterance_id")),
        )

    fraction_frames: dict[float, pd.DataFrame] = {}
    for fraction in fractions:
        count = max(1, int(len(df_full) * fraction))
        fraction_frames[round(fraction, 12)] = (
            df_full.iloc[:count].copy().reset_index(drop=True)
        )

    reference_hashes: dict[tuple[float, str, int], str] = {}
    if not existing.empty:
        for key, group in existing.groupby(
            ["dataset_fraction", "policy_mode", "workers"]
        ):
            hashes = set(group["trace_hash"].astype(str))
            if len(hashes) != 1:
                raise ValueError(
                    f"Existing timing hashes are unstable for {key}"
                )
            reference_hashes[
                (round(float(key[0]), 12), str(key[1]), int(key[2]))
            ] = next(iter(hashes))

    measured_rows = (
        existing.to_dict(orient="records") if not existing.empty else []
    )
    invocation_started = time.time()
    warmup_executions = 0
    measured_executions = 0

    conditions_needing_runs = [
        condition
        for condition in conditions
        if len(existing_ids[condition_key(condition)])
        < target_repetitions_for_condition(
            condition,
            full_fraction=args.full_fraction,
            workload_repetitions=args.workload_repetitions,
            worker_repetitions=args.worker_repetitions,
        )
    ]

    # Warm only configurations that will receive new measured executions in
    # this invocation. When resuming the original seven-repetition file, this
    # is exactly the 24 full-workload worker-scaling configurations.
    for warmup_index in range(1, args.warmups + 1):
        ordered = list(conditions_needing_runs)
        random.Random(args.order_seed + warmup_index - 1).shuffle(ordered)
        for condition in ordered:
            frame = fraction_frames[round(condition.dataset_fraction, 12)]
            _, summary, _ = run_replay(
                df=frame,
                cfg=cfg,
                policy_mode=condition.policy_mode,
                negative_labels=negative_labels,
                seed=args.policy_seed,
                workers=condition.workers,
                bc_actions=bc_actions,
            )
            if int(summary["authorization_execution_consistent"]) != 1:
                raise RuntimeError("Warm-up authorization invariant failed")
            if int(summary["row_count_match"]) != 1:
                raise RuntimeError("Warm-up row-count validation failed")
            if int(summary["fault_injected_count"]) != 0:
                raise RuntimeError("Warm-up unexpectedly injected a fault")
            warmup_executions += 1

    for repetition in range(1, args.worker_repetitions + 1):
        missing = []
        for condition in conditions:
            target = target_repetitions_for_condition(
                condition,
                full_fraction=args.full_fraction,
                workload_repetitions=args.workload_repetitions,
                worker_repetitions=args.worker_repetitions,
            )
            if (
                repetition <= target
                and repetition not in existing_ids[condition_key(condition)]
            ):
                missing.append(condition)

        random.Random(args.order_seed + 10_000 + repetition).shuffle(missing)

        for condition in missing:
            target = target_repetitions_for_condition(
                condition,
                full_fraction=args.full_fraction,
                workload_repetitions=args.workload_repetitions,
                worker_repetitions=args.worker_repetitions,
            )
            fraction_key = round(condition.dataset_fraction, 12)
            frame = fraction_frames[fraction_key]

            print(
                "[RUN] "
                f"rep={repetition}/{target}, "
                f"fraction={condition.dataset_fraction}, "
                f"policy={condition.policy_mode}, "
                f"workers={condition.workers}"
            )

            trace, summary, _ = run_replay(
                df=frame,
                cfg=cfg,
                policy_mode=condition.policy_mode,
                negative_labels=negative_labels,
                seed=args.policy_seed,
                workers=condition.workers,
                bc_actions=bc_actions,
            )

            if int(summary["authorization_execution_consistent"]) != 1:
                raise RuntimeError("Authorization-execution invariant failed")
            if int(summary["row_count_match"]) != 1:
                raise RuntimeError("Row-count validation failed")
            if int(summary["validation_passed"]) != 1:
                raise RuntimeError("Trace validation failed")
            if int(summary["fault_injected_count"]) != 0:
                raise RuntimeError("Timing run unexpectedly injected a fault")

            key = condition_key(condition)
            trace_hash = str(summary["trace_hash"])
            expected_hash = reference_hashes.get(key)
            if expected_hash is None:
                reference_hashes[key] = trace_hash
                hash_match = 1
            else:
                hash_match = int(trace_hash == expected_hash)
                if not hash_match:
                    raise RuntimeError(
                        "Timing action hash changed for "
                        f"fraction={condition.dataset_fraction}, "
                        f"policy={condition.policy_mode}, "
                        f"workers={condition.workers}"
                    )

            row: dict[str, Any] = {
                "dataset_fraction": float(condition.dataset_fraction),
                "workload_name": f"fraction_{condition.dataset_fraction}",
                "decision_points": int(summary["decision_points"]),
                "policy": condition.policy_mode,
                "policy_mode": condition.policy_mode,
                "policy_seed": int(args.policy_seed),
                "workers": int(condition.workers),
                "repetition": int(repetition),
                "runtime_seconds": float(summary["total_runtime_seconds"]),
                "total_runtime_seconds": float(
                    summary["total_runtime_seconds"]
                ),
                "throughput_points_per_second": float(
                    summary["throughput_points_per_second"]
                ),
                "trace_hash": trace_hash,
                "reference_hash": reference_hashes[key],
                "hash_match": int(hash_match),
                "unauthorized_invocations": int(
                    summary["unauthorized_invocations"]
                ),
                "authorization_execution_consistent": int(
                    summary["authorization_execution_consistent"]
                ),
                "row_count_match": int(summary["row_count_match"]),
                "validation_passed": int(summary["validation_passed"]),
                "fault_injected_count": int(summary["fault_injected_count"]),
                "intervention_rate": float(summary["intervention_rate"]),
            }
            row.update(summarize_stage_latency(trace))
            measured_rows.append(row)
            existing_ids[key].add(repetition)
            measured_executions += 1

            current = pd.DataFrame(measured_rows).sort_values(
                ["dataset_fraction", "policy_mode", "workers", "repetition"]
            )
            current.to_csv(raw_path, index=False)

    final = normalize_existing(pd.DataFrame(measured_rows))
    expected_rows = sum(
        target_repetitions_for_condition(
            condition,
            full_fraction=args.full_fraction,
            workload_repetitions=args.workload_repetitions,
            worker_repetitions=args.worker_repetitions,
        )
        for condition in conditions
    )
    if len(final) != expected_rows:
        raise RuntimeError(
            f"Incomplete timing study: expected={expected_rows}, "
            f"observed={len(final)}"
        )

    for condition in conditions:
        key = condition_key(condition)
        target = target_repetitions_for_condition(
            condition,
            full_fraction=args.full_fraction,
            workload_repetitions=args.workload_repetitions,
            worker_repetitions=args.worker_repetitions,
        )
        if existing_ids[key] != set(range(1, target + 1)):
            raise RuntimeError(
                f"Incomplete repetition set for {key}: "
                f"{sorted(existing_ids[key])}"
            )

    final = final.sort_values(
        ["dataset_fraction", "policy_mode", "workers", "repetition"]
    ).reset_index(drop=True)
    final.to_csv(raw_path, index=False)

    environment = {
        "python_version": sys.version,
        "git_commit": get_git_commit(),
        "config": str(args.config),
        "input_csv": str(input_csv),
    }
    environment_path.write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )

    non_full_configurations = sum(
        1
        for condition in conditions
        if abs(condition.dataset_fraction - args.full_fraction) >= 1e-12
    )
    full_configurations = len(conditions) - non_full_configurations

    manifest = {
        "study": "ReplayBench-PG targeted timing study",
        "config": str(args.config),
        "output": str(raw_path),
        "fractions": fractions,
        "policies": policies,
        "workers": workers,
        "full_fraction": args.full_fraction,
        "policy_seed": args.policy_seed,
        "order_seed": args.order_seed,
        "workload_repetitions": args.workload_repetitions,
        "worker_repetitions": args.worker_repetitions,
        "non_full_workload_configurations": non_full_configurations,
        "full_workload_worker_configurations": full_configurations,
        "unique_configurations": len(conditions),
        "expected_measured_rows": expected_rows,
        "completed_measured_rows": len(final),
        "warmups_per_active_configuration_in_this_invocation": args.warmups,
        "active_configurations_in_this_invocation": len(
            conditions_needing_runs
        ),
        "warmup_executions_in_this_invocation": warmup_executions,
        "measured_executions_in_this_invocation": measured_executions,
        "resume_mode": bool(args.resume),
        "invocation_started_unix": invocation_started,
        "invocation_completed_unix": time.time(),
        "all_hashes_stable": True,
        "all_authorization_execution_consistent": bool(
            final["authorization_execution_consistent"].dropna().eq(1).all()
        ),
        "all_row_counts_match": bool(
            final["row_count_match"].dropna().eq(1).all()
        ),
        "all_validation_passed": bool(
            final["validation_passed"].dropna().eq(1).all()
        ),
        "all_fault_counts_zero": bool(
            final["fault_injected_count"].dropna().eq(0).all()
        ),
        "all_unauthorized_invocations_zero": bool(
            final["unauthorized_invocations"].dropna().eq(0).all()
        ),
        "legacy_rows_without_validation_metadata": int(
            final["authorization_execution_consistent"].isna().sum()
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[DONE] Timing study complete")
    print(f"[OUT] {raw_path}")
    print(f"[OUT] {manifest_path}")
    print(f"[OUT] {environment_path}")


if __name__ == "__main__":
    main()