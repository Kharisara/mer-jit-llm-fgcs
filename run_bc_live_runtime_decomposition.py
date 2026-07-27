#!/usr/bin/env python3
"""Run the dedicated bc_live runtime-boundary study for Reviewer Comment 8.

This study does not replace or mutate the finalized 528-row mixed-policy timing
study. It reruns only the eight bc_live configurations needed to separate:

1. end-to-end in-memory benchmark runtime;
2. checkpoint-preparation runtime;
3. replay-only runtime;
4. post-replay validation runtime (reported so the total is auditable).

Design:
    - four non-full workload fractions, workers=1, seven repetitions each;
    - full workload, workers in {1,2,4,8}, fifteen repetitions each;
    - one untimed warm-up per active configuration by default.

Expected measured rows with the project configuration:
    4 * 7 + 4 * 15 = 88.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from run_fgcs_extended_benchmark import (
    ensure_dir,
    load_config,
    normalize_label,
    run_replay,
    summarize_stage_latency,
    validate_config,
)


@dataclass(frozen=True, order=True)
class BCCondition:
    dataset_fraction: float
    workers: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def condition_key(condition: BCCondition) -> tuple[float, int]:
    return (round(float(condition.dataset_fraction), 12), int(condition.workers))


def build_conditions(
    fractions: Iterable[float],
    workers: Iterable[int],
    *,
    full_fraction: float,
) -> list[BCCondition]:
    conditions = {
        BCCondition(float(fraction), 1)
        for fraction in fractions
        if not np.isclose(float(fraction), full_fraction)
    }
    conditions.update(
        BCCondition(float(full_fraction), int(worker)) for worker in workers
    )
    return sorted(conditions)


def target_repetitions(
    condition: BCCondition,
    *,
    full_fraction: float,
    workload_repetitions: int,
    worker_repetitions: int,
) -> int:
    return (
        worker_repetitions
        if np.isclose(condition.dataset_fraction, full_fraction)
        else workload_repetitions
    )


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int = 20_000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("At least two observations are required for a median CI")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, values.size), replace=True)
    medians = np.median(draws, axis=1)
    return (
        float(np.median(values)),
        float(np.quantile(medians, alpha / 2.0)),
        float(np.quantile(medians, 1.0 - alpha / 2.0)),
    )


def validate_existing(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    required = {
        "dataset_fraction",
        "workers",
        "repetition",
        "decision_points",
        "trace_hash",
        "end_to_end_runtime_seconds",
        "checkpoint_preparation_seconds",
        "replay_only_runtime_seconds",
        "post_replay_validation_seconds",
        "runtime_decomposition_valid",
        "authorization_execution_consistent",
        "row_count_match",
        "validation_passed",
        "fault_injected_count",
        "unauthorized_invocations",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Existing Comment 8 CSV cannot be resumed; missing columns: "
            f"{missing}"
        )

    frame = frame.copy()
    frame["dataset_fraction"] = pd.to_numeric(
        frame["dataset_fraction"], errors="raise"
    ).astype(float)
    frame["workers"] = pd.to_numeric(frame["workers"], errors="raise").astype(int)
    frame["repetition"] = pd.to_numeric(
        frame["repetition"], errors="raise"
    ).astype(int)

    numeric_positive = [
        "end_to_end_runtime_seconds",
        "checkpoint_preparation_seconds",
        "replay_only_runtime_seconds",
        "post_replay_validation_seconds",
    ]
    for column in numeric_positive:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)

    if (frame["end_to_end_runtime_seconds"] <= 0).any():
        raise ValueError("Existing end-to-end runtimes must be positive")
    if (frame["checkpoint_preparation_seconds"] <= 0).any():
        raise ValueError("Existing checkpoint-preparation runtimes must be positive")
    if (frame["replay_only_runtime_seconds"] <= 0).any():
        raise ValueError("Existing replay-only runtimes must be positive")
    if (frame["post_replay_validation_seconds"] < 0).any():
        raise ValueError("Existing post-replay validation runtimes cannot be negative")

    duplicate = frame.duplicated(
        ["dataset_fraction", "workers", "repetition"], keep=False
    )
    if duplicate.any():
        raise ValueError("Existing Comment 8 CSV contains duplicate repetitions")

    checks = {
        "runtime_decomposition_valid": 1,
        "authorization_execution_consistent": 1,
        "row_count_match": 1,
        "validation_passed": 1,
        "fault_injected_count": 0,
        "unauthorized_invocations": 0,
    }
    for column, expected in checks.items():
        values = pd.to_numeric(frame[column], errors="raise")
        if not values.eq(expected).all():
            raise ValueError(
                f"Existing Comment 8 CSV fails invariant {column}={expected}"
            )

    return frame.reset_index(drop=True)


def build_summary(frame: pd.DataFrame, *, bootstrap_seed: int) -> pd.DataFrame:
    metrics = [
        "end_to_end_runtime_seconds",
        "checkpoint_preparation_seconds",
        "replay_only_runtime_seconds",
        "post_replay_validation_seconds",
        "timed_execution_runtime_seconds",
    ]
    rows: list[dict[str, Any]] = []

    for group_index, (key, group) in enumerate(
        frame.groupby(["dataset_fraction", "workers"], sort=True)
    ):
        fraction, workers = key
        row: dict[str, Any] = {
            "dataset_fraction": float(fraction),
            "workers": int(workers),
            "decision_points": int(group["decision_points"].iloc[0]),
            "measured_repetitions": int(len(group)),
            "unique_trace_hashes": int(group["trace_hash"].nunique()),
            "all_runtime_decompositions_valid": int(
                pd.to_numeric(group["runtime_decomposition_valid"]).eq(1).all()
            ),
        }

        for metric_index, metric in enumerate(metrics):
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(float)
            estimate, lower, upper = bootstrap_median_ci(
                values,
                seed=bootstrap_seed + group_index * 100 + metric_index,
            )
            row[f"{metric}_median"] = estimate
            row[f"{metric}_q1"] = float(np.quantile(values, 0.25))
            row[f"{metric}_q3"] = float(np.quantile(values, 0.75))
            row[f"{metric}_median_ci95_low"] = lower
            row[f"{metric}_median_ci95_high"] = upper

        end_to_end = pd.to_numeric(
            group["end_to_end_runtime_seconds"], errors="raise"
        ).to_numpy(float)
        preparation = pd.to_numeric(
            group["checkpoint_preparation_seconds"], errors="raise"
        ).to_numpy(float)
        replay = pd.to_numeric(
            group["replay_only_runtime_seconds"], errors="raise"
        ).to_numpy(float)
        validation = pd.to_numeric(
            group["post_replay_validation_seconds"], errors="raise"
        ).to_numpy(float)

        row["checkpoint_preparation_share_median"] = float(
            np.median(preparation / end_to_end)
        )
        row["replay_only_share_median"] = float(np.median(replay / end_to_end))
        row["post_replay_validation_share_median"] = float(
            np.median(validation / end_to_end)
        )
        row["end_to_end_throughput_points_per_second_median"] = float(
            np.median(group["decision_points"].to_numpy(float) / end_to_end)
        )
        row["replay_only_throughput_points_per_second_median"] = float(
            np.median(group["decision_points"].to_numpy(float) / replay)
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["dataset_fraction", "workers"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the bc_live-only runtime decomposition study."
    )
    parser.add_argument(
        "--config",
        default="configs/fgcs_extended_benchmark.yaml",
        help="Base ReplayBench-PG benchmark YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_outputs/bc_live_runtime_decomposition",
    )
    parser.add_argument("--workload-repetitions", type=int, default=7)
    parser.add_argument("--worker-repetitions", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--policy-seed", type=int, default=1)
    parser.add_argument("--order-seed", type=int, default=20260714)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    parser.add_argument("--full-fraction", type=float, default=1.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retain valid completed rows and execute only missing repetitions.",
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

    config_path = Path(args.config)
    cfg = load_config(config_path)
    validate_config(cfg)

    dataset_cfg = cfg["dataset"]
    benchmark_cfg = cfg["benchmark"]
    policy_cfg = cfg["policy"]

    if "bc_live" not in [str(value) for value in benchmark_cfg["policy_modes"]]:
        raise ValueError("The base configuration does not enable bc_live")

    fractions = [float(value) for value in dataset_cfg["fractions"]]
    workers = sorted({int(value) for value in benchmark_cfg["workers"]})
    if 1 not in workers:
        raise ValueError("The Comment 8 design requires workers=1")
    if not any(np.isclose(value, args.full_fraction) for value in fractions):
        raise ValueError(
            f"Full fraction {args.full_fraction} is not present in the config"
        )

    conditions = build_conditions(
        fractions,
        workers,
        full_fraction=args.full_fraction,
    )
    expected_configuration_count = (len(fractions) - 1) + len(workers)
    if len(conditions) != expected_configuration_count:
        raise RuntimeError(
            "Unexpected bc_live condition count: "
            f"expected={expected_configuration_count}, observed={len(conditions)}"
        )

    input_csv = Path(str(dataset_cfg["input_csv"]))
    df_full = pd.read_csv(input_csv).reset_index(drop=True)
    if df_full.empty:
        raise ValueError(f"Input CSV has no rows: {input_csv}")

    fraction_frames: dict[float, pd.DataFrame] = {}
    for fraction in fractions:
        count = max(1, int(len(df_full) * fraction))
        fraction_frames[round(fraction, 12)] = (
            df_full.iloc[:count].copy().reset_index(drop=True)
        )

    negative_labels = {
        normalize_label(value)
        for value in policy_cfg.get("negative_labels", [])
    }

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    raw_path = output_dir / "bc_live_runtime_decomposition_raw.csv"
    summary_path = output_dir / "bc_live_runtime_decomposition_summary.csv"
    manifest_path = output_dir / "bc_live_runtime_decomposition_manifest.json"

    existing = pd.DataFrame()
    if args.resume and raw_path.exists():
        existing = validate_existing(pd.read_csv(raw_path))

    target_keys = {condition_key(condition) for condition in conditions}
    existing_ids: dict[tuple[float, int], set[int]] = {
        key: set() for key in target_keys
    }
    reference_hashes: dict[tuple[float, int], str] = {}

    if not existing.empty:
        existing_keys = {
            (round(float(row.dataset_fraction), 12), int(row.workers))
            for row in existing.itertuples(index=False)
        }
        unexpected = sorted(existing_keys - target_keys)
        if unexpected:
            raise ValueError(
                "Existing Comment 8 CSV contains unexpected configurations: "
                f"{unexpected}"
            )

        for key, group in existing.groupby(["dataset_fraction", "workers"]):
            normalized_key = (round(float(key[0]), 12), int(key[1]))
            hashes = set(group["trace_hash"].astype(str))
            if len(hashes) != 1:
                raise ValueError(
                    f"Existing action hashes are unstable for {normalized_key}"
                )
            reference_hashes[normalized_key] = next(iter(hashes))
            existing_ids[normalized_key] = set(
                pd.to_numeric(group["repetition"], errors="raise").astype(int)
            )

    measured_rows = existing.to_dict(orient="records") if not existing.empty else []
    invocation_started = time.time()
    warmup_executions = 0
    measured_executions = 0

    conditions_needing_runs = [
        condition
        for condition in conditions
        if len(existing_ids[condition_key(condition)])
        < target_repetitions(
            condition,
            full_fraction=args.full_fraction,
            workload_repetitions=args.workload_repetitions,
            worker_repetitions=args.worker_repetitions,
        )
    ]

    for warmup_index in range(1, args.warmups + 1):
        ordered = list(conditions_needing_runs)
        random.Random(args.order_seed + warmup_index - 1).shuffle(ordered)
        for condition in ordered:
            frame = fraction_frames[round(condition.dataset_fraction, 12)]
            _, summary, _ = run_replay(
                df=frame,
                cfg=cfg,
                policy_mode="bc_live",
                negative_labels=negative_labels,
                seed=args.policy_seed,
                workers=condition.workers,
                bc_actions=None,
            )
            for field, expected in {
                "runtime_decomposition_valid": 1,
                "authorization_execution_consistent": 1,
                "row_count_match": 1,
                "validation_passed": 1,
                "fault_injected_count": 0,
                "unauthorized_invocations": 0,
            }.items():
                if int(summary[field]) != expected:
                    raise RuntimeError(
                        f"Warm-up invariant failed: {field}={summary[field]}"
                    )
            warmup_executions += 1

    for repetition in range(1, args.worker_repetitions + 1):
        missing = []
        for condition in conditions:
            target = target_repetitions(
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
            target = target_repetitions(
                condition,
                full_fraction=args.full_fraction,
                workload_repetitions=args.workload_repetitions,
                worker_repetitions=args.worker_repetitions,
            )
            frame = fraction_frames[round(condition.dataset_fraction, 12)]
            print(
                "[RUN] "
                f"rep={repetition}/{target}, "
                f"fraction={condition.dataset_fraction}, "
                "policy=bc_live, "
                f"workers={condition.workers}"
            )

            trace, summary, _ = run_replay(
                df=frame,
                cfg=cfg,
                policy_mode="bc_live",
                negative_labels=negative_labels,
                seed=args.policy_seed,
                workers=condition.workers,
                bc_actions=None,
            )

            checks = {
                "runtime_decomposition_valid": 1,
                "authorization_execution_consistent": 1,
                "row_count_match": 1,
                "validation_passed": 1,
                "fault_injected_count": 0,
                "unauthorized_invocations": 0,
            }
            for field, expected in checks.items():
                if int(summary[field]) != expected:
                    raise RuntimeError(
                        f"Measured-run invariant failed: {field}={summary[field]}"
                    )

            key = condition_key(condition)
            trace_hash = str(summary["trace_hash"])
            expected_hash = reference_hashes.get(key)
            if expected_hash is None:
                reference_hashes[key] = trace_hash
                hash_match = 1
            else:
                hash_match = int(trace_hash == expected_hash)
                if hash_match != 1:
                    raise RuntimeError(
                        "bc_live action hash changed for "
                        f"fraction={condition.dataset_fraction}, "
                        f"workers={condition.workers}"
                    )

            end_to_end = float(summary["end_to_end_runtime_seconds"])
            preparation = float(summary["checkpoint_preparation_seconds"])
            replay_only = float(summary["replay_only_runtime_seconds"])
            validation = float(summary["post_replay_validation_seconds"])
            timed_execution = float(summary["timed_execution_runtime_seconds"])

            if preparation <= 0 or replay_only <= 0 or end_to_end <= 0:
                raise RuntimeError("Measured runtime phases must be positive")
            if validation < 0:
                raise RuntimeError("Post-replay validation runtime cannot be negative")

            row: dict[str, Any] = {
                "dataset_fraction": float(condition.dataset_fraction),
                "workload_name": f"fraction_{condition.dataset_fraction}",
                "decision_points": int(summary["decision_points"]),
                "policy": "bc_live",
                "policy_mode": "bc_live",
                "policy_seed": int(args.policy_seed),
                "workers": int(condition.workers),
                "repetition": int(repetition),
                "end_to_end_runtime_seconds": end_to_end,
                "checkpoint_preparation_seconds": preparation,
                "replay_only_runtime_seconds": replay_only,
                "post_replay_validation_seconds": validation,
                "timed_execution_runtime_seconds": timed_execution,
                "legacy_total_runtime_seconds": float(
                    summary["total_runtime_seconds"]
                ),
                "timed_runtime_decomposition_error_seconds": float(
                    summary["timed_runtime_decomposition_error_seconds"]
                ),
                "end_to_end_runtime_decomposition_error_seconds": float(
                    summary["end_to_end_runtime_decomposition_error_seconds"]
                ),
                "runtime_decomposition_tolerance_seconds": float(
                    summary["runtime_decomposition_tolerance_seconds"]
                ),
                "runtime_decomposition_valid": int(
                    summary["runtime_decomposition_valid"]
                ),
                "checkpoint_preparation_share": preparation / end_to_end,
                "replay_only_share": replay_only / end_to_end,
                "post_replay_validation_share": validation / end_to_end,
                "end_to_end_throughput_points_per_second": float(
                    summary["end_to_end_throughput_points_per_second"]
                ),
                "replay_only_throughput_points_per_second": float(
                    summary["replay_only_throughput_points_per_second"]
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
                "runtime_boundary_definition": str(
                    summary["runtime_boundary_definition"]
                ),
            }
            row.update(summarize_stage_latency(trace))
            measured_rows.append(row)
            existing_ids[key].add(repetition)
            measured_executions += 1

            current = pd.DataFrame(measured_rows).sort_values(
                ["dataset_fraction", "workers", "repetition"]
            )
            current.to_csv(raw_path, index=False)

    final = validate_existing(pd.DataFrame(measured_rows))
    expected_rows = sum(
        target_repetitions(
            condition,
            full_fraction=args.full_fraction,
            workload_repetitions=args.workload_repetitions,
            worker_repetitions=args.worker_repetitions,
        )
        for condition in conditions
    )
    if len(final) != expected_rows:
        raise RuntimeError(
            f"Incomplete Comment 8 study: expected={expected_rows}, "
            f"observed={len(final)}"
        )

    for condition in conditions:
        key = condition_key(condition)
        target = target_repetitions(
            condition,
            full_fraction=args.full_fraction,
            workload_repetitions=args.workload_repetitions,
            worker_repetitions=args.worker_repetitions,
        )
        required_ids = set(range(1, target + 1))
        if existing_ids[key] != required_ids:
            raise RuntimeError(
                f"Incomplete repetition set for {key}: "
                f"observed={sorted(existing_ids[key])}, "
                f"required={sorted(required_ids)}"
            )

    final = final.sort_values(
        ["dataset_fraction", "workers", "repetition"]
    ).reset_index(drop=True)
    final.to_csv(raw_path, index=False)

    summary_frame = build_summary(final, bootstrap_seed=args.bootstrap_seed)
    summary_frame.to_csv(summary_path, index=False)

    manifest = {
        "study": "ReplayBench-PG bc_live runtime decomposition",
        "reviewer_comment": 8,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "runner": "run_bc_live_runtime_decomposition.py",
        "base_runner": "run_fgcs_extended_benchmark.py",
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "input_csv": str(input_csv),
        "input_csv_sha256": sha256_file(input_csv),
        "policy_mode": "bc_live",
        "policy_seed": args.policy_seed,
        "order_seed": args.order_seed,
        "bootstrap_seed": args.bootstrap_seed,
        "fractions": fractions,
        "workers": workers,
        "full_fraction": args.full_fraction,
        "workload_repetitions": args.workload_repetitions,
        "worker_repetitions": args.worker_repetitions,
        "unique_configurations": len(conditions),
        "expected_measured_rows": expected_rows,
        "completed_measured_rows": len(final),
        "active_configurations_in_this_invocation": len(
            conditions_needing_runs
        ),
        "warmups_per_active_configuration_in_this_invocation": args.warmups,
        "warmup_executions_in_this_invocation": warmup_executions,
        "measured_executions_in_this_invocation": measured_executions,
        "resume_mode": bool(args.resume),
        "invocation_started_unix": invocation_started,
        "invocation_completed_unix": time.time(),
        "runtime_boundaries": {
            "end_to_end_runtime_seconds": (
                "run_start through in-memory trace reconstruction, receipt "
                "reconciliation, hashing, and validation; excludes output-file "
                "serialization"
            ),
            "checkpoint_preparation_seconds": (
                "bc_live model/checkpoint loading, state loading, batched "
                "inference, tensor/output conversion, and live action-map "
                "construction"
            ),
            "replay_only_runtime_seconds": (
                "per-record policy-gated replay using the already prepared "
                "bc_live action map"
            ),
            "post_replay_validation_seconds": (
                "in-memory trace sorting/reconstruction, downstream-receipt "
                "reconciliation, digest computation, and validation"
            ),
            "timed_execution_runtime_seconds": (
                "historical timing boundary retained for comparability: "
                "checkpoint preparation plus replay only"
            ),
        },
        "all_runtime_decompositions_valid": bool(
            final["runtime_decomposition_valid"].eq(1).all()
        ),
        "all_hashes_stable": bool(
            final.groupby(["dataset_fraction", "workers"])["trace_hash"]
            .nunique()
            .eq(1)
            .all()
        ),
        "all_authorization_execution_consistent": bool(
            final["authorization_execution_consistent"].eq(1).all()
        ),
        "all_row_counts_match": bool(final["row_count_match"].eq(1).all()),
        "all_validation_passed": bool(final["validation_passed"].eq(1).all()),
        "all_fault_counts_zero": bool(final["fault_injected_count"].eq(0).all()),
        "all_unauthorized_invocations_zero": bool(
            final["unauthorized_invocations"].eq(0).all()
        ),
        "raw_csv": str(raw_path),
        "raw_csv_sha256": sha256_file(raw_path),
        "summary_csv": str(summary_path),
        "summary_csv_sha256": sha256_file(summary_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[DONE] bc_live runtime decomposition complete")
    print(f"[OUT] {raw_path}")
    print(f"[OUT] {summary_path}")
    print(f"[OUT] {manifest_path}")
    print(
        "[CHECK] "
        f"configurations={len(conditions)}, measured_rows={len(final)}, "
        f"warmups_this_invocation={warmup_executions}"
    )


if __name__ == "__main__":
    main()
