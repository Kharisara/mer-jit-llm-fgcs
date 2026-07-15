#!/usr/bin/env python3
"""
Targeted warmed-up timing study for ReplayBench-PG.

Purpose
-------
This script separates performance repetitions from the benchmark's policy seeds.
It measures only the configurations needed for the paper's timing claims:

1. Workload-scaling arm:
   all configured fractions x all policy modes x 1 worker
2. Worker-scaling arm:
   full workload x all policy modes x all configured worker counts

The union normally contains 48 configurations for the published design:
(5 fractions x 6 policies x 1 worker) +
(1 full fraction x 6 policies x 4 workers) -
(1 overlapping full-fraction x 6 policies x 1 worker).

Each configuration receives untimed warm-up execution(s), followed by measured
repetitions. The policy seed remains fixed across repetitions so timing noise is
not conflated with stochastic policy variation. Configuration order is shuffled
within every warm-up and measurement round using a fixed order seed.

Expected outputs
----------------
paper_outputs/replaybench_timing_study/
    timing_repetitions_raw.csv
    timing_summary.csv
    timing_stage_latency_summary.csv
    timing_trace_consistency.csv
    timing_environment.json
    timing_study_manifest.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import run_fgcs_extended_benchmark as bench


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version(module: Any) -> Optional[str]:
    return getattr(module, "__version__", None) if module is not None else None


def quantile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q))


def summarize_repetitions(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["study_arm", "dataset_fraction", "workload_name", "decision_points", "policy_mode", "workers"]
    rows: List[Dict[str, Any]] = []

    for keys, group in raw_df.groupby(group_cols, dropna=False, sort=True):
        key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        runtimes = group["total_runtime_seconds"].astype(float)
        throughputs = group["throughput_points_per_second"].astype(float)
        mean_latencies = group["mean_latency_ms"].astype(float)
        p95_latencies = group["p95_latency_ms"].astype(float)

        runtime_mean = float(runtimes.mean())
        runtime_std = float(runtimes.std(ddof=1)) if len(runtimes) > 1 else 0.0

        rows.append({
            **key_map,
            "measured_repetitions": int(len(group)),
            "fixed_policy_seed": int(group["policy_seed"].iloc[0]),
            "runtime_median_seconds": float(runtimes.median()),
            "runtime_q1_seconds": quantile(runtimes, 0.25),
            "runtime_q3_seconds": quantile(runtimes, 0.75),
            "runtime_iqr_seconds": quantile(runtimes, 0.75) - quantile(runtimes, 0.25),
            "runtime_min_seconds": float(runtimes.min()),
            "runtime_max_seconds": float(runtimes.max()),
            "runtime_mean_seconds": runtime_mean,
            "runtime_std_seconds": runtime_std,
            "runtime_cv": runtime_std / runtime_mean if runtime_mean > 0 else 0.0,
            "throughput_median_points_per_second": float(throughputs.median()),
            "throughput_q1_points_per_second": quantile(throughputs, 0.25),
            "throughput_q3_points_per_second": quantile(throughputs, 0.75),
            "throughput_iqr_points_per_second": quantile(throughputs, 0.75) - quantile(throughputs, 0.25),
            "mean_latency_median_ms": float(mean_latencies.median()),
            "p95_latency_median_ms": float(p95_latencies.median()),
            "intervention_rate": float(group["intervention_rate"].median()),
            "unauthorized_invocations_total": int(group["unauthorized_invocations"].sum()),
            "fault_injected_count_total": int(group["fault_injected_count"].sum()),
            "unique_trace_hashes": int(group["trace_hash"].nunique()),
            "all_trace_hashes_match": bool(group["trace_hash"].nunique() == 1),
        })

    return pd.DataFrame(rows)


def summarize_stage_latency(raw_df: pd.DataFrame) -> pd.DataFrame:
    stage_metrics = [
        "state_loading_ms_mean",
        "state_loading_ms_p95",
        "policy_inference_ms_mean",
        "policy_inference_ms_p95",
        "gating_ms_mean",
        "gating_ms_p95",
        "generation_stub_ms_mean",
        "generation_stub_ms_p95",
        "logging_ms_mean",
        "logging_ms_p95",
        "total_latency_ms_mean",
        "total_latency_ms_p95",
    ]
    existing = [c for c in stage_metrics if c in raw_df.columns]
    group_cols = ["study_arm", "dataset_fraction", "workload_name", "decision_points", "policy_mode", "workers"]
    rows: List[Dict[str, Any]] = []

    for keys, group in raw_df.groupby(group_cols, dropna=False, sort=True):
        key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row: Dict[str, Any] = {**key_map, "measured_repetitions": int(len(group))}
        for col in existing:
            values = group[col].astype(float)
            row[f"{col}_median_across_repetitions"] = float(values.median())
            row[f"{col}_q1_across_repetitions"] = quantile(values, 0.25)
            row[f"{col}_q3_across_repetitions"] = quantile(values, 0.75)
        rows.append(row)

    return pd.DataFrame(rows)


def build_trace_consistency(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["study_arm", "dataset_fraction", "workload_name", "decision_points", "policy_mode", "workers"]
    return (
        raw_df.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            measured_repetitions=("repetition", "count"),
            unique_trace_hashes=("trace_hash", "nunique"),
            reference_trace_hash=("trace_hash", "first"),
            unauthorized_invocations_total=("unauthorized_invocations", "sum"),
            fault_injected_count_total=("fault_injected_count", "sum"),
        )
        .assign(all_trace_hashes_match=lambda d: d["unique_trace_hashes"] == 1)
    )


def build_configurations(
    fractions: Sequence[float],
    policies: Sequence[str],
    workers: Sequence[int],
) -> List[Dict[str, Any]]:
    if not fractions:
        raise ValueError("No dataset fractions configured")
    if not policies:
        raise ValueError("No policy modes configured")
    if not workers:
        raise ValueError("No worker settings configured")

    one_worker = 1 if 1 in workers else min(workers)
    full_fraction = max(fractions)

    configs: Dict[Tuple[float, str, int], Dict[str, Any]] = {}

    # RQ3 workload-scaling arm.
    for fraction in fractions:
        for policy in policies:
            key = (float(fraction), str(policy), int(one_worker))
            configs[key] = {
                "dataset_fraction": float(fraction),
                "policy_mode": str(policy),
                "workers": int(one_worker),
                "study_arm": "workload_scaling",
            }

    # RQ4 worker-scaling arm. Mark the overlap as both arms.
    for policy in policies:
        for worker_count in workers:
            key = (float(full_fraction), str(policy), int(worker_count))
            if key in configs:
                configs[key]["study_arm"] = "workload_and_worker_scaling"
            else:
                configs[key] = {
                    "dataset_fraction": float(full_fraction),
                    "policy_mode": str(policy),
                    "workers": int(worker_count),
                    "study_arm": "worker_scaling",
                }

    return list(configs.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ReplayBench-PG warmed-up timing study.")
    parser.add_argument("--config", default="configs/fgcs_extended_benchmark.yaml")
    parser.add_argument("--warmups", type=int, default=1, help="Untimed warm-up rounds per configuration.")
    parser.add_argument("--repetitions", type=int, default=7, help="Measured repetitions per configuration.")
    parser.add_argument("--policy-seed", type=int, default=1, help="Fixed policy seed used in every timing repetition.")
    parser.add_argument("--order-seed", type=int, default=20260714, help="Seed used only to shuffle configuration order.")
    parser.add_argument(
        "--output-dir",
        default="paper_outputs/replaybench_timing_study",
        help="Directory for timing-study outputs.",
    )
    parser.add_argument("--no-validate-files", action="store_true")
    args = parser.parse_args()

    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0")
    if args.repetitions < 3:
        raise ValueError("--repetitions must be >= 3; seven is recommended for the paper")

    config_path = Path(args.config)
    cfg = bench.load_config(config_path)
    if not args.no_validate_files:
        bench.validate_config(cfg)

    dataset_cfg = cfg["dataset"]
    benchmark_cfg = cfg["benchmark"]
    policy_cfg = cfg["policy"]

    input_csv = dataset_cfg["input_csv"]
    fractions = [float(x) for x in dataset_cfg["fractions"]]
    policies = list(benchmark_cfg.get("policy_modes", bench.DEFAULT_POLICY_ORDER))
    workers = [int(x) for x in benchmark_cfg["workers"]]
    negative_labels = {bench.normalize_label(x) for x in policy_cfg.get("negative_labels", [])}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_full = pd.read_csv(input_csv).reset_index(drop=True)
    if df_full.empty:
        raise ValueError(f"Input CSV has no rows: {input_csv}")

    bc_actions: Optional[Dict[int, int]] = None
    if "bc" in policies:
        bc_actions = bench.load_bc_reference_actions(
            bc_action_csv=policy_cfg.get("bc_action_csv", "paper_outputs/policy_first_outputs_bc.csv"),
            df_full=df_full,
            action_column=str(policy_cfg.get("bc_action_column", "action")),
            key_column=str(policy_cfg.get("bc_key_column", "utterance_id")),
        )

    df_by_fraction: Dict[float, pd.DataFrame] = {}
    for fraction in fractions:
        n = max(1, int(len(df_full) * fraction))
        df_by_fraction[fraction] = df_full.iloc[:n].copy().reset_index(drop=True)

    configurations = build_configurations(fractions, policies, workers)
    total_measured = len(configurations) * args.repetitions
    total_warmups = len(configurations) * args.warmups

    print("\n========== REPLAYBENCH-PG TIMING STUDY ==========")
    print(f"Configurations          : {len(configurations)}")
    print(f"Untimed warm-up runs    : {total_warmups}")
    print(f"Measured repetitions    : {total_measured}")
    print(f"Fixed policy seed       : {args.policy_seed}")
    print(f"Configuration order seed: {args.order_seed}")
    print(f"Output directory        : {output_dir}")
    print("=================================================\n")

    study_started = time.perf_counter()

    # Warm-up rounds. Each round uses a deterministic but different shuffle.
    for warmup_round in range(1, args.warmups + 1):
        ordered = configurations.copy()
        random.Random(args.order_seed + warmup_round).shuffle(ordered)
        print(f"[WARMUP] round {warmup_round}/{args.warmups}")
        for index, item in enumerate(ordered, start=1):
            fraction = float(item["dataset_fraction"])
            policy_mode = str(item["policy_mode"])
            worker_count = int(item["workers"])
            df = df_by_fraction[fraction]
            print(
                f"  [{index:02d}/{len(ordered)}] fraction={fraction:g}, "
                f"policy={policy_mode}, workers={worker_count}"
            )
            bench.run_replay(
                df=df,
                cfg=cfg,
                policy_mode=policy_mode,
                negative_labels=negative_labels,
                seed=args.policy_seed,
                workers=worker_count,
                bc_actions=bc_actions,
            )
            gc.collect()

    raw_rows: List[Dict[str, Any]] = []

    # Measured rounds. Re-shuffling every round reduces fixed ordering bias.
    for repetition in range(1, args.repetitions + 1):
        ordered = configurations.copy()
        random.Random(args.order_seed + 10_000 + repetition).shuffle(ordered)
        print(f"[MEASURE] repetition {repetition}/{args.repetitions}")

        for index, item in enumerate(ordered, start=1):
            fraction = float(item["dataset_fraction"])
            policy_mode = str(item["policy_mode"])
            worker_count = int(item["workers"])
            df = df_by_fraction[fraction]
            workload_name = f"fraction_{bench.sanitize_token(fraction)}"

            print(
                f"  [{index:02d}/{len(ordered)}] fraction={fraction:g}, "
                f"policy={policy_mode}, workers={worker_count}"
            )

            trace_df, summary, _ = bench.run_replay(
                df=df,
                cfg=cfg,
                policy_mode=policy_mode,
                negative_labels=negative_labels,
                seed=args.policy_seed,
                workers=worker_count,
                bc_actions=bc_actions,
            )
            stage = bench.summarize_stage_latency(trace_df)

            raw_rows.append({
                "study_arm": item["study_arm"],
                "repetition": int(repetition),
                "dataset_fraction": fraction,
                "workload_name": workload_name,
                "decision_points": int(len(df)),
                "policy_mode": policy_mode,
                "policy_seed": int(args.policy_seed),
                "workers": worker_count,
                **summary,
                **stage,
            })
            gc.collect()

        # Persist after every repetition so an interrupted study retains completed work.
        pd.DataFrame(raw_rows).to_csv(output_dir / "timing_repetitions_raw.partial.csv", index=False)

    raw_df = pd.DataFrame(raw_rows)
    summary_df = summarize_repetitions(raw_df)
    stage_df = summarize_stage_latency(raw_df)
    trace_df = build_trace_consistency(raw_df)

    raw_path = output_dir / "timing_repetitions_raw.csv"
    summary_path = output_dir / "timing_summary.csv"
    stage_path = output_dir / "timing_stage_latency_summary.csv"
    trace_path = output_dir / "timing_trace_consistency.csv"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    stage_df.to_csv(stage_path, index=False)
    trace_df.to_csv(trace_path, index=False)

    partial_path = output_dir / "timing_repetitions_raw.partial.csv"
    if partial_path.exists():
        partial_path.unlink()

    try:
        import torch
    except ImportError:
        torch = None

    environment = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "logical_cpu_count": os.cpu_count(),
        "numpy_version": package_version(np),
        "pandas_version": package_version(pd),
        "torch_version": package_version(torch),
        "torch_cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "torch_num_threads": int(torch.get_num_threads()) if torch is not None else None,
        "thread_environment": {
            name: os.environ.get(name)
            for name in [
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "PYTHONHASHSEED",
            ]
        },
    }

    environment_path = output_dir / "timing_environment.json"
    environment_path.write_text(json.dumps(environment, indent=2), encoding="utf-8")

    script_path = Path(__file__).resolve()
    benchmark_runner_path = Path(bench.__file__).resolve()
    elapsed_seconds = time.perf_counter() - study_started
    manifest = {
        "study_name": "ReplayBench-PG targeted warmed-up timing study",
        "design": {
            "workload_scaling_arm": "all fractions x all policies x one worker",
            "worker_scaling_arm": "full workload x all policies x all worker settings",
            "unique_configurations": len(configurations),
            "untimed_warmups_per_configuration": args.warmups,
            "measured_repetitions_per_configuration": args.repetitions,
            "fixed_policy_seed": args.policy_seed,
            "configuration_order_seed": args.order_seed,
            "aggregation": "median, quartiles, IQR; no inferential significance test",
        },
        "runtime": {
            "study_elapsed_seconds": elapsed_seconds,
            "warmup_runs": total_warmups,
            "measured_runs": total_measured,
        },
        "inputs": {
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path.resolve()),
            "input_csv": str(Path(input_csv).resolve()),
            "input_rows": int(len(df_full)),
            "fractions": fractions,
            "policies": policies,
            "workers": workers,
        },
        "software": {
            "timing_script": str(script_path),
            "timing_script_sha256": sha256_file(script_path),
            "benchmark_runner": str(benchmark_runner_path),
            "benchmark_runner_sha256": sha256_file(benchmark_runner_path),
        },
        "validation": {
            "all_configurations_have_expected_repetitions": bool(
                (summary_df["measured_repetitions"] == args.repetitions).all()
            ),
            "all_repetition_trace_hashes_match": bool(trace_df["all_trace_hashes_match"].all()),
            "unauthorized_invocations_total": int(raw_df["unauthorized_invocations"].sum()),
            "fault_injected_count_total": int(raw_df["fault_injected_count"].sum()),
        },
        "outputs": [
            str(raw_path),
            str(summary_path),
            str(stage_path),
            str(trace_path),
            str(environment_path),
        ],
    }
    manifest_path = output_dir / "timing_study_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n========== TIMING STUDY VALIDATION ==========")
    print(f"Raw measured rows            : {len(raw_df)}")
    print(f"Summary configurations       : {len(summary_df)}")
    print(f"Expected repetitions/config  : {args.repetitions}")
    print(f"All repetition counts valid  : {manifest['validation']['all_configurations_have_expected_repetitions']}")
    print(f"All trace hashes stable      : {manifest['validation']['all_repetition_trace_hashes_match']}")
    print(f"Unauthorized invocations     : {manifest['validation']['unauthorized_invocations_total']}")
    print(f"Injected faults              : {manifest['validation']['fault_injected_count_total']}")
    print("=============================================\n")
    print(f"[OUT] {raw_path}")
    print(f"[OUT] {summary_path}")
    print(f"[OUT] {stage_path}")
    print(f"[OUT] {trace_path}")
    print(f"[OUT] {environment_path}")
    print(f"[OUT] {manifest_path}")
    print("[DONE] ReplayBench-PG timing study complete.")


if __name__ == "__main__":
    main()
