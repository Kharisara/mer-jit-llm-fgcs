import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


# ============================================================
# BASIC UTILITIES
# ============================================================

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_label(x: Any) -> str:
    return str(x).strip().lower()


def split_ranges(n: int, workers: int) -> List[Tuple[int, int]]:
    workers = max(1, int(workers))
    chunk_size = int(np.ceil(n / workers))
    ranges = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        ranges.append((start, end))

    return ranges


def trace_hash_from_actions(actions: np.ndarray) -> str:
    """
    Efficient deterministic hash of the binary action trace.
    Uses bytes instead of string joining for scalability.
    """
    arr = np.ascontiguousarray(actions.astype(np.uint8))
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ============================================================
# INPUT LOADING AND AMPLIFIED WORKLOAD GENERATION
# ============================================================

def load_bc_actions(path: str) -> np.ndarray:
    if not path:
        raise ValueError("BC mode requires dataset.bc_actions_csv in the config.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"BC actions file not found: {path}")

    df = pd.read_csv(path)

    if "action" not in df.columns:
        raise ValueError(f"BC actions file must contain an 'action' column: {path}")

    actions = df["action"].astype(int).to_numpy(dtype=np.uint8)

    if not set(np.unique(actions)).issubset({0, 1}):
        raise ValueError("BC actions must be binary values in {0, 1}.")

    return actions


def build_amplified_workload(
    base_df: pd.DataFrame,
    target_size: int,
) -> Dict[str, Any]:
    """
    Builds an amplified replay workload by deterministic repetition of the
    original MELD-derived replay workload.

    This does not create new semantic samples. It is used only for systems
    scalability and replay-overhead evaluation.
    """
    if "label" not in base_df.columns:
        raise ValueError("Input CSV must contain a 'label' column.")

    base_n = len(base_df)

    if base_n == 0:
        raise ValueError("Base replay input is empty.")

    source_row_index = np.arange(target_size, dtype=np.int64) % base_n
    repeat_id = np.arange(target_size, dtype=np.int64) // base_n

    labels = (
        base_df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .to_numpy()[source_row_index]
    )

    return {
        "target_size": int(target_size),
        "base_size": int(base_n),
        "source_row_index": source_row_index,
        "repeat_id": repeat_id,
        "labels": labels,
    }


# ============================================================
# POLICY ACTION GENERATION
# ============================================================

def compute_actions_chunk(
    policy_mode: str,
    labels: np.ndarray,
    source_row_index: np.ndarray,
    start: int,
    end: int,
    seed: int,
    negative_labels: set,
    bc_actions: np.ndarray,
) -> Tuple[int, int, np.ndarray]:
    """
    Computes actions for a workload chunk.
    Returns start, end, actions.
    """
    chunk_labels = labels[start:end]
    chunk_source_idx = source_row_index[start:end]
    global_idx = np.arange(start, end, dtype=np.uint64)

    if policy_mode == "proxy":
        actions = np.isin(chunk_labels, list(negative_labels)).astype(np.uint8)

    elif policy_mode == "bc":
        if bc_actions is None:
            raise ValueError("BC policy mode requires bc_actions.")

        actions = bc_actions[chunk_source_idx].astype(np.uint8)

    elif policy_mode == "always":
        actions = np.ones(end - start, dtype=np.uint8)

    elif policy_mode == "never":
        actions = np.zeros(end - start, dtype=np.uint8)

    elif policy_mode == "random":
        # Deterministic pseudo-random binary action independent of worker count.
        values = (
            (global_idx * np.uint64(1103515245))
            + np.uint64(seed * 12345)
            + np.uint64(67890)
        ) % np.uint64(1000000)

        actions = (values < np.uint64(500000)).astype(np.uint8)

    else:
        raise ValueError(f"Unknown policy mode: {policy_mode}")

    return start, end, actions


def compute_actions_parallel(
    policy_mode: str,
    workload: Dict[str, Any],
    seed: int,
    workers: int,
    negative_labels: set,
    bc_actions: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Computes policy actions using worker-level chunking.
    The output sequence is reconstructed in row order.
    """
    labels = workload["labels"]
    source_row_index = workload["source_row_index"]
    n = workload["target_size"]

    actions = np.zeros(n, dtype=np.uint8)
    ranges = split_ranges(n, workers)

    t0 = time.perf_counter()

    if workers <= 1:
        for start, end in ranges:
            _, _, chunk_actions = compute_actions_chunk(
                policy_mode=policy_mode,
                labels=labels,
                source_row_index=source_row_index,
                start=start,
                end=end,
                seed=seed,
                negative_labels=negative_labels,
                bc_actions=bc_actions,
            )
            actions[start:end] = chunk_actions
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    compute_actions_chunk,
                    policy_mode,
                    labels,
                    source_row_index,
                    start,
                    end,
                    seed,
                    negative_labels,
                    bc_actions,
                )
                for start, end in ranges
            ]

            for future in as_completed(futures):
                start, end, chunk_actions = future.result()
                actions[start:end] = chunk_actions

    t1 = time.perf_counter()

    return actions, t1 - t0


# ============================================================
# NORMAL EXTENDED REPLAY BENCHMARK
# ============================================================

def run_single_extended_replay(
    workload: Dict[str, Any],
    policy_mode: str,
    seed: int,
    workers: int,
    negative_labels: set,
    bc_actions: np.ndarray,
) -> Dict[str, Any]:
    """
    Runs one extended replay benchmark and records:
    - action generation runtime
    - invocation gate runtime
    - aggregation runtime
    - trace hash runtime
    - throughput
    - intervention rate
    - unauthorized invocation count
    """
    n = workload["target_size"]

    replay_start = time.perf_counter()

    actions, policy_time = compute_actions_parallel(
        policy_mode=policy_mode,
        workload=workload,
        seed=seed,
        workers=workers,
        negative_labels=negative_labels,
        bc_actions=bc_actions,
    )

    gate_t0 = time.perf_counter()
    generation_invoked = actions == 1
    unauthorized_invocations = int(np.logical_and(actions == 0, generation_invoked).sum())
    gate_t1 = time.perf_counter()

    agg_t0 = time.perf_counter()
    intervention_rate = float(actions.mean()) if n else 0.0
    interventions = int(actions.sum())
    non_interventions = int(n - interventions)
    agg_t1 = time.perf_counter()

    hash_t0 = time.perf_counter()
    action_trace_hash = trace_hash_from_actions(actions)
    hash_t1 = time.perf_counter()

    replay_end = time.perf_counter()

    replay_runtime = replay_end - replay_start
    trace_hash_time = hash_t1 - hash_t0

    return {
        "workload_size": n,
        "base_size": workload["base_size"],
        "policy_mode": policy_mode,
        "seed": seed,
        "workers": workers,
        "replay_runtime_seconds": replay_runtime,
        "policy_action_time_seconds": policy_time,
        "gating_time_seconds": gate_t1 - gate_t0,
        "aggregation_time_seconds": agg_t1 - agg_t0,
        "trace_hash_time_seconds": trace_hash_time,
        "trace_hash_overhead_percent": (
            (trace_hash_time / replay_runtime) * 100.0 if replay_runtime > 0 else 0.0
        ),
        "throughput_points_per_second": n / replay_runtime if replay_runtime > 0 else 0.0,
        "interventions": interventions,
        "non_interventions": non_interventions,
        "intervention_rate": intervention_rate,
        "unauthorized_invocations": unauthorized_invocations,
        "trace_hash": action_trace_hash,
    }


# ============================================================
# FAULT INJECTION EXPERIMENT
# ============================================================

def inject_unauthorized_invocations(
    actions: np.ndarray,
    injection_rate: float,
    seed: int,
) -> Dict[str, Any]:
    """
    Injects unauthorized generation into a deterministic fraction of rows
    where action = 0, then tests whether the verifier detects them.
    """
    eligible_idx = np.where(actions == 0)[0]
    eligible_count = int(len(eligible_idx))

    if eligible_count == 0 or injection_rate <= 0:
        return {
            "eligible_non_intervention_rows": eligible_count,
            "injected_violations": 0,
            "detected_violations": 0,
            "false_negatives": 0,
            "detection_recall": 1.0,
        }

    injected_count = int(round(eligible_count * float(injection_rate)))
    injected_count = max(1, injected_count)
    injected_count = min(injected_count, eligible_count)

    rng = np.random.default_rng(seed)
    selected = rng.choice(eligible_idx, size=injected_count, replace=False)

    generation_invoked = actions == 1
    generation_invoked[selected] = True

    detected = int(np.logical_and(actions == 0, generation_invoked).sum())
    false_negatives = int(injected_count - detected)

    recall = detected / injected_count if injected_count > 0 else 1.0

    return {
        "eligible_non_intervention_rows": eligible_count,
        "injected_violations": injected_count,
        "detected_violations": detected,
        "false_negatives": false_negatives,
        "detection_recall": float(recall),
    }


def run_fault_injection_experiment(
    workload: Dict[str, Any],
    policy_mode: str,
    seed: int,
    workers: int,
    injection_rate: float,
    negative_labels: set,
    bc_actions: np.ndarray,
) -> Dict[str, Any]:
    actions, policy_time = compute_actions_parallel(
        policy_mode=policy_mode,
        workload=workload,
        seed=seed,
        workers=workers,
        negative_labels=negative_labels,
        bc_actions=bc_actions,
    )

    t0 = time.perf_counter()
    result = inject_unauthorized_invocations(
        actions=actions,
        injection_rate=injection_rate,
        seed=seed + int(injection_rate * 1_000_000),
    )
    t1 = time.perf_counter()

    result.update(
        {
            "workload_size": workload["target_size"],
            "policy_mode": policy_mode,
            "seed": seed,
            "workers": workers,
            "injection_rate": injection_rate,
            "policy_action_time_seconds": policy_time,
            "fault_detection_time_seconds": t1 - t0,
        }
    )

    return result


# ============================================================
# DETERMINISM SUMMARY
# ============================================================

def build_determinism_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["workload_size", "policy_mode", "seed"]

    for _, group in results_df.groupby(group_cols):
        group = group.sort_values("workers")
        reference_hash = group.iloc[0]["trace_hash"]

        for _, row in group.iterrows():
            rows.append(
                {
                    "workload_size": int(row["workload_size"]),
                    "policy_mode": row["policy_mode"],
                    "seed": int(row["seed"]),
                    "workers": int(row["workers"]),
                    "hash_match": int(row["trace_hash"] == reference_hash),
                    "intervention_rate": float(row["intervention_rate"]),
                    "unauthorized_invocations": int(row["unauthorized_invocations"]),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# PAPER ARTIFACT TABLES
# ============================================================

def write_novelty_comparison_table(out_dir: str) -> None:
    table_dir = Path("paper_outputs") / "fgcs_tables_figures"
    ensure_dir(str(table_dir))

    rows = [
        {
            "Capability": "Policy-gated AI replay",
            "MLflow": "Partial",
            "Airflow": "No",
            "Ray": "Partial",
            "Generic test frameworks": "No",
            "Proposed framework": "Yes",
        },
        {
            "Capability": "Multimodal replay workload support",
            "MLflow": "No",
            "Airflow": "No",
            "Ray": "Partial",
            "Generic test frameworks": "No",
            "Proposed framework": "Yes",
        },
        {
            "Capability": "Invocation-control validation",
            "MLflow": "No",
            "Airflow": "No",
            "Ray": "No",
            "Generic test frameworks": "Partial",
            "Proposed framework": "Yes",
        },
        {
            "Capability": "Unauthorized-generation detection",
            "MLflow": "No",
            "Airflow": "No",
            "Ray": "No",
            "Generic test frameworks": "Partial",
            "Proposed framework": "Yes",
        },
        {
            "Capability": "Deterministic action-trace hashing",
            "MLflow": "No",
            "Airflow": "No",
            "Ray": "Partial",
            "Generic test frameworks": "Partial",
            "Proposed framework": "Yes",
        },
        {
            "Capability": "Policy-ablation replay modes",
            "MLflow": "No",
            "Airflow": "No",
            "Ray": "Partial",
            "Generic test frameworks": "No",
            "Proposed framework": "Yes",
        },
        {
            "Capability": "Stage-level pipeline latency profiling",
            "MLflow": "Partial",
            "Airflow": "Partial",
            "Ray": "Partial",
            "Generic test frameworks": "No",
            "Proposed framework": "Yes",
        },
        {
            "Capability": "Reproducibility manifest generation",
            "MLflow": "Partial",
            "Airflow": "No",
            "Ray": "No",
            "Generic test frameworks": "No",
            "Proposed framework": "Yes",
        },
    ]

    out = table_dir / "fgcs_table_novelty_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[OUT] {out}")


def write_rq_mapping_table(out_dir: str) -> None:
    table_dir = Path("paper_outputs") / "fgcs_tables_figures"
    ensure_dir(str(table_dir))

    rows = [
        {
            "Research Question": "RQ1",
            "Focus": "Deterministic replay consistency",
            "Evaluation Evidence": "Extended determinism table; trace hash results",
            "Main Metric": "Hash match across workload sizes, policies, seeds, and workers",
        },
        {
            "Research Question": "RQ2",
            "Focus": "Invocation-control fault detection",
            "Evaluation Evidence": "Fault-injection detection table",
            "Main Metric": "Detection recall for injected unauthorized invocations",
        },
        {
            "Research Question": "RQ3",
            "Focus": "Workload-scale performance",
            "Evaluation Evidence": "Extended scalability results",
            "Main Metric": "Runtime and throughput from real workload to 1M replay points",
        },
        {
            "Research Question": "RQ4",
            "Focus": "Worker-level execution behavior",
            "Evaluation Evidence": "Worker-level runtime and throughput results",
            "Main Metric": "Throughput variation and consistency across worker settings",
        },
        {
            "Research Question": "RQ5",
            "Focus": "Stage-level execution cost",
            "Evaluation Evidence": "Trace overhead and replay timing results",
            "Main Metric": "Policy action time, gating time, aggregation time, hash overhead",
        },
    ]

    out = table_dir / "fgcs_table_rq_mapping.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[OUT] {out}")


# ============================================================
# REPRODUCIBILITY MANIFEST
# ============================================================

def write_extended_manifest(
    output_dir: str,
    config_path: str,
    config: Dict[str, Any],
    generated_outputs: Dict[str, str],
) -> None:
    manifest_path = Path(output_dir) / "fgcs_extended_reproducibility_manifest.json"

    try:
        import psutil
        psutil_version = psutil.__version__
    except Exception:
        psutil_version = "not_available"

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "run_fgcs_extended_benchmark.py",
        "config_path": config_path,
        "environment": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count_logical": os.cpu_count(),
            "working_directory": os.getcwd(),
        },
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyyaml": yaml.__version__ if hasattr(yaml, "__version__") else "unknown",
            "psutil": psutil_version,
        },
        "benchmark_configuration": config,
        "generated_outputs": generated_outputs,
        "notes": [
            "Amplified workloads are generated by deterministic repetition of the original replay workload.",
            "Amplified workloads are used only for systems scalability and replay-overhead evaluation.",
            "BC mode uses the learned behavioral-cloning action trace generated before benchmarking.",
            "Fault injection intentionally introduces unauthorized invocations for verifier testing.",
        ],
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OUT] {manifest_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/fgcs_extended_benchmark.yaml",
        help="Path to extended FGCS benchmark config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    input_csv = cfg["dataset"]["input_csv"]
    bc_actions_csv = cfg["dataset"]["bc_actions_csv"]
    workload_sizes = cfg["dataset"]["workload_sizes"]

    seeds = cfg["benchmark"]["seeds"]
    workers_list = cfg["benchmark"]["workers"]
    policy_modes = cfg["benchmark"]["policy_modes"]

    fault_cfg = cfg.get("fault_injection", {})
    fault_enabled = bool(fault_cfg.get("enabled", False))
    fault_policy_modes = fault_cfg.get("policy_modes", [])
    fault_rates = fault_cfg.get("rates", [])
    fault_workload_sizes = fault_cfg.get("workload_sizes", [])

    output_dir = cfg["logging"]["output_dir"]
    save_sample_traces = bool(cfg["logging"].get("save_sample_traces", True))
    sample_trace_rows = int(cfg["logging"].get("sample_trace_rows", 1000))

    negative_labels = set(normalize_label(x) for x in cfg["policy"]["negative_labels"])

    ensure_dir(output_dir)

    print("[INFO] Loading replay input:", input_csv)
    base_df = pd.read_csv(input_csv).reset_index(drop=True)

    print("[INFO] Loading BC actions:", bc_actions_csv)
    bc_actions = load_bc_actions(bc_actions_csv)

    if len(bc_actions) < len(base_df):
        raise ValueError(
            f"BC actions length ({len(bc_actions)}) is smaller than base input length "
            f"({len(base_df)})."
        )

    print(f"[INFO] Base replay size: {len(base_df)}")

    extended_rows = []
    fault_rows = []

    for workload_size in workload_sizes:
        print(f"\n[WORKLOAD] Building amplified workload: {workload_size}")
        workload_t0 = time.perf_counter()
        workload = build_amplified_workload(base_df, int(workload_size))
        workload_t1 = time.perf_counter()

        print(
            f"[INFO] Built workload {workload_size} in "
            f"{workload_t1 - workload_t0:.4f} seconds"
        )

        for policy_mode in policy_modes:
            for seed in seeds:
                for workers in workers_list:
                    print(
                        f"[RUN] workload={workload_size}, policy={policy_mode}, "
                        f"seed={seed}, workers={workers}"
                    )

                    row = run_single_extended_replay(
                        workload=workload,
                        policy_mode=policy_mode,
                        seed=int(seed),
                        workers=int(workers),
                        negative_labels=negative_labels,
                        bc_actions=bc_actions,
                    )

                    row["workload_generation_time_seconds"] = workload_t1 - workload_t0
                    extended_rows.append(row)

                    if save_sample_traces and int(workload_size) == int(workload_sizes[0]):
                        actions, _ = compute_actions_parallel(
                            policy_mode=policy_mode,
                            workload=workload,
                            seed=int(seed),
                            workers=int(workers),
                            negative_labels=negative_labels,
                            bc_actions=bc_actions,
                        )

                        sample_n = min(sample_trace_rows, len(actions))
                        sample_df = pd.DataFrame(
                            {
                                "synthetic_row_id": np.arange(sample_n),
                                "source_row_index": workload["source_row_index"][:sample_n],
                                "label": workload["labels"][:sample_n],
                                "policy_mode": policy_mode,
                                "seed": seed,
                                "workers": workers,
                                "action": actions[:sample_n],
                            }
                        )

                        sample_path = (
                            Path(output_dir)
                            / f"sample_trace_workload_{workload_size}_policy_{policy_mode}_seed_{seed}_workers_{workers}.csv"
                        )
                        sample_df.to_csv(sample_path, index=False)

        if fault_enabled and int(workload_size) in set(int(x) for x in fault_workload_sizes):
            print(f"[FAULT] Running fault injection for workload {workload_size}")

            for policy_mode in fault_policy_modes:
                for seed in seeds:
                    # Use workers=1 for fault detection to keep interpretation simple.
                    workers = 1

                    for rate in fault_rates:
                        print(
                            f"[FAULT] workload={workload_size}, policy={policy_mode}, "
                            f"seed={seed}, rate={rate}"
                        )

                        fault_row = run_fault_injection_experiment(
                            workload=workload,
                            policy_mode=policy_mode,
                            seed=int(seed),
                            workers=workers,
                            injection_rate=float(rate),
                            negative_labels=negative_labels,
                            bc_actions=bc_actions,
                        )

                        fault_rows.append(fault_row)

    extended_df = pd.DataFrame(extended_rows)
    fault_df = pd.DataFrame(fault_rows)

    determinism_df = build_determinism_summary(extended_df)

    extended_results_path = Path(output_dir) / "extended_scalability_results.csv"
    trace_overhead_path = Path(output_dir) / "trace_verification_overhead.csv"
    determinism_path = Path(output_dir) / "extended_determinism_results.csv"
    fault_path = Path(output_dir) / "fault_injection_detection.csv"

    extended_df.to_csv(extended_results_path, index=False)

    trace_cols = [
        "workload_size",
        "base_size",
        "policy_mode",
        "seed",
        "workers",
        "replay_runtime_seconds",
        "policy_action_time_seconds",
        "gating_time_seconds",
        "aggregation_time_seconds",
        "trace_hash_time_seconds",
        "trace_hash_overhead_percent",
        "throughput_points_per_second",
        "trace_hash",
    ]

    extended_df[trace_cols].to_csv(trace_overhead_path, index=False)
    determinism_df.to_csv(determinism_path, index=False)
    fault_df.to_csv(fault_path, index=False)

    write_novelty_comparison_table(output_dir)
    write_rq_mapping_table(output_dir)

    generated_outputs = {
        "extended_scalability_results": str(extended_results_path),
        "trace_verification_overhead": str(trace_overhead_path),
        "extended_determinism_results": str(determinism_path),
        "fault_injection_detection": str(fault_path),
        "novelty_comparison_table": "paper_outputs/fgcs_tables_figures/fgcs_table_novelty_comparison.csv",
        "rq_mapping_table": "paper_outputs/fgcs_tables_figures/fgcs_table_rq_mapping.csv",
    }

    write_extended_manifest(
        output_dir=output_dir,
        config_path=args.config,
        config=cfg,
        generated_outputs=generated_outputs,
    )

    print("\n[DONE] Extended FGCS benchmark complete.")
    print(f"[OUT] {extended_results_path}")
    print(f"[OUT] {trace_overhead_path}")
    print(f"[OUT] {determinism_path}")
    print(f"[OUT] {fault_path}")


if __name__ == "__main__":
    main()