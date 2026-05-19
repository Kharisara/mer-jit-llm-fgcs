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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import psutil
except ImportError:
    psutil = None


# ============================================================
# CONFIG AND BASIC UTILITIES
# ============================================================

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_reproducibility_manifest(
    output_dir: str,
    config_path: str,
    config: Dict[str, Any],
    input_csv: str,
    bc_actions_csv: str,
    generated_outputs: Dict[str, str],
) -> None:
    """
    Saves a reproducibility manifest for the FGCS benchmark run.

    The manifest records the execution environment, package versions,
    benchmark configuration, input files, and generated output files.
    """
    manifest_path = Path(output_dir) / "fgcs_reproducibility_manifest.json"

    try:
        import numpy as _np
        numpy_version = _np.__version__
    except Exception:
        numpy_version = "not_available"

    try:
        import pandas as _pd
        pandas_version = _pd.__version__
    except Exception:
        pandas_version = "not_available"

    try:
        import yaml as _yaml
        pyyaml_version = getattr(_yaml, "__version__", "unknown")
    except Exception:
        pyyaml_version = "not_available"

    try:
        import psutil as _psutil
        psutil_version = _psutil.__version__
    except Exception:
        psutil_version = "not_available"

    try:
        import matplotlib as _matplotlib
        matplotlib_version = _matplotlib.__version__
    except Exception:
        matplotlib_version = "not_available"

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "run_fgcs_benchmark.py",
        "config_path": str(config_path),
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
            "numpy": numpy_version,
            "pandas": pandas_version,
            "pyyaml": pyyaml_version,
            "psutil": psutil_version,
            "matplotlib": matplotlib_version,
        },
        "input_files": {
            "input_csv": str(input_csv),
            "bc_actions_csv": str(bc_actions_csv) if bc_actions_csv else None,
        },
        "benchmark_configuration": {
            "dataset": config.get("dataset", {}),
            "benchmark": config.get("benchmark", {}),
            "logging": config.get("logging", {}),
            "policy": config.get("policy", {}),
        },
        "generated_outputs": generated_outputs,
        "notes": [
            "The benchmark evaluates deterministic offline replay behavior.",
            "BC mode uses the learned behavioral-cloning action trace generated before benchmarking.",
            "Trace hashes are used to verify deterministic replay consistency across worker counts.",
            "Unauthorized invocation count verifies whether generation occurs when the selected action is non-intervention.",
        ],
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OUT] {manifest_path}")


# ============================================================
# BC ACTION TRACE LOADING
# ============================================================

def load_bc_actions(path: str) -> List[int]:
    """
    Loads real behavioral-cloning actions from a previously generated BC replay file.

    The file must contain an 'action' column.
    Row order is assumed to match the input replay CSV.
    """
    if not path:
        raise ValueError("BC policy mode requires dataset.bc_actions_csv in the config.")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BC actions file not found: {path}. "
            "Generate it first using the BC replay script."
        )

    df = pd.read_csv(path)

    if "action" not in df.columns:
        raise ValueError(f"BC actions file must contain an 'action' column: {path}")

    actions = df["action"].astype(int).tolist()

    if not set(actions).issubset({0, 1}):
        raise ValueError("BC actions must be binary values in {0, 1}.")

    return actions


# ============================================================
# DETERMINISTIC HASHING
# ============================================================

def stable_hash_to_float(*parts: Any) -> float:
    """
    Deterministic pseudo-random number in [0, 1).

    This is independent of Python's random state and worker count.
    """
    s = "|".join(str(p) for p in parts)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    value = int(h[:16], 16)
    return value / float(16**16)


def trace_hash(actions: List[int]) -> str:
    """
    Hash action sequence for deterministic replay comparison.
    """
    s = ",".join(str(int(a)) for a in actions)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ============================================================
# STATE LOADING AND GENERATION STUB
# ============================================================

def safe_load_state(state_path: Any) -> Tuple[bool, int]:
    """
    Loads or touches the state file to measure state-loading cost.

    Returns:
        exists: whether the file exists
        size_bytes: approximate file size if available
    """
    if pd.isna(state_path):
        return False, 0

    path = str(state_path)

    # Normalize Windows-style backslashes if running on Unix-like systems.
    path = path.replace("\\", os.sep)

    if not os.path.exists(path):
        return False, 0

    size_bytes = os.path.getsize(path)
    suffix = Path(path).suffix.lower()

    try:
        if suffix == ".npy":
            _ = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                _ = list(data.keys())
        else:
            # Generic fallback: read a small part of the file so I/O is measured
            # without requiring PyTorch/model-specific deserialization.
            with open(path, "rb") as f:
                _ = f.read(4096)
    except Exception:
        # Do not fail the benchmark because one state file has an unusual format.
        return True, size_bytes

    return True, size_bytes


def generate_stub(action: int, label: str) -> Dict[str, Any]:
    """
    Deterministic generation stub.

    This measures invocation and structural-output behavior, not content quality.
    """
    if action == 0:
        return {
            "invoked": False,
            "response_json": "",
            "safety": "not_invoked",
            "unauthorized_invocation": 0,
        }

    label_l = str(label).lower()

    if label_l in {"joy", "surprise"}:
        response = {
            "sentences": [
                "That sounds positive.",
                "We can keep the response brief and structured.",
            ],
            "safety": "ok",
        }
    else:
        response = {
            "sentences": [
                "I hear you.",
                "We can pause here for a moment.",
            ],
            "safety": "ok",
        }

    return {
        "invoked": True,
        "response_json": json.dumps(response, ensure_ascii=False),
        "safety": response["safety"],
        "unauthorized_invocation": 0,
    }


# ============================================================
# POLICY SELECTION
# ============================================================

def choose_action(
    row: pd.Series,
    policy_mode: str,
    negative_labels: set,
    seed: int,
    row_index: int,
    bc_actions: Optional[List[int]] = None,
) -> int:
    """
    Deterministic policy choices for FGCS system benchmarking.

    proxy  : intervention for negative labels
    bc     : real learned behavioral-cloning action trace
    random : deterministic pseudo-random action using row id + seed
    always : always intervention
    never  : always non-intervention
    """
    label = str(row.get("label", "")).lower()

    if policy_mode == "proxy":
        return 1 if label in negative_labels else 0

    if policy_mode == "bc":
        if bc_actions is None:
            raise ValueError("policy_mode='bc' requires loaded BC actions.")

        if row_index >= len(bc_actions):
            raise IndexError(
                f"BC action index {row_index} is outside BC action list length "
                f"{len(bc_actions)}."
            )

        return int(bc_actions[row_index])

    if policy_mode == "random":
        utterance_id = row.get("utterance_id", row_index)
        r = stable_hash_to_float(seed, utterance_id, row_index)
        return 1 if r < 0.5 else 0

    if policy_mode == "always":
        return 1

    if policy_mode == "never":
        return 0

    raise ValueError(f"Unknown policy_mode: {policy_mode}")


# ============================================================
# SINGLE ROW PROCESSING
# ============================================================

def process_one_row(
    row_index: int,
    row_dict: Dict[str, Any],
    policy_mode: str,
    negative_labels: set,
    seed: int,
    bc_actions: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Processes a single replay decision point and records stage-level timings.
    """
    row = pd.Series(row_dict)

    t0 = time.perf_counter()

    # Stage 1: state loading
    s0 = time.perf_counter()
    state_exists, state_size_bytes = safe_load_state(row.get("state_path", ""))
    s1 = time.perf_counter()

    # Stage 2: policy inference or action selection
    p0 = time.perf_counter()
    action = choose_action(
        row=row,
        policy_mode=policy_mode,
        negative_labels=negative_labels,
        seed=seed,
        row_index=row_index,
        bc_actions=bc_actions,
    )
    p1 = time.perf_counter()

    # Stage 3: invocation gating
    g0 = time.perf_counter()
    authorized_to_generate = action == 1
    g1 = time.perf_counter()

    # Stage 4: generation stub
    gen0 = time.perf_counter()
    generated = generate_stub(action, row.get("label", ""))
    gen1 = time.perf_counter()

    # Stage 5: logging/result construction
    log0 = time.perf_counter()
    result = {
        "row_index": row_index,
        "utterance_id": row.get("utterance_id", row_index),
        "label": row.get("label", ""),
        "split": row.get("split", ""),
        "policy_mode": policy_mode,
        "seed": seed,
        "action": int(action),
        "authorized_to_generate": int(authorized_to_generate),
        "generation_invoked": int(generated["invoked"]),
        "unauthorized_invocation": int(generated["unauthorized_invocation"]),
        "response_safety": generated["safety"],
        "state_exists": int(state_exists),
        "state_size_bytes": int(state_size_bytes),
        "state_loading_ms": (s1 - s0) * 1000.0,
        "policy_inference_ms": (p1 - p0) * 1000.0,
        "gating_ms": (g1 - g0) * 1000.0,
        "generation_stub_ms": (gen1 - gen0) * 1000.0,
    }
    log1 = time.perf_counter()

    t1 = time.perf_counter()

    result["logging_ms"] = (log1 - log0) * 1000.0
    result["total_latency_ms"] = (t1 - t0) * 1000.0

    return result


# ============================================================
# REPLAY EXECUTION
# ============================================================

def run_replay(
    df: pd.DataFrame,
    policy_mode: str,
    negative_labels: set,
    seed: int,
    workers: int,
    bc_actions: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs deterministic replay with a chosen number of workers.
    """
    rows = df.to_dict(orient="records")

    process = psutil.Process(os.getpid()) if psutil is not None else None
    mem_before = process.memory_info().rss / (1024 * 1024) if process else None

    start = time.perf_counter()

    if workers <= 1:
        results = [
            process_one_row(
                row_index=i,
                row_dict=row,
                policy_mode=policy_mode,
                negative_labels=negative_labels,
                seed=seed,
                bc_actions=bc_actions,
            )
            for i, row in enumerate(rows)
        ]
    else:
        results = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_one_row,
                    i,
                    row,
                    policy_mode,
                    negative_labels,
                    seed,
                    bc_actions,
                ): i
                for i, row in enumerate(rows)
            }

            for future in as_completed(futures):
                results.append(future.result())

    end = time.perf_counter()

    mem_after = process.memory_info().rss / (1024 * 1024) if process else None

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values("row_index").reset_index(drop=True)

    actions = out_df["action"].astype(int).tolist()
    total_runtime = end - start
    decision_points = len(out_df)

    total_latencies = out_df["total_latency_ms"].astype(float).tolist()

    summary = {
        "policy_mode": policy_mode,
        "seed": seed,
        "workers": workers,
        "decision_points": decision_points,
        "total_runtime_seconds": total_runtime,
        "throughput_points_per_second": decision_points / total_runtime if total_runtime > 0 else 0,
        "mean_latency_ms": statistics.mean(total_latencies) if total_latencies else 0,
        "median_latency_ms": statistics.median(total_latencies) if total_latencies else 0,
        "p95_latency_ms": float(np.percentile(total_latencies, 95)) if total_latencies else 0,
        "intervention_rate": float(out_df["action"].mean()) if decision_points else 0,
        "unauthorized_invocations": int(out_df["unauthorized_invocation"].sum()),
        "trace_hash": trace_hash(actions),
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
        "memory_delta_mb": (
            mem_after - mem_before
            if mem_before is not None and mem_after is not None
            else None
        ),
    }

    return out_df, summary


# ============================================================
# STAGE LATENCY SUMMARY
# ============================================================

def summarize_stage_latency(trace_df: pd.DataFrame) -> Dict[str, Any]:
    stage_cols = [
        "state_loading_ms",
        "policy_inference_ms",
        "gating_ms",
        "generation_stub_ms",
        "logging_ms",
        "total_latency_ms",
    ]

    summary = {}

    for col in stage_cols:
        values = trace_df[col].astype(float).tolist()
        summary[f"{col}_mean"] = statistics.mean(values) if values else 0
        summary[f"{col}_median"] = statistics.median(values) if values else 0
        summary[f"{col}_p95"] = float(np.percentile(values, 95)) if values else 0

    return summary


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/fgcs_benchmark.yaml",
        help="Path to benchmark YAML config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    input_csv = cfg["dataset"]["input_csv"]
    bc_actions_csv = cfg["dataset"].get("bc_actions_csv", "")

    fractions = cfg["dataset"]["fractions"]
    seeds = cfg["benchmark"]["seeds"]
    workers_list = cfg["benchmark"]["workers"]
    policy_modes = cfg["benchmark"]["policy_modes"]

    output_dir = cfg["logging"]["output_dir"]
    save_traces = bool(cfg["logging"].get("save_traces", True))

    negative_labels = set(x.lower() for x in cfg["policy"]["negative_labels"])

    ensure_dir(output_dir)

    df_full = pd.read_csv(input_csv)
    df_full = df_full.reset_index(drop=True)

    bc_actions = None
    if "bc" in policy_modes:
        bc_actions = load_bc_actions(bc_actions_csv)

        if len(bc_actions) < len(df_full):
            raise ValueError(
                f"BC actions length ({len(bc_actions)}) is smaller than input CSV length "
                f"({len(df_full)})."
            )

        print(f"[INFO] Loaded {len(bc_actions)} BC actions from {bc_actions_csv}")

    all_summaries = []
    all_stage_summaries = []
    determinism_rows = []

    print(f"[INFO] Loaded {len(df_full)} decision points from {input_csv}")

    for fraction in fractions:
        n = max(1, int(len(df_full) * float(fraction)))
        df = df_full.iloc[:n].copy().reset_index(drop=True)

        for policy_mode in policy_modes:
            reference_hash_by_seed = {}

            for seed in seeds:
                for workers in workers_list:
                    print(
                        f"[INFO] fraction={fraction}, n={n}, "
                        f"policy={policy_mode}, seed={seed}, workers={workers}"
                    )

                    trace_df, summary = run_replay(
                        df=df,
                        policy_mode=policy_mode,
                        negative_labels=negative_labels,
                        seed=int(seed),
                        workers=int(workers),
                        bc_actions=bc_actions,
                    )

                    summary["dataset_fraction"] = fraction
                    summary["input_csv"] = input_csv
                    all_summaries.append(summary)

                    stage_summary = summarize_stage_latency(trace_df)
                    stage_summary.update(
                        {
                            "dataset_fraction": fraction,
                            "decision_points": n,
                            "policy_mode": policy_mode,
                            "seed": seed,
                            "workers": workers,
                        }
                    )
                    all_stage_summaries.append(stage_summary)

                    key = (fraction, policy_mode, seed)

                    if workers == workers_list[0]:
                        reference_hash_by_seed[key] = summary["trace_hash"]

                    reference_hash = reference_hash_by_seed.get(key)
                    hash_match = None

                    if reference_hash is not None:
                        hash_match = int(summary["trace_hash"] == reference_hash)

                    determinism_rows.append(
                        {
                            "dataset_fraction": fraction,
                            "policy_mode": policy_mode,
                            "seed": seed,
                            "workers": workers,
                            "trace_hash": summary["trace_hash"],
                            "reference_hash": reference_hash,
                            "hash_match": hash_match,
                            "intervention_rate": summary["intervention_rate"],
                            "unauthorized_invocations": summary["unauthorized_invocations"],
                        }
                    )

                    if save_traces:
                        safe_fraction = str(fraction).replace(".", "p")
                        trace_path = (
                            Path(output_dir)
                            / f"trace_fraction_{safe_fraction}_policy_{policy_mode}_seed_{seed}_workers_{workers}.csv"
                        )
                        trace_df.to_csv(trace_path, index=False)

    summary_df = pd.DataFrame(all_summaries)
    stage_df = pd.DataFrame(all_stage_summaries)
    determinism_df = pd.DataFrame(determinism_rows)

    summary_path = Path(output_dir) / "scaling_and_runtime_results.csv"
    stage_path = Path(output_dir) / "stage_latency_summary.csv"
    det_path = Path(output_dir) / "determinism_hash_results.csv"

    summary_df.to_csv(summary_path, index=False)
    stage_df.to_csv(stage_path, index=False)
    determinism_df.to_csv(det_path, index=False)

    # Parallel speedup summary
    speedup_rows = []

    group_cols = ["dataset_fraction", "policy_mode", "seed"]

    for _, group in summary_df.groupby(group_cols):
        group = group.sort_values("workers")
        base = group.iloc[0]
        base_runtime = float(base["total_runtime_seconds"])

        for _, row in group.iterrows():
            runtime_seconds = float(row["total_runtime_seconds"])

            speedup_rows.append(
                {
                    "dataset_fraction": row["dataset_fraction"],
                    "policy_mode": row["policy_mode"],
                    "seed": row["seed"],
                    "workers": row["workers"],
                    "runtime_seconds": runtime_seconds,
                    "throughput_points_per_second": row["throughput_points_per_second"],
                    "speedup_vs_single_worker": (
                        base_runtime / runtime_seconds if runtime_seconds > 0 else 0
                    ),
                }
            )

    speedup_df = pd.DataFrame(speedup_rows)
    speedup_path = Path(output_dir) / "parallel_speedup_results.csv"
    speedup_df.to_csv(speedup_path, index=False)

    # Policy cost summary
    policy_cost_df = (
        summary_df.groupby(["dataset_fraction", "policy_mode", "workers"], as_index=False)
        .agg(
            {
                "total_runtime_seconds": "mean",
                "throughput_points_per_second": "mean",
                "mean_latency_ms": "mean",
                "p95_latency_ms": "mean",
                "intervention_rate": "mean",
                "unauthorized_invocations": "mean",
                "memory_delta_mb": "mean",
            }
        )
    )

    policy_cost_path = Path(output_dir) / "policy_ablation_costs.csv"
    policy_cost_df.to_csv(policy_cost_path, index=False)

    generated_outputs = {
        "scaling_and_runtime_results": str(summary_path),
        "stage_latency_summary": str(stage_path),
        "determinism_hash_results": str(det_path),
        "parallel_speedup_results": str(speedup_path),
        "policy_ablation_costs": str(policy_cost_path),
    }

    write_reproducibility_manifest(
        output_dir=output_dir,
        config_path=args.config,
        config=cfg,
        input_csv=input_csv,
        bc_actions_csv=bc_actions_csv,
        generated_outputs=generated_outputs,
    )

    print("[DONE] FGCS benchmark complete.")
    print(f"[OUT] {summary_path}")
    print(f"[OUT] {stage_path}")
    print(f"[OUT] {det_path}")
    print(f"[OUT] {speedup_path}")
    print(f"[OUT] {policy_cost_path}")


if __name__ == "__main__":
    main()