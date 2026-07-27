#!/usr/bin/env python3
"""
Reduced empirical Ray comparison for ReplayBench-PG.

Purpose
-------
This script evaluates a general-purpose Ray execution substrate using the same
policy-selection, action-flip, authorization/execution, canonical hashing, and
row-count validation logic used by run_fgcs_extended_benchmark.py.

The intended comparison is deliberately narrow:
    - clean deterministic replay
    - action-flip fault detection
    - dropped-row fault detection
    - ordered reconstruction after Ray task completion
    - authorization/execution logging
    - comparison with required ReplayBench-PG clean reference hashes

It does NOT claim that Ray cannot implement these capabilities. Instead, it
measures which validation functions remain application-level logic when Ray is
used as the task-execution substrate.

Recommended reduced design
--------------------------
    1 workload fraction
    x 3 policies
    x 3 seeds
    x 2 worker settings
    x 3 modes (clean, action_flip, dropped_row)
    = 54 conditions

Expected YAML structure
-----------------------
The script accepts either a standalone configuration or a small override that
inherits the existing ReplayBench-PG YAML through ``base_config``.

Example:

base_config: configs/fgcs_extended_benchmark.yaml

dataset:
  workload_fraction: 1.0

benchmark:
  policy_modes:
    - risk_proxy
    - random
    - never
  seeds:
    - 1
    - 2
    - 3
  workers:
    - 1
    - 4
  fault_modes:
    - clean
    - action_flip
    - dropped_row
  ray_chunk_size: 256

fault_injection:
  action_flip_probability: 0.01
  dropped_row_probability: 0.01

reference:
  determinism_csv: paper_outputs/fgcs_extended_benchmark/determinism_hash_results.csv
  require_reference_results: true

execution_semantics:
  task_retry_enabled: false
  task_failure_policy: fail_fast
  duplicate_delivery_supported: false

logging:
  output_dir: paper_outputs/ray_comparison
  save_traces: true

Outputs
-------
    ray_comparison_raw.csv
    ray_comparison_per_run.csv
    ray_comparison_summary.csv
    ray_validation_component_inventory.csv
    ray_comparison_manifest.json
    ray_comparison_effective_config.json
    traces/trace_*.csv                     when save_traces=true
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

try:
    import ray
except ImportError:
    ray = None

try:
    from run_fgcs_extended_benchmark import (
        authorization_execution_violation,
        base_action_for_row,
        canonical_action_hash,
        ensure_dir,
        execute_generation_stub,
        load_bc_reference_actions,
        load_config,
        maybe_fault_inject,
        no_generation_result,
        normalize_label,
        stable_hash_to_float,
        validate_trace,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import the shared ReplayBench-PG functions from "
        "run_fgcs_extended_benchmark.py. Place this script in the same project "
        "directory as the finalized benchmark runner."
    ) from exc


SUPPORTED_FAULT_MODES = {"clean", "action_flip", "dropped_row"}
DEFAULT_FAULT_MODES = ["clean", "action_flip", "dropped_row"]
DEFAULT_POLICIES = ["risk_proxy", "random", "never"]
DEFAULT_SEEDS = [1, 2, 3]
DEFAULT_WORKERS = [1, 4]


# ---------------------------------------------------------------------------
# Configuration and reproducibility utilities
# ---------------------------------------------------------------------------


def deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> Dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""
    result: Dict[str, Any] = dict(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_effective_config(path: str | Path) -> Dict[str, Any]:
    """
    Load a Ray comparison config and optionally inherit an existing benchmark
    config using the top-level ``base_config`` key.
    """
    config_path = Path(path)
    override = load_config(config_path)

    base_config_path = override.get("base_config")
    if not base_config_path:
        return override

    base_path = Path(str(base_config_path).replace("\\", os.sep))
    if not base_path.is_absolute():
        candidates = [
            Path.cwd() / base_path,
            config_path.resolve().parent / base_path,
        ]
        base_path = next(
            (candidate for candidate in candidates if candidate.exists()),
            candidates[0],
        )

    if not base_path.exists():
        raise FileNotFoundError(
            f"base_config was specified but not found: {base_path}"
        )

    base = load_config(base_path)
    override_without_pointer = {
        key: value
        for key, value in override.items()
        if key != "base_config"
    }
    effective = deep_merge(base, override_without_pointer)
    effective["_resolved_base_config"] = str(base_path)
    return effective


def normalize_comparison_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Normalize both the recommended sectioned schema and the earlier compact
    schema into one internal configuration.
    """
    dataset_cfg = cfg.get("dataset", {})
    benchmark_cfg = cfg.get("benchmark", {})
    policy_cfg = cfg.get("policy", {})
    logging_cfg = cfg.get("logging", {})
    fault_cfg = cfg.get("fault_injection", {})
    reference_cfg = cfg.get("reference", {})
    execution_cfg = cfg.get("execution_semantics", {})

    compact_input = cfg.get("input", {})

    input_csv = (
        dataset_cfg.get("input_csv")
        or compact_input.get("replay_csv")
    )
    if not input_csv:
        raise ValueError(
            "Missing dataset.input_csv (or input.replay_csv) in Ray config"
        )

    workload_fraction = float(
        dataset_cfg.get(
            "workload_fraction",
            compact_input.get("workload_fraction", 1.0),
        )
    )

    policy_modes = list(
        benchmark_cfg.get(
            "policy_modes",
            cfg.get("policies", DEFAULT_POLICIES),
        )
    )
    seeds = [
        int(value)
        for value in benchmark_cfg.get(
            "seeds",
            cfg.get("seeds", DEFAULT_SEEDS),
        )
    ]
    workers = [
        int(value)
        for value in benchmark_cfg.get(
            "workers",
            cfg.get("workers", DEFAULT_WORKERS),
        )
    ]
    fault_modes = [
        str(value).strip().lower()
        for value in benchmark_cfg.get(
            "fault_modes",
            cfg.get("fault_modes", DEFAULT_FAULT_MODES),
        )
    ]

    chunk_size = int(
        benchmark_cfg.get(
            "ray_chunk_size",
            cfg.get("chunk_size", 256),
        )
    )

    action_flip_probability = float(
        fault_cfg.get(
            "action_flip_probability",
            cfg.get("fault_rate", 0.01),
        )
    )
    dropped_row_probability = float(
        fault_cfg.get(
            "dropped_row_probability",
            cfg.get("fault_rate", 0.01),
        )
    )

    output_dir = str(
        logging_cfg.get(
            "output_dir",
            cfg.get("output_dir", "paper_outputs/ray_comparison"),
        )
    )
    save_traces = bool(logging_cfg.get("save_traces", True))

    determinism_csv = str(
        reference_cfg.get(
            "determinism_csv",
            "paper_outputs/fgcs_extended_benchmark/"
            "determinism_hash_results.csv",
        )
    )
    require_reference_results = bool(
        reference_cfg.get("require_reference_results", True)
    )

    retry_enabled = bool(
        execution_cfg.get("task_retry_enabled", False)
    )
    failure_policy = str(
        execution_cfg.get("task_failure_policy", "fail_fast")
    ).strip().lower()
    duplicate_delivery_supported = bool(
        execution_cfg.get("duplicate_delivery_supported", False)
    )

    normalized = {
        "dataset": {
            "input_csv": str(input_csv),
            "workload_fraction": workload_fraction,
        },
        "benchmark": {
            "policy_modes": policy_modes,
            "seeds": seeds,
            "workers": workers,
            "fault_modes": fault_modes,
            "ray_chunk_size": chunk_size,
        },
        "policy": dict(policy_cfg),
        "generation_stub": dict(cfg.get("generation_stub", {})),
        "fault_injection": {
            "action_flip_probability": action_flip_probability,
            "dropped_row_probability": dropped_row_probability,
        },
        "reference": {
            "determinism_csv": determinism_csv,
            "require_reference_results": require_reference_results,
        },
        "execution_semantics": {
            "task_retry_enabled": retry_enabled,
            "task_failure_policy": failure_policy,
            "duplicate_delivery_supported": duplicate_delivery_supported,
        },
        "logging": {
            "output_dir": output_dir,
            "save_traces": save_traces,
        },
    }

    if "_resolved_base_config" in cfg:
        normalized["_resolved_base_config"] = cfg[
            "_resolved_base_config"
        ]

    return normalized


def validate_comparison_config(cfg: Mapping[str, Any]) -> None:
    dataset_cfg = cfg["dataset"]
    benchmark_cfg = cfg["benchmark"]
    policy_cfg = cfg["policy"]
    fault_cfg = cfg["fault_injection"]
    reference_cfg = cfg["reference"]
    execution_cfg = cfg["execution_semantics"]

    input_csv = Path(
        str(dataset_cfg["input_csv"]).replace("\\", os.sep)
    )
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    fraction = float(dataset_cfg["workload_fraction"])
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            "dataset.workload_fraction must be in the interval (0, 1]"
        )

    policies = list(benchmark_cfg["policy_modes"])
    if not policies:
        raise ValueError("At least one policy mode is required")
    if "bc_live" in policies:
        raise ValueError(
            "bc_live is intentionally excluded from the reduced Ray "
            "comparison. The comparison targets orchestration and validation "
            "logic, not checkpoint-serving behavior."
        )

    seeds = [int(value) for value in benchmark_cfg["seeds"]]
    workers = [int(value) for value in benchmark_cfg["workers"]]
    fault_modes = [str(value) for value in benchmark_cfg["fault_modes"]]

    # The final manuscript comparison uses the preregistered reduced matrix.
    if abs(fraction - 1.0) > 1e-12:
        raise ValueError("The final Ray comparison requires workload_fraction=1.0")
    if policies != ["risk_proxy", "random", "never"]:
        raise ValueError(
            "The final Ray comparison requires policy order: "
            "risk_proxy, random, never"
        )
    if seeds != [1, 2, 3]:
        raise ValueError("The final Ray comparison requires seeds [1, 2, 3]")
    if workers != [1, 4]:
        raise ValueError("The final Ray comparison requires workers [1, 4]")
    if set(fault_modes) != SUPPORTED_FAULT_MODES:
        raise ValueError(
            "The final Ray comparison requires clean, action_flip, and "
            "dropped_row fault modes"
        )

    negative_labels = policy_cfg.get("negative_labels", [])
    if not negative_labels:
        raise ValueError(
            "risk_proxy requires non-empty policy.negative_labels. "
            "Use base_config to inherit the exact ReplayBench-PG policy logic."
        )

    if not bool(reference_cfg.get("require_reference_results", True)):
        raise ValueError(
            "Final Ray results must require external ReplayBench-PG reference "
            "hashes; set reference.require_reference_results=true"
        )

    if not workers or any(int(value) <= 0 for value in workers):
        raise ValueError("All worker settings must be positive integers")

    unknown_faults = sorted(
        set(fault_modes) - SUPPORTED_FAULT_MODES
    )
    if unknown_faults:
        raise ValueError(
            f"Unsupported fault modes: {unknown_faults}. "
            f"Supported modes: {sorted(SUPPORTED_FAULT_MODES)}"
        )
    if "clean" not in fault_modes:
        raise ValueError(
            "The Ray comparison must include the clean mode so fault runs "
            "have a clean reference sequence."
        )

    chunk_size = int(benchmark_cfg["ray_chunk_size"])
    if chunk_size <= 0:
        raise ValueError("benchmark.ray_chunk_size must be positive")

    for key in [
        "action_flip_probability",
        "dropped_row_probability",
    ]:
        probability = float(fault_cfg[key])
        if not (0.0 < probability < 1.0):
            raise ValueError(
                f"fault_injection.{key} must be between 0 and 1"
            )

    if bool(execution_cfg["task_retry_enabled"]):
        raise ValueError(
            "The comparison currently requires task_retry_enabled=false"
        )
    if execution_cfg["task_failure_policy"] != "fail_fast":
        raise ValueError(
            "The comparison currently supports only fail_fast execution"
        )
    if bool(execution_cfg["duplicate_delivery_supported"]):
        raise ValueError(
            "The comparison does not currently support duplicate delivery"
        )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
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


def sanitize_token(value: Any) -> str:
    return (
        str(value)
        .replace(".", "p")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def chunk_rows(
    rows: Sequence[Dict[str, Any]],
    chunk_size: int,
) -> List[List[Tuple[int, Dict[str, Any]]]]:
    indexed = list(enumerate(rows))
    return [
        indexed[start : start + chunk_size]
        for start in range(0, len(indexed), chunk_size)
    ]


# ---------------------------------------------------------------------------
# Shared Ray task logic
# ---------------------------------------------------------------------------


def process_chunk_impl(
    indexed_rows: Sequence[Tuple[int, Dict[str, Any]]],
    policy_mode: str,
    seed: int,
    negative_labels: set[str],
    random_probability: float,
    bc_actions: Optional[Mapping[int, int]],
    policy_cfg: Mapping[str, Any],
    generation_cfg: Mapping[str, Any],
    fault_mode: str,
    action_flip_probability: float,
) -> List[Dict[str, Any]]:
    """
    Process one chunk inside a Ray worker.

    Ray supplies task execution. Policy selection, ordering identifiers,
    authorization/execution logging, fault injection, and validation fields
    remain application-level logic.
    """
    fault_cfg: Dict[str, Any]

    if fault_mode == "action_flip":
        fault_cfg = {
            "enabled": True,
            "action_flip_probability": float(
                action_flip_probability
            ),
            "allowed_policy_modes": [policy_mode],
        }
    else:
        fault_cfg = {
            "enabled": False,
            "action_flip_probability": 0.0,
            "allowed_policy_modes": [policy_mode],
        }

    output: List[Dict[str, Any]] = []

    for row_index, row in indexed_rows:
        action_before_fault = base_action_for_row(
            row=row,
            row_index=int(row_index),
            policy_mode=policy_mode,
            negative_labels=negative_labels,
            seed=int(seed),
            random_p=float(random_probability),
            bc_actions=bc_actions,
            live_actions=None,
            policy_cfg=policy_cfg,
        )

        action, fault_injected = maybe_fault_inject(
            action=int(action_before_fault),
            row=row,
            row_index=int(row_index),
            policy_mode=policy_mode,
            seed=int(seed),
            fault_cfg=fault_cfg,
        )

        authorized = int(action) == 1

        if authorized:
            generated = execute_generation_stub(
                row=row,
                generation_cfg=generation_cfg,
            )
            generation_invoked = 1
        else:
            generated = no_generation_result()
            generation_invoked = 0

        unauthorized_invocation = (
            authorization_execution_violation(
                authorized=authorized,
                executed=generation_invoked,
            )
        )

        output.append(
            {
                "row_index": int(row_index),
                "utterance_id": row.get(
                    "utterance_id",
                    row_index,
                ),
                "source_record_id": row.get(
                    "source_record_id",
                    row.get("utterance_id", row_index),
                ),
                "policy_mode": policy_mode,
                "seed": int(seed),
                "action_before_fault": int(
                    action_before_fault
                ),
                "action": int(action),
                "action_flip_fault_injected": int(
                    fault_injected
                ),
                "authorized_to_generate": int(authorized),
                "generation_invoked": int(
                    generation_invoked
                ),
                "unauthorized_invocation": int(
                    unauthorized_invocation
                ),
                "response_safety": generated["safety"],
            }
        )

    return output


# ---------------------------------------------------------------------------
# Reference hashes and controlled row corruption
# ---------------------------------------------------------------------------


def load_reference_hashes(
    path: str | Path,
    workload_fraction: float,
) -> Dict[Tuple[str, int, int], str]:
    """
    Load ReplayBench-PG clean hashes keyed by policy, seed, and workers.

    The function selects rows matching the requested dataset fraction.
    """
    reference_path = Path(
        str(path).replace("\\", os.sep)
    )
    if not reference_path.exists():
        return {}

    frame = pd.read_csv(reference_path)

    required = {
        "dataset_fraction",
        "policy_mode",
        "seed",
        "workers",
        "trace_hash",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Reference determinism CSV is missing columns: {missing}"
        )

    fractions = pd.to_numeric(
        frame["dataset_fraction"],
        errors="coerce",
    )
    selected = frame[
        (fractions - float(workload_fraction)).abs() < 1e-12
    ].copy()

    lookup: Dict[Tuple[str, int, int], str] = {}

    for row in selected.to_dict(orient="records"):
        key = (
            str(row["policy_mode"]),
            int(row["seed"]),
            int(row["workers"]),
        )
        value = str(row["trace_hash"])
        if key in lookup and lookup[key] != value:
            raise ValueError(
                "Reference CSV contains conflicting hashes for "
                f"{key}"
            )
        lookup[key] = value

    return lookup


def select_dropped_indices(
    row_indices: Iterable[int],
    policy_mode: str,
    seed: int,
    probability: float,
) -> set[int]:
    """
    Select an exact deterministic number of rows for post-hoc removal.

    Rows are ranked by a stable hash. This avoids a zero-event fault condition
    on small reduced workloads.
    """
    indices = [int(value) for value in row_indices]
    if not indices:
        return set()

    count = max(1, int(len(indices) * float(probability)))
    count = min(count, len(indices))

    ranked = sorted(
        indices,
        key=lambda index: stable_hash_to_float(
            "ray_dropped_row",
            policy_mode,
            seed,
            index,
        ),
    )
    return set(ranked[:count])


def apply_dropped_row_fault(
    trace: pd.DataFrame,
    policy_mode: str,
    seed: int,
    probability: float,
) -> Tuple[pd.DataFrame, int]:
    dropped_indices = select_dropped_indices(
        trace["row_index"].astype(int).tolist(),
        policy_mode=policy_mode,
        seed=seed,
        probability=probability,
    )

    corrupted = trace[
        ~trace["row_index"].astype(int).isin(dropped_indices)
    ].copy()

    return corrupted.reset_index(drop=True), len(dropped_indices)


# ---------------------------------------------------------------------------
# Condition execution and reporting
# ---------------------------------------------------------------------------


def execute_condition(
    *,
    remote_process_chunk: Any,
    rows: Sequence[Dict[str, Any]],
    chunks: Sequence[Sequence[Tuple[int, Dict[str, Any]]]],
    policy_mode: str,
    seed: int,
    workers: int,
    fault_mode: str,
    negative_labels_ref: Any,
    policy_cfg_ref: Any,
    generation_cfg_ref: Any,
    bc_actions_ref: Any,
    random_probability: float,
    action_flip_probability: float,
    dropped_row_probability: float,
    expected_hash: Optional[str],
    reference_source: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute one Ray condition and validate the reconstructed trace.
    """
    start = time.perf_counter()

    futures = [
        remote_process_chunk.remote(
            chunk,
            policy_mode,
            int(seed),
            negative_labels_ref,
            float(random_probability),
            bc_actions_ref,
            policy_cfg_ref,
            generation_cfg_ref,
            fault_mode,
            float(action_flip_probability),
        )
        for chunk in chunks
    ]

    chunk_results = ray.get(futures)
    ray_execution_seconds = time.perf_counter() - start

    records = [
        record
        for chunk_output in chunk_results
        for record in chunk_output
    ]

    trace = pd.DataFrame(records)

    if trace.empty:
        raise RuntimeError(
            f"Ray produced no records for policy={policy_mode}, "
            f"seed={seed}, workers={workers}, fault={fault_mode}"
        )

    duplicate_count = int(
        trace["row_index"].duplicated().sum()
    )
    if duplicate_count:
        raise RuntimeError(
            "Ray returned duplicate row indices before post-hoc corruption: "
            f"{duplicate_count}"
        )

    trace = (
        trace.sort_values("row_index", kind="stable")
        .reset_index(drop=True)
    )

    if len(trace) != len(rows):
        raise RuntimeError(
            "Ray task execution did not return exactly one result per input "
            f"before post-hoc corruption: expected={len(rows)}, "
            f"observed={len(trace)}"
        )

    injected_events = int(
        trace["action_flip_fault_injected"].sum()
    )

    if fault_mode == "dropped_row":
        trace, injected_events = apply_dropped_row_fault(
            trace=trace,
            policy_mode=policy_mode,
            seed=seed,
            probability=dropped_row_probability,
        )

    if fault_mode != "clean" and injected_events <= 0:
        raise RuntimeError(
            f"Fault mode {fault_mode!r} injected zero events"
        )

    actions = trace["action"].astype(int).tolist()
    unauthorized_invocations = int(
        trace["unauthorized_invocation"].sum()
    )

    validation = validate_trace(
        actions=actions,
        expected_hash=expected_hash,
        observed_rows=len(trace),
        expected_rows=len(rows),
        unauthorized_invocations=unauthorized_invocations,
    )

    expected_flagged = int(fault_mode != "clean")
    detected_flag = int(
        validation["hash_match"] == 0
        or validation["row_count_match"] == 0
        or validation[
            "authorization_execution_consistent"
        ] == 0
    )

    detection_correct = int(
        detected_flag == expected_flagged
    )

    summary: Dict[str, Any] = {
        "policy_mode": policy_mode,
        "seed": int(seed),
        "workers": int(workers),
        "fault_mode": fault_mode,
        "expected_rows": int(len(rows)),
        "observed_rows": int(len(trace)),
        "injected_events": int(injected_events),
        "ray_task_count": int(len(chunks)),
        "ray_execution_seconds": float(
            ray_execution_seconds
        ),
        "throughput_points_per_second": (
            float(len(trace) / ray_execution_seconds)
            if ray_execution_seconds > 0
            else 0.0
        ),
        "expected_hash": expected_hash,
        "observed_hash": validation["actual_hash"],
        "hash_match": validation["hash_match"],
        "row_count_match": validation["row_count_match"],
        "unauthorized_invocations": (
            unauthorized_invocations
        ),
        "authorization_execution_consistent": validation[
            "authorization_execution_consistent"
        ],
        "validation_passed": validation[
            "validation_passed"
        ],
        "expected_flagged": expected_flagged,
        "detected_flag": detected_flag,
        "detection_correct": detection_correct,
        "reference_source": reference_source,
    }

    return trace, summary


def build_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for fault_mode, group in per_run.groupby(
        "fault_mode",
        sort=False,
    ):
        rows.append(
            {
                "fault_mode": fault_mode,
                "conditions": int(len(group)),
                "injected_events": int(
                    group["injected_events"].sum()
                ),
                "expected_flagged_runs": int(
                    group["expected_flagged"].sum()
                ),
                "detected_flagged_runs": int(
                    group["detected_flag"].sum()
                ),
                "correct_detection_runs": int(
                    group["detection_correct"].sum()
                ),
                "hash_mismatch_runs": int(
                    (group["hash_match"] == 0).sum()
                ),
                "row_count_mismatch_runs": int(
                    (group["row_count_match"] == 0).sum()
                ),
                "authorization_violation_runs": int(
                    (
                        group[
                            "authorization_execution_consistent"
                        ]
                        == 0
                    ).sum()
                ),
                "all_detection_expectations_met": int(
                    group["detection_correct"].eq(1).all()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_component_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component": "Distributed task execution",
                "ray_role_in_evaluated_implementation": "Distributed execution substrate",
                "application_logic_required": "No",
                "implementation": (
                    "ray.remote, ray.get, Ray CPU scheduling"
                ),
            },
            {
                "component": "Indexed ordered reconstruction",
                "ray_role_in_evaluated_implementation": "No task-specific primitive used",
                "application_logic_required": "Yes",
                "implementation": (
                    "Stable row_index returned by tasks and sorted "
                    "after completion"
                ),
            },
            {
                "component": "Policy-action generation",
                "ray_role_in_evaluated_implementation": "No task-specific primitive used",
                "application_logic_required": "Yes",
                "implementation": (
                    "Shared ReplayBench-PG base_action_for_row"
                ),
            },
            {
                "component": "Canonical action-sequence hashing",
                "ray_role_in_evaluated_implementation": "No task-specific primitive used",
                "application_logic_required": "Yes",
                "implementation": (
                    "Shared ReplayBench-PG canonical_action_hash"
                ),
            },
            {
                "component": (
                    "Authorization-execution consistency invariant"
                ),
                "ray_role_in_evaluated_implementation": "No task-specific primitive used",
                "application_logic_required": "Yes",
                "implementation": (
                    "Separate authorization and execution fields plus "
                    "shared invariant"
                ),
            },
            {
                "component": "Action-flip fault injection",
                "ray_role_in_evaluated_implementation": "No task-specific primitive used",
                "application_logic_required": "Yes",
                "implementation": (
                    "Shared ReplayBench-PG maybe_fault_inject"
                ),
            },
            {
                "component": "Dropped-row fault injection",
                "ray_role_in_evaluated_implementation": "No task-specific primitive used",
                "application_logic_required": "Yes",
                "implementation": (
                    "Deterministic post-hoc row removal"
                ),
            },
            {
                "component": "Row-count validation",
                "ray_role_in_evaluated_implementation": "No task-specific primitive used",
                "application_logic_required": "Yes",
                "implementation": (
                    "Shared ReplayBench-PG validate_trace"
                ),
            },
            {
                "component": "Reproducibility artifacts",
                "ray_role_in_evaluated_implementation": "No task-specific primitive used",
                "application_logic_required": "Yes",
                "implementation": (
                    "CSV summaries, trace files, effective config, "
                    "and JSON manifest"
                ),
            },
        ]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reduced empirical Ray comparison for ReplayBench-PG."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/ray_comparison.yaml",
        help="Path to the Ray comparison YAML configuration.",
    )
    args = parser.parse_args()

    if ray is None:
        raise RuntimeError(
            "Ray is not installed. Install it in the active environment "
            "with: python -m pip install ray"
        )

    raw_config = load_effective_config(args.config)
    cfg = normalize_comparison_config(raw_config)
    validate_comparison_config(cfg)

    dataset_cfg = cfg["dataset"]
    benchmark_cfg = cfg["benchmark"]
    policy_cfg = cfg["policy"]
    generation_cfg = cfg["generation_stub"]
    fault_cfg = cfg["fault_injection"]
    reference_cfg = cfg["reference"]
    logging_cfg = cfg["logging"]

    input_csv = Path(
        str(dataset_cfg["input_csv"]).replace("\\", os.sep)
    )
    workload_fraction = float(
        dataset_cfg["workload_fraction"]
    )
    policy_modes = [
        str(value)
        for value in benchmark_cfg["policy_modes"]
    ]
    seeds = [
        int(value)
        for value in benchmark_cfg["seeds"]
    ]
    workers_list = sorted(
        {int(value) for value in benchmark_cfg["workers"]}
    )
    fault_modes = [
        "clean",
        *[
            value
            for value in benchmark_cfg["fault_modes"]
            if value != "clean"
        ],
    ]
    chunk_size = int(
        benchmark_cfg["ray_chunk_size"]
    )

    output_dir = Path(logging_cfg["output_dir"])
    traces_dir = output_dir / "traces"
    save_traces = bool(logging_cfg["save_traces"])

    ensure_dir(output_dir)
    if save_traces:
        ensure_dir(traces_dir)

    df_full = pd.read_csv(input_csv).reset_index(drop=True)
    if df_full.empty:
        raise ValueError(f"Input CSV has no rows: {input_csv}")

    selected_rows = max(
        1,
        int(len(df_full) * workload_fraction),
    )
    df = (
        df_full.iloc[:selected_rows]
        .copy()
        .reset_index(drop=True)
    )
    rows = df.to_dict(orient="records")
    chunks = chunk_rows(rows, chunk_size)

    negative_labels = {
        normalize_label(value)
        for value in policy_cfg.get(
            "negative_labels",
            [],
        )
    }
    random_probability = float(
        policy_cfg.get(
            "random_intervention_probability",
            0.5,
        )
    )

    bc_actions: Optional[Dict[int, int]] = None
    if "bc" in policy_modes:
        bc_action_csv = policy_cfg.get(
            "bc_action_csv",
            "paper_outputs/policy_first_outputs_bc.csv",
        )
        bc_actions = load_bc_reference_actions(
            bc_action_csv=bc_action_csv,
            df_full=df_full,
            action_column=str(
                policy_cfg.get(
                    "bc_action_column",
                    "action",
                )
            ),
            key_column=str(
                policy_cfg.get(
                    "bc_key_column",
                    "utterance_id",
                )
            ),
        )

    reference_hashes = load_reference_hashes(
        reference_cfg["determinism_csv"],
        workload_fraction=workload_fraction,
    )
    reference_path_exists = Path(
        str(reference_cfg["determinism_csv"]).replace(
            "\\",
            os.sep,
        )
    ).exists()

    required_reference_keys = {
        (policy_mode, int(seed), int(workers))
        for policy_mode in policy_modes
        for seed in seeds
        for workers in workers_list
    }
    missing_reference_keys = sorted(
        required_reference_keys - set(reference_hashes)
    )
    if missing_reference_keys:
        raise RuntimeError(
            "The final Ray comparison requires an external ReplayBench-PG "
            "reference hash for every clean condition. Missing keys: "
            f"{missing_reference_keys}"
        )

    expected_conditions = (
        len(policy_modes)
        * len(seeds)
        * len(workers_list)
        * len(fault_modes)
    )

    print(f"[INFO] Ray version: {ray.__version__}")
    print(f"[INFO] Input: {input_csv}")
    print(
        f"[INFO] Selected rows: {selected_rows}/"
        f"{len(df_full)} (fraction={workload_fraction})"
    )
    print(f"[INFO] Policies: {policy_modes}")
    print(f"[INFO] Seeds: {seeds}")
    print(f"[INFO] Workers: {workers_list}")
    print(f"[INFO] Fault modes: {fault_modes}")
    print(f"[INFO] Chunk size: {chunk_size}")
    print(f"[INFO] Ray tasks per condition: {len(chunks)}")
    print(f"[INFO] Expected conditions: {expected_conditions}")
    print(
        "[INFO] External ReplayBench-PG reference hashes: "
        f"{len(reference_hashes)}"
    )

    remote_process_chunk = ray.remote(
        max_retries=0,
        retry_exceptions=False,
    )(process_chunk_impl)

    per_run_rows: List[Dict[str, Any]] = []
    ray_init_seconds_by_worker: Dict[int, float] = {}

    for workers in workers_list:
        if ray.is_initialized():
            ray.shutdown()

        init_start = time.perf_counter()
        ray.init(
            num_cpus=int(workers),
            include_dashboard=False,
            ignore_reinit_error=False,
            log_to_driver=False,
        )
        ray_init_seconds_by_worker[workers] = (
            time.perf_counter() - init_start
        )

        negative_labels_ref = ray.put(negative_labels)
        policy_cfg_ref = ray.put(policy_cfg)
        generation_cfg_ref = ray.put(generation_cfg)
        bc_actions_ref = (
            ray.put(bc_actions)
            if bc_actions is not None
            else None
        )

        try:
            for policy_mode in policy_modes:
                for seed in seeds:
                    for fault_mode in fault_modes:
                        external_key = (
                            policy_mode,
                            int(seed),
                            int(workers),
                        )
                        expected_hash = reference_hashes[external_key]
                        reference_source = "replaybench_determinism_csv"

                        print(
                            "[RUN] "
                            f"policy={policy_mode}, "
                            f"seed={seed}, "
                            f"workers={workers}, "
                            f"fault={fault_mode}"
                        )

                        trace, summary = execute_condition(
                            remote_process_chunk=(
                                remote_process_chunk
                            ),
                            rows=rows,
                            chunks=chunks,
                            policy_mode=policy_mode,
                            seed=seed,
                            workers=workers,
                            fault_mode=fault_mode,
                            negative_labels_ref=(
                                negative_labels_ref
                            ),
                            policy_cfg_ref=policy_cfg_ref,
                            generation_cfg_ref=(
                                generation_cfg_ref
                            ),
                            bc_actions_ref=bc_actions_ref,
                            random_probability=(
                                random_probability
                            ),
                            action_flip_probability=float(
                                fault_cfg[
                                    "action_flip_probability"
                                ]
                            ),
                            dropped_row_probability=float(
                                fault_cfg[
                                    "dropped_row_probability"
                                ]
                            ),
                            expected_hash=expected_hash,
                            reference_source=reference_source,
                        )

                        summary.update(
                            {
                                "workload_fraction": (
                                    workload_fraction
                                ),
                                "decision_points": (
                                    selected_rows
                                ),
                                "ray_chunk_size": chunk_size,
                                "ray_init_seconds": (
                                    ray_init_seconds_by_worker[
                                        workers
                                    ]
                                ),
                                "input_csv": str(input_csv),
                            }
                        )
                        per_run_rows.append(summary)

                        if save_traces:
                            trace_path = (
                                traces_dir
                                / (
                                    "trace_"
                                    f"fraction_{sanitize_token(workload_fraction)}_"
                                    f"policy_{sanitize_token(policy_mode)}_"
                                    f"seed_{seed}_"
                                    f"workers_{workers}_"
                                    f"fault_{sanitize_token(fault_mode)}.csv"
                                )
                            )
                            trace.to_csv(
                                trace_path,
                                index=False,
                            )
        finally:
            ray.shutdown()

    per_run = pd.DataFrame(per_run_rows)

    if len(per_run) != expected_conditions:
        raise RuntimeError(
            "Unexpected condition count: "
            f"expected={expected_conditions}, "
            f"observed={len(per_run)}"
        )

    if not per_run["detection_correct"].eq(1).all():
        failed = per_run.loc[
            per_run["detection_correct"] != 1,
            [
                "policy_mode",
                "seed",
                "workers",
                "fault_mode",
                "hash_match",
                "row_count_match",
                "authorization_execution_consistent",
                "expected_flagged",
                "detected_flag",
            ],
        ]
        raise RuntimeError(
            "One or more Ray comparison conditions did not meet the "
            "expected clean/fault validation outcome:\n"
            f"{failed.to_string(index=False)}"
        )

    raw_path = output_dir / "ray_comparison_raw.csv"
    per_run_path = output_dir / "ray_comparison_per_run.csv"
    summary_path = output_dir / "ray_comparison_summary.csv"
    inventory_path = (
        output_dir
        / "ray_validation_component_inventory.csv"
    )
    manifest_path = (
        output_dir
        / "ray_comparison_manifest.json"
    )
    effective_config_path = (
        output_dir
        / "ray_comparison_effective_config.json"
    )

    per_run.to_csv(raw_path, index=False)

    preferred_columns = [
        "workload_fraction",
        "decision_points",
        "policy_mode",
        "seed",
        "workers",
        "fault_mode",
        "injected_events",
        "ray_task_count",
        "ray_execution_seconds",
        "throughput_points_per_second",
        "expected_hash",
        "observed_hash",
        "hash_match",
        "expected_rows",
        "observed_rows",
        "row_count_match",
        "unauthorized_invocations",
        "authorization_execution_consistent",
        "expected_flagged",
        "detected_flag",
        "detection_correct",
        "reference_source",
    ]
    per_run[preferred_columns].to_csv(
        per_run_path,
        index=False,
    )

    summary_frame = build_summary(per_run)
    summary_frame.to_csv(summary_path, index=False)

    component_inventory = build_component_inventory()
    component_inventory.to_csv(
        inventory_path,
        index=False,
    )

    with open(
        effective_config_path,
        "w",
        encoding="utf-8",
    ) as file_obj:
        json.dump(cfg, file_obj, indent=2)

    external_clean = per_run[
        (
            per_run["fault_mode"] == "clean"
        )
        & (
            per_run["reference_source"]
            == "replaybench_determinism_csv"
        )
    ]

    manifest = {
        "experiment": "ReplayBench-PG Ray comparison",
        "purpose": (
            "Empirical comparison of a general-purpose Ray execution "
            "substrate with application-level policy-gated replay "
            "validation logic"
        ),
        "python_version": sys.version,
        "platform": platform.platform(),
        "ray_version": ray.__version__,
        "git_commit": get_git_commit(),
        "config_path": str(args.config),
        "resolved_base_config": cfg.get(
            "_resolved_base_config"
        ),
        "input_csv": str(input_csv),
        "input_sha256": file_sha256(input_csv),
        "full_input_rows": int(len(df_full)),
        "workload_fraction": workload_fraction,
        "decision_points": selected_rows,
        "policy_modes": policy_modes,
        "seeds": seeds,
        "workers": workers_list,
        "fault_modes": fault_modes,
        "ray_chunk_size": chunk_size,
        "expected_conditions": expected_conditions,
        "completed_conditions": int(len(per_run)),
        "ray_init_seconds_by_worker": {
            str(key): value
            for key, value in (
                ray_init_seconds_by_worker.items()
            )
        },
        "execution_semantics": cfg[
            "execution_semantics"
        ],
        "reference": {
            "determinism_csv": reference_cfg[
                "determinism_csv"
            ],
            "reference_file_exists": (
                reference_path_exists
            ),
            "loaded_reference_hashes": int(
                len(reference_hashes)
            ),
            "external_clean_comparisons": int(
                len(external_clean)
            ),
            "external_clean_hash_matches": int(
                external_clean["hash_match"].eq(1).sum()
            ),
        },
        "validation": {
            "ordered_reconstruction": (
                "Records sorted by stable row_index after Ray task "
                "completion"
            ),
            "action_hash": (
                "SHA-256 over canonical ordered integer actions"
            ),
            "authorization_execution_invariant": (
                "Violation when downstream execution=1 and prior "
                "authorization=0"
            ),
            "row_count_validation": True,
            "fault_classes": [
                "action_flip",
                "dropped_row",
            ],
        },
        "results": {
            "all_detection_expectations_met": bool(
                per_run["detection_correct"].eq(1).all()
            ),
            "clean_conditions": int(
                (per_run["fault_mode"] == "clean").sum()
            ),
            "clean_false_positive_detections": int(
                (
                    (per_run["fault_mode"] == "clean")
                    & (per_run["detected_flag"] == 1)
                ).sum()
            ),
            "fault_conditions": int(
                (per_run["fault_mode"] != "clean").sum()
            ),
            "fault_conditions_flagged": int(
                (
                    (per_run["fault_mode"] != "clean")
                    & (per_run["detected_flag"] == 1)
                ).sum()
            ),
        },
        "outputs": {
            "raw": str(raw_path),
            "per_run": str(per_run_path),
            "summary": str(summary_path),
            "component_inventory": str(inventory_path),
            "effective_config": str(
                effective_config_path
            ),
            "traces_saved": save_traces,
        },
    }

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file_obj:
        json.dump(manifest, file_obj, indent=2)

    print("[DONE] Ray comparison complete.")
    print(f"[OUT] {raw_path}")
    print(f"[OUT] {per_run_path}")
    print(f"[OUT] {summary_path}")
    print(f"[OUT] {inventory_path}")
    print(f"[OUT] {effective_config_path}")
    print(f"[OUT] {manifest_path}")


if __name__ == "__main__":
    main()
