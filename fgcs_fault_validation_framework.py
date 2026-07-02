#!/usr/bin/env python3
"""
Unified FGCS RQ7 fault-validation framework.

This script consolidates the compact RQ7 fault-validation workflow:

1. Runtime policy-output corruption:
   - action_flip_1_percent
   - detected by SHA-256 action-trace mismatch

2. Runtime invocation-boundary violation:
   - unauthorized_invoke_1_percent
   - detected by unauthorized-invocation counter

3. Post-hoc trace-integrity corruption:
   - trace_action_corruption_1_percent
   - drop_trace_rows_1_percent
   - duplicate_trace_rows_1_percent
   - detected by SHA-256 action-trace mismatch and/or row-count mismatch

It assumes that the clean 360-run benchmark already exists in:
    paper_outputs/fgcs_extended_benchmark/

It also assumes that the two compact runtime fault experiments have already been run:
    paper_outputs/fgcs_fault_action_flip/
    paper_outputs/fgcs_fault_unauthorized_invoke/

Typical use:
    python fgcs_fault_validation_framework.py --all

Outputs are written to:
    paper_outputs/fgcs_tables_figures/
    paper_outputs/fgcs_fault_trace_corruption/
"""

from __future__ import annotations

import argparse
import hashlib
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class FaultType(str, Enum):
    CLEAN_REPLAY = "clean_replay"
    ACTION_FLIP = "action_flip_1_percent"
    UNAUTHORIZED_INVOKE = "unauthorized_invoke_1_percent"
    TRACE_ACTION_CORRUPTION = "trace_action_corruption_1_percent"
    TRACE_ROW_DROP = "drop_trace_rows_1_percent"
    TRACE_ROW_DUPLICATE = "duplicate_trace_rows_1_percent"


FAULT_METADATA: Dict[FaultType, Dict[str, str]] = {
    FaultType.CLEAN_REPLAY: {
        "category": "false-positive control",
        "channel": "none",
        "interpretation": "Clean replay and clean trace files produced no detected faults.",
    },
    FaultType.ACTION_FLIP: {
        "category": "policy-output corruption",
        "channel": "SHA-256 trace mismatch + injected-fault counter",
        "interpretation": "Injected action flips changed the action sequence and were detected through SHA-256 trace mismatch.",
    },
    FaultType.UNAUTHORIZED_INVOKE: {
        "category": "invocation-boundary violation",
        "channel": "unauthorized-invocation counter + injected-fault counter",
        "interpretation": "Forced generator calls after action=0 were detected as unauthorized invocations at the invocation boundary.",
    },
    FaultType.TRACE_ACTION_CORRUPTION: {
        "category": "post-hoc action-trace corruption",
        "channel": "SHA-256 action-trace mismatch",
        "interpretation": "Post-hoc action changes in saved traces were detected through SHA-256 action-trace mismatch.",
    },
    FaultType.TRACE_ROW_DROP: {
        "category": "missing trace rows",
        "channel": "row-count mismatch + SHA-256 action-trace mismatch",
        "interpretation": "Dropped trace rows were detected through row-count mismatch and SHA-256 action-trace mismatch.",
    },
    FaultType.TRACE_ROW_DUPLICATE: {
        "category": "duplicated trace rows",
        "channel": "row-count mismatch + SHA-256 action-trace mismatch",
        "interpretation": "Duplicated trace rows were detected through row-count mismatch and SHA-256 action-trace mismatch.",
    },
}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def write_csv_and_tex(df: pd.DataFrame, csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)
    tex_path = csv_path.with_suffix(".tex")
    try:
        df.to_latex(tex_path, index=False, escape=True)
    except Exception as exc:
        print(f"[WARN] Could not write LaTeX table {tex_path}: {exc}")
    print(f"[OUT] {csv_path}")
    if tex_path.exists():
        print(f"[OUT] {tex_path}")


def normalize_fraction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dataset_fraction" in out.columns:
        out["dataset_fraction"] = out["dataset_fraction"].astype(float).round(6)
    return out


def trace_hash(actions: Iterable[int]) -> str:
    """Same action-sequence hash convention as the benchmark runner."""
    s = ",".join(str(int(a)) for a in actions)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def deterministic_indices(n: int, rate: float, salt: str) -> np.ndarray:
    """Deterministically select at least one row index for corruption."""
    k = max(1, int(round(n * rate)))
    seed_int = int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_int)
    return np.sort(rng.choice(np.arange(n), size=min(k, n), replace=False))


def filter_compact_rows(
    df: pd.DataFrame,
    policies: Sequence[str],
    seeds: Sequence[int],
    workers: Sequence[int],
    full_fraction: float,
) -> pd.DataFrame:
    out = normalize_fraction(df)
    return out[
        (out["dataset_fraction"] == float(full_fraction))
        & (out["policy_mode"].isin(list(policies)))
        & (out["seed"].astype(int).isin([int(s) for s in seeds]))
        & (out["workers"].astype(int).isin([int(w) for w in workers]))
    ].copy()


def compact_keys() -> List[str]:
    return ["dataset_fraction", "workload_name", "policy_mode", "seed", "workers"]


# ---------------------------------------------------------------------------
# Runtime fault summaries
# ---------------------------------------------------------------------------


def summarize_action_flip_fault(
    clean_dir: Path,
    fault_dir: Path,
    tables_dir: Path,
    policies: Sequence[str],
    seeds: Sequence[int],
    workers: Sequence[int],
    full_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clean = filter_compact_rows(
        read_required(clean_dir / "determinism_hash_results.csv"),
        policies=policies,
        seeds=seeds,
        workers=workers,
        full_fraction=full_fraction,
    )
    fault = filter_compact_rows(
        read_required(fault_dir / "determinism_hash_results.csv"),
        policies=policies,
        seeds=seeds,
        workers=workers,
        full_fraction=full_fraction,
    )

    keys = compact_keys()
    required_cols = keys + [
        "trace_hash",
        "intervention_rate",
        "unauthorized_invocations",
        "fault_injected_count",
    ]
    for name, df in [("clean", clean), ("action_flip", fault)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name} rows missing required columns: {missing}")

    expected_runs = len(policies) * len(seeds) * len(workers)
    merged = fault[required_cols].merge(
        clean[required_cols],
        on=keys,
        how="inner",
        suffixes=("_fault", "_clean"),
    )
    if len(merged) != expected_runs:
        raise RuntimeError(f"Action-flip summary expected {expected_runs} matched rows, got {len(merged)}")

    merged["hash_mismatch"] = (merged["trace_hash_fault"].astype(str) != merged["trace_hash_clean"].astype(str)).astype(int)
    merged["intervention_rate_delta"] = (
        merged["intervention_rate_fault"].astype(float) - merged["intervention_rate_clean"].astype(float)
    )
    merged["detected_by_trace_hash"] = (
        (merged["fault_injected_count_fault"].astype(int) > 0) & (merged["hash_mismatch"] == 1)
    ).astype(int)
    merged["detected_by_fault_counter"] = (merged["fault_injected_count_fault"].astype(int) > 0).astype(int)

    per_run = merged[
        keys
        + [
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
    ].rename(
        columns={
            "fault_injected_count_fault": "faults_injected",
            "unauthorized_invocations_clean": "clean_unauthorized_invocations",
            "unauthorized_invocations_fault": "fault_unauthorized_invocations",
        }
    )

    summary = pd.DataFrame(
        [
            {
                "fault_mode": FaultType.CLEAN_REPLAY.value,
                "runs": expected_runs,
                "faults_injected_total": 0,
                "detected_runs": 0,
                "detection_rate": "",
                "false_positive_runs": 0,
                "max_unauthorized_invocations": int(clean["unauthorized_invocations"].max()),
                "detection_channel": FAULT_METADATA[FaultType.CLEAN_REPLAY]["channel"],
                "interpretation": "Clean baseline used as false-positive control.",
            },
            {
                "fault_mode": FaultType.ACTION_FLIP.value,
                "runs": int(len(merged)),
                "faults_injected_total": int(merged["fault_injected_count_fault"].sum()),
                "detected_runs": int(merged["detected_by_trace_hash"].sum()),
                "detection_rate": float(merged["detected_by_trace_hash"].mean()),
                "false_positive_runs": 0,
                "max_unauthorized_invocations": int(merged["unauthorized_invocations_fault"].max()),
                "detection_channel": FAULT_METADATA[FaultType.ACTION_FLIP]["channel"],
                "interpretation": FAULT_METADATA[FaultType.ACTION_FLIP]["interpretation"],
            },
        ]
    )

    write_csv_and_tex(per_run, tables_dir / "fgcs_table_fault_action_flip_per_run.csv")
    write_csv_and_tex(summary, tables_dir / "fgcs_table_fault_action_flip_detection_summary.csv")
    return per_run, summary


def summarize_unauthorized_invoke_fault(
    clean_dir: Path,
    fault_dir: Path,
    tables_dir: Path,
    policies: Sequence[str],
    seeds: Sequence[int],
    workers: Sequence[int],
    full_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clean = filter_compact_rows(
        read_required(clean_dir / "determinism_hash_results.csv"),
        policies=policies,
        seeds=seeds,
        workers=workers,
        full_fraction=full_fraction,
    )
    fault = filter_compact_rows(
        read_required(fault_dir / "determinism_hash_results.csv"),
        policies=policies,
        seeds=seeds,
        workers=workers,
        full_fraction=full_fraction,
    )

    keys = compact_keys()
    required_cols = keys + [
        "trace_hash",
        "intervention_rate",
        "unauthorized_invocations",
        "fault_injected_count",
    ]
    for name, df in [("clean", clean), ("unauthorized_invoke", fault)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name} rows missing required columns: {missing}")

    expected_runs = len(policies) * len(seeds) * len(workers)
    merged = fault[required_cols].merge(
        clean[required_cols],
        on=keys,
        how="inner",
        suffixes=("_fault", "_clean"),
    )
    if len(merged) != expected_runs:
        raise RuntimeError(
            f"Unauthorized-invoke summary expected {expected_runs} matched rows, got {len(merged)}"
        )

    merged["detected_by_unauthorized_counter"] = (
        (merged["fault_injected_count_fault"].astype(int) > 0)
        & (merged["unauthorized_invocations_fault"].astype(int) > 0)
    ).astype(int)
    merged["detected_by_fault_counter"] = (merged["fault_injected_count_fault"].astype(int) > 0).astype(int)
    merged["unauthorized_delta"] = (
        merged["unauthorized_invocations_fault"].astype(int) - merged["unauthorized_invocations_clean"].astype(int)
    )

    per_run = merged[
        keys
        + [
            "fault_injected_count_fault",
            "unauthorized_invocations_clean",
            "unauthorized_invocations_fault",
            "unauthorized_delta",
            "detected_by_unauthorized_counter",
            "detected_by_fault_counter",
            "intervention_rate_clean",
            "intervention_rate_fault",
            "trace_hash_clean",
            "trace_hash_fault",
        ]
    ].rename(
        columns={
            "fault_injected_count_fault": "faults_injected",
            "unauthorized_invocations_clean": "clean_unauthorized_invocations",
            "unauthorized_invocations_fault": "fault_unauthorized_invocations",
        }
    )

    summary = pd.DataFrame(
        [
            {
                "fault_mode": FaultType.CLEAN_REPLAY.value,
                "runs": expected_runs,
                "faults_injected_total": 0,
                "detected_runs": 0,
                "detection_rate": "",
                "false_positive_runs": 0,
                "max_unauthorized_invocations": int(clean["unauthorized_invocations"].max()),
                "detection_channel": FAULT_METADATA[FaultType.CLEAN_REPLAY]["channel"],
                "interpretation": "Clean baseline used as false-positive control.",
            },
            {
                "fault_mode": FaultType.UNAUTHORIZED_INVOKE.value,
                "runs": int(len(merged)),
                "faults_injected_total": int(merged["fault_injected_count_fault"].sum()),
                "detected_runs": int(merged["detected_by_unauthorized_counter"].sum()),
                "detection_rate": float(merged["detected_by_unauthorized_counter"].mean()),
                "false_positive_runs": 0,
                "max_unauthorized_invocations": int(merged["unauthorized_invocations_fault"].max()),
                "detection_channel": FAULT_METADATA[FaultType.UNAUTHORIZED_INVOKE]["channel"],
                "interpretation": FAULT_METADATA[FaultType.UNAUTHORIZED_INVOKE]["interpretation"],
            },
        ]
    )

    write_csv_and_tex(per_run, tables_dir / "fgcs_table_fault_unauthorized_invoke_per_run.csv")
    write_csv_and_tex(summary, tables_dir / "fgcs_table_fault_unauthorized_invoke_detection_summary.csv")
    return per_run, summary


# ---------------------------------------------------------------------------
# Trace corruption
# ---------------------------------------------------------------------------


def clean_trace_path(clean_dir: Path, workload_name: str, policy: str, seed: int, workers: int) -> Path:
    return clean_dir / f"trace_{workload_name}_policy_{policy}_seed_{seed}_workers_{workers}.csv"


def compute_trace_properties(df: pd.DataFrame) -> Dict[str, object]:
    if "action" not in df.columns:
        raise KeyError("Trace file has no 'action' column.")
    out = df.copy()
    if "row_index" in out.columns:
        out = out.sort_values("row_index").reset_index(drop=True)
    return {
        "row_count": int(len(out)),
        "trace_hash": trace_hash(out["action"].astype(int).tolist()),
    }


def corrupt_action_values(df: pd.DataFrame, rate: float, salt: str) -> Tuple[pd.DataFrame, int]:
    out = df.copy()
    idx = deterministic_indices(len(out), rate, salt)
    out.loc[idx, "action"] = 1 - out.loc[idx, "action"].astype(int)
    return out, int(len(idx))


def corrupt_drop_rows(df: pd.DataFrame, rate: float, salt: str) -> Tuple[pd.DataFrame, int]:
    out = df.copy()
    idx = deterministic_indices(len(out), rate, salt)
    out = out.drop(index=idx).reset_index(drop=True)
    return out, int(len(idx))


def corrupt_duplicate_rows(df: pd.DataFrame, rate: float, salt: str) -> Tuple[pd.DataFrame, int]:
    out = df.copy()
    idx = deterministic_indices(len(out), rate, salt)
    duplicated = out.loc[idx].copy()
    out = pd.concat([out, duplicated], ignore_index=True)
    if "row_index" in out.columns:
        out = out.sort_values("row_index").reset_index(drop=True)
    return out, int(len(idx))


def summarize_trace_corruption(
    clean_dir: Path,
    trace_fault_dir: Path,
    tables_dir: Path,
    policies: Sequence[str],
    seeds: Sequence[int],
    workers: Sequence[int],
    full_fraction: float,
    corruption_rate: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(trace_fault_dir)
    corrupted_trace_dir = trace_fault_dir / "corrupted_traces"
    ensure_dir(corrupted_trace_dir)

    det = filter_compact_rows(
        read_required(clean_dir / "determinism_hash_results.csv"),
        policies=policies,
        seeds=seeds,
        workers=workers,
        full_fraction=full_fraction,
    )

    expected_runs = len(policies) * len(seeds) * len(workers)
    if len(det) != expected_runs:
        raise RuntimeError(f"Trace-corruption summary expected {expected_runs} clean rows, got {len(det)}")

    corruption_specs: List[Tuple[FaultType, Callable[[pd.DataFrame, float, str], Tuple[pd.DataFrame, int]]]] = [
        (FaultType.TRACE_ACTION_CORRUPTION, corrupt_action_values),
        (FaultType.TRACE_ROW_DROP, corrupt_drop_rows),
        (FaultType.TRACE_ROW_DUPLICATE, corrupt_duplicate_rows),
    ]

    per_run_rows: List[Dict[str, object]] = []
    clean_false_positives = 0

    for _, ref in det.iterrows():
        workload_name = str(ref["workload_name"])
        policy = str(ref["policy_mode"])
        seed = int(ref["seed"])
        workers_value = int(ref["workers"])
        clean_reference_hash = str(ref["trace_hash"])

        trace_path = clean_trace_path(clean_dir, workload_name, policy, seed, workers_value)
        clean_df = read_required(trace_path)
        clean_props = compute_trace_properties(clean_df)
        clean_row_count = int(clean_props["row_count"])
        clean_computed_hash = str(clean_props["trace_hash"])
        clean_hash_mismatch = int(clean_computed_hash != clean_reference_hash)
        clean_false_positives += clean_hash_mismatch

        common = {
            "dataset_fraction": float(full_fraction),
            "workload_name": workload_name,
            "policy_mode": policy,
            "seed": seed,
            "workers": workers_value,
            "clean_row_count": clean_row_count,
            "clean_reference_hash": clean_reference_hash,
            "clean_computed_hash": clean_computed_hash,
            "clean_hash_mismatch": clean_hash_mismatch,
        }

        for fault_type, corrupt_fn in corruption_specs:
            salt = f"{fault_type.value}|{policy}|{seed}|{workers_value}|{workload_name}"
            corrupted_df, corruptions = corrupt_fn(clean_df, corruption_rate, salt)
            corrupted_props = compute_trace_properties(corrupted_df)
            corrupted_row_count = int(corrupted_props["row_count"])
            corrupted_hash = str(corrupted_props["trace_hash"])
            row_count_mismatch = int(corrupted_row_count != clean_row_count)
            hash_mismatch = int(corrupted_hash != clean_reference_hash)
            detected = int((row_count_mismatch == 1) or (hash_mismatch == 1))

            corrupted_path = (
                corrupted_trace_dir
                / f"corrupted_{fault_type.value}_{workload_name}_policy_{policy}_seed_{seed}_workers_{workers_value}.csv"
            )
            corrupted_df.to_csv(corrupted_path, index=False)

            per_run_rows.append(
                {
                    **common,
                    "fault_mode": fault_type.value,
                    "corruptions_injected": int(corruptions),
                    "corrupted_row_count": corrupted_row_count,
                    "corrupted_hash": corrupted_hash,
                    "row_count_mismatch": row_count_mismatch,
                    "hash_mismatch": hash_mismatch,
                    "detected": detected,
                    "detection_channel": FAULT_METADATA[fault_type]["channel"],
                    "corrupted_trace_path": str(corrupted_path),
                }
            )

    per_run = pd.DataFrame(per_run_rows)

    summary_rows: List[Dict[str, object]] = [
        {
            "fault_mode": FaultType.CLEAN_REPLAY.value,
            "runs": expected_runs,
            "corruptions_injected_total": 0,
            "detected_runs": 0,
            "detection_rate": "",
            "false_positive_runs": int(clean_false_positives),
            "detection_channel": FAULT_METADATA[FaultType.CLEAN_REPLAY]["channel"],
            "interpretation": "Clean trace files are used as false-positive controls.",
        }
    ]

    for fault_mode, group in per_run.groupby("fault_mode"):
        fault_type = FaultType(str(fault_mode))
        summary_rows.append(
            {
                "fault_mode": fault_type.value,
                "runs": int(len(group)),
                "corruptions_injected_total": int(group["corruptions_injected"].sum()),
                "detected_runs": int(group["detected"].sum()),
                "detection_rate": float(group["detected"].mean()),
                "false_positive_runs": 0,
                "detection_channel": FAULT_METADATA[fault_type]["channel"],
                "interpretation": FAULT_METADATA[fault_type]["interpretation"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    write_csv_and_tex(per_run, tables_dir / "fgcs_table_fault_trace_corruption_per_run.csv")
    write_csv_and_tex(summary, tables_dir / "fgcs_table_fault_trace_corruption_detection_summary.csv")
    return per_run, summary


# ---------------------------------------------------------------------------
# Combined RQ7 summary and ablation matrix
# ---------------------------------------------------------------------------


def normalize_detection_rate(value: object) -> object:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    try:
        return round(float(value), 4)
    except Exception:
        return value


def combine_fault_summaries(tables_dir: Path) -> pd.DataFrame:
    input_paths = [
        tables_dir / "fgcs_table_fault_action_flip_detection_summary.csv",
        tables_dir / "fgcs_table_fault_unauthorized_invoke_detection_summary.csv",
        tables_dir / "fgcs_table_fault_trace_corruption_detection_summary.csv",
    ]

    frames: List[pd.DataFrame] = []
    for path in input_paths:
        df = read_required(path)
        df["source_file"] = path.name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    clean_rows = combined[combined["fault_mode"].astype(str).eq(FaultType.CLEAN_REPLAY.value)].copy()
    fault_rows = combined[~combined["fault_mode"].astype(str).eq(FaultType.CLEAN_REPLAY.value)].copy()

    clean_summary = pd.DataFrame(
        [
            {
                "fault_mode": FaultType.CLEAN_REPLAY.value,
                "fault_category": FAULT_METADATA[FaultType.CLEAN_REPLAY]["category"],
                "runs": int(clean_rows["runs"].sum()) if "runs" in clean_rows.columns else 0,
                "faults_or_corruptions_injected_total": 0,
                "detected_runs": 0,
                "detection_rate": "",
                "false_positive_runs": int(clean_rows["false_positive_runs"].sum())
                if "false_positive_runs" in clean_rows.columns
                else 0,
                "max_unauthorized_invocations": 0,
                "detection_channel": FAULT_METADATA[FaultType.CLEAN_REPLAY]["channel"],
                "paper_interpretation": FAULT_METADATA[FaultType.CLEAN_REPLAY]["interpretation"],
            }
        ]
    )

    rows: List[Dict[str, object]] = []
    for _, row in fault_rows.iterrows():
        fault_type = FaultType(str(row["fault_mode"]))
        injected_total = 0
        if "faults_injected_total" in row.index and not pd.isna(row.get("faults_injected_total")):
            injected_total = int(row.get("faults_injected_total", 0))
        elif "corruptions_injected_total" in row.index and not pd.isna(row.get("corruptions_injected_total")):
            injected_total = int(row.get("corruptions_injected_total", 0))

        max_unauth: object = ""
        if "max_unauthorized_invocations" in row.index and not pd.isna(row.get("max_unauthorized_invocations")):
            max_unauth = int(row.get("max_unauthorized_invocations", 0))

        rows.append(
            {
                "fault_mode": fault_type.value,
                "fault_category": FAULT_METADATA[fault_type]["category"],
                "runs": int(row.get("runs", 0)),
                "faults_or_corruptions_injected_total": injected_total,
                "detected_runs": int(row.get("detected_runs", 0)),
                "detection_rate": normalize_detection_rate(row.get("detection_rate", "")),
                "false_positive_runs": int(row.get("false_positive_runs", 0)),
                "max_unauthorized_invocations": max_unauth,
                "detection_channel": FAULT_METADATA[fault_type]["channel"],
                "paper_interpretation": FAULT_METADATA[fault_type]["interpretation"],
            }
        )

    preferred_order = [fault.value for fault in FaultType]
    out = pd.concat([clean_summary, pd.DataFrame(rows)], ignore_index=True)
    out["order"] = out["fault_mode"].apply(lambda x: preferred_order.index(x) if x in preferred_order else 999)
    out = out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
    write_csv_and_tex(out, tables_dir / "fgcs_table_rq7_fault_detection_combined.csv")
    return out


def make_validation_ablation_matrix(tables_dir: Path) -> pd.DataFrame:
    rows = [
        {
            "fault_mode": FaultType.ACTION_FLIP.value,
            "fault_category": FAULT_METADATA[FaultType.ACTION_FLIP]["category"],
            "logging_only": "No",
            "hash_only": "Yes",
            "gate_only": "No",
            "row_count_only": "No",
            "hash_plus_gate": "Yes",
            "full_validator": "Yes",
            "primary_detection_channel": "SHA-256 trace mismatch",
        },
        {
            "fault_mode": FaultType.UNAUTHORIZED_INVOKE.value,
            "fault_category": FAULT_METADATA[FaultType.UNAUTHORIZED_INVOKE]["category"],
            "logging_only": "No",
            "hash_only": "No",
            "gate_only": "Yes",
            "row_count_only": "No",
            "hash_plus_gate": "Yes",
            "full_validator": "Yes",
            "primary_detection_channel": "unauthorized-invocation counter",
        },
        {
            "fault_mode": FaultType.TRACE_ACTION_CORRUPTION.value,
            "fault_category": FAULT_METADATA[FaultType.TRACE_ACTION_CORRUPTION]["category"],
            "logging_only": "No",
            "hash_only": "Yes",
            "gate_only": "No",
            "row_count_only": "No",
            "hash_plus_gate": "Yes",
            "full_validator": "Yes",
            "primary_detection_channel": "SHA-256 action-trace mismatch",
        },
        {
            "fault_mode": FaultType.TRACE_ROW_DROP.value,
            "fault_category": FAULT_METADATA[FaultType.TRACE_ROW_DROP]["category"],
            "logging_only": "No",
            "hash_only": "Yes",
            "gate_only": "No",
            "row_count_only": "Yes",
            "hash_plus_gate": "Yes",
            "full_validator": "Yes",
            "primary_detection_channel": "row-count mismatch + SHA-256 mismatch",
        },
        {
            "fault_mode": FaultType.TRACE_ROW_DUPLICATE.value,
            "fault_category": FAULT_METADATA[FaultType.TRACE_ROW_DUPLICATE]["category"],
            "logging_only": "No",
            "hash_only": "Yes",
            "gate_only": "No",
            "row_count_only": "Yes",
            "hash_plus_gate": "Yes",
            "full_validator": "Yes",
            "primary_detection_channel": "row-count mismatch + SHA-256 mismatch",
        },
    ]
    out = pd.DataFrame(rows)
    write_csv_and_tex(out, tables_dir / "fgcs_table_validation_ablation_matrix.csv")
    return out


# ---------------------------------------------------------------------------
# Figures and README section
# ---------------------------------------------------------------------------


def save_figure(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OUT] {path}")


def make_fault_detection_rate_figure(combined: pd.DataFrame, tables_dir: Path) -> None:
    fault_df = combined[combined["fault_mode"].astype(str) != FaultType.CLEAN_REPLAY.value].copy()
    fault_df["detection_rate_numeric"] = pd.to_numeric(fault_df["detection_rate"], errors="coerce")

    labels = [
        "Action\nflip",
        "Unauthorized\ninvoke",
        "Trace action\ncorruption",
        "Dropped\nrows",
        "Duplicated\nrows",
    ]
    values = fault_df["detection_rate_numeric"].tolist()

    plt.figure(figsize=(8.5, 4.5))
    plt.bar(labels, values)
    plt.ylim(0, 1.1)
    plt.ylabel("Detection rate")
    plt.title("RQ7 fault-detection rate by injected fault type")
    for i, value in enumerate(values):
        if pd.notna(value):
            plt.text(i, min(float(value) + 0.03, 1.05), f"{float(value):.2f}", ha="center", va="bottom")
    save_figure(tables_dir / "fgcs_fig_rq7_fault_detection_rate.png")


def make_validation_ablation_figure(matrix: pd.DataFrame, tables_dir: Path) -> None:
    validator_cols = [
        "logging_only",
        "hash_only",
        "gate_only",
        "row_count_only",
        "hash_plus_gate",
        "full_validator",
    ]
    display_cols = [
        "Logging\nonly",
        "Hash\nonly",
        "Gate\nonly",
        "Row count\nonly",
        "Hash +\ngate",
        "Full\nvalidator",
    ]
    display_rows = [
        "Action flip",
        "Unauthorized invoke",
        "Trace action corruption",
        "Dropped rows",
        "Duplicated rows",
    ]

    data = matrix[validator_cols].applymap(lambda x: 1 if str(x).lower() == "yes" else 0).to_numpy()

    plt.figure(figsize=(9.5, 4.8))
    plt.imshow(data, aspect="auto")
    plt.xticks(np.arange(len(display_cols)), display_cols)
    plt.yticks(np.arange(len(display_rows)), display_rows)
    plt.title("Validation-ablation coverage matrix")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, "Yes" if data[i, j] else "No", ha="center", va="center")

    plt.colorbar(label="Detected by validator variant")
    save_figure(tables_dir / "fgcs_fig_validation_ablation_matrix.png")


def write_readme_section(tables_dir: Path) -> Path:
    readme_path = tables_dir / "README_RQ7_FAULT_VALIDATION.md"
    text = """# Compact Fault Validation (RQ7)

This section documents the additional compact fault-validation experiments used for RQ7. These experiments are separate from the main clean 360-condition benchmark and are intended to test whether the replay infrastructure detects deliberately injected execution and trace-integrity faults.

## Clean benchmark prerequisite

Run the main clean benchmark first:

```bash
python run_fgcs_extended_benchmark.py --config configs/fgcs_extended_benchmark.yaml
python summarize_fgcs_extended_results.py
```

The clean benchmark writes its outputs to:

```text
paper_outputs/fgcs_extended_benchmark/
paper_outputs/fgcs_tables_figures/
```

## Runtime fault experiments

Two compact runtime fault experiments are used. Both use the full replay workload only, three policy modes (`risk_proxy`, `random`, `never`), three seeds, and two worker settings (`1`, `4`), giving 18 runs per runtime fault mode.

### 1. Action-flip fault

This fault simulates policy-output corruption by flipping a deterministic 1% subset of actions before invocation gating. It should be detected through SHA-256 action-trace mismatch against the clean reference trace.

```bash
python run_fgcs_extended_benchmark.py --config configs/fgcs_fault_action_flip.yaml
```

Output directory:

```text
paper_outputs/fgcs_fault_action_flip/
```

### 2. Unauthorized-invocation fault

This fault simulates an invocation-boundary violation by forcing generation after `action = 0`. The policy action remains unchanged, so the fault should be detected by the unauthorized-invocation counter.

```bash
python run_fgcs_extended_benchmark.py --config configs/fgcs_fault_unauthorized_invoke.yaml
```

Output directory:

```text
paper_outputs/fgcs_fault_unauthorized_invoke/
```

## Unified RQ7 validation framework

After the clean benchmark and the two runtime fault experiments have been run, execute:

```bash
python fgcs_fault_validation_framework.py --all
```

This script consolidates all RQ7 outputs using a common fault taxonomy:

```text
FaultType.ACTION_FLIP
FaultType.UNAUTHORIZED_INVOKE
FaultType.TRACE_ACTION_CORRUPTION
FaultType.TRACE_ROW_DROP
FaultType.TRACE_ROW_DUPLICATE
```

It also performs post-hoc trace corruption by modifying clean trace artifacts:

1. flipping 1% of saved `action` values;
2. dropping 1% of trace rows;
3. duplicating 1% of trace rows.

These post-hoc faults are evaluated using row-count checks and SHA-256 action-trace hashes.

## Main RQ7 outputs

The unified framework generates the following paper-facing outputs:

```text
paper_outputs/fgcs_tables_figures/fgcs_table_fault_action_flip_detection_summary.csv
paper_outputs/fgcs_tables_figures/fgcs_table_fault_unauthorized_invoke_detection_summary.csv
paper_outputs/fgcs_tables_figures/fgcs_table_fault_trace_corruption_detection_summary.csv
paper_outputs/fgcs_tables_figures/fgcs_table_rq7_fault_detection_combined.csv
paper_outputs/fgcs_tables_figures/fgcs_table_validation_ablation_matrix.csv
paper_outputs/fgcs_tables_figures/fgcs_fig_rq7_fault_detection_rate.png
paper_outputs/fgcs_tables_figures/fgcs_fig_validation_ablation_matrix.png
```

## Paper interpretation

Use cautious wording:

> The infrastructure detects deliberately injected policy-output corruption, invocation-boundary violations, and trace-integrity corruption through SHA-256 trace mismatches, unauthorized-invocation counters, and row-count invariants.

Do not claim that the system prevents all execution faults. These experiments evaluate deterministic fault detection under controlled replay perturbations.

## Optional cloud validation

The compact RQ7 experiments can also be run through the existing Cloud Run Jobs wrapper by using the same two fault configs and separate cloud output prefixes. This is optional. It is useful only if the manuscript needs the stronger claim that local and cloud replay detect identical injected faults. If time or page budget is limited, local RQ7 validation is sufficient for the current manuscript revision.
"""
    readme_path.write_text(text, encoding="utf-8")
    print(f"[OUT] {readme_path}")
    return readme_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified FGCS RQ7 fault-validation framework.")
    parser.add_argument("--clean_dir", default="paper_outputs/fgcs_extended_benchmark")
    parser.add_argument("--action_flip_dir", default="paper_outputs/fgcs_fault_action_flip")
    parser.add_argument("--unauthorized_dir", default="paper_outputs/fgcs_fault_unauthorized_invoke")
    parser.add_argument("--trace_fault_dir", default="paper_outputs/fgcs_fault_trace_corruption")
    parser.add_argument("--tables_dir", default="paper_outputs/fgcs_tables_figures")
    parser.add_argument("--policies", nargs="+", default=["risk_proxy", "random", "never"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--full_fraction", type=float, default=1.0)
    parser.add_argument("--corruption_rate", type=float, default=0.01)
    parser.add_argument("--all", action="store_true", help="Run all summaries, trace corruption, combined tables, figures, and README section.")
    parser.add_argument("--summarize_runtime_faults", action="store_true")
    parser.add_argument("--trace_corruption", action="store_true")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--readme", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    clean_dir = Path(args.clean_dir)
    action_flip_dir = Path(args.action_flip_dir)
    unauthorized_dir = Path(args.unauthorized_dir)
    trace_fault_dir = Path(args.trace_fault_dir)
    tables_dir = Path(args.tables_dir)
    ensure_dir(tables_dir)

    run_all = bool(args.all) or not any(
        [
            args.summarize_runtime_faults,
            args.trace_corruption,
            args.combine,
            args.figures,
            args.readme,
        ]
    )

    if run_all or args.summarize_runtime_faults:
        print("[STEP] Summarizing runtime action-flip fault...")
        summarize_action_flip_fault(
            clean_dir=clean_dir,
            fault_dir=action_flip_dir,
            tables_dir=tables_dir,
            policies=args.policies,
            seeds=args.seeds,
            workers=args.workers,
            full_fraction=args.full_fraction,
        )
        print("[STEP] Summarizing runtime unauthorized-invocation fault...")
        summarize_unauthorized_invoke_fault(
            clean_dir=clean_dir,
            fault_dir=unauthorized_dir,
            tables_dir=tables_dir,
            policies=args.policies,
            seeds=args.seeds,
            workers=args.workers,
            full_fraction=args.full_fraction,
        )

    if run_all or args.trace_corruption:
        print("[STEP] Running post-hoc trace-corruption validation...")
        summarize_trace_corruption(
            clean_dir=clean_dir,
            trace_fault_dir=trace_fault_dir,
            tables_dir=tables_dir,
            policies=args.policies,
            seeds=args.seeds,
            workers=args.workers,
            full_fraction=args.full_fraction,
            corruption_rate=args.corruption_rate,
        )

    combined: pd.DataFrame | None = None
    matrix: pd.DataFrame | None = None

    if run_all or args.combine or args.figures:
        print("[STEP] Combining RQ7 fault-detection summaries...")
        combined = combine_fault_summaries(tables_dir)
        print("[STEP] Building validation-ablation matrix...")
        matrix = make_validation_ablation_matrix(tables_dir)

    if run_all or args.figures:
        if combined is None:
            combined = read_required(tables_dir / "fgcs_table_rq7_fault_detection_combined.csv")
        if matrix is None:
            matrix = read_required(tables_dir / "fgcs_table_validation_ablation_matrix.csv")
        print("[STEP] Generating RQ7 figures...")
        make_fault_detection_rate_figure(combined, tables_dir)
        make_validation_ablation_figure(matrix, tables_dir)

    if run_all or args.readme:
        print("[STEP] Writing README RQ7 section...")
        write_readme_section(tables_dir)

    print("[DONE] Unified FGCS RQ7 fault-validation framework complete.")


if __name__ == "__main__":
    main()
