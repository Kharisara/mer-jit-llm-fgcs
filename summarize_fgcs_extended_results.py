#!/usr/bin/env python
"""
summarize_fgcs_extended_results.py

Paper-facing summarization script for the FGCS extended benchmark outputs.

Expected current benchmark output directory:
    paper_outputs/fgcs_extended_benchmark/

Expected current benchmark files:
    scaling_and_runtime_results.csv
    stage_latency_summary.csv
    determinism_hash_results.csv
    parallel_speedup_results.csv
    policy_ablation_costs.csv
    live_bc_predictions.csv   (optional but recommended when bc_live is enabled)

Outputs:
    paper_outputs/fgcs_tables_figures/

The script is intentionally defensive:
- It accepts the current 360-run FGCS benchmark schema.
- It avoids the older "extended_scalability_results.csv" / "workload_size" schema.
- It does not require the old fault-injection CSVs.
- It produces CSV, LaTeX, PNG figures, and a compact Markdown summary.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Basic utilities
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def read_csv_required(path: Path) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Optional file not found; skipping: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def save_table(df: pd.DataFrame, path: Path, latex: bool = True) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[OUT] {path}")

    if latex:
        tex_path = path.with_suffix(".tex")
        try:
            df.to_latex(tex_path, index=False, escape=True)
            print(f"[OUT] {tex_path}")
        except Exception as exc:
            print(f"[WARN] Could not write LaTeX table {tex_path}: {exc}")


def save_figure(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OUT] {path}")


def safe_round(df: pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["float64", "float32"]).columns
    out[float_cols] = out[float_cols].round(decimals)
    return out


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def require_one_column(df: pd.DataFrame, candidates: Sequence[str], table_name: str) -> str:
    col = first_existing_column(df, candidates)
    if col is None:
        raise KeyError(
            f"{table_name}: none of these columns were found: {list(candidates)}. "
            f"Available columns: {list(df.columns)}"
        )
    return col


def add_workload_size_alias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Current benchmark schema usually uses decision_points, dataset_fraction,
    and workload_name. Some derived CSVs, especially parallel_speedup_results.csv,
    may omit decision_points. This alias keeps downstream table/figure code stable.

    Preferred workload_size meaning:
    1. decision_points, when available.
    2. n, if an older script used that name.
    3. dataset_fraction, only as a safe fallback when the file contains no absolute count.
    """
    out = df.copy()
    if "workload_size" not in out.columns:
        if "decision_points" in out.columns:
            out["workload_size"] = out["decision_points"]
        elif "n" in out.columns:
            out["workload_size"] = out["n"]
        elif "dataset_fraction" in out.columns:
            out["workload_size"] = out["dataset_fraction"]
    return out


def harmonize_workload_columns(df: pd.DataFrame, scaling_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach decision_points/workload_size to derived result files using the
    authoritative scaling_and_runtime_results.csv mapping. This fixes current
    files such as parallel_speedup_results.csv, which may contain only
    dataset_fraction and workload_name.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    if "decision_points" in out.columns and "workload_size" in out.columns:
        return out

    scale = scaling_df.copy()
    if "decision_points" not in scale.columns:
        scale = add_workload_size_alias(scale)
        if "workload_size" in scale.columns and "decision_points" not in scale.columns:
            scale["decision_points"] = scale["workload_size"]

    merge_keys = []
    for key in ["dataset_fraction", "workload_name"]:
        if key in out.columns and key in scale.columns:
            merge_keys.append(key)

    if merge_keys and "decision_points" in scale.columns and "decision_points" not in out.columns:
        mapping = (
            scale[merge_keys + ["decision_points"]]
            .drop_duplicates(subset=merge_keys)
            .copy()
        )
        out = out.merge(mapping, on=merge_keys, how="left")

    out = add_workload_size_alias(out)
    return out


def display_workload_value(value) -> object:
    """Return a clean workload value for tables without forcing floats to int unsafely."""
    try:
        f = float(value)
        if math.isfinite(f) and abs(f - round(f)) < 1e-9:
            return int(round(f))
        return round(f, 6)
    except Exception:
        return value


def numeric_sort_key(value) -> Tuple[int, float, str]:
    try:
        return (0, float(value), "")
    except Exception:
        return (1, 0.0, str(value))


def compact_policy_order(policies: Iterable[str]) -> List[str]:
    preferred = ["never", "proxy", "bc", "bc_live", "random", "always"]
    available = list(dict.fromkeys(str(p) for p in policies))
    ordered = [p for p in preferred if p in available]
    ordered += sorted([p for p in available if p not in ordered])
    return ordered


def get_runtime_col(df: pd.DataFrame, table_name: str) -> str:
    return require_one_column(
        df,
        [
            "runtime_seconds",
            "replay_runtime_seconds",
            "total_runtime_seconds",
            "wall_time_seconds",
            "elapsed_seconds",
        ],
        table_name,
    )


def get_throughput_col(df: pd.DataFrame, table_name: str) -> Optional[str]:
    return first_existing_column(
        df,
        [
            "throughput_points_per_second",
            "throughput",
            "points_per_second",
            "decision_points_per_second",
        ],
    )


def get_intervention_rate_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing_column(
        df,
        [
            "intervention_rate",
            "intervention_fraction",
            "mean_intervention_rate",
        ],
    )


def get_policy_time_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing_column(
        df,
        [
            "policy_action_time_seconds",
            "policy_time_seconds",
            "policy_runtime_seconds",
            "policy_inference_seconds",
            "mean_policy_inference_seconds",
            "policy_inference_ms",
            "mean_policy_inference_ms",
        ],
    )


def get_stage_latency_ms_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing_column(
        df,
        [
            "mean_ms",
            "latency_ms",
            "stage_latency_ms",
            "mean_latency_ms",
            "duration_ms",
            "policy_inference_ms",
            "state_loading_ms",
        ],
    )


def mean_or_nan(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return float("nan")
    return float(pd.to_numeric(series, errors="coerce").mean())


def max_or_nan(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return float("nan")
    return float(pd.to_numeric(series, errors="coerce").max())


def min_or_nan(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return float("nan")
    return float(pd.to_numeric(series, errors="coerce").min())


# ============================================================
# Validation and schema summary
# ============================================================

def validate_current_outputs(
    scaling_df: pd.DataFrame,
    speedup_df: pd.DataFrame,
    det_df: pd.DataFrame,
    policy_cost_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    live_bc_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(name: str, df: pd.DataFrame) -> None:
        rows.append(
            {
                "file": name,
                "rows": len(df),
                "columns": len(df.columns),
                "available_columns": ", ".join(df.columns.astype(str).tolist()),
            }
        )

    add("scaling_and_runtime_results.csv", scaling_df)
    add("parallel_speedup_results.csv", speedup_df)
    add("determinism_hash_results.csv", det_df)
    add("policy_ablation_costs.csv", policy_cost_df)
    add("stage_latency_summary.csv", stage_df)
    if len(live_bc_df):
        add("live_bc_predictions.csv", live_bc_df)

    return pd.DataFrame(rows)


def make_benchmark_design_table(scaling_df: pd.DataFrame) -> pd.DataFrame:
    scaling_df = add_workload_size_alias(scaling_df)

    n_rows = len(scaling_df)
    workloads = (
        scaling_df["workload_name"].nunique()
        if "workload_name" in scaling_df.columns
        else scaling_df["workload_size"].nunique()
    )
    policies = scaling_df["policy_mode"].nunique() if "policy_mode" in scaling_df.columns else np.nan
    seeds = scaling_df["seed"].nunique() if "seed" in scaling_df.columns else np.nan
    workers = scaling_df["workers"].nunique() if "workers" in scaling_df.columns else np.nan

    expected = np.nan
    if not any(pd.isna(x) for x in [workloads, policies, seeds, workers]):
        expected = int(workloads * policies * seeds * workers)

    fractions = (
        ", ".join(str(x) for x in sorted(scaling_df["dataset_fraction"].unique(), key=numeric_sort_key))
        if "dataset_fraction" in scaling_df.columns
        else ""
    )
    worker_values = (
        ", ".join(str(x) for x in sorted(scaling_df["workers"].unique(), key=numeric_sort_key))
        if "workers" in scaling_df.columns
        else ""
    )
    policies_list = (
        ", ".join(compact_policy_order(scaling_df["policy_mode"].unique()))
        if "policy_mode" in scaling_df.columns
        else ""
    )

    return pd.DataFrame(
        [
            {"item": "Observed benchmark rows", "value": n_rows},
            {"item": "Expected benchmark rows", "value": expected},
            {"item": "Workload settings", "value": workloads},
            {"item": "Dataset fractions", "value": fractions},
            {"item": "Policy modes", "value": policies_list},
            {"item": "Seeds", "value": seeds},
            {"item": "Worker settings", "value": worker_values},
        ]
    )


# ============================================================
# Table generation
# ============================================================

def make_runtime_scaling_table(scaling_df: pd.DataFrame) -> pd.DataFrame:
    df = add_workload_size_alias(scaling_df)
    runtime_col = get_runtime_col(df, "scaling_and_runtime_results.csv")
    throughput_col = get_throughput_col(df, "scaling_and_runtime_results.csv")
    intervention_col = get_intervention_rate_col(df)

    group_cols = [
        c for c in ["dataset_fraction", "workload_name", "workload_size", "policy_mode", "workers"]
        if c in df.columns
    ]

    agg: Dict[str, object] = {
        runtime_col: ["mean", "std", "min", "max"],
    }

    if throughput_col:
        agg[throughput_col] = ["mean", "std", "max"]

    if intervention_col:
        agg[intervention_col] = ["mean", "std"]

    for optional in [
        "unauthorized_invocations",
        "fault_injected_count",
        "state_missing_count",
        "memory_delta_mb",
    ]:
        if optional in df.columns:
            agg[optional] = "max"

    table = df.groupby(group_cols, as_index=False).agg(agg)
    table.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in table.columns
    ]

    rename = {
        f"{runtime_col}_mean": "runtime_seconds_mean",
        f"{runtime_col}_std": "runtime_seconds_std",
        f"{runtime_col}_min": "runtime_seconds_min",
        f"{runtime_col}_max": "runtime_seconds_max",
    }
    if throughput_col:
        rename.update(
            {
                f"{throughput_col}_mean": "throughput_points_per_second_mean",
                f"{throughput_col}_std": "throughput_points_per_second_std",
                f"{throughput_col}_max": "throughput_points_per_second_max",
            }
        )
    if intervention_col:
        rename.update(
            {
                f"{intervention_col}_mean": "intervention_rate_mean",
                f"{intervention_col}_std": "intervention_rate_std",
            }
        )

    table = table.rename(columns=rename)
    sort_cols = [c for c in ["dataset_fraction", "workload_size", "policy_mode", "workers"] if c in table.columns]
    return safe_round(table.sort_values(sort_cols))


def make_full_workload_runtime_table(scaling_df: pd.DataFrame) -> pd.DataFrame:
    df = add_workload_size_alias(scaling_df)
    runtime_col = get_runtime_col(df, "scaling_and_runtime_results.csv")
    throughput_col = get_throughput_col(df, "scaling_and_runtime_results.csv")
    intervention_col = get_intervention_rate_col(df)

    max_workload = df["workload_size"].max()
    full = df[df["workload_size"] == max_workload].copy()

    group_cols = ["policy_mode", "workers"]
    agg: Dict[str, object] = {
        runtime_col: ["mean", "std"],
    }
    if throughput_col:
        agg[throughput_col] = ["mean", "std", "max"]
    if intervention_col:
        agg[intervention_col] = "mean"

    for optional in ["unauthorized_invocations", "fault_injected_count", "state_missing_count"]:
        if optional in full.columns:
            agg[optional] = "max"

    table = full.groupby(group_cols, as_index=False).agg(agg)
    table.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in table.columns
    ]

    table = table.rename(
        columns={
            f"{runtime_col}_mean": "runtime_seconds_mean",
            f"{runtime_col}_std": "runtime_seconds_std",
            f"{throughput_col}_mean" if throughput_col else "": "throughput_points_per_second_mean",
            f"{throughput_col}_std" if throughput_col else "": "throughput_points_per_second_std",
            f"{throughput_col}_max" if throughput_col else "": "throughput_points_per_second_max",
            f"{intervention_col}_mean" if intervention_col else "": "intervention_rate_mean",
        }
    )
    table = table[[c for c in table.columns if c != ""]]
    table.insert(0, "workload_size", display_workload_value(max_workload))
    return safe_round(table.sort_values(["policy_mode", "workers"]))


def make_speedup_summary_table(speedup_df: pd.DataFrame) -> pd.DataFrame:
    df = add_workload_size_alias(speedup_df)

    group_cols = [
        c for c in ["dataset_fraction", "workload_name", "workload_size", "policy_mode", "workers"]
        if c in df.columns
    ]

    agg: Dict[str, object] = {}
    for col in [
        "runtime_seconds",
        "throughput_points_per_second",
        "speedup_vs_single_worker",
        "throughput_gain_vs_single_worker",
    ]:
        if col in df.columns:
            agg[col] = ["mean", "std", "min", "max"]

    table = df.groupby(group_cols, as_index=False).agg(agg)
    table.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in table.columns
    ]

    sort_cols = [c for c in ["dataset_fraction", "workload_size", "policy_mode", "workers"] if c in table.columns]
    return safe_round(table.sort_values(sort_cols))


def make_full_workload_speedup_table(speedup_df: pd.DataFrame) -> pd.DataFrame:
    df = add_workload_size_alias(speedup_df)
    max_workload = df["workload_size"].max()
    full = df[df["workload_size"] == max_workload].copy()

    group_cols = ["policy_mode", "workers"]
    agg_cols = [
        c for c in [
            "runtime_seconds",
            "throughput_points_per_second",
            "speedup_vs_single_worker",
            "throughput_gain_vs_single_worker",
        ]
        if c in full.columns
    ]

    table = full.groupby(group_cols, as_index=False)[agg_cols].agg(["mean", "std", "max"])
    table.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in table.columns
    ]
    table = table.reset_index()
    table.insert(0, "workload_size", display_workload_value(max_workload))
    return safe_round(table.sort_values(["policy_mode", "workers"]))


def make_determinism_table(det_df: pd.DataFrame) -> pd.DataFrame:
    df = add_workload_size_alias(det_df)

    group_cols = [
        c for c in ["dataset_fraction", "workload_name", "workload_size", "policy_mode"]
        if c in df.columns
    ]

    agg: Dict[str, object] = {}

    if "trace_hash" in df.columns:
        agg["trace_hash"] = pd.Series.nunique
    if "hash_match" in df.columns:
        agg["hash_match"] = "min"
    if "intervention_rate_delta" in df.columns:
        agg["intervention_rate_delta"] = ["max", "mean"]
    if "intervention_rate" in df.columns:
        agg["intervention_rate"] = ["mean", "std"]
    if "reference_intervention_rate" in df.columns:
        agg["reference_intervention_rate"] = "mean"
    if "unauthorized_invocations" in df.columns:
        agg["unauthorized_invocations"] = "max"
    if "fault_injected_count" in df.columns:
        agg["fault_injected_count"] = "max"

    table = df.groupby(group_cols, as_index=False).agg(agg)
    table.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in table.columns
    ]

    table = table.rename(
        columns={
            "trace_hash_nunique": "unique_trace_hashes",
            "hash_match_min": "minimum_hash_match",
            "intervention_rate_delta_max": "max_intervention_rate_delta",
            "intervention_rate_delta_mean": "mean_intervention_rate_delta",
            "intervention_rate_mean": "mean_intervention_rate",
            "intervention_rate_std": "std_intervention_rate",
            "reference_intervention_rate_mean": "reference_intervention_rate",
            "unauthorized_invocations_max": "max_unauthorized_invocations",
            "fault_injected_count_max": "max_fault_injected_count",
        }
    )

    if "unique_trace_hashes" in table.columns:
        table["deterministic_replay_interpretation"] = np.where(
            table["policy_mode"].astype(str).eq("random"),
            "seed-dependent stochastic baseline",
            np.where(table["unique_trace_hashes"].eq(1), "stable across seeds/workers", "check trace variation"),
        )

    sort_cols = [c for c in ["dataset_fraction", "workload_size", "policy_mode"] if c in table.columns]
    return safe_round(table.sort_values(sort_cols))


def make_determinism_compact_table(det_df: pd.DataFrame) -> pd.DataFrame:
    df = add_workload_size_alias(det_df)

    group_cols = ["policy_mode"]
    agg: Dict[str, object] = {}
    if "trace_hash" in df.columns:
        agg["trace_hash"] = pd.Series.nunique
    if "hash_match" in df.columns:
        agg["hash_match"] = "min"
    if "unauthorized_invocations" in df.columns:
        agg["unauthorized_invocations"] = "max"
    if "fault_injected_count" in df.columns:
        agg["fault_injected_count"] = "max"
    if "intervention_rate_delta" in df.columns:
        agg["intervention_rate_delta"] = "max"

    table = df.groupby(group_cols, as_index=False).agg(agg)
    table = table.rename(
        columns={
            "trace_hash": "unique_trace_hashes_across_all_workloads",
            "hash_match": "minimum_hash_match",
            "unauthorized_invocations": "max_unauthorized_invocations",
            "fault_injected_count": "max_fault_injected_count",
            "intervention_rate_delta": "max_intervention_rate_delta",
        }
    )

    # Add expected number of unique hashes based on observed workloads.
    if "workload_name" in df.columns:
        observed_workloads = df["workload_name"].nunique()
    else:
        observed_workloads = df["workload_size"].nunique()

    if "unique_trace_hashes_across_all_workloads" in table.columns:
        table["interpretation"] = np.where(
            table["policy_mode"].astype(str).eq("random"),
            "stochastic policy varies across seeds as expected",
            np.where(
                table["unique_trace_hashes_across_all_workloads"].eq(observed_workloads),
                "one stable hash per workload",
                "inspect unexpected trace variation",
            ),
        )

    return safe_round(table.sort_values("policy_mode"))


def make_policy_ablation_table(policy_cost_df: pd.DataFrame) -> pd.DataFrame:
    df = add_workload_size_alias(policy_cost_df)
    runtime_col = get_runtime_col(df, "policy_ablation_costs.csv")
    throughput_col = get_throughput_col(df, "policy_ablation_costs.csv")
    intervention_col = get_intervention_rate_col(df)
    policy_time_col = get_policy_time_col(df)

    group_cols = [
        c for c in ["dataset_fraction", "workload_name", "workload_size", "policy_mode", "workers"]
        if c in df.columns
    ]

    agg: Dict[str, object] = {
        runtime_col: ["mean", "std"],
    }
    if throughput_col:
        agg[throughput_col] = ["mean", "std"]
    if intervention_col:
        agg[intervention_col] = ["mean", "std"]
    if policy_time_col:
        agg[policy_time_col] = ["mean", "std"]

    for optional in [
        "unauthorized_invocations",
        "fault_injected_count",
        "memory_delta_mb",
        "state_missing_count",
    ]:
        if optional in df.columns:
            agg[optional] = "max"

    table = df.groupby(group_cols, as_index=False).agg(agg)
    table.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in table.columns
    ]

    rename = {
        f"{runtime_col}_mean": "runtime_seconds_mean",
        f"{runtime_col}_std": "runtime_seconds_std",
    }
    if throughput_col:
        rename.update(
            {
                f"{throughput_col}_mean": "throughput_points_per_second_mean",
                f"{throughput_col}_std": "throughput_points_per_second_std",
            }
        )
    if intervention_col:
        rename.update(
            {
                f"{intervention_col}_mean": "intervention_rate_mean",
                f"{intervention_col}_std": "intervention_rate_std",
            }
        )
    if policy_time_col:
        rename.update(
            {
                f"{policy_time_col}_mean": "policy_timing_mean",
                f"{policy_time_col}_std": "policy_timing_std",
            }
        )

    table = table.rename(columns=rename)
    sort_cols = [c for c in ["dataset_fraction", "workload_size", "policy_mode", "workers"] if c in table.columns]
    return safe_round(table.sort_values(sort_cols))


def make_full_workload_policy_table(policy_cost_df: pd.DataFrame) -> pd.DataFrame:
    df = add_workload_size_alias(policy_cost_df)
    max_workload = df["workload_size"].max()
    full = df[df["workload_size"] == max_workload].copy()

    runtime_col = get_runtime_col(full, "policy_ablation_costs.csv")
    throughput_col = get_throughput_col(full, "policy_ablation_costs.csv")
    intervention_col = get_intervention_rate_col(full)
    policy_time_col = get_policy_time_col(full)

    group_cols = ["policy_mode", "workers"]
    agg: Dict[str, object] = {
        runtime_col: ["mean", "std"],
    }
    if throughput_col:
        agg[throughput_col] = ["mean", "std"]
    if intervention_col:
        agg[intervention_col] = ["mean", "std"]
    if policy_time_col:
        agg[policy_time_col] = ["mean", "std"]

    for optional in [
        "unauthorized_invocations",
        "fault_injected_count",
        "memory_delta_mb",
        "state_missing_count",
    ]:
        if optional in full.columns:
            agg[optional] = "max"

    table = full.groupby(group_cols, as_index=False).agg(agg)
    table.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in table.columns
    ]

    rename = {
        f"{runtime_col}_mean": "runtime_seconds_mean",
        f"{runtime_col}_std": "runtime_seconds_std",
    }
    if throughput_col:
        rename.update(
            {
                f"{throughput_col}_mean": "throughput_points_per_second_mean",
                f"{throughput_col}_std": "throughput_points_per_second_std",
            }
        )
    if intervention_col:
        rename.update(
            {
                f"{intervention_col}_mean": "intervention_rate_mean",
                f"{intervention_col}_std": "intervention_rate_std",
            }
        )
    if policy_time_col:
        rename.update(
            {
                f"{policy_time_col}_mean": "policy_timing_mean",
                f"{policy_time_col}_std": "policy_timing_std",
            }
        )

    table = table.rename(columns=rename)
    table.insert(0, "workload_size", display_workload_value(max_workload))
    return safe_round(table.sort_values(["policy_mode", "workers"]))


def make_stage_latency_table(stage_df: pd.DataFrame) -> pd.DataFrame:
    if stage_df.empty:
        return pd.DataFrame()

    df = add_workload_size_alias(stage_df)

    group_cols = [
        c for c in [
            "dataset_fraction",
            "workload_name",
            "workload_size",
            "policy_mode",
            "workers",
            "stage",
            "stage_name",
            "metric",
        ]
        if c in df.columns
    ]

    # Aggregate all numeric columns not already in group_cols.
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in group_cols and c not in ["seed"]
    ]

    if not group_cols or not numeric_cols:
        return safe_round(df)

    table = df.groupby(group_cols, as_index=False)[numeric_cols].mean()
    sort_cols = [c for c in ["dataset_fraction", "workload_size", "policy_mode", "workers", "stage", "stage_name"] if c in table.columns]
    return safe_round(table.sort_values(sort_cols))


def make_live_bc_summary_table(live_bc_df: pd.DataFrame) -> pd.DataFrame:
    if live_bc_df.empty:
        return pd.DataFrame()

    df = add_workload_size_alias(live_bc_df)

    # Normalize action column naming.
    action_col = first_existing_column(df, ["bc_live_action", "action", "predicted_action"])
    if action_col is None:
        raise KeyError(
            "live_bc_predictions.csv: no action column found. "
            "Expected one of: bc_live_action, action, predicted_action."
        )

    group_cols = [
        c for c in ["dataset_fraction", "workload_name", "workload_size", "seed", "workers"]
        if c in df.columns
    ]

    rows: List[Dict[str, object]] = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        action_values = pd.to_numeric(group[action_col], errors="coerce")
        row["rows"] = len(group)
        row["bc_live_intervention_rate"] = float(action_values.mean())
        row["bc_live_action_0_count"] = int((action_values == 0).sum())
        row["bc_live_action_1_count"] = int((action_values == 1).sum())

        for col in ["state_loading_ms", "policy_inference_ms"]:
            if col in group.columns:
                row[f"{col}_mean"] = mean_or_nan(group[col])
                row[f"{col}_std"] = float(pd.to_numeric(group[col], errors="coerce").std())

        rows.append(row)

    table = pd.DataFrame(rows)
    sort_cols = [c for c in ["dataset_fraction", "workload_size", "seed", "workers"] if c in table.columns]
    return safe_round(table.sort_values(sort_cols))


def make_live_bc_compact_table(live_bc_df: pd.DataFrame) -> pd.DataFrame:
    if live_bc_df.empty:
        return pd.DataFrame()

    df = add_workload_size_alias(live_bc_df)
    action_col = first_existing_column(df, ["bc_live_action", "action", "predicted_action"])
    if action_col is None:
        return pd.DataFrame()

    action_values = pd.to_numeric(df[action_col], errors="coerce")
    rows = [
        {"metric": "Live BC prediction rows", "value": len(df)},
        {"metric": "Live BC intervention rate", "value": float(action_values.mean())},
        {"metric": "Live BC action=0 count", "value": int((action_values == 0).sum())},
        {"metric": "Live BC action=1 count", "value": int((action_values == 1).sum())},
    ]

    for col in ["state_loading_ms", "policy_inference_ms"]:
        if col in df.columns:
            rows.append({"metric": f"{col} mean", "value": mean_or_nan(df[col])})
            rows.append({"metric": f"{col} max", "value": max_or_nan(df[col])})

    return safe_round(pd.DataFrame(rows))


def make_summary_findings_table(
    scaling_df: pd.DataFrame,
    det_df: pd.DataFrame,
    speedup_df: pd.DataFrame,
    policy_cost_df: pd.DataFrame,
    live_bc_df: pd.DataFrame,
) -> pd.DataFrame:
    scaling = add_workload_size_alias(scaling_df)
    det = add_workload_size_alias(det_df)
    speedup = add_workload_size_alias(speedup_df)
    policy_cost = add_workload_size_alias(policy_cost_df)

    runtime_col = get_runtime_col(scaling, "scaling_and_runtime_results.csv")
    throughput_col = get_throughput_col(scaling, "scaling_and_runtime_results.csv")
    intervention_col = get_intervention_rate_col(scaling)

    max_workload = scaling["workload_size"].max()
    full_scaling = scaling[scaling["workload_size"] == max_workload].copy()

    n_runs = len(scaling)
    policy_modes = compact_policy_order(scaling["policy_mode"].unique()) if "policy_mode" in scaling.columns else []
    worker_values = sorted(scaling["workers"].unique().tolist()) if "workers" in scaling.columns else []
    workload_count = scaling["workload_size"].nunique() if "workload_size" in scaling.columns else np.nan

    max_throughput = max_or_nan(full_scaling[throughput_col]) if throughput_col else float("nan")
    min_runtime = min_or_nan(full_scaling[runtime_col])
    max_runtime = max_or_nan(full_scaling[runtime_col])

    if "trace_hash" in det.columns:
        hash_summary = det.groupby("policy_mode")["trace_hash"].nunique().to_dict()
    else:
        hash_summary = {}

    max_unauthorized = (
        int(pd.to_numeric(det["unauthorized_invocations"], errors="coerce").max())
        if "unauthorized_invocations" in det.columns
        else np.nan
    )

    max_speedup = (
        max_or_nan(speedup["speedup_vs_single_worker"])
        if "speedup_vs_single_worker" in speedup.columns
        else float("nan")
    )

    live_bc_intervention_rate = np.nan
    if not live_bc_df.empty:
        action_col = first_existing_column(live_bc_df, ["bc_live_action", "action", "predicted_action"])
        if action_col:
            live_bc_intervention_rate = float(pd.to_numeric(live_bc_df[action_col], errors="coerce").mean())

    rows = [
        {
            "finding": "Benchmark coverage",
            "observed_value": f"{n_runs} runs",
            "paper_interpretation": (
                f"The benchmark covers {workload_count} workload fractions, "
                f"{len(policy_modes)} policies, and {len(worker_values)} worker settings."
            ),
        },
        {
            "finding": "Policy modes",
            "observed_value": ", ".join(policy_modes),
            "paper_interpretation": "The evaluation includes fixed, stochastic, proxy, offline BC, and live BC policy modes.",
        },
        {
            "finding": "Maximum replay workload",
            "observed_value": f"{max_workload:,} decision points",
            "paper_interpretation": "The benchmark reaches the full MELD-derived replay workload.",
        },
        {
            "finding": "Full-workload runtime range",
            "observed_value": f"{min_runtime:.6f}–{max_runtime:.6f} seconds",
            "paper_interpretation": "Runtime varies by policy mode and worker configuration under deterministic replay.",
        },
        {
            "finding": "Maximum full-workload throughput",
            "observed_value": f"{max_throughput:.6f} decision points/s" if not math.isnan(max_throughput) else "N/A",
            "paper_interpretation": "Replay throughput is directly measurable across workload and worker settings.",
        },
        {
            "finding": "Maximum observed speedup",
            "observed_value": f"{max_speedup:.6f}x" if not math.isnan(max_speedup) else "N/A",
            "paper_interpretation": "Parallel workers provide measurable acceleration, though gains depend on policy and workload.",
        },
        {
            "finding": "Deterministic trace hashes",
            "observed_value": str(hash_summary),
            "paper_interpretation": (
                "Deterministic policies should produce one stable trace per workload; "
                "the random baseline is expected to vary across seeds."
            ),
        },
        {
            "finding": "Unauthorized invocations",
            "observed_value": f"{max_unauthorized}",
            "paper_interpretation": "The policy-first gate prevented unauthorized generator invocation in normal replay.",
        },
        {
            "finding": "Live BC action rate",
            "observed_value": f"{live_bc_intervention_rate:.6f}" if not math.isnan(live_bc_intervention_rate) else "N/A",
            "paper_interpretation": (
                "The live BC policy executed end-to-end; action diversity should be reported conservatively."
            ),
        },
    ]

    if intervention_col and intervention_col in full_scaling.columns:
        full_interventions = (
            full_scaling.groupby("policy_mode")[intervention_col]
            .mean()
            .sort_index()
            .to_dict()
        )
        rows.append(
            {
                "finding": "Full-workload mean intervention rates",
                "observed_value": str({k: round(float(v), 6) for k, v in full_interventions.items()}),
                "paper_interpretation": "Intervention frequency is policy-dependent and execution-identifiable under replay.",
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Figure generation
# ============================================================

def plot_runtime_vs_workload(scaling_df: pd.DataFrame, out_path: Path) -> None:
    df = add_workload_size_alias(scaling_df)
    runtime_col = get_runtime_col(df, "scaling_and_runtime_results.csv")

    min_workers = df["workers"].min() if "workers" in df.columns else None
    if min_workers is not None:
        df = df[df["workers"] == min_workers].copy()

    fig_df = (
        df.groupby(["workload_size", "policy_mode"], as_index=False)[runtime_col]
        .mean()
        .sort_values(["policy_mode", "workload_size"])
    )

    plt.figure()
    for policy_mode, group in fig_df.groupby("policy_mode"):
        plt.plot(group["workload_size"], group[runtime_col], marker="o", label=str(policy_mode))

    plt.xlabel("Replay workload size (decision points)")
    plt.ylabel("Runtime seconds")
    title = "Runtime vs workload size"
    if min_workers is not None:
        title += f" (workers={min_workers})"
    plt.title(title)
    plt.xscale("log")
    plt.legend()
    save_figure(out_path)


def plot_throughput_vs_workload(scaling_df: pd.DataFrame, out_path: Path) -> None:
    df = add_workload_size_alias(scaling_df)
    throughput_col = get_throughput_col(df, "scaling_and_runtime_results.csv")
    if throughput_col is None:
        print("[WARN] No throughput column found; skipping throughput-vs-workload figure.")
        return

    min_workers = df["workers"].min() if "workers" in df.columns else None
    if min_workers is not None:
        df = df[df["workers"] == min_workers].copy()

    fig_df = (
        df.groupby(["workload_size", "policy_mode"], as_index=False)[throughput_col]
        .mean()
        .sort_values(["policy_mode", "workload_size"])
    )

    plt.figure()
    for policy_mode, group in fig_df.groupby("policy_mode"):
        plt.plot(group["workload_size"], group[throughput_col], marker="o", label=str(policy_mode))

    plt.xlabel("Replay workload size (decision points)")
    plt.ylabel("Throughput: decision points per second")
    title = "Throughput vs workload size"
    if min_workers is not None:
        title += f" (workers={min_workers})"
    plt.title(title)
    plt.xscale("log")
    plt.legend()
    save_figure(out_path)


def plot_worker_throughput_at_full_workload(scaling_df: pd.DataFrame, out_path: Path) -> None:
    df = add_workload_size_alias(scaling_df)
    throughput_col = get_throughput_col(df, "scaling_and_runtime_results.csv")
    if throughput_col is None:
        print("[WARN] No throughput column found; skipping worker throughput figure.")
        return

    max_workload = df["workload_size"].max()
    df = df[df["workload_size"] == max_workload].copy()

    fig_df = (
        df.groupby(["policy_mode", "workers"], as_index=False)[throughput_col]
        .mean()
        .sort_values(["policy_mode", "workers"])
    )

    plt.figure()
    for policy_mode, group in fig_df.groupby("policy_mode"):
        plt.plot(group["workers"], group[throughput_col], marker="o", label=str(policy_mode))

    plt.xlabel("Number of workers")
    plt.ylabel("Throughput: decision points per second")
    plt.title(f"Worker-level throughput at {int(max_workload):,} replay points")
    plt.legend()
    save_figure(out_path)


def plot_speedup_at_full_workload(speedup_df: pd.DataFrame, out_path: Path) -> None:
    if "speedup_vs_single_worker" not in speedup_df.columns:
        print("[WARN] speedup_vs_single_worker not found; skipping speedup figure.")
        return

    df = add_workload_size_alias(speedup_df)
    max_workload = df["workload_size"].max()
    df = df[df["workload_size"] == max_workload].copy()

    fig_df = (
        df.groupby(["policy_mode", "workers"], as_index=False)["speedup_vs_single_worker"]
        .mean()
        .sort_values(["policy_mode", "workers"])
    )

    plt.figure()
    for policy_mode, group in fig_df.groupby("policy_mode"):
        plt.plot(group["workers"], group["speedup_vs_single_worker"], marker="o", label=str(policy_mode))

    plt.xlabel("Number of workers")
    plt.ylabel("Speedup vs single worker")
    plt.title(f"Parallel speedup at {int(max_workload):,} replay points")
    plt.legend()
    save_figure(out_path)


def plot_intervention_rate_at_full_workload(scaling_df: pd.DataFrame, out_path: Path) -> None:
    df = add_workload_size_alias(scaling_df)
    intervention_col = get_intervention_rate_col(df)
    if intervention_col is None:
        print("[WARN] No intervention-rate column found; skipping intervention-rate figure.")
        return

    max_workload = df["workload_size"].max()
    df = df[df["workload_size"] == max_workload].copy()

    fig_df = (
        df.groupby("policy_mode", as_index=False)[intervention_col]
        .mean()
        .sort_values("policy_mode")
    )

    plt.figure()
    plt.bar(fig_df["policy_mode"].astype(str), fig_df[intervention_col])
    plt.xlabel("Policy mode")
    plt.ylabel("Mean intervention rate")
    plt.title(f"Policy-level intervention rates at {int(max_workload):,} replay points")
    plt.xticks(rotation=30, ha="right")
    save_figure(out_path)


def plot_policy_runtime_at_full_workload(policy_cost_df: pd.DataFrame, out_path: Path) -> None:
    df = add_workload_size_alias(policy_cost_df)
    runtime_col = get_runtime_col(df, "policy_ablation_costs.csv")

    max_workload = df["workload_size"].max()
    df = df[df["workload_size"] == max_workload].copy()

    fig_df = (
        df.groupby("policy_mode", as_index=False)[runtime_col]
        .mean()
        .sort_values("policy_mode")
    )

    plt.figure()
    plt.bar(fig_df["policy_mode"].astype(str), fig_df[runtime_col])
    plt.xlabel("Policy mode")
    plt.ylabel("Mean runtime seconds")
    plt.title(f"Policy runtime comparison at {int(max_workload):,} replay points")
    plt.xticks(rotation=30, ha="right")
    save_figure(out_path)


def plot_live_bc_action_distribution(live_bc_df: pd.DataFrame, out_path: Path) -> None:
    if live_bc_df.empty:
        print("[WARN] live_bc_predictions.csv not found; skipping live BC action-distribution figure.")
        return

    action_col = first_existing_column(live_bc_df, ["bc_live_action", "action", "predicted_action"])
    if action_col is None:
        print("[WARN] No live BC action column found; skipping live BC action-distribution figure.")
        return

    counts = live_bc_df[action_col].value_counts(dropna=False).sort_index()

    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Live BC predicted action")
    plt.ylabel("Count")
    plt.title("Live BC action distribution")
    save_figure(out_path)


# ============================================================
# Markdown report
# ============================================================

def write_markdown_summary(
    out_path: Path,
    design_table: pd.DataFrame,
    summary_findings: pd.DataFrame,
    determinism_compact: pd.DataFrame,
    live_bc_compact: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# FGCS Extended Benchmark Summary")
    lines.append("")
    lines.append("## Benchmark design")
    lines.append("")
    lines.append(design_table.to_markdown(index=False))
    lines.append("")
    lines.append("## Main findings")
    lines.append("")
    lines.append(summary_findings.to_markdown(index=False))
    lines.append("")
    lines.append("## Determinism compact summary")
    lines.append("")
    lines.append(determinism_compact.to_markdown(index=False))
    lines.append("")

    if not live_bc_compact.empty:
        lines.append("## Live BC compact summary")
        lines.append("")
        lines.append(live_bc_compact.to_markdown(index=False))
        lines.append("")

    lines.append("## Safe paper wording")
    lines.append("")
    lines.append(
        "The extended benchmark executed deterministic offline replay across multiple "
        "workload fractions, policy modes, random seeds, and worker settings. "
        "Trace-level hashes provide an audit mechanism for reproducibility. "
        "Deterministic policies produced stable trace behavior, whereas the random "
        "baseline varied across seeds as expected. The live BC policy was evaluated "
        "as an execution-level policy mode; its observed action distribution should "
        "be interpreted as checkpoint-dependent behavior rather than evidence of "
        "policy optimality."
    )
    lines.append("")

    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OUT] {out_path}")


# ============================================================
# Console validation
# ============================================================

def print_validation_summary(
    scaling_df: pd.DataFrame,
    speedup_df: pd.DataFrame,
    det_df: pd.DataFrame,
    policy_cost_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    live_bc_df: pd.DataFrame,
) -> None:
    scaling = add_workload_size_alias(scaling_df)

    print("\n========== FGCS EXTENDED RESULT VALIDATION ==========")
    print(f"Scaling/runtime rows       : {len(scaling_df)}")
    print(f"Parallel speedup rows      : {len(speedup_df)}")
    print(f"Determinism rows           : {len(det_df)}")
    print(f"Policy ablation rows       : {len(policy_cost_df)}")
    print(f"Stage latency rows         : {len(stage_df)}")
    print(f"Live BC prediction rows    : {len(live_bc_df)}")

    if "workload_size" in scaling.columns:
        print("\nWorkload sizes:")
        print(sorted(scaling["workload_size"].unique().tolist()))

    if "dataset_fraction" in scaling.columns:
        print("\nDataset fractions:")
        print(sorted(scaling["dataset_fraction"].unique().tolist()))

    if "policy_mode" in scaling.columns:
        print("\nPolicy modes:")
        print(compact_policy_order(scaling["policy_mode"].unique()))

    if "workers" in scaling.columns:
        print("\nWorkers:")
        print(sorted(scaling["workers"].unique().tolist()))

    if "trace_hash" in det_df.columns and "policy_mode" in det_df.columns:
        print("\nUnique trace hashes by policy:")
        print(det_df.groupby("policy_mode")["trace_hash"].nunique())

    if not live_bc_df.empty:
        action_col = first_existing_column(live_bc_df, ["bc_live_action", "action", "predicted_action"])
        if action_col:
            print("\nLive BC action distribution:")
            print(live_bc_df[action_col].value_counts(normalize=True, dropna=False))

    print("=====================================================\n")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready tables and figures from the current FGCS extended benchmark outputs."
    )
    parser.add_argument(
        "--input_dir",
        default="paper_outputs/fgcs_extended_benchmark",
        help="Directory containing current FGCS benchmark CSV outputs.",
    )
    parser.add_argument(
        "--out_dir",
        default="paper_outputs/fgcs_tables_figures",
        help="Directory where paper-ready tables, figures, and summary files will be saved.",
    )
    parser.add_argument(
        "--no_latex",
        action="store_true",
        help="Disable LaTeX .tex table export.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    scaling_path = input_dir / "scaling_and_runtime_results.csv"
    stage_path = input_dir / "stage_latency_summary.csv"
    det_path = input_dir / "determinism_hash_results.csv"
    speedup_path = input_dir / "parallel_speedup_results.csv"
    policy_cost_path = input_dir / "policy_ablation_costs.csv"
    live_bc_path = input_dir / "live_bc_predictions.csv"

    scaling_df = read_csv_required(scaling_path)
    stage_df = read_csv_required(stage_path)
    det_df = read_csv_required(det_path)
    speedup_df = read_csv_required(speedup_path)
    policy_cost_df = read_csv_required(policy_cost_path)
    live_bc_df = read_csv_optional(live_bc_path)

    # Normalize schemas before any table/figure functions run.
    # In the current benchmark, parallel_speedup_results.csv may not include
    # decision_points, so we recover it from scaling_and_runtime_results.csv.
    scaling_df = add_workload_size_alias(scaling_df)
    stage_df = harmonize_workload_columns(stage_df, scaling_df)
    det_df = harmonize_workload_columns(det_df, scaling_df)
    speedup_df = harmonize_workload_columns(speedup_df, scaling_df)
    policy_cost_df = harmonize_workload_columns(policy_cost_df, scaling_df)
    live_bc_df = harmonize_workload_columns(live_bc_df, scaling_df)

    print_validation_summary(
        scaling_df=scaling_df,
        speedup_df=speedup_df,
        det_df=det_df,
        policy_cost_df=policy_cost_df,
        stage_df=stage_df,
        live_bc_df=live_bc_df,
    )

    latex = not args.no_latex

    # --------------------------------------------------------
    # Validation/schema tables
    # --------------------------------------------------------
    schema_table = validate_current_outputs(
        scaling_df=scaling_df,
        speedup_df=speedup_df,
        det_df=det_df,
        policy_cost_df=policy_cost_df,
        stage_df=stage_df,
        live_bc_df=live_bc_df,
    )
    design_table = make_benchmark_design_table(scaling_df)

    save_table(schema_table, out_dir / "fgcs_table_output_schema_summary.csv", latex=False)
    save_table(design_table, out_dir / "fgcs_table_benchmark_design.csv", latex=latex)

    # --------------------------------------------------------
    # Paper-facing tables
    # --------------------------------------------------------
    runtime_scaling_table = make_runtime_scaling_table(scaling_df)
    full_runtime_table = make_full_workload_runtime_table(scaling_df)
    speedup_table = make_speedup_summary_table(speedup_df)
    full_speedup_table = make_full_workload_speedup_table(speedup_df)
    determinism_table = make_determinism_table(det_df)
    determinism_compact = make_determinism_compact_table(det_df)
    policy_table = make_policy_ablation_table(policy_cost_df)
    full_policy_table = make_full_workload_policy_table(policy_cost_df)
    stage_table = make_stage_latency_table(stage_df)
    live_bc_table = make_live_bc_summary_table(live_bc_df)
    live_bc_compact = make_live_bc_compact_table(live_bc_df)
    summary_findings = make_summary_findings_table(
        scaling_df=scaling_df,
        det_df=det_df,
        speedup_df=speedup_df,
        policy_cost_df=policy_cost_df,
        live_bc_df=live_bc_df,
    )

    save_table(runtime_scaling_table, out_dir / "fgcs_table_runtime_scaling_all.csv", latex=latex)
    save_table(full_runtime_table, out_dir / "fgcs_table_runtime_full_workload.csv", latex=latex)
    save_table(speedup_table, out_dir / "fgcs_table_parallel_speedup_all.csv", latex=latex)
    save_table(full_speedup_table, out_dir / "fgcs_table_parallel_speedup_full_workload.csv", latex=latex)
    save_table(determinism_table, out_dir / "fgcs_table_determinism_by_workload_policy.csv", latex=latex)
    save_table(determinism_compact, out_dir / "fgcs_table_determinism_compact.csv", latex=latex)
    save_table(policy_table, out_dir / "fgcs_table_policy_ablation_all.csv", latex=latex)
    save_table(full_policy_table, out_dir / "fgcs_table_policy_ablation_full_workload.csv", latex=latex)
    save_table(stage_table, out_dir / "fgcs_table_stage_latency_summary.csv", latex=latex)
    save_table(summary_findings, out_dir / "fgcs_table_summary_findings.csv", latex=latex)

    if not live_bc_table.empty:
        save_table(live_bc_table, out_dir / "fgcs_table_live_bc_summary.csv", latex=latex)
    if not live_bc_compact.empty:
        save_table(live_bc_compact, out_dir / "fgcs_table_live_bc_compact.csv", latex=latex)

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    plot_runtime_vs_workload(
        scaling_df,
        out_dir / "fgcs_fig_runtime_vs_workload.png",
    )
    plot_throughput_vs_workload(
        scaling_df,
        out_dir / "fgcs_fig_throughput_vs_workload.png",
    )
    plot_worker_throughput_at_full_workload(
        scaling_df,
        out_dir / "fgcs_fig_worker_throughput_full_workload.png",
    )
    plot_speedup_at_full_workload(
        speedup_df,
        out_dir / "fgcs_fig_speedup_full_workload.png",
    )
    plot_intervention_rate_at_full_workload(
        scaling_df,
        out_dir / "fgcs_fig_intervention_rate_full_workload.png",
    )
    plot_policy_runtime_at_full_workload(
        policy_cost_df,
        out_dir / "fgcs_fig_policy_runtime_full_workload.png",
    )
    plot_live_bc_action_distribution(
        live_bc_df,
        out_dir / "fgcs_fig_live_bc_action_distribution.png",
    )

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------
    write_markdown_summary(
        out_path=out_dir / "fgcs_extended_benchmark_summary.md",
        design_table=design_table,
        summary_findings=summary_findings,
        determinism_compact=determinism_compact,
        live_bc_compact=live_bc_compact,
    )

    print("[DONE] FGCS extended result summarization complete.")
    print(f"[OUT_DIR] {out_dir}")


if __name__ == "__main__":
    main()
