#!/usr/bin/env python3
"""Summarize the 15-repetition ReplayBench-PG timing study.

The script validates the measured timing matrix, reports runtime medians and
IQRs, and computes median paired worker speedups with percentile-bootstrap
95% confidence intervals.

Expected measured design
------------------------
Workload-scaling arm:
    all configured workload fractions x all policies x 1 worker
Worker-scaling arm:
    full workload x all policies x all configured workers

The two arms overlap at full-workload/one-worker. With the current paper
configuration this produces 48 unique configurations. The 24 non-full
workload-scaling configurations retain seven measured repetitions, while the
24 full-workload worker-scaling configurations require fifteen repetitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = (
    "paper_outputs/replaybench_timing_study/timing_repetitions_raw.csv"
)
DEFAULT_OUTPUT_DIR = "paper_outputs/replaybench_timing_study"


def paired_bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 20260722,
) -> tuple[float, float, float]:
    """Return median and percentile-bootstrap confidence interval.

    ``values`` must already contain paired speedups, one value per matched
    timing repetition: runtime_1_worker / runtime_k_workers.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if len(values) < 2:
        raise ValueError("At least two paired values are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not np.isfinite(values).all():
        raise ValueError("values contain NaN or infinity")
    if (values <= 0).any():
        raise ValueError("paired speedup values must be positive")

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        low=0,
        high=len(values),
        size=(bootstrap_samples, len(values)),
    )
    bootstrap_medians = np.median(values[indices], axis=1)

    alpha = 1.0 - confidence
    lower = float(np.quantile(bootstrap_medians, alpha / 2.0))
    upper = float(np.quantile(bootstrap_medians, 1.0 - alpha / 2.0))
    estimate = float(np.median(values))
    return estimate, lower, upper


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        "Missing required column. Expected one of: " + ", ".join(candidates)
    )


def normalize_timing_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported historical column names into one schema."""
    if raw.empty:
        raise ValueError("Timing input is empty")

    policy_col = _first_existing_column(raw, ["policy", "policy_mode"])
    runtime_col = _first_existing_column(
        raw,
        ["runtime_seconds", "total_runtime_seconds"],
    )
    fraction_col = _first_existing_column(
        raw,
        ["dataset_fraction", "workload_fraction"],
    )
    repetition_col = _first_existing_column(
        raw,
        ["repetition", "measured_repetition"],
    )
    workers_col = _first_existing_column(raw, ["workers"])

    frame = raw.rename(
        columns={
            policy_col: "policy",
            runtime_col: "runtime_seconds",
            fraction_col: "dataset_fraction",
            repetition_col: "repetition",
            workers_col: "workers",
        }
    ).copy()

    # Support raw files that include warm-ups. Final summaries use measured
    # repetitions only.
    if "is_warmup" in frame.columns:
        warm = frame["is_warmup"].astype(str).str.lower().isin(
            {"1", "true", "yes"}
        )
        frame = frame.loc[~warm].copy()
    elif "run_type" in frame.columns:
        frame = frame.loc[
            frame["run_type"].astype(str).str.lower() != "warmup"
        ].copy()

    frame["policy"] = frame["policy"].astype(str)
    frame["workers"] = pd.to_numeric(frame["workers"], errors="raise").astype(int)
    frame["repetition"] = pd.to_numeric(
        frame["repetition"], errors="raise"
    ).astype(int)
    frame["runtime_seconds"] = pd.to_numeric(
        frame["runtime_seconds"], errors="raise"
    ).astype(float)
    frame["dataset_fraction"] = pd.to_numeric(
        frame["dataset_fraction"], errors="raise"
    ).astype(float)

    if not np.isfinite(frame["runtime_seconds"]).all():
        raise ValueError("runtime_seconds contains NaN or infinity")
    if (frame["runtime_seconds"] <= 0).any():
        raise ValueError("runtime_seconds must be positive")
    if (frame["workers"] <= 0).any():
        raise ValueError("workers must be positive")
    if (frame["repetition"] <= 0).any():
        raise ValueError("repetition identifiers must be positive")
    if not frame["dataset_fraction"].between(0.0, 1.0, inclusive="right").all():
        raise ValueError("dataset_fraction must be in (0, 1]")

    duplicate_mask = frame.duplicated(
        ["dataset_fraction", "policy", "workers", "repetition"],
        keep=False,
    )
    if duplicate_mask.any():
        duplicates = frame.loc[
            duplicate_mask,
            ["dataset_fraction", "policy", "workers", "repetition"],
        ]
        raise ValueError(
            "Duplicate timing rows found:\n"
            + duplicates.to_string(index=False)
        )

    return frame.reset_index(drop=True)


def validate_timing_design(
    frame: pd.DataFrame,
    workload_repetitions: int = 7,
    worker_repetitions: int = 15,
    full_fraction: float = 1.0,
) -> dict[str, Any]:
    """Validate the mixed 7/15-repetition timing design.

    Non-full workload-scaling configurations retain seven measured
    repetitions. Full-workload worker-scaling configurations require fifteen
    measured repetitions. This matches the controlled revision plan: the
    original seven-repetition study is retained and eight additional runs are
    added only to the 24 full-workload worker configurations.
    """
    if workload_repetitions < 2:
        raise ValueError("workload_repetitions must be at least 2")
    if worker_repetitions < workload_repetitions:
        raise ValueError(
            "worker_repetitions must be at least workload_repetitions"
        )

    policies = sorted(frame["policy"].unique().tolist())
    workers = sorted(frame["workers"].unique().tolist())
    fractions = sorted(frame["dataset_fraction"].unique().tolist())

    full_mask = np.isclose(frame["dataset_fraction"], full_fraction)
    full = frame.loc[full_mask]
    non_full = frame.loc[~full_mask]
    if full.empty:
        raise ValueError("No full-workload timing rows were found")

    invalid_non_full = non_full.loc[non_full["workers"] != 1]
    if not invalid_non_full.empty:
        raise ValueError(
            "Non-full workload fractions must contain only one-worker "
            "workload-scaling configurations"
        )

    expected_full_pairs = {
        (policy, worker)
        for policy in policies
        for worker in workers
    }
    actual_full_pairs = set(
        full[["policy", "workers"]].itertuples(index=False, name=None)
    )
    missing_full_pairs = sorted(expected_full_pairs - actual_full_pairs)
    if missing_full_pairs:
        raise ValueError(
            "Full-workload worker-scaling configurations are missing: "
            f"{missing_full_pairs}"
        )

    expected_non_full_pairs = {
        (fraction, policy, 1)
        for fraction in fractions
        if not np.isclose(fraction, full_fraction)
        for policy in policies
    }
    actual_non_full_pairs = set(
        non_full[["dataset_fraction", "policy", "workers"]]
        .itertuples(index=False, name=None)
    )
    missing_non_full_pairs = sorted(
        expected_non_full_pairs - actual_non_full_pairs
    )
    if missing_non_full_pairs:
        raise ValueError(
            "Workload-scaling configurations are missing: "
            f"{missing_non_full_pairs}"
        )

    bad_counts: list[tuple[float, str, int, int, int]] = []
    bad_ids: list[tuple[float, str, int, list[int], list[int]]] = []

    for key, group in frame.groupby(
        ["dataset_fraction", "policy", "workers"],
        sort=True,
    ):
        fraction, policy, worker = key
        expected = (
            worker_repetitions
            if np.isclose(float(fraction), full_fraction)
            else workload_repetitions
        )
        actual_count = int(group["repetition"].nunique())
        if actual_count != expected:
            bad_counts.append(
                (float(fraction), str(policy), int(worker), actual_count, expected)
            )

        expected_ids = set(range(1, expected + 1))
        actual_ids = set(group["repetition"].astype(int))
        if actual_ids != expected_ids:
            bad_ids.append(
                (
                    float(fraction),
                    str(policy),
                    int(worker),
                    sorted(actual_ids),
                    sorted(expected_ids),
                )
            )

    if bad_counts:
        raise ValueError(
            "Timing configurations have incorrect measured-repetition counts: "
            f"{bad_counts}"
        )
    if bad_ids:
        raise ValueError(
            "Timing repetition identifiers do not match the required ranges: "
            f"{bad_ids}"
        )

    invariant_checks: dict[str, bool] = {}
    legacy_missing_counts: dict[str, int] = {}

    # The original seven-repetition rows predate several explicit validation
    # columns. Validate every value that was actually recorded, while allowing
    # missing legacy metadata. Determinism is still checked independently
    # below by requiring one trace hash per configuration.
    checks = {
        "hash_match": 1,
        "authorization_execution_consistent": 1,
        "row_count_match": 1,
        "validation_passed": 1,
        "fault_injected_count": 0,
        "unauthorized_invocations": 0,
    }

    for column, expected in checks.items():
        if column not in frame.columns:
            continue

        numeric = pd.to_numeric(frame[column], errors="coerce")
        recorded = numeric.dropna()
        missing_count = int(numeric.isna().sum())
        legacy_missing_counts[column] = missing_count

        invalid_mask = numeric.notna() & numeric.ne(expected)
        ok = not bool(invalid_mask.any())
        invariant_checks[column] = ok

        if not ok:
            identifying_columns = [
                name
                for name in [
                    "dataset_fraction",
                    "policy",
                    "workers",
                    "repetition",
                    "trace_hash",
                    column,
                ]
                if name in frame.columns
            ]
            invalid_rows = frame.loc[invalid_mask, identifying_columns]
            raise ValueError(
                f"Timing functional check failed: recorded {column} values "
                f"must equal {expected}.\n"
                f"{invalid_rows.to_string(index=False)}"
            )

        # A present column containing only missing values provides no usable
        # validation evidence and is therefore rejected.
        if recorded.empty:
            raise ValueError(
                f"Timing functional check failed: {column} is present but "
                "contains no recorded values"
            )

    if "trace_hash" not in frame.columns:
        raise ValueError(
            "Timing determinism check requires the trace_hash column"
        )

    if frame["trace_hash"].isna().any():
        missing_hash_rows = frame.loc[
            frame["trace_hash"].isna(),
            ["dataset_fraction", "policy", "workers", "repetition"],
        ]
        raise ValueError(
            "Timing determinism check failed: trace_hash contains missing "
            "values.\n"
            f"{missing_hash_rows.to_string(index=False)}"
        )

    unique_hashes = frame.groupby(
        ["dataset_fraction", "policy", "workers"],
        dropna=False,
    )["trace_hash"].nunique(dropna=False)

    unstable_hashes = unique_hashes.loc[unique_hashes.ne(1)]
    if not unstable_hashes.empty:
        raise ValueError(
            "Action-sequence hashes changed across timing repetitions:\n"
            f"{unstable_hashes.to_string()}"
        )

    invariant_checks["stable_hash_per_configuration"] = True

    expected_rows = (
        len(expected_non_full_pairs) * workload_repetitions
        + len(expected_full_pairs) * worker_repetitions
    )
    if len(frame) != expected_rows:
        raise ValueError(
            f"Timing matrix row count mismatch: expected={expected_rows}, "
            f"observed={len(frame)}"
        )

    return {
        "workload_repetitions": workload_repetitions,
        "worker_repetitions": worker_repetitions,
        "measured_rows": int(len(frame)),
        "expected_measured_rows": int(expected_rows),
        "unique_configurations": int(
            frame[["dataset_fraction", "policy", "workers"]]
            .drop_duplicates()
            .shape[0]
        ),
        "workload_scaling_configurations": int(len(expected_non_full_pairs)),
        "worker_scaling_configurations": int(len(expected_full_pairs)),
        "policies": policies,
        "workers": workers,
        "fractions": fractions,
        "full_fraction": full_fraction,
        "invariant_checks": invariant_checks,
        "legacy_missing_validation_values": legacy_missing_counts,
    }

def calculate_runtime_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(
        ["dataset_fraction", "policy", "workers"],
        as_index=False,
    )["runtime_seconds"]

    summary = grouped.agg(
        repetitions="count",
        runtime_median_seconds="median",
        runtime_q1_seconds=lambda values: float(np.quantile(values, 0.25)),
        runtime_q3_seconds=lambda values: float(np.quantile(values, 0.75)),
        runtime_min_seconds="min",
        runtime_max_seconds="max",
    )
    summary["runtime_iqr_seconds"] = (
        summary["runtime_q3_seconds"] - summary["runtime_q1_seconds"]
    )
    return summary.sort_values(
        ["dataset_fraction", "policy", "workers"]
    ).reset_index(drop=True)


def _bootstrap_seed(base_seed: int, policy: str, workers: int) -> int:
    token = f"{base_seed}|{policy}|{workers}".encode("utf-8")
    digest = hashlib.sha256(token).hexdigest()
    return int(digest[:8], 16)


def calculate_worker_speedups(
    raw: pd.DataFrame,
    full_fraction: float = 1.0,
    worker_repetitions: int = 15,
    workload_repetitions: int = 7,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260722,
) -> pd.DataFrame:
    """Calculate paired speedups for full-workload worker configurations."""
    frame = normalize_timing_frame(raw)
    validate_timing_design(
        frame,
        workload_repetitions=workload_repetitions,
        worker_repetitions=worker_repetitions,
        full_fraction=full_fraction,
    )

    full = frame.loc[
        np.isclose(frame["dataset_fraction"], full_fraction)
    ].copy()
    output: list[dict[str, object]] = []

    for policy, policy_data in full.groupby("policy", sort=True):
        baseline = (
            policy_data.loc[policy_data["workers"] == 1]
            [["repetition", "runtime_seconds"]]
            .rename(columns={"runtime_seconds": "runtime_1"})
        )

        if len(baseline) != worker_repetitions:
            raise ValueError(
                f"{policy} has an incomplete one-worker baseline"
            )

        for workers in sorted(policy_data["workers"].unique()):
            comparison = (
                policy_data.loc[policy_data["workers"] == workers]
                [["repetition", "runtime_seconds"]]
                .rename(columns={"runtime_seconds": "runtime_k"})
            )

            paired = baseline.merge(
                comparison,
                on="repetition",
                how="inner",
                validate="one_to_one",
            ).sort_values("repetition")

            if len(paired) != worker_repetitions:
                raise ValueError(
                    f"Incomplete pairing for {policy}, workers={workers}"
                )

            paired["speedup"] = paired["runtime_1"] / paired["runtime_k"]

            estimate, lower, upper = paired_bootstrap_ci(
                paired["speedup"].to_numpy(dtype=float),
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                seed=_bootstrap_seed(bootstrap_seed, policy, int(workers)),
            )

            runtime_k = paired["runtime_k"].to_numpy(dtype=float)
            output.append(
                {
                    "policy": policy,
                    "workers": int(workers),
                    "paired_repetitions": int(len(paired)),
                    "runtime_median_seconds": float(np.median(runtime_k)),
                    "runtime_q1_seconds": float(np.quantile(runtime_k, 0.25)),
                    "runtime_q3_seconds": float(np.quantile(runtime_k, 0.75)),
                    "runtime_iqr_seconds": float(
                        np.quantile(runtime_k, 0.75)
                        - np.quantile(runtime_k, 0.25)
                    ),
                    "median_paired_speedup": estimate,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "confidence": confidence,
                    "bootstrap_samples": int(bootstrap_samples),
                }
            )

    return pd.DataFrame(output).sort_values(
        ["policy", "workers"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize and validate ReplayBench-PG timing results."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workload-repetitions", type=int, default=7)
    parser.add_argument("--worker-repetitions", type=int, default=15)
    parser.add_argument("--full-fraction", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Timing input not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    frame = normalize_timing_frame(raw)
    design = validate_timing_design(
        frame,
        workload_repetitions=args.workload_repetitions,
        worker_repetitions=args.worker_repetitions,
        full_fraction=args.full_fraction,
    )

    runtime_summary = calculate_runtime_summary(frame)
    speedup_summary = calculate_worker_speedups(
        frame,
        full_fraction=args.full_fraction,
        worker_repetitions=args.worker_repetitions,
        workload_repetitions=args.workload_repetitions,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    normalized_path = output_dir / "timing_repetitions_normalized.csv"
    runtime_path = output_dir / "timing_runtime_summary_mixed_7_15.csv"
    speedup_path = output_dir / "timing_worker_speedup_paired_ci.csv"
    manifest_path = output_dir / "timing_summary_manifest.json"

    frame.to_csv(normalized_path, index=False)
    runtime_summary.to_csv(runtime_path, index=False)
    speedup_summary.to_csv(speedup_path, index=False)

    manifest = {
        "input": str(input_path),
        "outputs": {
            "normalized": str(normalized_path),
            "runtime_summary": str(runtime_path),
            "paired_speedup_summary": str(speedup_path),
        },
        "design": design,
        "statistics": {
            "runtime_location": "median",
            "runtime_dispersion": "interquartile range",
            "speedup_definition": "paired runtime_1 / runtime_k",
            "speedup_location": "median of paired speedups",
            "confidence_interval": "percentile bootstrap",
            "confidence": 0.95,
            "bootstrap_samples": int(args.bootstrap_samples),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[DONE] Timing summary complete")
    print(f"[OUT] {normalized_path}")
    print(f"[OUT] {runtime_path}")
    print(f"[OUT] {speedup_path}")
    print(f"[OUT] {manifest_path}")


if __name__ == "__main__":
    main()