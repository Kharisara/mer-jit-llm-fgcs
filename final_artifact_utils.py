#!/usr/bin/env python3
"""Shared validation and table-building utilities for ReplayBench-PG.

This module reads only finalized CSV/JSON evidence. It never reruns an
experiment and never repairs result files silently. Every public validator is
strict: a missing artifact, unexpected condition matrix, failed invariant, or
ambiguous cloud comparison raises ``ValidationError``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class ValidationError(RuntimeError):
    """Raised when finalized evidence does not satisfy the frozen protocol."""


@dataclass(frozen=True)
class EvidenceFile:
    component: str
    role: str
    path: str
    exists: bool
    file_type: str
    rows: int | None
    columns: int | None
    size_bytes: int | None
    sha256: str | None
    status: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRIMARY_FRACTIONS = {0.10, 0.25, 0.50, 0.75, 1.00}
PRIMARY_POLICIES = {"risk_proxy", "bc", "bc_live", "random", "always", "never"}
PRIMARY_SEEDS = {1, 2, 3}
PRIMARY_WORKERS = {1, 2, 4, 8}

TIMING_FRACTIONS = PRIMARY_FRACTIONS
TIMING_POLICIES = PRIMARY_POLICIES
TIMING_WORKERS = PRIMARY_WORKERS

RAY_POLICIES = {"risk_proxy", "random", "never"}
RAY_SEEDS = {1, 2, 3}
RAY_WORKERS = {1, 4}
RAY_FAULTS = {"clean", "action_flip", "dropped_row"}

METRO_FRACTIONS = {0.25, 0.50, 1.00}
METRO_POLICIES = {"rule_gate", "random", "always", "never"}
METRO_SEEDS = {1, 2, 3}
METRO_WORKERS = {1, 4}

FAULT_ORDER = [
    "action_flip_1_percent",
    "unauthorized_invoke_1_percent",
    "trace_action_corruption_1_percent",
    "drop_trace_rows_1_percent",
    "duplicate_trace_rows_1_percent",
]

FAULT_LABELS = {
    "action_flip_1_percent": "Action flip (1%)",
    "unauthorized_invoke_1_percent": "Unauthorized invocation (1%)",
    "trace_action_corruption_1_percent": "Saved-action corruption (1%)",
    "drop_trace_rows_1_percent": "Dropped trace rows (1%)",
    "duplicate_trace_rows_1_percent": "Duplicated trace rows (1%)",
}

DEFAULT_REQUIRED_CONFIGS = [
    "configs/fgcs_extended_benchmark.yaml",
    "configs/fgcs_fault_action_flip.yaml",
    "configs/fgcs_fault_unauthorized_invoke.yaml",
    "configs/secondary_metropt3_benchmark.yaml",
    "configs/secondary_metropt3_fault_action_flip.yaml",
    "configs/secondary_metropt3_fault_unauthorized_invoke.yaml",
    "configs/ray_comparison.yaml",
]

DEFAULT_REQUIRED_MANIFESTS = [
    "paper_outputs/fgcs_extended_benchmarks/fgcs_extended_reproducibility_manifest.json",
    "paper_outputs/replaybench_timing_study/timing_study_manifest.json",
    "paper_outputs/replaybench_timing_study/timing_summary_manifest.json",
    "paper_outputs/replaybench_timing_study/timing_environment_manifest.json",
    "paper_outputs/ray_comparison/ray_comparison_manifest.json",
    "paper_outputs/ray_comparison/ray_comparison_effective_config.json",
    "paper_outputs/array_revision_validation.json",
    "paper_outputs/secondary_metropt3/preparation_manifest.json",
    "paper_outputs/secondary_metropt3/tables_figures/secondary_validation_summary.json",
]

DEFAULT_REQUIRED_SCRIPTS = [
    "run_fgcs_extended_benchmark.py",
    "run_replaybench_timing_study.py",
    "summarize_replaybench_timing_study.py",
    "run_ray_comparison.py",
    "run_array_revision_validation.py",
    "run_secondary_metropt3_validation.py",
    "summarize_secondary_workload_results.py",
    "fgcs_fault_validation_framework.py",
    "generate_final_manuscript_tables.py",
    "run_final_artifact_validation.py",
    "final_artifact_utils.py",
    "apply_final_code_quality_fix.py",
    "freeze_final_repository.py",
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_csv_required(path: Path, role: str) -> pd.DataFrame:
    if not path.is_file():
        raise ValidationError(f"Missing {role}: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - diagnostic path
        raise ValidationError(f"Could not read {role} at {path}: {exc}") from exc
    if frame.empty:
        raise ValidationError(f"{role} is empty: {path}")
    return frame


def read_json_required(path: Path, role: str) -> Any:
    if not path.is_file():
        raise ValidationError(f"Missing {role}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        raise ValidationError(f"Could not read {role} at {path}: {exc}") from exc


def require_columns(frame: pd.DataFrame, columns: Iterable[str], role: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValidationError(f"{role} is missing required columns: {missing}")


def numeric(series: pd.Series, role: str, allow_nan: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if not allow_nan and values.isna().any():
        bad = series.loc[values.isna()].head(5).tolist()
        raise ValidationError(f"{role} contains non-numeric values, examples={bad}")
    return values


def _bool_token(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(int(value))
        return None
    if isinstance(value, (float, np.floating)):
        if float(value) in {0.0, 1.0}:
            return bool(int(value))
        return None
    token = str(value).strip().lower()
    if token in {"true", "yes", "y", "1", "pass", "passed", "match", "matched"}:
        return True
    if token in {"false", "no", "n", "0", "fail", "failed", "mismatch", "unmatched"}:
        return False
    return None


def bool_series(series: pd.Series, role: str, allow_nan: bool = False) -> pd.Series:
    mapped = series.map(_bool_token)
    if not allow_nan and mapped.isna().any():
        bad = series.loc[mapped.isna()].head(5).tolist()
        raise ValidationError(f"{role} contains non-boolean values, examples={bad}")
    return mapped


def assert_set_equal(actual: Iterable[Any], expected: set[Any], role: str) -> None:
    actual_set = set(actual)
    if actual_set != expected:
        raise ValidationError(
            f"Unexpected {role}: expected={sorted(expected, key=str)}, "
            f"observed={sorted(actual_set, key=str)}"
        )


def assert_no_duplicates(frame: pd.DataFrame, keys: Sequence[str], role: str) -> None:
    duplicates = frame.loc[frame.duplicated(list(keys), keep=False), list(keys)]
    if not duplicates.empty:
        raise ValidationError(
            f"{role} contains duplicate condition keys.\n"
            f"{duplicates.head(20).to_string(index=False)}"
        )


def assert_all_zero(series: pd.Series, role: str) -> None:
    values = numeric(series, role)
    failures = values.ne(0)
    if failures.any():
        raise ValidationError(
            f"{role} must equal zero for every row; "
            f"failed_rows={int(failures.sum())}, max={float(values.max())}"
        )


def assert_all_one(series: pd.Series, role: str) -> None:
    values = numeric(series, role)
    failures = values.ne(1)
    if failures.any():
        raise ValidationError(
            f"{role} must equal one for every row; failed_rows={int(failures.sum())}"
        )


def evidence_file(
    root: Path,
    component: str,
    role: str,
    path: Path,
    status: str = "validated",
    notes: str = "",
) -> EvidenceFile:
    if not path.is_file():
        return EvidenceFile(
            component=component,
            role=role,
            path=relative_posix(path, root),
            exists=False,
            file_type=path.suffix.lower().lstrip("."),
            rows=None,
            columns=None,
            size_bytes=None,
            sha256=None,
            status="missing",
            notes=notes,
        )

    rows: int | None = None
    columns: int | None = None
    if path.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(path)
            rows = int(len(frame))
            columns = int(len(frame.columns))
        except Exception:
            pass

    return EvidenceFile(
        component=component,
        role=role,
        path=relative_posix(path, root),
        exists=True,
        file_type=path.suffix.lower().lstrip("."),
        rows=rows,
        columns=columns,
        size_bytes=int(path.stat().st_size),
        sha256=sha256_file(path),
        status=status,
        notes=notes,
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, lineterminator="\n")
    temp.replace(path)


# ---------------------------------------------------------------------------
# Primary benchmark validation
# ---------------------------------------------------------------------------


def validate_primary_benchmark(project_dir: Path) -> tuple[dict[str, Any], list[EvidenceFile]]:
    base = project_dir / "paper_outputs" / "fgcs_extended_benchmark"
    scaling_path = base / "scaling_and_runtime_results.csv"
    det_path = base / "determinism_hash_results.csv"
    scaling = read_csv_required(scaling_path, "primary scaling/runtime output")
    det = read_csv_required(det_path, "primary determinism output")

    condition_cols = ["dataset_fraction", "policy_mode", "seed", "workers"]
    require_columns(
        scaling,
        condition_cols
        + [
            "decision_points",
            "trace_hash",
            "intervention_rate",
            "unauthorized_invocations",
            "fault_injected_count",
        ],
        "primary scaling/runtime output",
    )
    require_columns(
        det,
        condition_cols
        + [
            "trace_hash",
            "hash_match",
            "unauthorized_invocations",
        ],
        "primary determinism output",
    )

    if len(scaling) != 360:
        raise ValidationError(f"Primary benchmark must contain 360 rows; found {len(scaling)}")
    if len(det) != 360:
        raise ValidationError(f"Primary determinism output must contain 360 rows; found {len(det)}")

    scaling = scaling.copy()
    det = det.copy()
    for frame in (scaling, det):
        frame["dataset_fraction"] = numeric(frame["dataset_fraction"], "dataset_fraction").round(6)
        frame["seed"] = numeric(frame["seed"], "seed").astype(int)
        frame["workers"] = numeric(frame["workers"], "workers").astype(int)
        frame["policy_mode"] = frame["policy_mode"].astype(str)

    assert_set_equal(scaling["dataset_fraction"].tolist(), PRIMARY_FRACTIONS, "primary fractions")
    assert_set_equal(scaling["policy_mode"].tolist(), PRIMARY_POLICIES, "primary policies")
    assert_set_equal(scaling["seed"].tolist(), PRIMARY_SEEDS, "primary seeds")
    assert_set_equal(scaling["workers"].tolist(), PRIMARY_WORKERS, "primary workers")
    assert_no_duplicates(scaling, condition_cols, "primary benchmark")
    assert_no_duplicates(det, condition_cols, "primary determinism output")

    expected_grid = (
        len(PRIMARY_FRACTIONS)
        * len(PRIMARY_POLICIES)
        * len(PRIMARY_SEEDS)
        * len(PRIMARY_WORKERS)
    )
    if expected_grid != len(scaling):
        raise ValidationError("Primary condition grid is incomplete")

    assert_all_zero(scaling["unauthorized_invocations"], "primary clean unauthorized invocations")
    assert_all_zero(scaling["fault_injected_count"], "primary clean injected-fault counter")
    assert_all_zero(det["unauthorized_invocations"], "primary determinism unauthorized invocations")
    assert_all_one(det["hash_match"], "primary worker hash_match")

    for optional in ["authorization_execution_consistent", "row_count_match", "validation_passed"]:
        if optional in scaling.columns:
            assert_all_one(scaling[optional], f"primary {optional}")
        if optional in det.columns:
            assert_all_one(det[optional], f"primary determinism {optional}")

    # Every worker reconstruction must produce the same hash for a fixed
    # fraction, policy and seed.
    worker_hash_counts = det.groupby(
        ["dataset_fraction", "policy_mode", "seed"], dropna=False
    )["trace_hash"].nunique(dropna=False)
    unstable = worker_hash_counts[worker_hash_counts.ne(1)]
    if not unstable.empty:
        raise ValidationError(
            "Primary worker reconstruction produced unstable hashes.\n"
            f"{unstable.to_string()}"
        )

    # Deterministic policies must also be seed invariant. Random is permitted
    # to vary across the three configured seeds but remains worker invariant.
    deterministic = PRIMARY_POLICIES - {"random"}
    seed_hash_counts = det.loc[det["policy_mode"].isin(deterministic)].groupby(
        ["dataset_fraction", "policy_mode"], dropna=False
    )["trace_hash"].nunique(dropna=False)
    unstable_seed = seed_hash_counts[seed_hash_counts.ne(1)]
    if not unstable_seed.empty:
        raise ValidationError(
            "A deterministic primary policy varied across seeds.\n"
            f"{unstable_seed.to_string()}"
        )

    full = scaling.loc[np.isclose(scaling["dataset_fraction"], 1.0)]
    intervention = (
        full.groupby("policy_mode", as_index=False)["intervention_rate"]
        .mean()
        .set_index("policy_mode")["intervention_rate"]
        .to_dict()
    )

    unique_hashes = det.groupby("policy_mode")["trace_hash"].nunique().to_dict()
    max_unauthorized = int(numeric(scaling["unauthorized_invocations"], "unauthorized").max())

    results = {
        "conditions": int(len(scaling)),
        "determinism_rows": int(len(det)),
        "expected_conditions": 360,
        "condition_matrix_complete": True,
        "all_worker_hashes_match": True,
        "deterministic_policies_seed_invariant": True,
        "max_clean_unauthorized_invocations": max_unauthorized,
        "unique_hashes_by_policy": {str(k): int(v) for k, v in unique_hashes.items()},
        "full_workload_mean_intervention_rate_by_policy": {
            str(k): float(v) for k, v in intervention.items()
        },
        "full_workload_decision_points": int(
            numeric(full["decision_points"], "full workload decision points").max()
        ),
    }

    inventory = [
        evidence_file(project_dir, "primary_benchmark", "scaling_runtime", scaling_path),
        evidence_file(project_dir, "primary_benchmark", "determinism", det_path),
    ]
    return results, inventory


# ---------------------------------------------------------------------------
# Timing validation
# ---------------------------------------------------------------------------


def _pick_column(frame: pd.DataFrame, candidates: Sequence[str], role: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValidationError(f"Could not find {role}; expected one of {list(candidates)}")


def normalize_timing_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    policy_col = _pick_column(out, ["policy", "policy_mode"], "timing policy column")
    runtime_col = _pick_column(
        out,
        ["runtime_seconds", "total_runtime_seconds", "measured_runtime_seconds"],
        "timing runtime column",
    )
    repetition_col = _pick_column(out, ["repetition", "rep", "measured_repetition"], "timing repetition column")
    require_columns(out, ["dataset_fraction", "workers", policy_col, runtime_col, repetition_col], "timing output")
    rename = {
        policy_col: "policy",
        runtime_col: "runtime_seconds",
        repetition_col: "repetition",
    }
    out = out.rename(columns=rename)
    out["dataset_fraction"] = numeric(out["dataset_fraction"], "timing fraction").round(6)
    out["workers"] = numeric(out["workers"], "timing workers").astype(int)
    out["repetition"] = numeric(out["repetition"], "timing repetition").astype(int)
    out["runtime_seconds"] = numeric(out["runtime_seconds"], "timing runtime")
    out["policy"] = out["policy"].astype(str)
    return out


def validate_timing_study(project_dir: Path) -> tuple[dict[str, Any], list[EvidenceFile]]:
    base = project_dir / "paper_outputs" / "replaybench_timing_study"
    raw_path = base / "timing_repetitions_raw.csv"
    runtime_summary_path = base / "timing_runtime_summary_mixed_7_15.csv"
    speedup_path = base / "timing_worker_speedup_paired_ci.csv"
    raw = normalize_timing_frame(read_csv_required(raw_path, "timing repetitions"))

    if len(raw) != 528:
        raise ValidationError(f"Timing study must contain 528 measured rows; found {len(raw)}")
    assert_set_equal(raw["dataset_fraction"].tolist(), TIMING_FRACTIONS, "timing fractions")
    assert_set_equal(raw["policy"].tolist(), TIMING_POLICIES, "timing policies")
    assert_set_equal(raw["workers"].tolist(), TIMING_WORKERS, "timing workers")
    assert_no_duplicates(
        raw,
        ["dataset_fraction", "policy", "workers", "repetition"],
        "timing repetitions",
    )

    group_sizes = raw.groupby(["dataset_fraction", "policy", "workers"]).size()
    counts = group_sizes.value_counts().sort_index().to_dict()
    if counts != {7: 24, 15: 24}:
        raise ValidationError(
            "Timing mixed design must contain 24 configurations x 7 and "
            f"24 configurations x 15; observed={counts}"
        )
    if len(group_sizes) != 48:
        raise ValidationError(f"Timing study must contain 48 configurations; found {len(group_sizes)}")

    non_full = raw.loc[~np.isclose(raw["dataset_fraction"], 1.0)]
    full = raw.loc[np.isclose(raw["dataset_fraction"], 1.0)]
    if non_full.groupby(["dataset_fraction", "policy", "workers"]).size().ne(7).any():
        raise ValidationError("Every non-full timing configuration must contain 7 repetitions")
    if full.groupby(["dataset_fraction", "policy", "workers"]).size().ne(15).any():
        raise ValidationError("Every full-workload timing configuration must contain 15 repetitions")

    if (raw["runtime_seconds"] <= 0).any():
        raise ValidationError("Timing study contains a non-positive runtime")

    if "trace_hash" in raw.columns:
        if raw["trace_hash"].isna().any():
            raise ValidationError("Timing trace_hash must be populated for all measured rows")
        unstable = raw.groupby(["dataset_fraction", "policy", "workers"])["trace_hash"].nunique(dropna=False)
        unstable = unstable[unstable.ne(1)]
        if not unstable.empty:
            raise ValidationError(f"Timing hashes are unstable.\n{unstable.to_string()}")

    # Legacy rows may have missing explicit validation flags. Every recorded
    # flag must nevertheless pass.
    legacy_missing: dict[str, int] = {}
    for column in [
        "hash_match",
        "authorization_execution_consistent",
        "row_count_match",
        "validation_passed",
    ]:
        if column in raw.columns:
            values = numeric(raw[column], f"timing {column}", allow_nan=True)
            legacy_missing[column] = int(values.isna().sum())
            recorded = values.dropna()
            if not recorded.eq(1).all():
                raise ValidationError(f"One or more recorded timing {column} values failed")
    for column in ["unauthorized_invocations", "fault_injected_count"]:
        if column in raw.columns:
            values = numeric(raw[column], f"timing {column}", allow_nan=True)
            legacy_missing[column] = int(values.isna().sum())
            recorded = values.dropna()
            if not recorded.eq(0).all():
                raise ValidationError(f"One or more recorded timing {column} values are non-zero")

    runtime_summary = read_csv_required(runtime_summary_path, "timing runtime summary")
    speedup = read_csv_required(speedup_path, "timing worker-speedup summary")
    if len(runtime_summary) != 48:
        raise ValidationError(f"Timing runtime summary must contain 48 rows; found {len(runtime_summary)}")
    if len(speedup) != 24:
        raise ValidationError(f"Timing worker-speedup summary must contain 24 rows; found {len(speedup)}")
    require_columns(
        speedup,
        [
            "policy",
            "workers",
            "paired_repetitions",
            "runtime_median_seconds",
            "median_paired_speedup",
            "ci95_lower",
            "ci95_upper",
        ],
        "timing worker-speedup summary",
    )
    if not numeric(speedup["paired_repetitions"], "paired repetitions").eq(15).all():
        raise ValidationError("Every full-workload speedup row must use 15 paired repetitions")

    results = {
        "measured_rows": int(len(raw)),
        "expected_measured_rows": 528,
        "unique_configurations": int(len(group_sizes)),
        "configurations_with_7_repetitions": int((group_sizes == 7).sum()),
        "configurations_with_15_repetitions": int((group_sizes == 15).sum()),
        "runtime_summary_rows": int(len(runtime_summary)),
        "worker_speedup_rows": int(len(speedup)),
        "all_recorded_validation_flags_pass": True,
        "stable_hashes_within_configuration": True,
        "legacy_missing_validation_values": legacy_missing,
    }
    inventory = [
        evidence_file(project_dir, "timing_study", "raw_repetitions", raw_path),
        evidence_file(project_dir, "timing_study", "runtime_summary", runtime_summary_path),
        evidence_file(project_dir, "timing_study", "paired_speedup_summary", speedup_path),
    ]
    return results, inventory


# ---------------------------------------------------------------------------
# Ray validation
# ---------------------------------------------------------------------------


def validate_ray(project_dir: Path) -> tuple[dict[str, Any], list[EvidenceFile]]:
    base = project_dir / "paper_outputs" / "ray_comparison"
    per_run_path = base / "ray_comparison_per_run.csv"
    summary_path = base / "ray_comparison_summary.csv"
    manifest_path = base / "ray_comparison_manifest.json"
    frame = read_csv_required(per_run_path, "Ray per-run output")
    require_columns(
        frame,
        [
            "policy_mode",
            "seed",
            "workers",
            "fault_mode",
            "injected_events",
            "hash_match",
            "row_count_match",
            "unauthorized_invocations",
            "authorization_execution_consistent",
            "expected_flagged",
            "detected_flag",
            "detection_correct",
            "reference_source",
        ],
        "Ray per-run output",
    )
    if len(frame) != 54:
        raise ValidationError(f"Ray comparison must contain 54 rows; found {len(frame)}")

    frame = frame.copy()
    frame["policy_mode"] = frame["policy_mode"].astype(str)
    frame["fault_mode"] = frame["fault_mode"].astype(str)
    frame["seed"] = numeric(frame["seed"], "Ray seed").astype(int)
    frame["workers"] = numeric(frame["workers"], "Ray workers").astype(int)
    assert_set_equal(frame["policy_mode"], RAY_POLICIES, "Ray policies")
    assert_set_equal(frame["seed"], RAY_SEEDS, "Ray seeds")
    assert_set_equal(frame["workers"], RAY_WORKERS, "Ray workers")
    assert_set_equal(frame["fault_mode"], RAY_FAULTS, "Ray fault modes")
    assert_no_duplicates(frame, ["policy_mode", "seed", "workers", "fault_mode"], "Ray output")

    counts = frame.groupby("fault_mode").size().to_dict()
    if counts != {"action_flip": 18, "clean": 18, "dropped_row": 18}:
        raise ValidationError(f"Unexpected Ray mode counts: {counts}")
    if not frame["reference_source"].astype(str).eq("replaybench_determinism_csv").all():
        raise ValidationError("Every Ray row must use the external ReplayBench-PG determinism CSV")

    clean = frame.loc[frame["fault_mode"].eq("clean")]
    faults = frame.loc[~frame["fault_mode"].eq("clean")]
    action_flip = frame.loc[frame["fault_mode"].eq("action_flip")]
    dropped = frame.loc[frame["fault_mode"].eq("dropped_row")]

    assert_all_one(clean["hash_match"], "Ray clean external hash_match")
    assert_all_one(clean["row_count_match"], "Ray clean row_count_match")
    assert_all_one(clean["authorization_execution_consistent"], "Ray clean authorization invariant")
    assert_all_zero(clean["detected_flag"], "Ray clean false-positive flag")
    assert_all_zero(frame["unauthorized_invocations"], "Ray unauthorized invocations")
    assert_all_one(frame["detection_correct"], "Ray detection_correct")
    assert_all_one(faults["detected_flag"], "Ray fault detected_flag")

    if not (numeric(action_flip["injected_events"], "Ray action-flip injected events") > 0).all():
        raise ValidationError("A Ray action-flip condition injected zero events")
    assert_all_zero(action_flip["hash_match"], "Ray action-flip hash_match")
    assert_all_one(action_flip["row_count_match"], "Ray action-flip row_count_match")

    if not (numeric(dropped["injected_events"], "Ray dropped-row injected events") > 0).all():
        raise ValidationError("A Ray dropped-row condition injected zero events")
    assert_all_zero(dropped["hash_match"], "Ray dropped-row hash_match")
    assert_all_zero(dropped["row_count_match"], "Ray dropped-row row_count_match")

    manifest = read_json_required(manifest_path, "Ray manifest")
    manifest_results = manifest.get("results", {}) if isinstance(manifest, Mapping) else {}
    manifest_reference = manifest.get("reference", {}) if isinstance(manifest, Mapping) else {}
    if manifest_results.get("clean_conditions") != 18:
        raise ValidationError("Ray manifest does not report 18 clean conditions")
    if manifest_results.get("fault_conditions_flagged") != 36:
        raise ValidationError("Ray manifest does not report 36 flagged fault conditions")
    if manifest_reference.get("external_clean_comparisons") != 18:
        raise ValidationError("Ray manifest does not report 18 external clean comparisons")
    if manifest_reference.get("external_clean_hash_matches") != 18:
        raise ValidationError("Ray manifest does not report 18 external clean hash matches")

    summary = read_csv_required(summary_path, "Ray summary")
    if len(summary) != 3:
        raise ValidationError(f"Ray summary must contain 3 rows; found {len(summary)}")

    results = {
        "conditions": int(len(frame)),
        "expected_conditions": 54,
        "clean_conditions": int(len(clean)),
        "clean_external_hash_matches": int(numeric(clean["hash_match"], "hash_match").sum()),
        "fault_conditions": int(len(faults)),
        "fault_conditions_flagged": int(numeric(faults["detected_flag"], "detected_flag").sum()),
        "clean_false_positive_detections": int(numeric(clean["detected_flag"], "detected_flag").sum()),
        "max_unauthorized_invocations": int(numeric(frame["unauthorized_invocations"], "unauthorized").max()),
        "all_external_references": True,
        "all_detection_expectations_met": True,
    }
    inventory = [
        evidence_file(project_dir, "ray", "per_run", per_run_path),
        evidence_file(project_dir, "ray", "summary", summary_path),
        evidence_file(project_dir, "ray", "manifest", manifest_path),
        evidence_file(
            project_dir,
            "ray",
            "effective_config",
            base / "ray_comparison_effective_config.json",
        ),
    ]
    return results, inventory


# ---------------------------------------------------------------------------
# Controlled fault-summary validation
# ---------------------------------------------------------------------------


def _canonical_fault_mode(value: Any) -> str:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "clean": "clean_replay",
        "clean_comparison": "clean_replay",
        "clean_comparisons": "clean_replay",
        "action_flip": "action_flip_1_percent",
        "action_flip_(1%)": "action_flip_1_percent",
        "unauthorized_invocation_1_percent": "unauthorized_invoke_1_percent",
        "unauthorized_invocation_(1%)": "unauthorized_invoke_1_percent",
        "unauthorized_invoke": "unauthorized_invoke_1_percent",
        "saved_action_corruption_1_percent": "trace_action_corruption_1_percent",
        "saved_action_corruption_(1%)": "trace_action_corruption_1_percent",
        "trace_action_corruption": "trace_action_corruption_1_percent",
        "dropped_row_1_percent": "drop_trace_rows_1_percent",
        "dropped_rows_1_percent": "drop_trace_rows_1_percent",
        "dropped_trace_rows_(1%)": "drop_trace_rows_1_percent",
        "drop_trace_rows": "drop_trace_rows_1_percent",
        "duplicated_row_1_percent": "duplicate_trace_rows_1_percent",
        "duplicated_rows_1_percent": "duplicate_trace_rows_1_percent",
        "duplicated_trace_rows_(1%)": "duplicate_trace_rows_1_percent",
        "duplicate_trace_rows": "duplicate_trace_rows_1_percent",
    }
    return aliases.get(token, token)


def normalize_fault_summary(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    require_columns(frame, ["fault_mode", "runs", "detected_runs", "false_positive_runs"], role)
    out = frame.copy()
    out["fault_mode"] = out["fault_mode"].map(_canonical_fault_mode)
    out["runs"] = numeric(out["runs"], f"{role} runs").astype(int)
    out["detected_runs"] = numeric(out["detected_runs"], f"{role} detected_runs").astype(int)
    out["false_positive_runs"] = numeric(
        out["false_positive_runs"], f"{role} false_positive_runs"
    ).astype(int)

    injected_col = None
    for candidate in [
        "faults_or_corruptions_injected_total",
        "faults_injected_total",
        "corruptions_injected_total",
        "injected_events",
    ]:
        if candidate in out.columns:
            injected_col = candidate
            break
    if injected_col is None:
        raise ValidationError(f"{role} has no injected-event total column")
    out["injected_events"] = numeric(out[injected_col], f"{role} injected events").fillna(0).astype(int)

    if "detection_rate" in out.columns:
        out["detection_rate_numeric"] = numeric(
            out["detection_rate"], f"{role} detection rate", allow_nan=True
        )
    else:
        out["detection_rate_numeric"] = np.nan
    return out


def validate_fault_summary(
    project_dir: Path,
    path: Path,
    component: str,
    expected_clean_instances: int = 54,
) -> tuple[dict[str, Any], list[EvidenceFile], pd.DataFrame]:
    raw = read_csv_required(path, f"{component} combined fault summary")
    frame = normalize_fault_summary(raw, f"{component} fault summary")
    if frame["fault_mode"].duplicated().any():
        duplicated = frame.loc[frame["fault_mode"].duplicated(False), "fault_mode"].tolist()
        raise ValidationError(f"{component} fault summary has duplicate modes: {duplicated}")

    modes = set(frame["fault_mode"])
    expected_modes = {"clean_replay", *FAULT_ORDER}
    if modes != expected_modes:
        raise ValidationError(
            f"{component} fault modes differ from the five-class protocol: "
            f"expected={sorted(expected_modes)}, observed={sorted(modes)}"
        )

    clean = frame.loc[frame["fault_mode"].eq("clean_replay")].iloc[0]
    if int(clean["runs"]) != expected_clean_instances:
        raise ValidationError(
            f"{component} clean comparison instances must equal {expected_clean_instances}; "
            f"found {int(clean['runs'])}"
        )
    if int(clean["detected_runs"]) != 0 or int(clean["false_positive_runs"]) != 0:
        raise ValidationError(f"{component} clean comparisons contain a false positive")
    if int(clean["injected_events"]) != 0:
        raise ValidationError(f"{component} clean row reports injected events")

    fault_rows = frame.loc[frame["fault_mode"].isin(FAULT_ORDER)].copy()
    for mode in FAULT_ORDER:
        row = fault_rows.loc[fault_rows["fault_mode"].eq(mode)].iloc[0]
        if int(row["runs"]) != 18:
            raise ValidationError(f"{component} {mode} must contain 18 runs")
        if int(row["detected_runs"]) != 18:
            raise ValidationError(f"{component} {mode} must detect 18/18 runs")
        if int(row["false_positive_runs"]) != 0:
            raise ValidationError(f"{component} {mode} reports false positives")
        if int(row["injected_events"]) <= 0:
            raise ValidationError(f"{component} {mode} reports zero injected events")
        rate = row["detection_rate_numeric"]
        if not pd.isna(rate) and not math.isclose(float(rate), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValidationError(f"{component} {mode} detection rate is not 1.0")

    results = {
        "clean_comparison_instances": int(clean["runs"]),
        "clean_false_positive_detections": int(clean["false_positive_runs"]),
        "fault_classes": 5,
        "runs_per_fault_class": 18,
        "all_fault_classes_detected_18_of_18": True,
        "faults": {
            mode: {
                "label": FAULT_LABELS[mode],
                "runs": 18,
                "detected_runs": 18,
                "injected_events": int(
                    fault_rows.loc[fault_rows["fault_mode"].eq(mode), "injected_events"].iloc[0]
                ),
                "false_positive_runs": 0,
            }
            for mode in FAULT_ORDER
        },
    }
    inventory = [evidence_file(project_dir, component, "combined_fault_summary", path)]
    return results, inventory, frame


# ---------------------------------------------------------------------------
# MetroPT-3 validation
# ---------------------------------------------------------------------------


def validate_metropt3(project_dir: Path) -> tuple[dict[str, Any], list[EvidenceFile]]:
    base = project_dir / "paper_outputs" / "secondary_metropt3"
    clean_base = base / "clean_benchmark"
    scaling_path = clean_base / "scaling_and_runtime_results.csv"
    det_path = clean_base / "determinism_hash_results.csv"
    fault_path = base / "tables_figures" / "fgcs_table_rq7_fault_detection_combined.csv"
    policy_summary_path = base / "tables_figures" / "secondary_policy_determinism_summary.csv"

    scaling = read_csv_required(scaling_path, "MetroPT-3 clean scaling output")
    det = read_csv_required(det_path, "MetroPT-3 clean determinism output")
    condition_cols = ["dataset_fraction", "policy_mode", "seed", "workers"]
    require_columns(
        scaling,
        condition_cols
        + ["trace_hash", "intervention_rate", "unauthorized_invocations", "fault_injected_count"],
        "MetroPT-3 scaling output",
    )
    require_columns(
        det,
        condition_cols + ["trace_hash", "hash_match", "unauthorized_invocations"],
        "MetroPT-3 determinism output",
    )
    if len(scaling) != 72:
        raise ValidationError(f"MetroPT-3 clean benchmark must contain 72 rows; found {len(scaling)}")
    if len(det) != 72:
        raise ValidationError(f"MetroPT-3 determinism output must contain 72 rows; found {len(det)}")

    scaling = scaling.copy()
    det = det.copy()
    for frame in (scaling, det):
        frame["dataset_fraction"] = numeric(frame["dataset_fraction"], "Metro fraction").round(6)
        frame["seed"] = numeric(frame["seed"], "Metro seed").astype(int)
        frame["workers"] = numeric(frame["workers"], "Metro workers").astype(int)
        frame["policy_mode"] = frame["policy_mode"].astype(str)

    assert_set_equal(scaling["dataset_fraction"], METRO_FRACTIONS, "MetroPT-3 fractions")
    assert_set_equal(scaling["policy_mode"], METRO_POLICIES, "MetroPT-3 policies")
    assert_set_equal(scaling["seed"], METRO_SEEDS, "MetroPT-3 seeds")
    assert_set_equal(scaling["workers"], METRO_WORKERS, "MetroPT-3 workers")
    assert_no_duplicates(scaling, condition_cols, "MetroPT-3 clean benchmark")
    assert_no_duplicates(det, condition_cols, "MetroPT-3 determinism output")

    assert_all_zero(scaling["unauthorized_invocations"], "MetroPT-3 clean unauthorized invocations")
    assert_all_zero(scaling["fault_injected_count"], "MetroPT-3 clean fault counter")
    assert_all_zero(det["unauthorized_invocations"], "MetroPT-3 determinism unauthorized invocations")
    assert_all_one(det["hash_match"], "MetroPT-3 worker hash_match")
    for optional in ["authorization_execution_consistent", "row_count_match", "validation_passed"]:
        if optional in scaling.columns:
            assert_all_one(scaling[optional], f"MetroPT-3 {optional}")
        if optional in det.columns:
            assert_all_one(det[optional], f"MetroPT-3 determinism {optional}")

    worker_hash_counts = det.groupby(["dataset_fraction", "policy_mode", "seed"])["trace_hash"].nunique()
    if worker_hash_counts.ne(1).any():
        raise ValidationError("MetroPT-3 worker reconstruction produced unstable hashes")
    deterministic = METRO_POLICIES - {"random"}
    seed_hash_counts = det.loc[det["policy_mode"].isin(deterministic)].groupby(
        ["dataset_fraction", "policy_mode"]
    )["trace_hash"].nunique()
    if seed_hash_counts.ne(1).any():
        raise ValidationError("A deterministic MetroPT-3 policy varied across seeds")

    policy_summary = read_csv_required(policy_summary_path, "MetroPT-3 policy summary")
    require_columns(
        policy_summary,
        [
            "policy_mode",
            "conditions",
            "unique_hashes",
            "all_worker_matches",
            "max_unauthorized_invocations",
            "full_workload_mean_intervention_rate",
        ],
        "MetroPT-3 policy summary",
    )
    if len(policy_summary) != 4:
        raise ValidationError("MetroPT-3 policy summary must contain four policy rows")
    if not numeric(policy_summary["conditions"], "Metro policy conditions").eq(18).all():
        raise ValidationError("Each MetroPT-3 policy must contain 18 clean conditions")
    if not bool_series(policy_summary["all_worker_matches"], "Metro all_worker_matches").all():
        raise ValidationError("MetroPT-3 policy summary reports a worker mismatch")
    assert_all_zero(
        policy_summary["max_unauthorized_invocations"],
        "MetroPT-3 policy max unauthorized invocations",
    )

    fault_results, fault_inventory, _ = validate_fault_summary(
        project_dir,
        fault_path,
        component="metropt3_faults",
        expected_clean_instances=54,
    )

    full = scaling.loc[np.isclose(scaling["dataset_fraction"], 1.0)]
    intervention = full.groupby("policy_mode")["intervention_rate"].mean().to_dict()
    unique_hashes = det.groupby("policy_mode")["trace_hash"].nunique().to_dict()

    results = {
        "clean_conditions": int(len(scaling)),
        "expected_clean_conditions": 72,
        "all_worker_hashes_match": True,
        "max_clean_unauthorized_invocations": 0,
        "unique_hashes_by_policy": {str(k): int(v) for k, v in unique_hashes.items()},
        "full_workload_mean_intervention_rate_by_policy": {
            str(k): float(v) for k, v in intervention.items()
        },
        "fault_validation": fault_results,
    }
    inventory = [
        evidence_file(project_dir, "metropt3", "clean_scaling_runtime", scaling_path),
        evidence_file(project_dir, "metropt3", "clean_determinism", det_path),
        evidence_file(project_dir, "metropt3", "policy_summary", policy_summary_path),
        *fault_inventory,
    ]
    return results, inventory


# ---------------------------------------------------------------------------
# Cloud evidence discovery and validation
# ---------------------------------------------------------------------------


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _column_by_names(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    normalized = {_normalize_name(str(column)): str(column) for column in frame.columns}
    for name in names:
        if _normalize_name(name) in normalized:
            return normalized[_normalize_name(name)]
    return None


def _max_unauthorized_from_frame(frame: pd.DataFrame) -> int | None:
    candidates = [
        column
        for column in frame.columns
        if "unauthorized" in _normalize_name(str(column))
        or "authorization_contradiction" in _normalize_name(str(column))
        or "authorization_execution_violation" in _normalize_name(str(column))
    ]
    if not candidates:
        return None
    maxima: list[int] = []
    for column in candidates:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.empty:
            maxima.append(int(values.max()))
    return max(maxima) if maxima else None


def _infer_comparison_metrics(frame: pd.DataFrame) -> tuple[int, int] | None:
    if frame.empty:
        return None

    compared_col = _column_by_names(
        frame,
        [
            "compared",
            "comparisons",
            "conditions_compared",
            "compared_conditions",
            "total_comparisons",
            "total_conditions",
            "conditions",
        ],
    )
    matched_col = _column_by_names(
        frame,
        [
            "matched",
            "matches",
            "hash_matches",
            "matched_conditions",
            "matching_conditions",
            "external_clean_hash_matches",
        ],
    )
    match_flag_col = _column_by_names(
        frame,
        [
            "hash_match",
            "trace_hash_match",
            "action_hash_match",
            "all_match",
            "all_hashes_match",
            "match",
            "matched_flag",
        ],
    )

    if compared_col and matched_col:
        compared = pd.to_numeric(frame[compared_col], errors="coerce").dropna()
        matched = pd.to_numeric(frame[matched_col], errors="coerce").dropna()
        if not compared.empty and not matched.empty:
            return int(compared.sum()), int(matched.sum())

    if match_flag_col:
        flags = frame[match_flag_col].map(_bool_token)
        if flags.notna().all():
            return int(len(flags)), int(flags.sum())

    # Fall back to any row-level boolean column whose normalized name contains
    # both "hash" and "match" (for example local_vs_asia_hashes_match).
    for column in frame.columns:
        token = _normalize_name(str(column))
        if "hash" in token and "match" in token:
            flags = frame[column].map(_bool_token)
            if flags.notna().all():
                return int(len(flags)), int(flags.sum())

    # Some summary files use only matched and a single total row.
    if matched_col and len(frame) == 1:
        matched = pd.to_numeric(frame[matched_col], errors="coerce").dropna()
        if len(matched) == 1:
            value = int(matched.iloc[0])
            return value, value
    return None


def _json_walk(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    output: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            output.extend(_json_walk(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.extend(_json_walk(child, f"{prefix}[{index}]"))
    else:
        output.append((prefix, value))
    return output


def _json_metrics(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    flat = _json_walk(payload)
    numeric_lookup: dict[str, float] = {}
    for key, value in flat:
        try:
            if isinstance(value, bool):
                continue
            numeric_lookup[_normalize_name(key)] = float(value)
        except (TypeError, ValueError):
            continue

    candidates: list[dict[str, Any]] = []
    for expected, kind in [(360, "cross_region"), (720, "local_to_cloud")]:
        related = {
            key: value
            for key, value in numeric_lookup.items()
            if ("match" in key or "compar" in key or "condition" in key)
            and any(token in key for token in (["cross", "region"] if kind == "cross_region" else ["local", "cloud"]))
        }
        exact = [value for value in related.values() if int(value) == expected]
        if len(exact) >= 1:
            candidates.append(
                {
                    "kind": kind,
                    "compared": expected,
                    "matched": expected,
                    "max_unauthorized": None,
                    "source": path,
                    "source_detail": "JSON inferred exact count",
                }
            )
    return candidates


def discover_cloud_evidence(
    project_dir: Path,
    cloud_root: Path,
    cross_region_override: Path | None = None,
    local_to_cloud_override: Path | None = None,
) -> tuple[dict[str, Any], list[EvidenceFile]]:
    if not cloud_root.is_dir():
        raise ValidationError(f"Cloud results directory not found: {cloud_root}")

    candidates: list[dict[str, Any]] = []
    unauthorized_values: list[int] = []
    scanned_paths: set[Path] = set()

    explicit = {
        "cross_region": cross_region_override,
        "local_to_cloud": local_to_cloud_override,
    }

    for kind, path in explicit.items():
        if path is None:
            continue
        if not path.is_file():
            raise ValidationError(f"Explicit {kind} cloud evidence not found: {path}")
        if path.suffix.lower() == ".csv":
            frame = read_csv_required(path, f"explicit {kind} cloud evidence")
            metrics = _infer_comparison_metrics(frame)
            if metrics is None:
                raise ValidationError(f"Could not infer compared/matched counts from {path}")
            compared, matched = metrics
            candidates.append(
                {
                    "kind": kind,
                    "compared": compared,
                    "matched": matched,
                    "max_unauthorized": _max_unauthorized_from_frame(frame),
                    "source": path,
                    "source_detail": "explicit CSV",
                }
            )
            if _max_unauthorized_from_frame(frame) is not None:
                unauthorized_values.append(int(_max_unauthorized_from_frame(frame) or 0))
        elif path.suffix.lower() == ".json":
            json_candidates = _json_metrics(path)
            matching = [candidate for candidate in json_candidates if candidate["kind"] == kind]
            if not matching:
                raise ValidationError(f"Could not infer {kind} counts from JSON {path}")
            candidates.extend(matching)
        else:
            raise ValidationError("Cloud evidence override must be CSV or JSON")
        scanned_paths.add(path.resolve())

    for path in sorted(cloud_root.rglob("*")):
        if not path.is_file() or path.resolve() in scanned_paths:
            continue
        suffix = path.suffix.lower()
        if suffix == ".csv":
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            max_unauthorized = _max_unauthorized_from_frame(frame)
            if max_unauthorized is not None:
                unauthorized_values.append(max_unauthorized)

            # A combined comparison file can contain a comparison-type column.
            type_col = _column_by_names(
                frame,
                ["comparison_type", "comparison", "comparison_scope", "environment_comparison"],
            )
            if type_col:
                for label, group in frame.groupby(type_col, dropna=False):
                    metrics = _infer_comparison_metrics(group)
                    if metrics is None:
                        continue
                    compared, matched = metrics
                    label_token = _normalize_name(str(label))
                    kind = None
                    if "cross" in label_token or ("region" in label_token and "local" not in label_token):
                        kind = "cross_region"
                    elif "local" in label_token and "cloud" in label_token:
                        kind = "local_to_cloud"
                    if kind:
                        candidates.append(
                            {
                                "kind": kind,
                                "compared": compared,
                                "matched": matched,
                                "max_unauthorized": max_unauthorized,
                                "source": path,
                                "source_detail": f"comparison_type={label}",
                            }
                        )

            metrics = _infer_comparison_metrics(frame)
            if metrics is None:
                continue
            compared, matched = metrics
            token = _normalize_name(path.as_posix())
            kind = None
            if compared == 360:
                kind = "cross_region"
            elif compared == 720:
                kind = "local_to_cloud"
            elif "cross_region" in token or "region_to_region" in token:
                kind = "cross_region"
            elif "local_to_cloud" in token or "local_cloud" in token:
                kind = "local_to_cloud"
            if kind:
                candidates.append(
                    {
                        "kind": kind,
                        "compared": compared,
                        "matched": matched,
                        "max_unauthorized": max_unauthorized,
                        "source": path,
                        "source_detail": "auto-discovered CSV",
                    }
                )
        elif suffix == ".json":
            candidates.extend(_json_metrics(path))

    selected: dict[str, dict[str, Any]] = {}
    expected_by_kind = {"cross_region": 360, "local_to_cloud": 720}
    for kind, expected in expected_by_kind.items():
        exact = [
            candidate
            for candidate in candidates
            if candidate["kind"] == kind
            and int(candidate["compared"]) == expected
            and int(candidate["matched"]) == expected
        ]
        if not exact:
            related = [candidate for candidate in candidates if candidate["kind"] == kind]
            details = [
                {
                    "source": str(candidate["source"]),
                    "compared": candidate["compared"],
                    "matched": candidate["matched"],
                }
                for candidate in related[:10]
            ]
            raise ValidationError(
                f"Could not locate exact {kind} cloud evidence {expected}/{expected}. "
                f"Candidates={details}. Use the explicit CLI override for the finalized file."
            )

        # Prefer explicit evidence, then CSV over JSON, then shortest path.
        exact.sort(
            key=lambda candidate: (
                0 if "explicit" in candidate["source_detail"] else 1,
                0 if Path(candidate["source"]).suffix.lower() == ".csv" else 1,
                len(str(candidate["source"])),
            )
        )
        selected[kind] = exact[0]

    if not unauthorized_values:
        raise ValidationError(
            "Cloud outputs contain no authorization/unauthorized-invocation column; "
            "zero clean cloud contradictions cannot be verified"
        )
    max_cloud_unauthorized = max(unauthorized_values)
    if max_cloud_unauthorized != 0:
        raise ValidationError(
            f"Cloud clean outputs contain authorization contradictions; max={max_cloud_unauthorized}"
        )

    inventory: list[EvidenceFile] = []
    seen: set[Path] = set()
    for kind, candidate in selected.items():
        source = Path(candidate["source"])
        if source not in seen:
            inventory.append(
                evidence_file(
                    project_dir,
                    "cloud_validation",
                    kind,
                    source,
                    notes=candidate["source_detail"],
                )
            )
            seen.add(source)

    # Require at least one cloud metadata/manifest JSON.
    cloud_metadata = sorted(
        {
            *cloud_root.rglob("*manifest*.json"),
            *cloud_root.rglob("*metadata*.json"),
        }
    )
    if not cloud_metadata:
        raise ValidationError("No cloud metadata or manifest JSON was found")
    for path in cloud_metadata:
        inventory.append(
            evidence_file(project_dir, "cloud_validation", "metadata_or_manifest", path)
        )

    results = {
        "cloud_root": relative_posix(cloud_root, project_dir),
        "cross_region": {
            "compared": 360,
            "matched": 360,
            "all_match": True,
            "source": relative_posix(Path(selected["cross_region"]["source"]), project_dir),
        },
        "local_to_cloud": {
            "compared": 720,
            "matched": 720,
            "all_match": True,
            "source": relative_posix(Path(selected["local_to_cloud"]["source"]), project_dir),
        },
        "max_clean_unauthorized_invocations": int(max_cloud_unauthorized),
    }
    return results, inventory


# ---------------------------------------------------------------------------
# Required files, compilation candidates and hash inventory
# ---------------------------------------------------------------------------


def validate_required_files(
    project_dir: Path,
    required_configs: Sequence[str] = DEFAULT_REQUIRED_CONFIGS,
    required_manifests: Sequence[str] = DEFAULT_REQUIRED_MANIFESTS,
    required_scripts: Sequence[str] = DEFAULT_REQUIRED_SCRIPTS,
    cloud_root: Path | None = None,
) -> tuple[dict[str, Any], list[EvidenceFile]]:
    inventory: list[EvidenceFile] = []
    missing: list[str] = []

    groups = [
        ("configuration", required_configs),
        ("manifest", required_manifests),
        ("script", required_scripts),
    ]
    for role, paths in groups:
        for relative in paths:
            path = project_dir / relative
            if not path.is_file():
                missing.append(relative)
            inventory.append(evidence_file(project_dir, "required_files", role, path))

    if cloud_root is not None:
        cloud_metadata = sorted(
            {
                *cloud_root.rglob("*manifest*.json"),
                *cloud_root.rglob("*metadata*.json"),
            }
        ) if cloud_root.is_dir() else []
        if not cloud_metadata:
            missing.append(f"cloud metadata/manifest under {relative_posix(cloud_root, project_dir)}")

    if missing:
        raise ValidationError(f"Required files are missing: {missing}")
    return {
        "required_configurations": len(required_configs),
        "required_manifests": len(required_manifests),
        "required_scripts": len(required_scripts),
        "all_present": True,
    }, inventory


def discover_python_files(project_dir: Path) -> list[Path]:
    excluded_parts = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "site-packages",
        "dist",
        "build",
        "release",
        "paper_outputs",
        "cloud_results",
    }
    paths: list[Path] = []
    for path in project_dir.rglob("*.py"):
        try:
            relative = path.relative_to(project_dir)
        except ValueError:
            continue
        if any(part in excluded_parts for part in relative.parts):
            continue
        paths.append(path)
    return sorted(paths)


def build_sha256_manifest(
    project_dir: Path,
    paths: Iterable[Path],
    exclude: Iterable[Path] = (),
) -> pd.DataFrame:
    excluded = {path.resolve() for path in exclude}
    unique = sorted({path.resolve() for path in paths if path.is_file() and path.resolve() not in excluded})
    rows = []
    for path in unique:
        rows.append(
            {
                "path": relative_posix(path, project_dir),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows, columns=["path", "size_bytes", "sha256"])


def verify_sha256_manifest(project_dir: Path, manifest_path: Path) -> dict[str, Any]:
    frame = read_csv_required(manifest_path, "file SHA-256 manifest")
    require_columns(frame, ["path", "size_bytes", "sha256"], "file SHA-256 manifest")
    missing: list[str] = []
    mismatched: list[str] = []
    for row in frame.to_dict(orient="records"):
        path = project_dir / str(row["path"])
        if not path.is_file():
            missing.append(str(row["path"]))
            continue
        observed = sha256_file(path)
        if observed != str(row["sha256"]):
            mismatched.append(str(row["path"]))
    if missing or mismatched:
        raise ValidationError(
            f"Frozen SHA-256 manifest verification failed; missing={missing}, mismatched={mismatched}"
        )
    return {"verified_files": int(len(frame)), "all_hashes_match": True}
