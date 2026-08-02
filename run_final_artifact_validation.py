#!/usr/bin/env python3
"""Master final validator for the frozen ReplayBench-PG evidence package.

The validator does not rerun experiments. It validates the completed primary,
common timing, bc_live phase-decomposition, execution-integrity,
validator-selectivity, Ray, MetroPT-3, controlled-fault, and cloud outputs;
compiles every repository Python file; runs the complete pytest suite; and emits
a machine-readable final validation package.

Default outputs
---------------
paper_outputs/final_validation/final_validation_manifest.json
paper_outputs/final_validation/final_results_inventory.csv
paper_outputs/final_validation/final_claims_numbers.json
paper_outputs/final_validation/file_sha256_manifest.csv
"""

from __future__ import annotations

import argparse
import json
import py_compile
import re
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from comment12_artifact_validation import validate_comment12_artifacts
from comment13_artifact_validation import validate_comment13_artifacts
from final_artifact_utils import (
    DEFAULT_REQUIRED_CONFIGS,
    DEFAULT_REQUIRED_MANIFESTS,
    DEFAULT_REQUIRED_SCRIPTS,
    FAULT_CLEAN_VALIDATOR_APPLICATIONS,
    FAULT_RUNS_PER_CLASS,
    FAULT_UNIQUE_CLEAN_REFERENCES,
    FAULT_VALIDATOR_WORKFLOWS,
    ValidationError,
    atomic_write_csv,
    atomic_write_json,
    assert_all_one,
    assert_all_zero,
    assert_no_duplicates,
    bool_series,
    build_sha256_manifest,
    discover_cloud_evidence,
    discover_python_files,
    evidence_file,
    json_safe,
    numeric,
    read_csv_required,
    read_json_required,
    relative_posix,
    require_columns,
    sha256_file,
    validate_fault_summary,
    validate_metropt3,
    validate_primary_benchmark,
    validate_ray,
    validate_required_files,
    validate_timing_study,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compile_all_python(project_dir: Path) -> dict[str, Any]:
    python_files = discover_python_files(project_dir)
    if not python_files:
        raise ValidationError("No repository Python files were discovered")

    compiled: list[str] = []
    with tempfile.TemporaryDirectory(prefix="replaybench_pycompile_") as temp_dir:
        temp_root = Path(temp_dir)
        for index, path in enumerate(python_files):
            cfile = temp_root / f"{index:05d}.pyc"
            try:
                py_compile.compile(str(path), cfile=str(cfile), doraise=True)
            except py_compile.PyCompileError as exc:
                raise ValidationError(
                    f"Python compilation failed for {relative_posix(path, project_dir)}: {exc}"
                ) from exc
            compiled.append(relative_posix(path, project_dir))

    return {
        "python_files_discovered": int(len(python_files)),
        "python_files_compiled": int(len(compiled)),
        "all_python_files_compile": True,
        "compiled_files": compiled,
    }


def run_all_tests(project_dir: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(
        command,
        cwd=project_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout or ""
    print("[CMD] " + " ".join(command))
    print(output, end="" if output.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise ValidationError(
            "The complete pytest suite failed.\n"
            f"Exit code: {completed.returncode}\n"
            f"Output:\n{output}"
        )

    passed_match = re.search(r"(?P<count>\d+)\s+passed", output)
    if passed_match is None:
        raise ValidationError(
            "pytest returned success but the number of passed tests could not be parsed"
        )
    passed = int(passed_match.group("count"))
    if passed <= 0:
        raise ValidationError("pytest did not execute any passing tests")
    return {
        "command": command,
        "exit_code": int(completed.returncode),
        "tests_passed": passed,
        "all_tests_passed": True,
        "output": output.strip(),
    }


def validate_code_quality_fix(project_dir: Path) -> dict[str, Any]:
    path = project_dir / "fgcs_fault_validation_framework.py"
    if not path.is_file():
        raise ValidationError(f"Fault-validation framework not found: {path}")
    text = path.read_text(encoding="utf-8")
    deprecated_patterns = [
        "matrix[validator_cols].applymap(",
        ".applymap(lambda x: 1 if str(x).lower() == \"yes\" else 0)",
    ]
    present = [pattern for pattern in deprecated_patterns if pattern in text]
    if present:
        raise ValidationError(
            "Deprecated DataFrame.applymap usage remains in "
            "fgcs_fault_validation_framework.py. Run:\n"
            "  python apply_final_code_quality_fix.py --apply"
        )
    if "matrix[validator_cols].map(" not in text:
        raise ValidationError(
            "Expected DataFrame.map replacement was not found in "
            "fgcs_fault_validation_framework.py"
        )
    return {
        "file": relative_posix(path, project_dir),
        "deprecated_applymap_removed": True,
        "dataframe_map_replacement_present": True,
    }



def _resolve_evidence_base(
    project_dir: Path,
    candidates: list[str],
    required_names: list[str],
    role: str,
) -> Path:
    """Return the first candidate directory containing every required file."""

    checked: list[str] = []
    for relative in candidates:
        base = project_dir / relative
        checked.append(relative_posix(base, project_dir))
        if base.is_dir() and all((base / name).is_file() for name in required_names):
            return base
    raise ValidationError(
        f"Could not locate complete {role} evidence. Checked={checked}; "
        f"required_files={required_names}"
    )


def _require_manifest_true(manifest: dict[str, Any], keys: list[str], role: str) -> None:
    for key in keys:
        value = manifest.get(key)
        if value is not True:
            raise ValidationError(f"{role} field {key} must be true; found {value!r}")


def _require_manifest_int(
    manifest: dict[str, Any], key: str, expected: int, role: str
) -> None:
    try:
        observed = int(manifest.get(key))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{role} field {key} is not an integer") from exc
    if observed != expected:
        raise ValidationError(
            f"{role} field {key} must equal {expected}; found {observed}"
        )


def validate_bc_live_runtime_decomposition(
    project_dir: Path,
) -> tuple[dict[str, Any], list[Any]]:
    base = _resolve_evidence_base(
        project_dir,
        ["paper_outputs/bc_live_runtime_decomposition"],
        [
            "bc_live_runtime_decomposition_raw.csv",
            "bc_live_runtime_decomposition_summary.csv",
            "bc_live_runtime_decomposition_manifest.json",
        ],
        "bc_live runtime-decomposition",
    )
    raw_path = base / "bc_live_runtime_decomposition_raw.csv"
    summary_path = base / "bc_live_runtime_decomposition_summary.csv"
    manifest_path = base / "bc_live_runtime_decomposition_manifest.json"

    raw = read_csv_required(raw_path, "bc_live runtime-decomposition raw output")
    require_columns(
        raw,
        [
            "dataset_fraction",
            "decision_points",
            "policy_mode",
            "policy_seed",
            "workers",
            "repetition",
            "end_to_end_runtime_seconds",
            "checkpoint_preparation_seconds",
            "replay_only_runtime_seconds",
            "post_replay_validation_seconds",
            "timed_execution_runtime_seconds",
            "runtime_decomposition_tolerance_seconds",
            "runtime_decomposition_valid",
            "trace_hash",
            "reference_hash",
            "hash_match",
            "unauthorized_invocations",
            "authorization_execution_consistent",
            "row_count_match",
            "validation_passed",
            "fault_injected_count",
        ],
        "bc_live runtime-decomposition raw output",
    )
    if len(raw) != 88:
        raise ValidationError(
            f"bc_live decomposition must contain 88 measured rows; found {len(raw)}"
        )

    raw = raw.copy()
    raw["dataset_fraction"] = numeric(
        raw["dataset_fraction"], "bc_live decomposition fraction"
    ).round(6)
    raw["workers"] = numeric(raw["workers"], "bc_live decomposition workers").astype(int)
    raw["policy_seed"] = numeric(
        raw["policy_seed"], "bc_live decomposition policy seed"
    ).astype(int)
    raw["repetition"] = numeric(
        raw["repetition"], "bc_live decomposition repetition"
    ).astype(int)
    raw["policy_mode"] = raw["policy_mode"].astype(str)

    if set(raw["policy_mode"]) != {"bc_live"}:
        raise ValidationError("bc_live decomposition contains a non-bc_live policy")
    if set(raw["policy_seed"]) != {1}:
        raise ValidationError("bc_live decomposition must use policy seed 1")
    assert_no_duplicates(
        raw,
        ["dataset_fraction", "workers", "repetition"],
        "bc_live decomposition repetitions",
    )

    group_sizes = raw.groupby(["dataset_fraction", "workers"]).size().to_dict()
    expected_groups = {
        (0.10, 1): 7,
        (0.25, 1): 7,
        (0.50, 1): 7,
        (0.75, 1): 7,
        (1.00, 1): 15,
        (1.00, 2): 15,
        (1.00, 4): 15,
        (1.00, 8): 15,
    }
    normalized_groups = {
        (round(float(fraction), 6), int(workers)): int(count)
        for (fraction, workers), count in group_sizes.items()
    }
    if normalized_groups != expected_groups:
        raise ValidationError(
            "Unexpected bc_live decomposition mixed-repetition design: "
            f"{normalized_groups}"
        )

    for column in [
        "runtime_decomposition_valid",
        "hash_match",
        "authorization_execution_consistent",
        "row_count_match",
        "validation_passed",
    ]:
        assert_all_one(raw[column], f"bc_live decomposition {column}")
    for column in ["unauthorized_invocations", "fault_injected_count"]:
        assert_all_zero(raw[column], f"bc_live decomposition {column}")

    for column in [
        "end_to_end_runtime_seconds",
        "checkpoint_preparation_seconds",
        "replay_only_runtime_seconds",
        "post_replay_validation_seconds",
        "timed_execution_runtime_seconds",
    ]:
        values = numeric(raw[column], f"bc_live decomposition {column}")
        if values.le(0).any():
            raise ValidationError(f"bc_live decomposition {column} must be positive")

    end_to_end = numeric(raw["end_to_end_runtime_seconds"], "bc_live end-to-end")
    components = (
        numeric(raw["checkpoint_preparation_seconds"], "bc_live preparation")
        + numeric(raw["replay_only_runtime_seconds"], "bc_live replay-only")
        + numeric(raw["post_replay_validation_seconds"], "bc_live validation")
    )
    tolerance = numeric(
        raw["runtime_decomposition_tolerance_seconds"],
        "bc_live decomposition tolerance",
    )
    if (end_to_end.sub(components).abs() > tolerance.add(1e-12)).any():
        raise ValidationError("bc_live end-to-end phase decomposition is inconsistent")

    unstable = raw.groupby(["dataset_fraction", "workers"])["trace_hash"].nunique()
    if unstable.ne(1).any():
        raise ValidationError("bc_live decomposition trace hashes are unstable")
    if not raw["trace_hash"].astype(str).eq(raw["reference_hash"].astype(str)).all():
        raise ValidationError("bc_live decomposition trace/reference hashes differ")

    summary = read_csv_required(summary_path, "bc_live runtime-decomposition summary")
    require_columns(
        summary,
        [
            "dataset_fraction",
            "workers",
            "measured_repetitions",
            "unique_trace_hashes",
            "all_runtime_decompositions_valid",
            "end_to_end_runtime_seconds_median",
            "checkpoint_preparation_seconds_median",
            "replay_only_runtime_seconds_median",
            "post_replay_validation_seconds_median",
            "checkpoint_preparation_share_median",
            "replay_only_share_median",
            "post_replay_validation_share_median",
        ],
        "bc_live runtime-decomposition summary",
    )
    if len(summary) != 8:
        raise ValidationError(
            f"bc_live decomposition summary must contain 8 rows; found {len(summary)}"
        )
    summary = summary.copy()
    summary["dataset_fraction"] = numeric(
        summary["dataset_fraction"], "bc_live summary fraction"
    ).round(6)
    summary["workers"] = numeric(summary["workers"], "bc_live summary workers").astype(int)
    summary["measured_repetitions"] = numeric(
        summary["measured_repetitions"], "bc_live summary repetitions"
    ).astype(int)
    summary_groups = {
        (round(float(row.dataset_fraction), 6), int(row.workers)): int(
            row.measured_repetitions
        )
        for row in summary.itertuples(index=False)
    }
    if summary_groups != expected_groups:
        raise ValidationError("bc_live summary design differs from the raw design")
    assert_all_one(summary["unique_trace_hashes"], "bc_live summary unique hashes")
    assert_all_one(
        summary["all_runtime_decompositions_valid"],
        "bc_live summary decomposition status",
    )

    manifest_obj = read_json_required(manifest_path, "bc_live decomposition manifest")
    if not isinstance(manifest_obj, dict):
        raise ValidationError("bc_live decomposition manifest must be a JSON object")
    _require_manifest_int(manifest_obj, "unique_configurations", 8, "bc_live manifest")
    _require_manifest_int(manifest_obj, "expected_measured_rows", 88, "bc_live manifest")
    _require_manifest_int(manifest_obj, "completed_measured_rows", 88, "bc_live manifest")
    _require_manifest_true(
        manifest_obj,
        [
            "all_runtime_decompositions_valid",
            "all_hashes_stable",
            "all_authorization_execution_consistent",
            "all_row_counts_match",
            "all_validation_passed",
            "all_fault_counts_zero",
            "all_unauthorized_invocations_zero",
        ],
        "bc_live manifest",
    )
    if str(manifest_obj.get("raw_csv_sha256", "")).lower() != sha256_file(raw_path):
        raise ValidationError("bc_live raw CSV SHA-256 differs from its manifest")
    if str(manifest_obj.get("summary_csv_sha256", "")).lower() != sha256_file(summary_path):
        raise ValidationError("bc_live summary CSV SHA-256 differs from its manifest")

    full_one = summary.loc[
        summary["dataset_fraction"].eq(1.0) & summary["workers"].eq(1)
    ]
    if len(full_one) != 1:
        raise ValidationError("bc_live summary lacks the full-workload one-worker row")
    row = full_one.iloc[0]
    results = {
        "measured_executions": 88,
        "unique_configurations": 8,
        "configurations_x_7_repetitions": 4,
        "configurations_x_15_repetitions": 4,
        "all_runtime_decompositions_valid": True,
        "all_trace_hashes_stable": True,
        "max_authorization_contradictions": 0,
        "full_workload_one_worker": {
            "end_to_end_runtime_seconds_median": float(
                row["end_to_end_runtime_seconds_median"]
            ),
            "checkpoint_preparation_seconds_median": float(
                row["checkpoint_preparation_seconds_median"]
            ),
            "replay_only_runtime_seconds_median": float(
                row["replay_only_runtime_seconds_median"]
            ),
            "post_replay_validation_seconds_median": float(
                row["post_replay_validation_seconds_median"]
            ),
            "checkpoint_preparation_share_median": float(
                row["checkpoint_preparation_share_median"]
            ),
            "replay_only_share_median": float(row["replay_only_share_median"]),
            "post_replay_validation_share_median": float(
                row["post_replay_validation_share_median"]
            ),
        },
    }
    inventory = [
        evidence_file(project_dir, "bc_live_runtime_decomposition", "raw", raw_path),
        evidence_file(
            project_dir, "bc_live_runtime_decomposition", "summary", summary_path
        ),
        evidence_file(
            project_dir, "bc_live_runtime_decomposition", "manifest", manifest_path
        ),
    ]
    return results, inventory


def validate_execution_integrity_validation(
    project_dir: Path,
) -> tuple[dict[str, Any], list[Any]]:
    names = [
        "execution_receipt_validation_per_run.csv",
        "execution_receipt_validation_summary_corrected.csv",
        "record_bound_corruption_validation_per_run.csv",
        "record_bound_corruption_validation_summary.csv",
        "execution_integrity_manifest.json",
    ]
    base = _resolve_evidence_base(
        project_dir,
        ["paper_outputs/execution_integrity_validation", "revision_docs/evidence"],
        names,
        "execution-integrity",
    )
    receipt_path = base / names[0]
    receipt_summary_path = base / names[1]
    record_path = base / names[2]
    record_summary_path = base / names[3]
    manifest_path = base / names[4]

    receipt = read_csv_required(receipt_path, "execution-integrity receipt per-run output")
    require_columns(
        receipt,
        [
            "fault_mode",
            "policy_mode",
            "seed",
            "workers",
            "decision_points",
            "faults_injected",
            "receipt_rows",
            "missing_receipts",
            "unlogged_downstream_calls",
            "duplicate_downstream_calls",
            "mismatched_correlation_ids",
            "unauthorized_downstream_calls",
            "receipt_validation_passed",
            "detected_as_expected",
            "action_hash",
            "record_trace_hash",
            "config_manifest_hash",
            "config_bound_trace_hash",
        ],
        "execution-integrity receipt per-run output",
    )
    if len(receipt) != 90:
        raise ValidationError(
            f"Execution-integrity receipt output must contain 90 instances; found {len(receipt)}"
        )
    receipt = receipt.copy()
    receipt["fault_mode"] = receipt["fault_mode"].astype(str)
    receipt["policy_mode"] = receipt["policy_mode"].astype(str)
    receipt["seed"] = numeric(receipt["seed"], "receipt seed").astype(int)
    receipt["workers"] = numeric(receipt["workers"], "receipt workers").astype(int)
    expected_receipt_modes = {
        "clean",
        "unlogged_downstream_call",
        "false_execution_log",
        "duplicate_downstream_call",
        "mismatched_correlation_id",
    }
    if set(receipt["fault_mode"]) != expected_receipt_modes:
        raise ValidationError("Execution-integrity receipt modes are incomplete")
    if set(receipt["policy_mode"]) != {"risk_proxy", "random", "always"}:
        raise ValidationError("Execution-integrity policies are incomplete")
    if set(receipt["seed"]) != {1, 2, 3} or set(receipt["workers"]) != {1, 4}:
        raise ValidationError("Execution-integrity seed/worker matrix is incomplete")
    assert_no_duplicates(
        receipt,
        ["fault_mode", "policy_mode", "seed", "workers"],
        "execution-integrity receipt output",
    )
    mode_counts = receipt.groupby("fault_mode").size().to_dict()
    if mode_counts != {mode: 18 for mode in expected_receipt_modes}:
        raise ValidationError(f"Unexpected execution-integrity mode counts: {mode_counts}")
    assert_all_one(receipt["detected_as_expected"], "receipt detected_as_expected")

    clean = receipt.loc[receipt["fault_mode"].eq("clean")]
    faults = receipt.loc[~receipt["fault_mode"].eq("clean")]
    assert_all_zero(clean["faults_injected"], "clean receipt injected faults")
    assert_all_one(clean["receipt_validation_passed"], "clean receipt validation")
    anomaly_columns = [
        "missing_receipts",
        "unlogged_downstream_calls",
        "duplicate_downstream_calls",
        "mismatched_correlation_ids",
        "unauthorized_downstream_calls",
    ]
    for column in anomaly_columns:
        assert_all_zero(clean[column], f"clean receipt {column}")
    if not (numeric(faults["faults_injected"], "receipt injected faults") > 0).all():
        raise ValidationError("A receipt-fault execution injected zero events")
    assert_all_zero(faults["receipt_validation_passed"], "fault receipt validation")

    expected_injected = {
        "unlogged_downstream_call": 1184,
        "false_execution_log": 1166,
        "duplicate_downstream_call": 1238,
        "mismatched_correlation_id": 1252,
    }
    observed_injected = (
        receipt.groupby("fault_mode")["faults_injected"]
        .apply(lambda s: int(numeric(s, "receipt injected totals").sum()))
        .to_dict()
    )
    for mode, expected in expected_injected.items():
        if observed_injected.get(mode) != expected:
            raise ValidationError(
                f"Receipt mode {mode} injected total must equal {expected}; "
                f"found {observed_injected.get(mode)}"
            )

    mode_channel = {
        "unlogged_downstream_call": "unlogged_downstream_calls",
        "false_execution_log": "missing_receipts",
        "duplicate_downstream_call": "duplicate_downstream_calls",
        "mismatched_correlation_id": "mismatched_correlation_ids",
    }
    for mode, channel in mode_channel.items():
        selected = receipt.loc[receipt["fault_mode"].eq(mode)]
        observed = int(numeric(selected[channel], f"{mode} {channel}").sum())
        if observed != expected_injected[mode]:
            raise ValidationError(
                f"Receipt mode {mode} anomaly total differs from injected events"
            )
    mismatch_rows = receipt.loc[
        receipt["fault_mode"].eq("mismatched_correlation_id")
    ]
    if int(numeric(mismatch_rows["missing_receipts"], "mismatch missing receipts").sum()) != 1252:
        raise ValidationError(
            "Mismatched-correlation controls must also produce 1,252 missing matching receipts"
        )

    clean_receipts = int(numeric(clean["receipt_rows"], "clean receipt rows").sum())
    all_receipts = int(numeric(receipt["receipt_rows"], "all receipt rows").sum())
    if clean_receipts != 117_786 or all_receipts != 589_002:
        raise ValidationError(
            "Execution-integrity receipt totals differ from the frozen evidence: "
            f"clean={clean_receipts}, all={all_receipts}"
        )

    for column in [
        "action_hash",
        "record_trace_hash",
        "config_manifest_hash",
        "config_bound_trace_hash",
    ]:
        if not receipt[column].astype(str).str.fullmatch(r"[0-9a-fA-F]{64}").all():
            raise ValidationError(f"Execution-integrity {column} contains an invalid digest")

    clean_worker_action = clean.groupby(["policy_mode", "seed"])["action_hash"].nunique()
    clean_worker_record = clean.groupby(["policy_mode", "seed"])["record_trace_hash"].nunique()
    if clean_worker_action.ne(1).any() or clean_worker_record.ne(1).any():
        raise ValidationError("Clean execution-integrity worker reconstructions are unstable")

    receipt_summary = read_csv_required(
        receipt_summary_path, "execution-integrity corrected receipt summary"
    )
    require_columns(
        receipt_summary,
        [
            "fault_mode",
            "execution_instances",
            "correctly_classified_instances",
            "injected_events",
            "validation_passes",
            "anomaly_detected_instances",
            "false_positive_instances",
            "false_negative_instances",
        ],
        "execution-integrity corrected receipt summary",
    )
    if len(receipt_summary) != 5:
        raise ValidationError("Execution-integrity receipt summary must contain five rows")
    if not numeric(
        receipt_summary["execution_instances"], "receipt summary instances"
    ).eq(18).all():
        raise ValidationError("Each execution-integrity receipt mode must contain 18 instances")
    if not numeric(
        receipt_summary["correctly_classified_instances"],
        "receipt summary classifications",
    ).eq(18).all():
        raise ValidationError("Every receipt instance must be correctly classified")
    assert_all_zero(
        receipt_summary["false_positive_instances"], "receipt summary false positives"
    )
    assert_all_zero(
        receipt_summary["false_negative_instances"], "receipt summary false negatives"
    )

    record = read_csv_required(record_path, "record-bound corruption per-run output")
    require_columns(
        record,
        [
            "corruption_mode",
            "policy_mode",
            "seed",
            "workers",
            "detected",
            "clean_record_trace_hash",
            "corrupted_record_trace_hash",
            "clean_config_bound_trace_hash",
            "corrupted_config_bound_trace_hash",
        ],
        "record-bound corruption per-run output",
    )
    if len(record) != 90:
        raise ValidationError(
            f"Record-bound corruption output must contain 90 applications; found {len(record)}"
        )
    expected_record_modes = {
        "row_reordering",
        "replay_id_action_reassignment",
        "authorization_field_corruption",
        "execution_field_corruption",
        "configuration_label_corruption",
    }
    if set(record["corruption_mode"].astype(str)) != expected_record_modes:
        raise ValidationError("Record-bound corruption modes are incomplete")
    if record.groupby("corruption_mode").size().to_dict() != {
        mode: 18 for mode in expected_record_modes
    }:
        raise ValidationError("Every record-bound corruption mode must contain 18 applications")
    assert_all_one(record["detected"], "record-bound corruption detection")
    config_mismatch = record["clean_config_bound_trace_hash"].astype(str).ne(
        record["corrupted_config_bound_trace_hash"].astype(str)
    )
    if not config_mismatch.all():
        raise ValidationError("Every record/configuration corruption must change H_RC")
    non_config = record.loc[
        ~record["corruption_mode"].astype(str).eq("configuration_label_corruption")
    ]
    if not non_config["clean_record_trace_hash"].astype(str).ne(
        non_config["corrupted_record_trace_hash"].astype(str)
    ).all():
        raise ValidationError("A record-state corruption failed to change H_R")
    config_only = record.loc[
        record["corruption_mode"].astype(str).eq("configuration_label_corruption")
    ]
    if not config_only["clean_record_trace_hash"].astype(str).eq(
        config_only["corrupted_record_trace_hash"].astype(str)
    ).all():
        raise ValidationError("Configuration-label corruption unexpectedly changed H_R")

    record_summary = read_csv_required(
        record_summary_path, "record-bound corruption summary"
    )
    require_columns(
        record_summary,
        ["corruption_mode", "runs", "detected_runs"],
        "record-bound corruption summary",
    )
    if len(record_summary) != 5:
        raise ValidationError("Record-bound corruption summary must contain five rows")
    if not numeric(record_summary["runs"], "record summary runs").eq(18).all():
        raise ValidationError("Each record corruption summary row must contain 18 applications")
    if not numeric(
        record_summary["detected_runs"], "record summary detected runs"
    ).eq(18).all():
        raise ValidationError("Every record corruption application must be detected")

    manifest_obj = read_json_required(manifest_path, "execution-integrity manifest")
    if not isinstance(manifest_obj, dict):
        raise ValidationError("Execution-integrity manifest must be a JSON object")
    _require_manifest_int(manifest_obj, "decision_points", 11_351, "execution-integrity manifest")
    _require_manifest_int(manifest_obj, "conditions_per_mode", 18, "execution-integrity manifest")
    _require_manifest_true(
        manifest_obj,
        ["receipt_validation_all_expected", "record_corruption_all_detected"],
        "execution-integrity manifest",
    )

    results = {
        "receipt_execution_instances": 90,
        "clean_receipt_execution_instances": 18,
        "receipt_fault_execution_instances": 72,
        "clean_downstream_receipts_reconciled": clean_receipts,
        "semantic_receipt_digests_recomputed": all_receipts,
        "receipt_faults": {
            mode: {
                "execution_instances": 18,
                "injected_events": expected,
                "correctly_classified": 18,
            }
            for mode, expected in expected_injected.items()
        },
        "record_configuration_corruption_applications": 90,
        "record_configuration_corruption_applications_detected": 90,
        "clean_false_positive_executions": 0,
        "receipt_fault_false_negative_executions": 0,
        "all_expected_outcomes_met": True,
    }
    inventory = [
        evidence_file(project_dir, "execution_integrity_validation", "receipt_per_run", receipt_path),
        evidence_file(
            project_dir,
            "execution_integrity_validation",
            "receipt_summary",
            receipt_summary_path,
        ),
        evidence_file(project_dir, "execution_integrity_validation", "record_per_run", record_path),
        evidence_file(
            project_dir,
            "execution_integrity_validation",
            "record_summary",
            record_summary_path,
        ),
        evidence_file(project_dir, "execution_integrity_validation", "manifest", manifest_path),
    ]
    return results, inventory


def validate_validator_selectivity_validation(
    project_dir: Path,
) -> tuple[dict[str, Any], list[Any]]:
    names = [
        "validator_selectivity_metrics.json",
        "validator_selectivity_manifest.json",
        "runtime_fault_selectivity_summary.csv",
        "runtime_fault_selectivity_per_execution.csv",
        "runtime_fault_event_localization.csv",
        "record_integrity_selectivity_summary.csv",
        "record_integrity_selectivity_per_application.csv",
        "posthoc_trace_selectivity_summary.csv",
        "posthoc_trace_selectivity_per_application.csv",
        "clean_reference_executions.csv",
        "benign_negative_controls_summary.csv",
        "benign_negative_controls_per_execution.csv",
    ]
    base = _resolve_evidence_base(
        project_dir,
        [
            "paper_outputs/validator_selectivity_validation",
            "evidence/validator_selectivity_validation",
        ],
        names,
        "validator-selectivity",
    )
    paths = {name: base / name for name in names}

    metrics = read_json_required(paths[names[0]], "validator-selectivity metrics")
    manifest_obj = read_json_required(paths[names[1]], "validator-selectivity manifest")
    if not isinstance(metrics, dict) or not isinstance(manifest_obj, dict):
        raise ValidationError("Validator-selectivity metrics and manifest must be JSON objects")

    exact_counts = {
        "decision_points": 11_351,
        "independent_benign_negative_control_executions": 24,
        "independent_positive_fault_executions": 216,
        "run_level_true_negatives": 24,
        "run_level_true_positives": 216,
        "run_level_false_positives": 0,
        "run_level_false_negatives": 0,
        "event_level_true_positives": 3_240,
        "event_level_false_positives": 0,
        "event_level_false_negatives": 0,
        "posthoc_validator_applications": 258,
        "posthoc_validator_failures": 0,
        "unique_clean_reference_executions": 8,
    }
    for role, payload in [("metrics", metrics), ("manifest", manifest_obj)]:
        for key, expected in exact_counts.items():
            _require_manifest_int(payload, key, expected, f"validator-selectivity {role}")
    if manifest_obj.get("smoke_mode") is not False:
        raise ValidationError("Validator-selectivity final manifest must record smoke_mode=false")
    _require_manifest_true(
        manifest_obj,
        [
            "all_benign_controls_passed",
            "all_runtime_fault_executions_detected",
            "all_supported_events_localized",
            "all_posthoc_applications_detected",
        ],
        "validator-selectivity manifest",
    )
    for payload_name, payload in [("metrics", metrics), ("manifest", manifest_obj)]:
        for key in [
            "run_level_false_positive_rate",
            "run_level_false_negative_rate",
        ]:
            if float(payload.get(key, -1)) != 0.0:
                raise ValidationError(f"Validator-selectivity {payload_name} {key} must be 0")
        for key in ["event_level_precision", "event_level_recall"]:
            if float(payload.get(key, -1)) != 1.0:
                raise ValidationError(f"Validator-selectivity {payload_name} {key} must be 1")

    clean_refs = read_csv_required(
        paths["clean_reference_executions.csv"], "validator-selectivity clean references"
    )
    require_columns(
        clean_refs,
        [
            "experiment_instance_id",
            "policy_mode",
            "seed",
            "workers",
            "decision_points",
            "receipt_validation_passed",
            "anomaly_findings",
        ],
        "validator-selectivity clean references",
    )
    if len(clean_refs) != 8:
        raise ValidationError("Validator-selectivity must retain 8 unique clean references")
    if clean_refs["experiment_instance_id"].astype(str).duplicated().any():
        raise ValidationError("Validator-selectivity clean reference IDs are duplicated")
    assert_all_one(clean_refs["receipt_validation_passed"], "selectivity clean receipt status")
    assert_all_zero(clean_refs["anomaly_findings"], "selectivity clean anomalies")

    benign = read_csv_required(
        paths["benign_negative_controls_per_execution.csv"],
        "validator-selectivity benign controls",
    )
    require_columns(
        benign,
        [
            "experiment_instance_id",
            "benign_mode",
            "receipt_validation_passed",
            "anomaly_findings",
            "action_hash_preserved",
            "record_hash_preserved",
            "config_bound_hash_preserved",
            "false_positive",
            "correctly_classified",
        ],
        "validator-selectivity benign controls",
    )
    if len(benign) != 24:
        raise ValidationError("Validator-selectivity must contain 24 benign executions")
    if benign.groupby("benign_mode").size().to_dict() != {
        "timing_fields_changed": 6,
        "metadata_column_order_changed": 6,
        "permitted_logging_format_changed": 6,
        "completion_order_changed_then_reconstructed": 6,
    }:
        raise ValidationError("Validator-selectivity benign mode matrix is incomplete")
    for column in [
        "receipt_validation_passed",
        "action_hash_preserved",
        "record_hash_preserved",
        "config_bound_hash_preserved",
        "correctly_classified",
    ]:
        assert_all_one(benign[column], f"selectivity benign {column}")
    for column in ["anomaly_findings", "false_positive"]:
        assert_all_zero(benign[column], f"selectivity benign {column}")

    runtime = read_csv_required(
        paths["runtime_fault_selectivity_per_execution.csv"],
        "validator-selectivity runtime faults",
    )
    require_columns(
        runtime,
        [
            "experiment_instance_id",
            "fault_mode",
            "profile",
            "policy_mode",
            "seed",
            "workers",
            "injected_events",
            "validator_triggered",
            "event_true_positives",
            "event_false_positives",
            "event_false_negatives",
            "correctly_classified",
        ],
        "validator-selectivity runtime faults",
    )
    if len(runtime) != 216:
        raise ValidationError("Validator-selectivity must contain 216 runtime fault executions")
    expected_fault_modes = {
        "action_flip",
        "unauthorized_invocation",
        "unlogged_downstream_call",
        "false_execution_log",
        "duplicate_downstream_call",
        "mismatched_correlation_id",
    }
    expected_profiles = {
        "single_first",
        "single_last",
        "single_seeded",
        "rate_0_01_percent",
        "rate_0_1_percent",
        "rate_1_percent",
    }
    if set(runtime["fault_mode"].astype(str)) != expected_fault_modes:
        raise ValidationError("Validator-selectivity runtime fault modes are incomplete")
    if set(runtime["profile"].astype(str)) != expected_profiles:
        raise ValidationError("Validator-selectivity profiles are incomplete")
    if not runtime.groupby("fault_mode").size().eq(36).all():
        raise ValidationError("Each selectivity runtime fault mode must contain 36 executions")
    if not runtime.groupby("profile").size().eq(36).all():
        raise ValidationError("Each selectivity profile must contain 36 executions")
    if not (numeric(runtime["injected_events"], "selectivity injected events") > 0).all():
        raise ValidationError("A selectivity runtime execution injected zero events")
    assert_all_one(runtime["validator_triggered"], "selectivity runtime trigger")
    assert_all_one(runtime["correctly_classified"], "selectivity runtime classification")
    assert_all_zero(runtime["event_false_positives"], "selectivity event false positives")
    assert_all_zero(runtime["event_false_negatives"], "selectivity event false negatives")
    if int(numeric(runtime["injected_events"], "selectivity injected total").sum()) != 3_240:
        raise ValidationError("Selectivity runtime injected-event total must equal 3,240")
    if int(numeric(runtime["event_true_positives"], "selectivity event TPs").sum()) != 3_240:
        raise ValidationError("Selectivity runtime event true positives must equal 3,240")

    localization = read_csv_required(
        paths["runtime_fault_event_localization.csv"],
        "validator-selectivity event localization",
    )
    require_columns(
        localization,
        [
            "experiment_instance_id",
            "fault_mode",
            "profile",
            "replay_point_id",
            "was_injected",
            "was_localized",
        ],
        "validator-selectivity event localization",
    )
    if len(localization) != 3_240:
        raise ValidationError("Validator-selectivity localization must contain 3,240 rows")
    assert_all_one(localization["was_injected"], "selectivity localization injected")
    assert_all_one(localization["was_localized"], "selectivity localization found")

    posthoc = read_csv_required(
        paths["posthoc_trace_selectivity_per_application.csv"],
        "validator-selectivity post-hoc trace applications",
    )
    require_columns(
        posthoc,
        [
            "application_id",
            "fault_mode",
            "profile",
            "injected_events",
            "validator_triggered",
            "correctly_classified",
        ],
        "validator-selectivity post-hoc trace applications",
    )
    if len(posthoc) != 108:
        raise ValidationError("Selectivity must contain 108 post-hoc trace applications")
    assert_all_one(posthoc["validator_triggered"], "selectivity post-hoc trigger")
    assert_all_one(posthoc["correctly_classified"], "selectivity post-hoc classification")

    record = read_csv_required(
        paths["record_integrity_selectivity_per_application.csv"],
        "validator-selectivity record/configuration applications",
    )
    require_columns(
        record,
        [
            "application_id",
            "corruption_mode",
            "profile",
            "validator_triggered",
            "correctly_classified",
        ],
        "validator-selectivity record/configuration applications",
    )
    if len(record) != 150:
        raise ValidationError(
            "Selectivity must contain 150 record/configuration applications"
        )
    assert_all_one(record["validator_triggered"], "selectivity record trigger")
    assert_all_one(record["correctly_classified"], "selectivity record classification")
    if len(posthoc) + len(record) != 258:
        raise ValidationError("Selectivity post-execution applications must total 258")

    # Validate the compact summaries against their per-unit evidence.
    benign_summary = read_csv_required(
        paths["benign_negative_controls_summary.csv"], "selectivity benign summary"
    )
    if len(benign_summary) != 4:
        raise ValidationError("Selectivity benign summary must contain four rows")
    assert_all_zero(benign_summary["false_positive_executions"], "benign summary FPs")

    runtime_summary = read_csv_required(
        paths["runtime_fault_selectivity_summary.csv"], "selectivity runtime summary"
    )
    if len(runtime_summary) != 36:
        raise ValidationError("Selectivity runtime summary must contain 36 rows")
    if not numeric(
        runtime_summary["correctly_classified"], "runtime summary classifications"
    ).eq(6).all():
        raise ValidationError("Every runtime summary cell must classify 6/6 executions")

    posthoc_summary = read_csv_required(
        paths["posthoc_trace_selectivity_summary.csv"], "selectivity post-hoc summary"
    )
    if len(posthoc_summary) != 18:
        raise ValidationError("Selectivity post-hoc summary must contain 18 rows")
    if int(numeric(posthoc_summary["validator_applications"], "post-hoc summary apps").sum()) != 108:
        raise ValidationError("Post-hoc summary applications must total 108")
    if not numeric(
        posthoc_summary["correctly_classified"], "post-hoc summary classification"
    ).eq(numeric(posthoc_summary["validator_applications"], "post-hoc summary apps")).all():
        raise ValidationError("A post-hoc summary cell is not fully detected")

    record_summary = read_csv_required(
        paths["record_integrity_selectivity_summary.csv"], "selectivity record summary"
    )
    if len(record_summary) != 25:
        raise ValidationError("Selectivity record summary must contain 25 rows")
    if int(numeric(record_summary["validator_applications"], "record summary apps").sum()) != 150:
        raise ValidationError("Record/configuration summary applications must total 150")
    if not numeric(
        record_summary["correctly_classified"], "record summary classification"
    ).eq(numeric(record_summary["validator_applications"], "record summary apps")).all():
        raise ValidationError("A record/configuration summary cell is not fully detected")

    results = {
        "unique_clean_reference_executions": 8,
        "independent_benign_negative_control_executions": 24,
        "run_level_true_negatives": 24,
        "run_level_false_positives": 0,
        "independent_positive_fault_executions": 216,
        "run_level_true_positives": 216,
        "run_level_false_negatives": 0,
        "legacy_ground_truth_aware_runtime_events": 3_240,
        "legacy_event_accounting_matches_injection_manifest": True,
        "label_independent_localization_claim_superseded": True,
        "authoritative_label_independent_component": (
            "phase1_label_independent_validation"
        ),
        "posthoc_trace_applications": 108,
        "record_configuration_applications": 150,
        "posthoc_validator_applications": 258,
        "posthoc_validator_failures": 0,
        "all_expected_outcomes_met": True,
    }
    inventory = [
        evidence_file(project_dir, "validator_selectivity_validation", name, path)
        for name, path in paths.items()
    ]
    return results, inventory


def validate_phase1_label_independent_validation(
    project_dir: Path,
) -> tuple[dict[str, Any], list[Any]]:
    base = project_dir / "paper_outputs" / "phase1_label_independent_validation"
    names = [
        "generic_validator_findings.jsonl",
        "ground_truth_manifest.jsonl",
        "validator_input_separation_audit.csv",
        "per_evidence_scored_results.csv",
        "event_localization_results.csv",
        "baseline_comparison_by_fault_class.csv",
        "phase1_validation_manifest.json",
    ]
    paths = {name: base / name for name in names}
    for name, path in paths.items():
        if not path.is_file():
            raise ValidationError(f"Missing Phase-1 label-independent evidence: {path}")

    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValidationError(
                        f"Invalid JSONL in {path} at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValidationError(
                        f"JSONL row in {path} at line {line_number} is not an object"
                    )
                rows.append(value)
        return rows

    findings_rows = read_jsonl(paths["generic_validator_findings.jsonl"])
    truth_rows = read_jsonl(paths["ground_truth_manifest.jsonl"])
    if len(findings_rows) != 270 or len(truth_rows) != 270:
        raise ValidationError(
            "Phase-1 findings and ground truth must each contain 270 records"
        )
    finding_ids = [str(row.get("evidence_id", "")) for row in findings_rows]
    truth_ids = [str(row.get("evidence_id", "")) for row in truth_rows]
    if len(set(finding_ids)) != 270 or len(set(truth_ids)) != 270:
        raise ValidationError("Phase-1 evidence IDs must be unique")
    if set(finding_ids) != set(truth_ids):
        raise ValidationError("Phase-1 findings and ground truth do not align one-to-one")

    forbidden_finding_fields = {
        "fault_mode",
        "positive_control",
        "target_replay_point_ids",
        "target_count",
        "injection_seed",
        "target_indices_json",
        "target_indices_sha256",
        "expected_trigger",
        "was_injected",
        "receipt_fault_mode",
        "receipt_fault_injected",
        "action_flip_fault_injected",
        "unauthorized_invoke_fault_injected",
    }
    leaked = sorted(
        {
            field
            for row in findings_rows
            for field in forbidden_finding_fields.intersection(row.keys())
        }
    )
    if leaked:
        raise ValidationError(
            f"Generic validator findings leak ground-truth fields: {leaked}"
        )

    audit = read_csv_required(
        paths["validator_input_separation_audit.csv"],
        "Phase-1 validator-input separation audit",
    )
    require_columns(
        audit,
        [
            "evidence_id",
            "forbidden_columns_in_validator_input",
            "validator_input_label_independent",
        ],
        "Phase-1 validator-input separation audit",
    )
    if len(audit) != 270:
        raise ValidationError("Phase-1 input-separation audit must contain 270 rows")
    assert_all_one(
        audit["validator_input_label_independent"],
        "Phase-1 label-independent validator inputs",
    )
    forbidden_values = audit["forbidden_columns_in_validator_input"].fillna("").astype(str)
    if forbidden_values.str.strip().ne("").any():
        raise ValidationError("A Phase-1 validator input retained a forbidden field")

    scored = read_csv_required(
        paths["per_evidence_scored_results.csv"],
        "Phase-1 per-evidence scored results",
    )
    require_columns(
        scored,
        [
            "evidence_id",
            "fault_mode",
            "positive_control",
            "source_kind",
            "primary_validator_triggered",
            "full_validator_triggered",
            "primary_correct",
            "full_correct",
            "event_localization_supported",
            "event_true_positives",
            "event_false_positives",
            "event_false_negatives",
        ],
        "Phase-1 per-evidence scored results",
    )
    if len(scored) != 270:
        raise ValidationError("Phase-1 scored results must contain 270 evidence units")
    assert_no_duplicates(scored, ["evidence_id"], "Phase-1 scored results")
    positive = scored.loc[numeric(scored["positive_control"], "positive control").eq(1)]
    negative = scored.loc[numeric(scored["positive_control"], "positive control").eq(0)]
    if len(positive) != 228 or len(negative) != 42:
        raise ValidationError(
            f"Phase-1 expected 228 positive and 42 negative units; "
            f"found {len(positive)} and {len(negative)}"
        )
    if int(numeric(positive["full_validator_triggered"], "full positive detection").sum()) != 228:
        raise ValidationError("Full validator did not flag all 228 positive units")
    if int(numeric(negative["full_validator_triggered"], "full negative detection").sum()) != 0:
        raise ValidationError("Full validator flagged a clean or benign unit")
    if int(numeric(positive["primary_validator_triggered"], "primary positive detection").sum()) != 84:
        raise ValidationError("Primary validator detection total must equal 84/228")
    assert_all_one(positive["full_correct"], "Phase-1 full positive classifications")
    assert_all_one(negative["full_correct"], "Phase-1 full negative classifications")

    clean_units = scored["fault_mode"].astype(str).eq("clean")
    benign_units = scored["source_kind"].astype(str).eq("post_execution_benign_control")
    receipt_units = scored["source_kind"].astype(str).eq("frozen_receipt_execution")
    if int(clean_units.sum()) != 18 or int(benign_units.sum()) != 24:
        raise ValidationError("Phase-1 clean/benign accounting must equal 18 and 24")
    if int(receipt_units.sum()) != 90:
        raise ValidationError("Phase-1 receipt execution accounting must equal 90")

    localization = read_csv_required(
        paths["event_localization_results.csv"],
        "Phase-1 event-localization results",
    )
    require_columns(
        localization,
        [
            "evidence_id",
            "event_localization_supported",
            "event_true_positives",
            "event_false_positives",
            "event_false_negatives",
        ],
        "Phase-1 event-localization results",
    )
    if len(localization) != 270:
        raise ValidationError("Phase-1 localization output must contain 270 rows")
    supported = localization.loc[
        numeric(
            localization["event_localization_supported"],
            "localization support",
        ).eq(1)
    ]
    localized_events = int(
        numeric(supported["event_true_positives"], "localized true positives").sum()
    )
    localization_fp = int(
        numeric(supported["event_false_positives"], "localized false positives").sum()
    )
    localization_fn = int(
        numeric(supported["event_false_negatives"], "localized false negatives").sum()
    )
    if localized_events != 4_906 or localization_fp != 0 or localization_fn != 0:
        raise ValidationError(
            "Phase-1 label-independent localization must equal "
            "4,906 TP, 0 FP, and 0 FN"
        )

    comparison = read_csv_required(
        paths["baseline_comparison_by_fault_class.csv"],
        "Phase-1 baseline comparison",
    )
    require_columns(
        comparison,
        [
            "fault_mode",
            "evidence_units",
            "primary_detected",
            "full_detected",
        ],
        "Phase-1 baseline comparison",
    )
    expected_modes = {
        "clean",
        "saved_action_corruption",
        "dropped_rows",
        "duplicated_rows",
        "logged_unauthorized_invocation",
        "unlogged_downstream_call",
        "false_execution_log",
        "duplicate_downstream_call",
        "mismatched_correlation_id",
        "row_reordering",
        "replay_id_action_reassignment",
        "authorization_field_corruption",
        "execution_field_corruption",
        "configuration_label_corruption",
        "timing_fields_changed",
        "metadata_column_order_changed",
        "permitted_logging_format_changed",
        "completion_order_changed_then_reconstructed",
    }
    if set(comparison["fault_mode"].astype(str)) != expected_modes:
        raise ValidationError("Phase-1 comparison fault/control modes are incomplete")

    manifest = read_json_required(
        paths["phase1_validation_manifest.json"],
        "Phase-1 validation manifest",
    )
    expected_manifest_values = {
        "status": "passed",
        "generic_validator_findings": 270,
        "ground_truth_records": 270,
        "receipt_execution_instances": 90,
        "clean_reference_units": 18,
        "benign_control_units": 24,
        "negative_control_units": 42,
        "positive_control_units": 228,
        "primary_detected_positive_units": 84,
        "full_detected_positive_units": 228,
        "full_false_positive_units": 0,
        "label_independent_localized_events": 4_906,
        "localization_false_positives": 0,
        "localization_false_negatives": 0,
    }
    for key, expected in expected_manifest_values.items():
        if manifest.get(key) != expected:
            raise ValidationError(
                f"Phase-1 manifest {key} must equal {expected!r}; "
                f"found {manifest.get(key)!r}"
            )
    file_entries = manifest.get("files")
    if not isinstance(file_entries, dict):
        raise ValidationError("Phase-1 manifest files entry must be an object")
    for name, metadata in file_entries.items():
        path = base / name
        if not path.is_file():
            raise ValidationError(f"Phase-1 manifest references a missing file: {path}")
        if not isinstance(metadata, dict):
            raise ValidationError(f"Invalid Phase-1 file metadata for {name}")
        if int(metadata.get("bytes", -1)) != path.stat().st_size:
            raise ValidationError(f"Phase-1 byte count mismatch for {name}")
        if str(metadata.get("sha256", "")) != sha256_file(path):
            raise ValidationError(f"Phase-1 SHA-256 mismatch for {name}")

    results = {
        "schema_version": manifest.get("schema_version"),
        "status": "passed",
        "generic_validator_findings": 270,
        "ground_truth_records": 270,
        "receipt_execution_instances": 90,
        "clean_reference_units": 18,
        "benign_control_units": 24,
        "negative_control_units": 42,
        "positive_control_units": 228,
        "primary_detected_positive_units": 84,
        "full_detected_positive_units": 228,
        "full_false_positive_units": 0,
        "label_independent_localized_events": localized_events,
        "localization_false_positives": localization_fp,
        "localization_false_negatives": localization_fn,
        "ground_truth_fields_absent_from_findings": True,
        "validator_inputs_label_independent": True,
        "primary_invariants_recomputed": True,
        "findings_frozen_before_scoring": True,
    }
    inventory = [
        evidence_file(
            project_dir,
            "phase1_label_independent_validation",
            name,
            path,
        )
        for name, path in paths.items()
    ]
    return results, inventory


def make_claims_numbers(
    primary: dict[str, Any],
    timing: dict[str, Any],
    bc_live_decomposition: dict[str, Any],
    execution_integrity: dict[str, Any],
    validator_selectivity: dict[str, Any],
    phase1_label_independent: dict[str, Any],
    ray: dict[str, Any],
    primary_faults: dict[str, Any],
    metro: dict[str, Any],
    cloud: dict[str, Any],
    comment13: dict[str, Any],
    tests: dict[str, Any],
    compilation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "primary_benchmark": {
            "conditions_completed": primary["conditions"],
            "conditions_expected": 360,
            "full_workload_decision_points": primary["full_workload_decision_points"],
            "max_clean_authorization_contradictions": primary[
                "max_clean_unauthorized_invocations"
            ],
            "unique_hashes_by_policy": primary["unique_hashes_by_policy"],
            "full_workload_mean_intervention_rate_by_policy": primary[
                "full_workload_mean_intervention_rate_by_policy"
            ],
        },
        "timing_study": {
            "measured_rows": timing["measured_rows"],
            "configurations": timing["unique_configurations"],
            "configurations_x_7_repetitions": timing[
                "configurations_with_7_repetitions"
            ],
            "configurations_x_15_repetitions": timing[
                "configurations_with_15_repetitions"
            ],
            "paired_worker_speedup_rows": timing["worker_speedup_rows"],
        },
        "bc_live_runtime_decomposition": bc_live_decomposition,
        "execution_integrity_validation": execution_integrity,
        "validator_selectivity_validation": validator_selectivity,
        "phase1_label_independent_validation": phase1_label_independent,
        "ray_validation": {
            "conditions": ray["conditions"],
            "clean_conditions": ray["clean_conditions"],
            "clean_external_hash_matches": ray["clean_external_hash_matches"],
            "fault_conditions": ray["fault_conditions"],
            "fault_conditions_flagged": ray["fault_conditions_flagged"],
            "clean_false_positive_detections": ray[
                "clean_false_positive_detections"
            ],
            "max_authorization_contradictions": ray["max_unauthorized_invocations"],
        },
        "primary_controlled_faults": primary_faults,
        "metropt3": {
            "clean_conditions": metro["clean_conditions"],
            "max_clean_authorization_contradictions": metro[
                "max_clean_unauthorized_invocations"
            ],
            "unique_hashes_by_policy": metro["unique_hashes_by_policy"],
            "full_workload_mean_intervention_rate_by_policy": metro[
                "full_workload_mean_intervention_rate_by_policy"
            ],
            "provenance": metro["provenance"],
            "controlled_faults": metro["fault_validation"],
        },
        "cloud_validation": {
            "cross_region_compared": cloud["cross_region"]["compared"],
            "cross_region_matched": cloud["cross_region"]["matched"],
            "local_to_cloud_compared": cloud["local_to_cloud"]["compared"],
            "local_to_cloud_matched": cloud["local_to_cloud"]["matched"],
            "max_clean_authorization_contradictions": cloud[
                "max_clean_unauthorized_invocations"
            ],
        },
        "environment_comparison": comment13,
        "software_validation": {
            "python_files_compiled": compilation["python_files_compiled"],
            "all_python_files_compile": compilation["all_python_files_compile"],
            "tests_passed": tests["tests_passed"],
            "all_tests_passed": tests["all_tests_passed"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly validate every finalized ReplayBench-PG evidence component."
    )
    parser.add_argument(
        "--project-dir",
        default=str(SCRIPT_DIR),
        help="ReplayBench-PG repository root.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_outputs/final_validation",
        help="Directory for the four final validation outputs.",
    )
    parser.add_argument(
        "--cloud-root",
        default="cloud_results/cloud360_riskproxy_20260702",
        help="Root containing finalized regional and local-to-cloud outputs.",
    )
    parser.add_argument(
        "--cloud-cross-region-csv",
        default=None,
        help="Optional explicit finalized CSV/JSON for the 360 cross-region comparisons.",
    )
    parser.add_argument(
        "--cloud-local-to-cloud-csv",
        default=None,
        help="Optional explicit finalized CSV/JSON for the 720 local-to-cloud comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    output_dir = (project_dir / args.output_dir).resolve()
    cloud_root = (project_dir / args.cloud_root).resolve()
    cross_override = (
        (project_dir / args.cloud_cross_region_csv).resolve()
        if args.cloud_cross_region_csv
        else None
    )
    local_override = (
        (project_dir / args.cloud_local_to_cloud_csv).resolve()
        if args.cloud_local_to_cloud_csv
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "final_validation_manifest.json"
    inventory_path = output_dir / "final_results_inventory.csv"
    claims_path = output_dir / "final_claims_numbers.json"
    sha_path = output_dir / "file_sha256_manifest.csv"

    started_at = utc_now()
    component_results: dict[str, Any] = {}
    inventory_rows = []

    validators: list[tuple[str, Callable[[], Any]]] = [
        ("required_files", lambda: validate_required_files(
            project_dir,
            required_configs=[
                *DEFAULT_REQUIRED_CONFIGS,
                "configs/execution_integrity_validation.yaml",
                "configs/validator_selectivity_validation.yaml",
            ],
            required_manifests=DEFAULT_REQUIRED_MANIFESTS,
            required_scripts=[
                *DEFAULT_REQUIRED_SCRIPTS,
                "run_bc_live_runtime_decomposition.py",
                "run_execution_integrity_validation.py",
                "run_validator_selectivity_validation.py",
                "run_phase1_label_independent_validation.py",
                "score_phase1_label_independent_validation.py",
                "replaybench/generic_validator.py",
            ],
            cloud_root=cloud_root,
        )),
        ("code_quality", lambda: (validate_code_quality_fix(project_dir), [])),
        ("comment12_traceability", lambda: validate_comment12_artifacts(project_dir)),
        ("comment13_environment_comparison", lambda: validate_comment13_artifacts(project_dir)),
        ("primary_benchmark", lambda: validate_primary_benchmark(project_dir)),
        ("timing_study", lambda: validate_timing_study(project_dir)),
        (
            "bc_live_runtime_decomposition",
            lambda: validate_bc_live_runtime_decomposition(project_dir),
        ),
        ("ray_validation", lambda: validate_ray(project_dir)),
        (
            "execution_integrity_validation",
            lambda: validate_execution_integrity_validation(project_dir),
        ),
        (
            "validator_selectivity_validation",
            lambda: validate_validator_selectivity_validation(project_dir),
        ),
        (
            "phase1_label_independent_validation",
            lambda: validate_phase1_label_independent_validation(project_dir),
        ),
        ("primary_controlled_faults", lambda: validate_fault_summary(
            project_dir,
            project_dir
            / "paper_outputs"
            / "fgcs_tables_figures"
            / "fgcs_table_rq7_fault_detection_combined.csv",
            component="primary_faults",
            expected_clean_validator_applications=(
                FAULT_CLEAN_VALIDATOR_APPLICATIONS
            ),
            expected_unique_clean_references=FAULT_UNIQUE_CLEAN_REFERENCES,
            expected_validator_workflows=FAULT_VALIDATOR_WORKFLOWS,
            expected_runs_per_fault_class=FAULT_RUNS_PER_CLASS,
        )),
        ("metropt3", lambda: validate_metropt3(project_dir)),
        ("cloud_validation", lambda: discover_cloud_evidence(
            project_dir,
            cloud_root,
            cross_region_override=cross_override,
            local_to_cloud_override=local_override,
        )),
    ]

    current_component = "initialization"
    try:
        for current_component, validator in validators:
            print(f"[CHECK] {current_component}")
            returned = validator()
            # validate_fault_summary additionally returns a normalized DataFrame.
            if isinstance(returned, tuple) and len(returned) == 3:
                result, component_inventory, _ = returned
            else:
                result, component_inventory = returned
            component_results[current_component] = result
            inventory_rows.extend(component_inventory)
            print(f"[PASS] {current_component}")

        current_component = "python_compilation"
        print("[CHECK] python_compilation")
        compilation = compile_all_python(project_dir)
        component_results[current_component] = compilation
        print(f"[PASS] python_compilation ({compilation['python_files_compiled']} files)")

        current_component = "pytest"
        print("[CHECK] pytest")
        tests = run_all_tests(project_dir)
        component_results[current_component] = tests
        print(f"[PASS] pytest ({tests['tests_passed']} passed)")

        primary_faults = component_results["primary_controlled_faults"]
        claims = make_claims_numbers(
            primary=component_results["primary_benchmark"],
            timing=component_results["timing_study"],
            bc_live_decomposition=component_results[
                "bc_live_runtime_decomposition"
            ],
            execution_integrity=component_results[
                "execution_integrity_validation"
            ],
            validator_selectivity=component_results[
                "validator_selectivity_validation"
            ],
            phase1_label_independent=component_results[
                "phase1_label_independent_validation"
            ],
            ray=component_results["ray_validation"],
            primary_faults=primary_faults,
            metro=component_results["metropt3"],
            cloud=component_results["cloud_validation"],
            comment13=component_results["comment13_environment_comparison"],
            tests=tests,
            compilation=compilation,
        )

        # Add the code-quality target to the artifact inventory.
        inventory_rows.append(
            evidence_file(
                project_dir,
                "code_quality",
                "fault_validation_framework",
                project_dir / "fgcs_fault_validation_framework.py",
            )
        )

        # Deduplicate file inventory by path and role.
        inventory_frame = pd.DataFrame([row.to_dict() for row in inventory_rows])
        if inventory_frame.empty:
            raise ValidationError("Final evidence inventory is unexpectedly empty")
        inventory_frame = (
            inventory_frame.drop_duplicates(["component", "role", "path"], keep="last")
            .sort_values(["component", "role", "path"])
            .reset_index(drop=True)
        )
        atomic_write_csv(inventory_path, inventory_frame)
        atomic_write_json(claims_path, claims)

        passed_manifest = {
            "schema_version": "1.1",
            "status": "passed",
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "project_dir": project_dir.as_posix(),
            "output_dir": relative_posix(output_dir, project_dir),
            "strict_requirements": {
                "primary_benchmark_conditions": 360,
                "timing_rows": 528,
                "timing_configurations_x_7": 24,
                "timing_configurations_x_15": 24,
                "bc_live_decomposition_rows": 88,
                "bc_live_decomposition_configurations": 8,
                "execution_integrity_clean_instances": 18,
                "execution_integrity_receipt_fault_instances": 72,
                "execution_integrity_record_config_applications": 90,
                "validator_selectivity_benign_executions": 24,
                "validator_selectivity_runtime_fault_executions": 216,
                "legacy_validator_selectivity_ground_truth_aware_events": 3240,
                "validator_selectivity_posthoc_applications": 258,
                "phase1_generic_validator_findings": 270,
                "phase1_positive_control_units": 228,
                "phase1_negative_control_units": 42,
                "phase1_primary_detected_positive_units": 84,
                "phase1_full_detected_positive_units": 228,
                "phase1_label_independent_localized_events": 4906,
                "phase1_localization_false_positives": 0,
                "phase1_localization_false_negatives": 0,
                "ray_conditions": 54,
                "ray_clean_external_hash_matches": "18/18",
                "ray_fault_conditions_flagged": "36/36",
                "metropt3_clean_conditions": 72,
                "metropt3_canonical_source_rows": 1_516_948,
                "metropt3_prepared_replay_rows": 20_000,
                "metropt3_unique_full_source_rows": 1_516_948,
                "metropt3_duplicate_full_source_rows": 0,
                "primary_fault_classes": "5 classes, 18/18 each",
                "metropt3_fault_classes": "5 classes, 18/18 each",
                "fault_unique_clean_reference_executions": (
                    FAULT_UNIQUE_CLEAN_REFERENCES
                ),
                "fault_clean_validator_workflows": FAULT_VALIDATOR_WORKFLOWS,
                "fault_clean_validator_applications": (
                    FAULT_CLEAN_VALIDATOR_APPLICATIONS
                ),
                "cloud_cross_region_matches": "360/360",
                "cloud_local_to_cloud_matches": "720/720",
                "comment13_environment_comparison": True,
                "clean_authorization_contradictions": 0,
                "all_python_files_compile": True,
                "all_tests_pass": True,
            },
            "components": component_results,
            "outputs": {
                "manifest": relative_posix(manifest_path, project_dir),
                "inventory": relative_posix(inventory_path, project_dir),
                "claims_numbers": relative_posix(claims_path, project_dir),
                "sha256_manifest": relative_posix(sha_path, project_dir),
            },
        }
        atomic_write_json(manifest_path, passed_manifest)

        # Hash every validated evidence file, every required/configuration file,
        # every repository Python file, and the three already-written final files.
        hash_paths = {
            project_dir / path
            for path in inventory_frame.loc[inventory_frame["exists"].eq(True), "path"].astype(str)
        }
        hash_paths.update(discover_python_files(project_dir))
        hash_paths.update({manifest_path, inventory_path, claims_path})
        final_table_dir = project_dir / "paper_outputs" / "final_manuscript_tables"
        if final_table_dir.is_dir():
            hash_paths.update(final_table_dir.glob("*.csv"))
            hash_paths.update(final_table_dir.glob("*.json"))

        sha_frame = build_sha256_manifest(
            project_dir,
            hash_paths,
            exclude=[sha_path],
        )
        if sha_frame.empty:
            raise ValidationError("SHA-256 manifest would be empty")
        atomic_write_csv(sha_path, sha_frame)

        print("[DONE] Final ReplayBench-PG validation passed")
        print(f"[OUT] {manifest_path}")
        print(f"[OUT] {inventory_path}")
        print(f"[OUT] {claims_path}")
        print(f"[OUT] {sha_path}")

    except Exception as exc:
        failed_manifest = {
            "schema_version": "1.1",
            "status": "failed",
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "failed_component": current_component,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "validated_components_before_failure": component_results,
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(manifest_path, failed_manifest)
        print(f"[FAIL] {current_component}: {exc}", file=sys.stderr)
        print(f"[OUT] Failure manifest: {manifest_path}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
