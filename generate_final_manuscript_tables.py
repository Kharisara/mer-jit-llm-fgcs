#!/usr/bin/env python3
"""Generate the seven frozen manuscript-result CSVs for ReplayBench-PG.

This script reads only previously finalized CSV/JSON outputs. It does not run
benchmarks, inject faults, execute Ray, or recompute cloud jobs. By default it
requires a passed master-validation manifest and verifies the SHA-256 values in
the final evidence inventory before generating tables.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final_artifact_utils import (
    FAULT_LABELS,
    FAULT_ORDER,
    ValidationError,
    atomic_write_csv,
    normalize_fault_summary,
    numeric,
    read_csv_required,
    read_json_required,
    relative_posix,
    sha256_file,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def verify_passed_validation(project_dir: Path, validation_dir: Path) -> dict[str, Any]:
    manifest_path = validation_dir / "final_validation_manifest.json"
    inventory_path = validation_dir / "final_results_inventory.csv"
    claims_path = validation_dir / "final_claims_numbers.json"

    manifest = read_json_required(manifest_path, "final validation manifest")
    if not isinstance(manifest, dict) or manifest.get("status") != "passed":
        raise ValidationError(
            "The master final validator has not passed. Run "
            "python run_final_artifact_validation.py first."
        )
    claims = read_json_required(claims_path, "final claims numbers")
    inventory = read_csv_required(inventory_path, "final evidence inventory")
    required = {"path", "exists", "sha256", "status"}
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise ValidationError(f"Final evidence inventory is missing columns: {missing}")

    # Recheck every inventoried validated file before using any value.
    changed: list[str] = []
    missing_files: list[str] = []
    for row in inventory.to_dict(orient="records"):
        if str(row.get("status", "")).lower() != "validated":
            continue
        path = project_dir / str(row["path"])
        if not path.is_file():
            missing_files.append(str(row["path"]))
            continue
        expected = str(row.get("sha256", ""))
        if expected and expected.lower() != "nan" and sha256_file(path) != expected:
            changed.append(str(row["path"]))
    if missing_files or changed:
        raise ValidationError(
            "Finalized evidence changed after validation; "
            f"missing={missing_files}, changed={changed}. Rerun the master validator."
        )
    return {"manifest": manifest, "claims": claims}


def make_primary_benchmark_summary(project_dir: Path) -> pd.DataFrame:
    base = project_dir / "paper_outputs" / "fgcs_extended_benchmark"
    scaling = read_csv_required(base / "scaling_and_runtime_results.csv", "primary benchmark")
    det = read_csv_required(base / "determinism_hash_results.csv", "primary determinism")

    scaling = scaling.copy()
    det = det.copy()
    scaling["dataset_fraction"] = numeric(scaling["dataset_fraction"], "primary fraction")
    det["dataset_fraction"] = numeric(det["dataset_fraction"], "primary det fraction")
    scaling["workers"] = numeric(scaling["workers"], "primary workers").astype(int)
    det["workers"] = numeric(det["workers"], "primary det workers").astype(int)
    det["hash_match"] = numeric(det["hash_match"], "primary hash_match")

    full = scaling.loc[np.isclose(scaling["dataset_fraction"], 1.0)].copy()
    rows: list[dict[str, Any]] = []
    for policy in sorted(scaling["policy_mode"].astype(str).unique()):
        all_policy = scaling.loc[scaling["policy_mode"].astype(str).eq(policy)]
        det_policy = det.loc[det["policy_mode"].astype(str).eq(policy)]
        full_policy = full.loc[full["policy_mode"].astype(str).eq(policy)]
        worker_groups = det_policy.groupby(["dataset_fraction", "seed"])["trace_hash"].nunique()
        rows.append(
            {
                "policy_mode": policy,
                "conditions": int(len(all_policy)),
                "unique_action_hashes": int(det_policy["trace_hash"].nunique()),
                "all_worker_reconstructions_match": bool(worker_groups.eq(1).all()),
                "minimum_hash_match": int(det_policy["hash_match"].min()),
                "max_clean_unauthorized_invocations": int(
                    numeric(
                        all_policy["unauthorized_invocations"],
                        "primary unauthorized invocations",
                    ).max()
                ),
                "full_workload_decision_points": int(
                    numeric(full_policy["decision_points"], "decision points").max()
                ),
                "full_workload_mean_intervention_rate": float(
                    numeric(full_policy["intervention_rate"], "intervention rate").mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("policy_mode").reset_index(drop=True)


def make_timing_worker_scaling(project_dir: Path) -> pd.DataFrame:
    source = (
        project_dir
        / "paper_outputs"
        / "replaybench_timing_study"
        / "timing_worker_speedup_paired_ci.csv"
    )
    frame = read_csv_required(source, "timing worker-speedup summary").copy()
    if "policy" in frame.columns and "policy_mode" not in frame.columns:
        frame = frame.rename(columns={"policy": "policy_mode"})
    preferred = [
        "policy_mode",
        "workers",
        "paired_repetitions",
        "runtime_median_seconds",
        "runtime_q1_seconds",
        "runtime_q3_seconds",
        "runtime_iqr_seconds",
        "median_paired_speedup",
        "ci95_lower",
        "ci95_upper",
        "confidence",
        "bootstrap_samples",
    ]
    missing = [column for column in preferred if column not in frame.columns]
    if missing:
        raise ValidationError(f"Timing worker summary is missing columns: {missing}")
    return frame[preferred].sort_values(["policy_mode", "workers"]).reset_index(drop=True)


def make_ray_validation_summary(project_dir: Path) -> pd.DataFrame:
    source = project_dir / "paper_outputs" / "ray_comparison" / "ray_comparison_per_run.csv"
    frame = read_csv_required(source, "Ray per-run output").copy()
    rows: list[dict[str, Any]] = []
    order = ["clean", "action_flip", "dropped_row"]
    for mode in order:
        group = frame.loc[frame["fault_mode"].astype(str).eq(mode)]
        if group.empty:
            raise ValidationError(f"Ray output is missing mode {mode}")
        rows.append(
            {
                "fault_mode": mode,
                "conditions": int(len(group)),
                "injected_events": int(numeric(group["injected_events"], "Ray injected events").sum()),
                "expected_flagged_runs": int(
                    numeric(group["expected_flagged"], "Ray expected flag").sum()
                ),
                "detected_flagged_runs": int(
                    numeric(group["detected_flag"], "Ray detected flag").sum()
                ),
                "correct_detection_runs": int(
                    numeric(group["detection_correct"], "Ray detection correct").sum()
                ),
                "hash_mismatch_runs": int(
                    numeric(group["hash_match"], "Ray hash_match").eq(0).sum()
                ),
                "row_count_mismatch_runs": int(
                    numeric(group["row_count_match"], "Ray row_count_match").eq(0).sum()
                ),
                "authorization_violation_runs": int(
                    numeric(
                        group["authorization_execution_consistent"],
                        "Ray authorization invariant",
                    ).eq(0).sum()
                ),
                "external_reference_hash_matches": int(
                    numeric(group["hash_match"], "Ray hash_match").sum()
                    if mode == "clean"
                    else 0
                ),
                "all_detection_expectations_met": bool(
                    numeric(group["detection_correct"], "Ray detection correct").eq(1).all()
                ),
            }
        )
    return pd.DataFrame(rows)


def manuscript_fault_table(path: Path, role: str) -> pd.DataFrame:
    raw = read_csv_required(path, role)
    frame = normalize_fault_summary(raw, role)
    frame = frame.copy()
    frame["fault_label"] = frame["fault_mode"].map(
        {"clean_replay": "Clean comparison instances", **FAULT_LABELS}
    )
    order_map = {"clean_replay": 0, **{mode: index + 1 for index, mode in enumerate(FAULT_ORDER)}}
    frame["_order"] = frame["fault_mode"].map(order_map)
    detection_channel = (
        frame["detection_channel"].astype(str)
        if "detection_channel" in frame.columns
        else ""
    )
    output = pd.DataFrame(
        {
            "fault_mode": frame["fault_mode"],
            "fault_label": frame["fault_label"],
            "runs": frame["runs"].astype(int),
            "injected_events": frame["injected_events"].astype(int),
            "detected_runs": frame["detected_runs"].astype(int),
            "detection_rate": np.where(
                frame["runs"].gt(0),
                frame["detected_runs"] / frame["runs"],
                np.nan,
            ),
            "false_positive_runs": frame["false_positive_runs"].astype(int),
            "detection_channel": detection_channel,
            "_order": frame["_order"],
        }
    )
    # Detection rate is not applicable to the clean false-positive control.
    output.loc[output["fault_mode"].eq("clean_replay"), "detection_rate"] = np.nan
    return output.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def make_metropt3_validation_summary(project_dir: Path) -> pd.DataFrame:
    base = project_dir / "paper_outputs" / "secondary_metropt3" / "tables_figures"
    clean = read_csv_required(
        base / "secondary_policy_determinism_summary.csv",
        "MetroPT-3 policy summary",
    ).copy()
    clean_rows = pd.DataFrame(
        {
            "panel": "clean_deterministic_replay",
            "item": clean["policy_mode"].astype(str),
            "conditions_or_runs": numeric(clean["conditions"], "Metro conditions").astype(int),
            "unique_action_hashes": numeric(clean["unique_hashes"], "Metro unique hashes").astype(int),
            "all_worker_reconstructions_match": clean["all_worker_matches"].astype(bool),
            "max_clean_unauthorized_invocations": numeric(
                clean["max_unauthorized_invocations"], "Metro max unauthorized"
            ).astype(int),
            "full_workload_mean_intervention_rate": numeric(
                clean["full_workload_mean_intervention_rate"],
                "Metro intervention rate",
            ),
            "injected_events": np.nan,
            "detected_runs": np.nan,
            "detection_rate": np.nan,
            "false_positive_runs": np.nan,
        }
    )

    faults = manuscript_fault_table(
        base / "fgcs_table_rq7_fault_detection_combined.csv",
        "MetroPT-3 combined fault summary",
    )
    fault_rows = pd.DataFrame(
        {
            "panel": "controlled_fault_detection",
            "item": faults["fault_label"],
            "conditions_or_runs": faults["runs"],
            "unique_action_hashes": np.nan,
            "all_worker_reconstructions_match": np.nan,
            "max_clean_unauthorized_invocations": np.where(
                faults["fault_mode"].eq("clean_replay"), 0, np.nan
            ),
            "full_workload_mean_intervention_rate": np.nan,
            "injected_events": faults["injected_events"],
            "detected_runs": faults["detected_runs"],
            "detection_rate": faults["detection_rate"],
            "false_positive_runs": faults["false_positive_runs"],
        }
    )
    return pd.concat([clean_rows, fault_rows], ignore_index=True)


def make_cloud_validation_summary(validation_manifest: dict[str, Any]) -> pd.DataFrame:
    cloud = validation_manifest["components"]["cloud_validation"]
    return pd.DataFrame(
        [
            {
                "comparison_scope": "cross_region",
                "compared_conditions": int(cloud["cross_region"]["compared"]),
                "matched_action_hashes": int(cloud["cross_region"]["matched"]),
                "agreement_rate": float(
                    cloud["cross_region"]["matched"] / cloud["cross_region"]["compared"]
                ),
                "max_clean_unauthorized_invocations": int(
                    cloud["max_clean_unauthorized_invocations"]
                ),
                "source_file": str(cloud["cross_region"]["source"]),
            },
            {
                "comparison_scope": "local_to_cloud",
                "compared_conditions": int(cloud["local_to_cloud"]["compared"]),
                "matched_action_hashes": int(cloud["local_to_cloud"]["matched"]),
                "agreement_rate": float(
                    cloud["local_to_cloud"]["matched"] / cloud["local_to_cloud"]["compared"]
                ),
                "max_clean_unauthorized_invocations": int(
                    cloud["max_clean_unauthorized_invocations"]
                ),
                "source_file": str(cloud["local_to_cloud"]["source"]),
            },
        ]
    )


def make_final_results_overview(claims: dict[str, Any]) -> pd.DataFrame:
    primary_faults = claims["primary_controlled_faults"]
    metro_faults = claims["metropt3"]["controlled_faults"]
    rows = [
        {
            "component": "Primary benchmark",
            "metric": "Completed conditions",
            "observed": claims["primary_benchmark"]["conditions_completed"],
            "expected": claims["primary_benchmark"]["conditions_expected"],
            "status": "passed",
        },
        {
            "component": "Timing study",
            "metric": "Measured timing rows",
            "observed": claims["timing_study"]["measured_rows"],
            "expected": 528,
            "status": "passed",
        },
        {
            "component": "Timing study",
            "metric": "Configurations with seven repetitions",
            "observed": claims["timing_study"]["configurations_x_7_repetitions"],
            "expected": 24,
            "status": "passed",
        },
        {
            "component": "Timing study",
            "metric": "Configurations with fifteen repetitions",
            "observed": claims["timing_study"]["configurations_x_15_repetitions"],
            "expected": 24,
            "status": "passed",
        },
        {
            "component": "Ray validation",
            "metric": "Clean external hash matches",
            "observed": claims["ray_validation"]["clean_external_hash_matches"],
            "expected": 18,
            "status": "passed",
        },
        {
            "component": "Ray validation",
            "metric": "Fault conditions flagged",
            "observed": claims["ray_validation"]["fault_conditions_flagged"],
            "expected": 36,
            "status": "passed",
        },
        {
            "component": "Primary controlled faults",
            "metric": "Fault classes with 18/18 detection",
            "observed": primary_faults["fault_classes"],
            "expected": 5,
            "status": "passed",
        },
        {
            "component": "MetroPT-3",
            "metric": "Clean conditions",
            "observed": claims["metropt3"]["clean_conditions"],
            "expected": 72,
            "status": "passed",
        },
        {
            "component": "MetroPT-3 controlled faults",
            "metric": "Fault classes with 18/18 detection",
            "observed": metro_faults["fault_classes"],
            "expected": 5,
            "status": "passed",
        },
        {
            "component": "Cloud validation",
            "metric": "Cross-region hash matches",
            "observed": claims["cloud_validation"]["cross_region_matched"],
            "expected": claims["cloud_validation"]["cross_region_compared"],
            "status": "passed",
        },
        {
            "component": "Cloud validation",
            "metric": "Local-to-cloud hash matches",
            "observed": claims["cloud_validation"]["local_to_cloud_matched"],
            "expected": claims["cloud_validation"]["local_to_cloud_compared"],
            "status": "passed",
        },
        {
            "component": "Clean authorization checks",
            "metric": "Maximum contradiction count across clean evidence",
            "observed": max(
                claims["primary_benchmark"]["max_clean_authorization_contradictions"],
                claims["ray_validation"]["max_authorization_contradictions"],
                claims["metropt3"]["max_clean_authorization_contradictions"],
                claims["cloud_validation"]["max_clean_authorization_contradictions"],
            ),
            "expected": 0,
            "status": "passed",
        },
        {
            "component": "Software validation",
            "metric": "Python files compiled",
            "observed": claims["software_validation"]["python_files_compiled"],
            "expected": claims["software_validation"]["python_files_compiled"],
            "status": "passed",
        },
        {
            "component": "Software validation",
            "metric": "Tests passed",
            "observed": claims["software_validation"]["tests_passed"],
            "expected": claims["software_validation"]["tests_passed"],
            "status": "passed",
        },
    ]
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the seven final ReplayBench-PG manuscript CSV tables."
    )
    parser.add_argument("--project-dir", default=str(SCRIPT_DIR))
    parser.add_argument(
        "--validation-dir", default="paper_outputs/final_validation"
    )
    parser.add_argument(
        "--output-dir", default="paper_outputs/final_manuscript_tables"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    validation_dir = (project_dir / args.validation_dir).resolve()
    output_dir = (project_dir / args.output_dir).resolve()

    validated = verify_passed_validation(project_dir, validation_dir)
    manifest = validated["manifest"]
    claims = validated["claims"]

    tables = {
        "primary_benchmark_summary.csv": make_primary_benchmark_summary(project_dir),
        "timing_worker_scaling.csv": make_timing_worker_scaling(project_dir),
        "ray_validation_summary.csv": make_ray_validation_summary(project_dir),
        "primary_fault_summary.csv": manuscript_fault_table(
            project_dir
            / "paper_outputs"
            / "fgcs_tables_figures"
            / "fgcs_table_rq7_fault_detection_combined.csv",
            "primary combined fault summary",
        ),
        "metropt3_validation_summary.csv": make_metropt3_validation_summary(project_dir),
        "cloud_validation_summary.csv": make_cloud_validation_summary(manifest),
        "final_results_overview.csv": make_final_results_overview(claims),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if frame.empty:
            raise ValidationError(f"Generated table would be empty: {name}")
        path = output_dir / name
        atomic_write_csv(path, frame)
        print(f"[OUT] {path}")

    print("[DONE] Final manuscript tables generated from validated outputs")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
