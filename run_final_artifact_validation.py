#!/usr/bin/env python3
"""Master final validator for the frozen ReplayBench-PG evidence package.

The validator does not rerun experiments. It validates the completed primary,
timing, Ray, MetroPT-3, fault, and cloud outputs; compiles every repository
Python file; runs the complete pytest suite; and emits a machine-readable final
validation package.

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

from final_artifact_utils import (
    DEFAULT_REQUIRED_CONFIGS,
    DEFAULT_REQUIRED_MANIFESTS,
    DEFAULT_REQUIRED_SCRIPTS,
    ValidationError,
    atomic_write_csv,
    atomic_write_json,
    build_sha256_manifest,
    discover_cloud_evidence,
    discover_python_files,
    evidence_file,
    json_safe,
    relative_posix,
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


def make_claims_numbers(
    primary: dict[str, Any],
    timing: dict[str, Any],
    ray: dict[str, Any],
    primary_faults: dict[str, Any],
    metro: dict[str, Any],
    cloud: dict[str, Any],
    tests: dict[str, Any],
    compilation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
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
            required_configs=DEFAULT_REQUIRED_CONFIGS,
            required_manifests=DEFAULT_REQUIRED_MANIFESTS,
            required_scripts=DEFAULT_REQUIRED_SCRIPTS,
            cloud_root=cloud_root,
        )),
        ("code_quality", lambda: (validate_code_quality_fix(project_dir), [])),
        ("primary_benchmark", lambda: validate_primary_benchmark(project_dir)),
        ("timing_study", lambda: validate_timing_study(project_dir)),
        ("ray_validation", lambda: validate_ray(project_dir)),
        ("primary_controlled_faults", lambda: validate_fault_summary(
            project_dir,
            project_dir
            / "paper_outputs"
            / "fgcs_tables_figures"
            / "fgcs_table_rq7_fault_detection_combined.csv",
            component="primary_faults",
            expected_clean_instances=54,
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
            ray=component_results["ray_validation"],
            primary_faults=primary_faults,
            metro=component_results["metropt3"],
            cloud=component_results["cloud_validation"],
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
            "schema_version": "1.0",
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
                "ray_conditions": 54,
                "ray_clean_external_hash_matches": "18/18",
                "ray_fault_conditions_flagged": "36/36",
                "metropt3_clean_conditions": 72,
                "primary_fault_classes": "5 classes, 18/18 each",
                "metropt3_fault_classes": "5 classes, 18/18 each",
                "cloud_cross_region_matches": "360/360",
                "cloud_local_to_cloud_matches": "720/720",
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
            "schema_version": "1.0",
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
