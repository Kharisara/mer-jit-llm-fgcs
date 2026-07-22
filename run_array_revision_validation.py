#!/usr/bin/env python3
"""Orchestrate and validate the Array revision experiments.

The wrapper can execute the timing and Ray experiments, or validate already
completed outputs. It fails fast when the mixed 7/15-repetition timing design, external
Ray references, condition counts, clean outcomes, or controlled-fault outcomes
do not match the final revision protocol.
"""

from __future__ import annotations

import argparse
import json
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from summarize_replaybench_timing_study import (
    normalize_timing_frame,
    validate_timing_design,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def run_checked(command: list[str], cwd: Path) -> None:
    print("[CMD] " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def compile_scripts(project_dir: Path) -> list[str]:
    names = [
        "run_fgcs_extended_benchmark.py",
        "run_replaybench_timing_study.py",
        "summarize_replaybench_timing_study.py",
        "run_ray_comparison.py",
        "run_array_revision_validation.py",
    ]
    compiled: list[str] = []
    for name in names:
        path = project_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Required script not found: {path}")
        py_compile.compile(str(path), doraise=True)
        compiled.append(str(path))
    return compiled


def validate_ray_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Ray per-run output not found: {path}")

    frame = pd.read_csv(path)
    required = {
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
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Ray output is missing columns: {missing}")

    if len(frame) != 54:
        raise ValueError(
            f"Ray comparison must contain 54 conditions; found {len(frame)}"
        )

    expected_policies = {"risk_proxy", "random", "never"}
    expected_seeds = {1, 2, 3}
    expected_workers = {1, 4}
    expected_faults = {"clean", "action_flip", "dropped_row"}

    if set(frame["policy_mode"].astype(str)) != expected_policies:
        raise ValueError("Unexpected Ray policy set")
    if set(pd.to_numeric(frame["seed"]).astype(int)) != expected_seeds:
        raise ValueError("Unexpected Ray seed set")
    if set(pd.to_numeric(frame["workers"]).astype(int)) != expected_workers:
        raise ValueError("Unexpected Ray worker set")
    if set(frame["fault_mode"].astype(str)) != expected_faults:
        raise ValueError("Unexpected Ray fault-mode set")

    duplicate_mask = frame.duplicated(
        ["policy_mode", "seed", "workers", "fault_mode"], keep=False
    )
    if duplicate_mask.any():
        raise ValueError("Ray output contains duplicate conditions")

    counts = frame.groupby("fault_mode").size().to_dict()
    if counts != {"action_flip": 18, "clean": 18, "dropped_row": 18}:
        raise ValueError(f"Unexpected Ray condition counts: {counts}")

    if not frame["reference_source"].eq(
        "replaybench_determinism_csv"
    ).all():
        raise ValueError(
            "Every Ray condition must use an external ReplayBench-PG hash"
        )

    clean = frame.loc[frame["fault_mode"] == "clean"]
    if not pd.to_numeric(clean["hash_match"]).eq(1).all():
        raise ValueError("One or more clean Ray hashes do not match")
    if not pd.to_numeric(clean["row_count_match"]).eq(1).all():
        raise ValueError("One or more clean Ray row counts do not match")
    if not pd.to_numeric(clean["authorization_execution_consistent"]).eq(1).all():
        raise ValueError("A clean Ray authorization invariant failed")
    if not pd.to_numeric(clean["detected_flag"]).eq(0).all():
        raise ValueError("A clean Ray condition produced a false positive")

    action_flip = frame.loc[frame["fault_mode"] == "action_flip"]
    if not (pd.to_numeric(action_flip["injected_events"]) > 0).all():
        raise ValueError("An action-flip condition injected zero events")
    if not pd.to_numeric(action_flip["hash_match"]).eq(0).all():
        raise ValueError("Every action-flip condition must mismatch the hash")
    if not pd.to_numeric(action_flip["row_count_match"]).eq(1).all():
        raise ValueError("Action-flip must preserve row cardinality")

    dropped = frame.loc[frame["fault_mode"] == "dropped_row"]
    if not (pd.to_numeric(dropped["injected_events"]) > 0).all():
        raise ValueError("A dropped-row condition removed zero rows")
    if not pd.to_numeric(dropped["hash_match"]).eq(0).all():
        raise ValueError("Every dropped-row condition must mismatch the hash")
    if not pd.to_numeric(dropped["row_count_match"]).eq(0).all():
        raise ValueError("Every dropped-row condition must fail row count")

    faults = frame.loc[frame["fault_mode"] != "clean"]
    if not pd.to_numeric(faults["detected_flag"]).eq(1).all():
        raise ValueError("One or more Ray fault conditions were not flagged")
    if not pd.to_numeric(frame["detection_correct"]).eq(1).all():
        raise ValueError("One or more Ray detection expectations failed")
    if not pd.to_numeric(frame["unauthorized_invocations"]).eq(0).all():
        raise ValueError(
            "The selected Ray fault classes must not create unauthorized calls"
        )

    return {
        "conditions": int(len(frame)),
        "clean_conditions": int(len(clean)),
        "action_flip_conditions": int(len(action_flip)),
        "dropped_row_conditions": int(len(dropped)),
        "clean_hash_matches": int(pd.to_numeric(clean["hash_match"]).sum()),
        "fault_conditions_flagged": int(
            pd.to_numeric(faults["detected_flag"]).sum()
        ),
        "all_external_references": True,
        "all_expectations_met": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and validate the ReplayBench-PG Array revision."
    )
    parser.add_argument(
        "--project-dir",
        default=str(SCRIPT_DIR),
        help="Repository directory containing these scripts.",
    )
    parser.add_argument(
        "--timing-config",
        default="configs/fgcs_extended_benchmark.yaml",
    )
    parser.add_argument(
        "--ray-config",
        default="configs/ray_comparison.yaml",
    )
    parser.add_argument(
        "--timing-output-dir",
        default="paper_outputs/replaybench_timing_study",
    )
    parser.add_argument(
        "--ray-output-dir",
        default="paper_outputs/ray_comparison",
    )
    parser.add_argument("--run-timing", action="store_true")
    parser.add_argument("--resume-timing", action="store_true")
    parser.add_argument("--run-ray", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    compiled = compile_scripts(project_dir)

    timing_output_dir = project_dir / args.timing_output_dir
    ray_output_dir = project_dir / args.ray_output_dir

    if args.run_timing:
        command = [
            sys.executable,
            "run_replaybench_timing_study.py",
            "--config",
            args.timing_config,
            "--output-dir",
            args.timing_output_dir,
            "--workload-repetitions",
            "7",
            "--worker-repetitions",
            "15",
            "--warmups",
            "1",
        ]
        if args.resume_timing:
            command.append("--resume")
        run_checked(command, project_dir)

    timing_raw = timing_output_dir / "timing_repetitions_raw.csv"
    if not timing_raw.exists():
        raise FileNotFoundError(
            f"Timing results not found: {timing_raw}. Use --run-timing."
        )

    timing_frame = normalize_timing_frame(pd.read_csv(timing_raw))
    timing_validation = validate_timing_design(
        timing_frame,
        workload_repetitions=7,
        worker_repetitions=15,
        full_fraction=1.0,
    )

    run_checked(
        [
            sys.executable,
            "summarize_replaybench_timing_study.py",
            "--input",
            str(timing_raw),
            "--output-dir",
            args.timing_output_dir,
            "--workload-repetitions",
            "7",
            "--worker-repetitions",
            "15",
        ],
        project_dir,
    )

    if args.run_ray:
        run_checked(
            [
                sys.executable,
                "run_ray_comparison.py",
                "--config",
                args.ray_config,
            ],
            project_dir,
        )

    ray_per_run = ray_output_dir / "ray_comparison_per_run.csv"
    ray_validation = validate_ray_results(ray_per_run)

    tests_run = False
    if not args.skip_tests:
        run_checked(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_array_revision_validation.py",
            ],
            project_dir,
        )
        tests_run = True

    output_manifest = project_dir / "paper_outputs" / "array_revision_validation.json"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "passed",
        "compiled_scripts": compiled,
        "tests_run": tests_run,
        "timing_validation": timing_validation,
        "ray_validation": ray_validation,
        "timing_outputs": {
            "raw": str(timing_raw),
            "runtime_summary": str(
                timing_output_dir / "timing_runtime_summary_mixed_7_15.csv"
            ),
            "paired_speedup_summary": str(
                timing_output_dir / "timing_worker_speedup_paired_ci.csv"
            ),
        },
        "ray_outputs": {
            "per_run": str(ray_per_run),
            "summary": str(ray_output_dir / "ray_comparison_summary.csv"),
            "manifest": str(ray_output_dir / "ray_comparison_manifest.json"),
        },
    }
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[DONE] Array revision validation passed")
    print(f"[OUT] {output_manifest}")


if __name__ == "__main__":
    main()
