#!/usr/bin/env python3
"""Run the complete compact MetroPT-3 secondary-workload validation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", help="Path to MetroPT3(AirCompressor).csv")
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--skip-preparation", action="store_true")
    parser.add_argument("--allow-row-count-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    py = sys.executable
    output_root = root / "paper_outputs" / "secondary_metropt3"
    replay_csv = output_root / "replay_input_metropt3.csv"

    if not args.skip_preparation:
        if not args.source_csv:
            raise ValueError("--source-csv is required unless --skip-preparation is used")
        prep = [
            py,
            str(root / "scripts" / "prepare_metropt3_secondary_workload.py"),
            "--source-csv",
            args.source_csv,
            "--sample-size",
            str(args.sample_size),
        ]
        if args.allow_row_count_mismatch:
            prep.append("--allow-row-count-mismatch")
        run(prep)
    elif not replay_csv.exists():
        raise FileNotFoundError(replay_csv)

    run([py, "run_fgcs_extended_benchmark.py", "--config", "configs/secondary_metropt3_benchmark.yaml"])
    run([py, "run_fgcs_extended_benchmark.py", "--config", "configs/secondary_metropt3_fault_action_flip.yaml"])
    run([py, "run_fgcs_extended_benchmark.py", "--config", "configs/secondary_metropt3_fault_unauthorized_invoke.yaml"])

    run(
        [
            py,
            "fgcs_fault_validation_framework.py",
            "--all",
            "--clean_dir",
            "paper_outputs/secondary_metropt3/clean_benchmark",
            "--action_flip_dir",
            "paper_outputs/secondary_metropt3/fault_action_flip",
            "--unauthorized_dir",
            "paper_outputs/secondary_metropt3/fault_unauthorized_invoke",
            "--trace_fault_dir",
            "paper_outputs/secondary_metropt3/fault_trace_corruption",
            "--tables_dir",
            "paper_outputs/secondary_metropt3/tables_figures",
            "--policies",
            "rule_gate",
            "random",
            "never",
            "--seeds",
            "1",
            "2",
            "3",
            "--workers",
            "1",
            "4",
        ]
    )
    run([py, "summarize_secondary_workload_results.py"])
    print("[DONE] MetroPT-3 secondary-workload validation completed.")


if __name__ == "__main__":
    main()
