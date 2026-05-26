"""
run_full_reproduction.py

Full reproduction runner for the FGCS deterministic replay and benchmarking artifact.

This script runs:
1. Extended FGCS deterministic replay benchmark
2. Fault-injection evaluation
3. Trace-verification overhead measurement
4. Paper-ready table and figure generation

Expected prepared inputs:
- configs/fgcs_extended_benchmark.yaml
- paper_outputs/replay_input_clean.csv
- paper_outputs/policy_first_outputs_bc.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "fgcs_extended_benchmark.yaml"
REPLAY_INPUT = PROJECT_ROOT / "paper_outputs" / "replay_input_clean.csv"
BC_TRACE = PROJECT_ROOT / "paper_outputs" / "policy_first_outputs_bc.csv"

BENCHMARK_SCRIPT = PROJECT_ROOT / "run_fgcs_extended_benchmark.py"
SUMMARY_SCRIPT = PROJECT_ROOT / "summarize_fgcs_extended_results.py"

RAW_OUTPUT_DIR = PROJECT_ROOT / "paper_outputs" / "fgcs_extended_benchmarks"
TABLE_FIGURE_DIR = PROJECT_ROOT / "paper_outputs" / "fgcs_tables_figures"


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required {description} not found:\n{path}\n\n"
            "This FGCS artifact reproduces the benchmark from prepared "
            "MELD-derived replay inputs. It does not recreate the full raw "
            "MELD preprocessing pipeline."
        )


def run_cmd(cmd: list[str]) -> None:
    print("\n[CMD]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full FGCS benchmark reproduction pipeline."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to FGCS extended benchmark YAML configuration.",
    )

    parser.add_argument(
        "--skip_benchmark",
        action="store_true",
        help="Skip benchmark execution and only run result summarization.",
    )

    parser.add_argument(
        "--skip_summary",
        action="store_true",
        help="Skip summary table/figure generation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    print("\n================================================")
    print(" FGCS DETERMINISTIC REPLAY REPRODUCTION PIPELINE")
    print("================================================")
    print(f"[INFO] Project root : {PROJECT_ROOT}")
    print(f"[INFO] Config path  : {config_path}")

    require_file(config_path, "benchmark configuration file")
    require_file(REPLAY_INPUT, "prepared replay input CSV")
    require_file(BC_TRACE, "prepared learned-policy trace CSV")
    require_file(BENCHMARK_SCRIPT, "extended benchmark script")
    require_file(SUMMARY_SCRIPT, "extended summarization script")

    if not args.skip_benchmark:
        print("\n[STEP 1] Running extended FGCS benchmark")
        run_cmd([
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--config",
            str(config_path),
        ])
    else:
        print("\n[STEP 1] Skipping benchmark execution")

    if not args.skip_summary:
        print("\n[STEP 2] Generating paper-ready tables and figures")
        run_cmd([
            sys.executable,
            str(SUMMARY_SCRIPT),
        ])
    else:
        print("\n[STEP 2] Skipping summarization")

    print("\n[DONE] FGCS reproduction pipeline completed.")
    print(f"[OUT] Raw benchmark outputs : {RAW_OUTPUT_DIR}")
    print(f"[OUT] Tables and figures    : {TABLE_FIGURE_DIR}")


if __name__ == "__main__":
    main()