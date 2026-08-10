"""Run the active ReplayBench-PG v2.6.0 four-policy workflow.

The active primary replay input is regenerated from the historical local source
before execution unless ``--skip-prepare`` is supplied. Learned-policy artifacts
and state embeddings are not active reproduction inputs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "fgcs_extended_benchmark.yaml"
REPLAY_INPUT = PROJECT_ROOT / "paper_outputs" / "replay_input_v260.csv"
PREP_SCRIPT = PROJECT_ROOT / "scripts" / "prepare_primary_replay_v260.py"
BENCHMARK_SCRIPT = PROJECT_ROOT / "run_fgcs_extended_benchmark.py"
SUMMARY_SCRIPT = PROJECT_ROOT / "summarize_fgcs_extended_results.py"
RAW_OUTPUT_DIR = PROJECT_ROOT / "paper_outputs" / "fgcs_extended_benchmark"
TABLE_FIGURE_DIR = PROJECT_ROOT / "paper_outputs" / "fgcs_tables_figures"


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required {description} not found: {path}")


def run_cmd(cmd: list[str]) -> None:
    print("\n[CMD]", " ".join(str(value) for value in cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ReplayBench-PG v2.6.0 four-policy workflow."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    print("\n====================================================")
    print(" REPLAYBENCH-PG v2.6.0 FOUR-POLICY REPRODUCTION")
    print("====================================================")
    print(f"[INFO] Project root : {PROJECT_ROOT}")
    print(f"[INFO] Config path  : {config_path}")

    require_file(config_path, "benchmark configuration")
    require_file(PREP_SCRIPT, "primary-input preparation script")
    require_file(BENCHMARK_SCRIPT, "benchmark script")
    require_file(SUMMARY_SCRIPT, "summarization script")

    if not args.skip_prepare:
        run_cmd([sys.executable, str(PREP_SCRIPT)])
    require_file(REPLAY_INPUT, "prepared v2.6.0 replay input")

    if not args.skip_benchmark:
        run_cmd([sys.executable, str(BENCHMARK_SCRIPT), "--config", str(config_path)])
    else:
        print("\n[STEP 1] Skipping benchmark execution")

    if not args.skip_summary:
        run_cmd([sys.executable, str(SUMMARY_SCRIPT)])
    else:
        print("\n[STEP 2] Skipping summarization")

    print("\n[DONE] ReplayBench-PG four-policy workflow completed")
    print(f"[OUT] Active replay input    : {REPLAY_INPUT}")
    print(f"[OUT] Raw benchmark outputs : {RAW_OUTPUT_DIR}")
    print(f"[OUT] Tables and figures    : {TABLE_FIGURE_DIR}")


if __name__ == "__main__":
    main()
