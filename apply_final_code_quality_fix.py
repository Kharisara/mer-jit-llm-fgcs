#!/usr/bin/env python3
"""Apply or verify the pandas DataFrame.map code-quality correction.

This changes only the deprecated validation-ablation conversion in
``fgcs_fault_validation_framework.py``. It does not rerun any experiment and it
does not alter any result CSV.
"""

from __future__ import annotations

import argparse
import py_compile
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace the deprecated DataFrame.applymap call with DataFrame.map."
    )
    parser.add_argument("--project-dir", default=str(SCRIPT_DIR))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Verify the correction only.")
    mode.add_argument("--apply", action="store_true", help="Apply the correction in place.")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create fgcs_fault_validation_framework.py.bak before editing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    path = project_dir / "fgcs_fault_validation_framework.py"
    if not path.is_file():
        raise FileNotFoundError(f"Target file not found: {path}")

    text = path.read_text(encoding="utf-8")
    corrected_token = "matrix[validator_cols].map("
    deprecated_token = "matrix[validator_cols].applymap("

    if args.check:
        if deprecated_token in text:
            raise RuntimeError(
                "Deprecated applymap call is still present. Run this script with --apply."
            )
        if corrected_token not in text:
            raise RuntimeError(
                "Neither the deprecated target nor the expected map replacement was found. "
                "Inspect the target file manually before freezing."
            )
        py_compile.compile(str(path), doraise=True)
        print("[PASS] DataFrame.map correction is present and the file compiles")
        return

    if corrected_token in text and deprecated_token not in text:
        py_compile.compile(str(path), doraise=True)
        print("[DONE] Correction was already applied; no file change was needed")
        return

    occurrences = text.count(deprecated_token)
    if occurrences != 1:
        raise RuntimeError(
            "Expected exactly one targeted applymap occurrence, "
            f"but found {occurrences}. No change was made."
        )

    if not args.no_backup:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        print(f"[BACKUP] {backup}")

    updated = text.replace(deprecated_token, corrected_token, 1)
    path.write_text(updated, encoding="utf-8", newline="\n")
    py_compile.compile(str(path), doraise=True)
    print(f"[DONE] Replaced DataFrame.applymap with DataFrame.map in {path}")
    print("[NOTE] Experimental outputs were not rerun or modified")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
