#!/usr/bin/env python3
"""Validate the aligned ReplayBench-PG submission package.

This compact check is supplementary to the repository master validator. It
verifies that the three newly aligned evidence families are present in the
canonical final manifest, claims JSON, results inventory, and retained files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    final_dir = ROOT / "paper_outputs" / "final_validation"
    manifest = json.loads((final_dir / "final_validation_manifest.json").read_text())
    claims = json.loads((final_dir / "final_claims_numbers.json").read_text())
    inventory = pd.read_csv(final_dir / "final_results_inventory.csv")

    expected = {
        "bc_live_runtime_decomposition": {
            "measured_executions": 88,
            "unique_configurations": 8,
        },
        "execution_integrity": {
            "clean_receipt_instances_passed": 18,
            "receipt_fault_instances_rejected": 72,
            "record_configuration_corruptions_detected": 90,
        },
        "validator_selectivity": {
            "independent_benign_negative_control_executions": 24,
            "independent_positive_fault_executions": 216,
            "supported_runtime_events": 3240,
            "posthoc_validator_applications": 258,
        },
    }
    for component, fields in expected.items():
        require(component in manifest["components"], f"manifest missing {component}")
        require(component in claims, f"claims JSON missing {component}")
        require(component in set(inventory["component"]), f"inventory missing {component}")
        for field, value in fields.items():
            require(claims[component].get(field) == value, f"{component}.{field} != {value}")

    aligned = inventory[inventory["component"].isin(expected)]
    require(not aligned.empty, "aligned inventory rows are empty")
    for row in aligned.itertuples(index=False):
        path = ROOT / row.path
        require(path.is_file(), f"missing retained evidence file: {row.path}")
        require(int(path.stat().st_size) == int(row.size_bytes), f"size mismatch: {row.path}")
        require(sha256(path) == row.sha256, f"SHA-256 mismatch: {row.path}")

    tex = (ROOT / "main.tex").read_text(encoding="utf-8")
    require("deterministic 1\\% subset" not in tex, "obsolete exact-1% wording remains")
    require((ROOT / "main.pdf").is_file(), "compiled main.pdf is missing")
    print("PASS: aligned submission package validation succeeded")


if __name__ == "__main__":
    main()
