#!/usr/bin/env python3
"""Validate the corrected ReplayBench-PG v2.6.0 evidence package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    final_dir = ROOT / "paper_outputs" / "final_validation"
    manifest = json.loads((final_dir / "final_validation_manifest.json").read_text())
    claims = json.loads((final_dir / "final_claims_numbers.json").read_text())
    inventory = pd.read_csv(final_dir / "final_results_inventory.csv")

    require(
        claims["primary_benchmark"].get("conditions_expected") == 240,
        "primary benchmark is not the 240-condition four-policy design",
    )
    require(
        claims["timing_study"].get("measured_rows") == 352,
        "timing study is not the 352-execution four-policy design",
    )
    require(
        "bc_live_runtime_decomposition" not in claims,
        "retired bc_live decomposition remains in claims",
    )

    expected = {
        "execution_integrity_validation": {
            "clean_receipt_instances_passed": 18,
            "receipt_fault_instances_rejected": 72,
            "record_configuration_corruptions_detected": 90,
        },
        "validator_selectivity_validation": {
            "independent_benign_negative_control_executions": 24,
            "independent_positive_fault_executions": 216,
            "supported_runtime_events": 3240,
            "posthoc_validator_applications": 258,
        },
        "phase1_label_independent_validation": {
            "generic_validator_findings": 270,
            "positive_control_units": 228,
            "negative_control_units": 42,
        },
    }

    for component, fields in expected.items():
        require(component in manifest["components"], f"manifest missing {component}")
        require(component in claims, f"claims JSON missing {component}")
        require(component in set(inventory["component"]), f"inventory missing {component}")
        for field, value in fields.items():
            require(
                claims[component].get(field) == value,
                f"{component}.{field} != {value}",
            )

    aligned = inventory[inventory["component"].isin(expected)]
    require(not aligned.empty, "aligned inventory rows are empty")
    for row in aligned.itertuples(index=False):
        evidence_path = ROOT / row.path
        require(evidence_path.is_file(), f"missing retained evidence file: {row.path}")
        require(
            int(evidence_path.stat().st_size) == int(row.size_bytes),
            f"size mismatch: {row.path}",
        )
        require(sha256(evidence_path) == row.sha256, f"SHA-256 mismatch: {row.path}")

    print("PASS: v2.6.0 submission package validation succeeded")


if __name__ == "__main__":
    main()
