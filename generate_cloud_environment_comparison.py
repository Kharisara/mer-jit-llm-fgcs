#!/usr/bin/env python3
"""
Generate the Comment 13 local-versus-cloud environment comparison artifacts.

Outputs:
  paper_outputs/final_manuscript_tables/cloud_environment_comparison.csv
  paper_outputs/environment/cloud_environment_comparison.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent

ASIA_METADATA = (
    ROOT
    / "cloud_results"
    / "cloud360_riskproxy_20260702"
    / "asia-southeast1"
    / "cloud_run_metadata.json"
)
US_METADATA = (
    ROOT
    / "cloud_results"
    / "cloud360_riskproxy_20260702"
    / "us-central1"
    / "cloud_run_metadata.json"
)
LOCK_FILE = ROOT / "paper_outputs" / "environment" / "pip_freeze_lock.txt"

OUTPUT_CSV = (
    ROOT
    / "paper_outputs"
    / "final_manuscript_tables"
    / "cloud_environment_comparison.csv"
)
OUTPUT_JSON = (
    ROOT
    / "paper_outputs"
    / "environment"
    / "cloud_environment_comparison.json"
)

BASE_IMAGE = (
    "python@sha256:"
    "57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de"
)
FINAL_IMAGE = (
    "sha256:"
    "bfcc798c003790c81544f712b9fa403d997ea5200a084ea74a79db651e62fcfc"
)

EXPECTED_PLATFORM = "Linux-6.9.12-x86_64-with-glibc2.41"
EXPECTED_CLOUD_PYTHON_PREFIX = "3.12.13"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    if not LOCK_FILE.is_file():
        raise FileNotFoundError(
            f"Dependency lock is missing: {LOCK_FILE}\n"
            "Run collect_comment12_environment.ps1 first."
        )

    asia = load_json(ASIA_METADATA)
    us = load_json(US_METADATA)

    require(
        asia.get("region") == "asia-southeast1",
        "Asia metadata has an unexpected region.",
    )
    require(
        us.get("region") == "us-central1",
        "US metadata has an unexpected region.",
    )
    require(
        asia.get("return_code") == 0 and us.get("return_code") == 0,
        "One or both archived Cloud Run jobs did not complete successfully.",
    )
    require(
        asia.get("platform") == EXPECTED_PLATFORM,
        f"Unexpected Asia platform: {asia.get('platform')!r}",
    )
    require(
        us.get("platform") == EXPECTED_PLATFORM,
        f"Unexpected US platform: {us.get('platform')!r}",
    )
    require(
        str(asia.get("python_version", "")).startswith(
            EXPECTED_CLOUD_PYTHON_PREFIX
        ),
        f"Unexpected Asia Python version: {asia.get('python_version')!r}",
    )
    require(
        str(us.get("python_version", "")).startswith(
            EXPECTED_CLOUD_PYTHON_PREFIX
        ),
        f"Unexpected US Python version: {us.get('python_version')!r}",
    )
    require(
        asia.get("platform") == us.get("platform"),
        "The two cloud regions report different container platforms.",
    )
    require(
        asia.get("python_version") == us.get("python_version"),
        "The two cloud regions report different Python builds.",
    )

    lock_sha = sha256_file(LOCK_FILE)

    columns = [
        "property",
        "local_workstation",
        "asia-southeast1",
        "us-central1",
    ]

    rows = [
        {
            "property": "execution_environment",
            "local_workstation": "Native Microsoft Windows",
            "asia-southeast1": "Google Cloud Run Jobs Linux container",
            "us-central1": "Google Cloud Run Jobs Linux container",
        },
        {
            "property": "region",
            "local_workstation": "Local workstation, Sri Lanka",
            "asia-southeast1": "asia-southeast1",
            "us-central1": "us-central1",
        },
        {
            "property": "operating_system",
            "local_workstation": (
                "Microsoft Windows 11 Home; version 10.0.26200; build 26200"
            ),
            "asia-southeast1": (
                "Debian GNU/Linux 13 (trixie); Linux 6.9.12; glibc 2.41"
            ),
            "us-central1": (
                "Debian GNU/Linux 13 (trixie); Linux 6.9.12; glibc 2.41"
            ),
        },
        {
            "property": "cpu_architecture",
            "local_workstation": "x86_64 (64-bit)",
            "asia-southeast1": "x86_64",
            "us-central1": "x86_64",
        },
        {
            "property": "compute_resources",
            "local_workstation": (
                "Intel Core i7-10750H CPU at 2.60 GHz; "
                "6 physical cores; 12 logical processors"
            ),
            "asia-southeast1": (
                "2 allocated vCPU; provider host CPU model not controlled"
            ),
            "us-central1": (
                "2 allocated vCPU; provider host CPU model not controlled"
            ),
        },
        {
            "property": "memory",
            "local_workstation": "15.8 GiB installed RAM",
            "asia-southeast1": "4 GiB allocated memory",
            "us-central1": "4 GiB allocated memory",
        },
        {
            "property": "python_version",
            "local_workstation": "Python 3.12.2",
            "asia-southeast1": str(asia["python_version"]),
            "us-central1": str(us["python_version"]),
        },
        {
            "property": "dependency_lock_sha256",
            "local_workstation": lock_sha,
            "asia-southeast1": lock_sha,
            "us-central1": lock_sha,
        },
        {
            "property": "container_base_image",
            "local_workstation": "Not applicable",
            "asia-southeast1": BASE_IMAGE,
            "us-central1": BASE_IMAGE,
        },
        {
            "property": "final_container_image",
            "local_workstation": "Not applicable",
            "asia-southeast1": FINAL_IMAGE,
            "us-central1": FINAL_IMAGE,
        },
        {
            "property": "locale",
            "local_workstation": "en-GB",
            "asia-southeast1": "C.UTF-8",
            "us-central1": "C.UTF-8",
        },
        {
            "property": "time_zone",
            "local_workstation": (
                "Sri Lanka Standard Time; UTC+05:30"
            ),
            "asia-southeast1": "UTC; +0000",
            "us-central1": "UTC; +0000",
        },
        {
            "property": "resource_or_power_mode",
            "local_workstation": "Windows Balanced power plan",
            "asia-southeast1": (
                "Cloud Run Jobs second-generation execution environment; "
                "2 vCPU; 4 GiB; zero retries"
            ),
            "us-central1": (
                "Cloud Run Jobs second-generation execution environment; "
                "2 vCPU; 4 GiB; zero retries"
            ),
        },
        {
            "property": "benchmark_configuration",
            "local_workstation": str(asia["config_path"]),
            "asia-southeast1": str(asia["config_path"]),
            "us-central1": str(us["config_path"]),
        },
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    json_payload = {
        "schema": "replaybench-pg-cloud-environment-comparison-v1",
        "source_metadata": {
            "asia-southeast1": {
                "path": ASIA_METADATA.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(ASIA_METADATA),
                "region": asia["region"],
                "platform": asia["platform"],
                "python_version": asia["python_version"],
                "return_code": asia["return_code"],
            },
            "us-central1": {
                "path": US_METADATA.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(US_METADATA),
                "region": us["region"],
                "platform": us["platform"],
                "python_version": us["python_version"],
                "return_code": us["return_code"],
            },
        },
        "dependency_lock": {
            "path": LOCK_FILE.relative_to(ROOT).as_posix(),
            "sha256": lock_sha,
        },
        "container": {
            "base_image_digest": BASE_IMAGE,
            "final_image_digest": FINAL_IMAGE,
            "inspected_os": "Debian GNU/Linux 13 (trixie)",
            "inspected_kernel": "Linux 6.9.12",
            "architecture": "x86_64",
            "locale": "C.UTF-8",
            "time_zone": "UTC +0000",
        },
        "local": {
            "os": "Microsoft Windows 11 Home",
            "version": "10.0.26200",
            "build": "26200",
            "architecture": "x86_64 (64-bit)",
            "cpu": "Intel Core i7-10750H CPU at 2.60 GHz",
            "logical_processors": 12,
            "ram_gib": 15.8,
            "python_version": "3.12.2",
            "locale": "en-GB",
            "ui_locale": "en-GB",
            "time_zone": "Sri Lanka Standard Time",
            "utc_offset": "+05:30",
            "power_plan": "Balanced",
        },
        "comparison_csv": {
            "path": OUTPUT_CSV.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(OUTPUT_CSV),
            "row_count": len(rows),
        },
        "interpretation": (
            "Portability and environment-consistency validation across a "
            "native Windows workstation and the same digest-pinned Linux "
            "container in two regions of one cloud provider."
        ),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(json_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("[PASS] Comment 13 cloud environment comparison generated")
    print(f"[OUT]  {OUTPUT_CSV}")
    print(f"[SHA]  {sha256_file(OUTPUT_CSV)}")
    print(f"[OUT]  {OUTPUT_JSON}")
    print(f"[SHA]  {sha256_file(OUTPUT_JSON)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
