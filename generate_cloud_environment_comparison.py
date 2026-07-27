#!/usr/bin/env python3
"""Generate Comment 13 local-versus-cloud environment comparison evidence.

The generator does not rerun benchmark experiments. It reads the archived
regional Cloud Run metadata, inspects the exact digest-pinned container image,
captures the image's dependency lock, and writes deterministic comparison
artifacts for manuscript and release validation.

Outputs
-------
paper_outputs/environment/cloud_container_environment.json
paper_outputs/environment/cloud_pip_freeze_lock.txt
paper_outputs/environment/cloud_environment_comparison.json
paper_outputs/final_manuscript_tables/cloud_environment_comparison.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import locale
import os
import subprocess
import sys
import time
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
ENVIRONMENT_MANIFEST = (
    ROOT
    / "paper_outputs"
    / "environment"
    / "execution_environment_manifest.json"
)

CONTAINER_ENVIRONMENT_JSON = (
    ROOT
    / "paper_outputs"
    / "environment"
    / "cloud_container_environment.json"
)
CLOUD_LOCK_FILE = (
    ROOT
    / "paper_outputs"
    / "environment"
    / "cloud_pip_freeze_lock.txt"
)
OUTPUT_JSON = (
    ROOT
    / "paper_outputs"
    / "environment"
    / "cloud_environment_comparison.json"
)
OUTPUT_CSV = (
    ROOT
    / "paper_outputs"
    / "final_manuscript_tables"
    / "cloud_environment_comparison.csv"
)

EXPECTED_REGIONS = ("asia-southeast1", "us-central1")
EXPECTED_PLATFORM = "Linux-6.9.12-x86_64-with-glibc2.41"
EXPECTED_CLOUD_PYTHON_PREFIX = "3.12.13"
EXPECTED_VCPU = 2
EXPECTED_MEMORY_GIB = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {role}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not parse {role} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must contain a JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run_command(command: list[str], role: str) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout or ""
    if completed.returncode != 0:
        raise RuntimeError(
            f"{role} failed with exit code {completed.returncode}.\n"
            f"Command: {' '.join(command)}\n"
            f"Output:\n{output}"
        )
    return output


def capture_windows_locale() -> dict[str, str]:
    if os.name == "nt":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "[ordered]@{"
                "culture=(Get-Culture).Name;"
                "ui_culture=(Get-UICulture).Name;"
                "time_zone=(Get-TimeZone).Id;"
                "utc_offset=([DateTimeOffset]::Now.Offset.ToString())"
                "} | ConvertTo-Json -Compress"
            ),
        ]
        output = run_command(command, "Windows locale capture").strip()
        value = json.loads(output)
        if not isinstance(value, dict):
            raise ValueError("Windows locale capture did not return an object")
        return {
            "culture": str(value.get("culture", "")).strip(),
            "ui_culture": str(value.get("ui_culture", "")).strip(),
            "time_zone": str(value.get("time_zone", "")).strip(),
            "utc_offset": str(value.get("utc_offset", "")).strip(),
        }

    current_locale = locale.setlocale(locale.LC_ALL, None)
    offset_seconds = -time.timezone
    sign = "+" if offset_seconds >= 0 else "-"
    offset_seconds = abs(offset_seconds)
    hours, remainder = divmod(offset_seconds, 3600)
    minutes = remainder // 60
    return {
        "culture": current_locale,
        "ui_culture": current_locale,
        "time_zone": time.tzname[0] if time.tzname else "unknown",
        "utc_offset": f"{sign}{hours:02d}:{minutes:02d}:00",
    }


def capture_container_environment(final_image_uri: str) -> dict[str, Any]:
    probe = r'''
import json
import locale
import os
import platform
import time
from pathlib import Path

os_release = {}
path = Path("/etc/os-release")
if path.is_file():
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os_release[key] = value.strip().strip('"')

payload = {
    "platform": platform.platform(),
    "system": platform.system(),
    "release": platform.release(),
    "machine": platform.machine(),
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "locale": locale.setlocale(locale.LC_ALL, None),
    "lang": os.environ.get("LANG"),
    "language": os.environ.get("LANGUAGE"),
    "lc_all": os.environ.get("LC_ALL"),
    "tz_env": os.environ.get("TZ"),
    "time_zone_names": list(time.tzname),
    "utc_offset_seconds": -time.timezone,
    "os_release": os_release,
}
print(json.dumps(payload, sort_keys=True))
'''.strip()

    output = run_command(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "python",
            final_image_uri,
            "-c",
            probe,
        ],
        "Digest-pinned container environment inspection",
    ).strip()
    try:
        value = json.loads(output.splitlines()[-1])
    except Exception as exc:
        raise ValueError(
            "Could not parse container-environment probe output:\n" + output
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("Container-environment probe did not return an object")
    return value


def capture_cloud_lock(final_image_uri: str) -> str:
    output = run_command(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "python",
            final_image_uri,
            "-m",
            "pip",
            "freeze",
            "--all",
        ],
        "Digest-pinned container dependency capture",
    )
    lines = sorted(
        {
            line.strip()
            for line in output.replace("\r\n", "\n").split("\n")
            if line.strip()
        },
        key=str.casefold,
    )
    if not lines:
        raise ValueError("Container dependency capture returned no packages")
    return "\n".join(lines) + "\n"


def format_utc_offset(seconds: int) -> str:
    sign = "+" if seconds >= 0 else "-"
    seconds = abs(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    return f"{sign}{hours:02d}:{minutes:02d}"


def main() -> int:
    asia = read_json(ASIA_METADATA, "asia-southeast1 Cloud Run metadata")
    us = read_json(US_METADATA, "us-central1 Cloud Run metadata")
    environment = read_json(
        ENVIRONMENT_MANIFEST,
        "Comment 12 execution-environment manifest",
    )

    require(asia.get("region") == EXPECTED_REGIONS[0], "Unexpected Asia region")
    require(us.get("region") == EXPECTED_REGIONS[1], "Unexpected US region")
    require(
        asia.get("return_code") == 0 and us.get("return_code") == 0,
        "One or both archived regional jobs did not complete successfully",
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
        asia.get("platform") == us.get("platform"),
        "Regional Cloud Run platforms differ",
    )
    require(
        asia.get("python_version") == us.get("python_version"),
        "Regional Cloud Run Python builds differ",
    )
    require(
        str(asia.get("python_version", "")).startswith(
            EXPECTED_CLOUD_PYTHON_PREFIX
        ),
        f"Unexpected cloud Python build: {asia.get('python_version')!r}",
    )

    local = environment.get("local")
    python_env = environment.get("python")
    container = environment.get("container")
    if not isinstance(local, dict):
        raise ValueError("Environment manifest local field must be an object")
    if not isinstance(python_env, dict):
        raise ValueError("Environment manifest python field must be an object")
    if not isinstance(container, dict):
        raise ValueError("Environment manifest container field must be an object")

    final_image_uri = str(container.get("final_image_uri", "")).strip()
    final_image_digest = str(container.get("final_image_digest", "")).strip()
    base_image_digest = str(container.get("base_image_repo_digest", "")).strip()
    require(final_image_uri, "Environment manifest is missing final_image_uri")
    require(final_image_digest, "Environment manifest is missing final_image_digest")
    require(base_image_digest, "Environment manifest is missing base_image_repo_digest")
    require(
        final_image_uri.endswith("@" + final_image_digest),
        "Final image URI is not pinned to the recorded digest",
    )

    local_lock_relative = str(python_env.get("pip_freeze_lock", "")).strip()
    if not local_lock_relative:
        raise ValueError("Environment manifest is missing pip_freeze_lock")
    local_lock_path = ROOT / local_lock_relative
    require(local_lock_path.is_file(), f"Missing local lock: {local_lock_path}")
    local_lock_sha = sha256_file(local_lock_path)
    require(
        local_lock_sha
        == str(python_env.get("pip_freeze_lock_sha256", "")).lower(),
        "Local dependency-lock digest does not match Comment 12 manifest",
    )

    local_locale = capture_windows_locale()
    for key, value in local_locale.items():
        require(bool(value), f"Local {key} capture is empty")

    container_environment = capture_container_environment(final_image_uri)
    require(
        container_environment.get("machine") == "x86_64",
        "Pinned container is not x86_64",
    )
    require(
        str(container_environment.get("python_version", "")).startswith(
            EXPECTED_CLOUD_PYTHON_PREFIX
        ),
        "Pinned container Python version differs from archived metadata",
    )
    require(
        str(asia["python_version"]).startswith(
            str(container_environment["python_version"])
        ),
        "Archived cloud Python build differs from the pinned image",
    )

    os_release = container_environment.get("os_release")
    if not isinstance(os_release, dict):
        raise ValueError("Container probe did not return os-release metadata")
    require(
        os_release.get("PRETTY_NAME") == "Debian GNU/Linux 13 (trixie)",
        f"Unexpected container OS: {os_release.get('PRETTY_NAME')!r}",
    )

    cloud_lock = capture_cloud_lock(final_image_uri)

    CONTAINER_ENVIRONMENT_JSON.parent.mkdir(parents=True, exist_ok=True)
    CONTAINER_ENVIRONMENT_JSON.write_text(
        json.dumps(container_environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    CLOUD_LOCK_FILE.write_text(
        cloud_lock,
        encoding="utf-8",
        newline="\n",
    )
    cloud_lock_sha = sha256_file(CLOUD_LOCK_FILE)

    local_os = (
        f"{local.get('os_caption')}; version {local.get('os_version')}; "
        f"build {local.get('os_build_number')}"
    )
    local_arch = str(local.get("os_architecture", "")).strip()
    local_cpu = str(local.get("cpu_name", "")).strip()
    local_physical = int(local.get("physical_cores"))
    local_logical = int(local.get("logical_processors"))
    local_ram = float(local.get("installed_ram_gib"))
    local_python = str(python_env.get("version", "")).strip()

    container_os = str(os_release["PRETTY_NAME"])
    inspection_platform = str(container_environment["platform"])
    archived_cloud_platform = str(asia["platform"])
    container_python = str(container_environment["python_version"])
    container_locale = str(
        container_environment.get("lang")
        or container_environment.get("locale")
        or ""
    ).strip()
    zone_names = container_environment.get("time_zone_names")
    if not isinstance(zone_names, list) or not zone_names:
        raise ValueError("Container probe did not return time-zone names")
    cloud_time_zone = str(zone_names[0])
    cloud_utc_offset = format_utc_offset(
        int(container_environment.get("utc_offset_seconds", 0))
    )

    rows = [
        {
            "property": "execution_environment",
            "local_workstation": "Native Microsoft Windows",
            EXPECTED_REGIONS[0]: "Google Cloud Run Jobs Linux container",
            EXPECTED_REGIONS[1]: "Google Cloud Run Jobs Linux container",
        },
        {
            "property": "region",
            "local_workstation": "Local workstation",
            EXPECTED_REGIONS[0]: EXPECTED_REGIONS[0],
            EXPECTED_REGIONS[1]: EXPECTED_REGIONS[1],
        },
        {
            "property": "operating_system",
            "local_workstation": local_os,
            EXPECTED_REGIONS[0]: (
                f"{container_os}; archived runtime {archived_cloud_platform}"
            ),
            EXPECTED_REGIONS[1]: (
                f"{container_os}; archived runtime {archived_cloud_platform}"
            ),
        },
        {
            "property": "cpu_architecture",
            "local_workstation": local_arch,
            EXPECTED_REGIONS[0]: str(container_environment["machine"]),
            EXPECTED_REGIONS[1]: str(container_environment["machine"]),
        },
        {
            "property": "compute_resources",
            "local_workstation": (
                f"{local_cpu}; {local_physical} physical cores; "
                f"{local_logical} logical processors"
            ),
            EXPECTED_REGIONS[0]: (
                f"{EXPECTED_VCPU} allocated vCPU; "
                "provider host CPU model not controlled"
            ),
            EXPECTED_REGIONS[1]: (
                f"{EXPECTED_VCPU} allocated vCPU; "
                "provider host CPU model not controlled"
            ),
        },
        {
            "property": "memory",
            "local_workstation": f"{local_ram:.1f} GiB installed RAM",
            EXPECTED_REGIONS[0]: f"{EXPECTED_MEMORY_GIB} GiB allocated memory",
            EXPECTED_REGIONS[1]: f"{EXPECTED_MEMORY_GIB} GiB allocated memory",
        },
        {
            "property": "python_version",
            "local_workstation": local_python,
            EXPECTED_REGIONS[0]: str(asia["python_version"]),
            EXPECTED_REGIONS[1]: str(us["python_version"]),
        },
        {
            "property": "dependency_lock_sha256",
            "local_workstation": local_lock_sha,
            EXPECTED_REGIONS[0]: cloud_lock_sha,
            EXPECTED_REGIONS[1]: cloud_lock_sha,
        },
        {
            "property": "container_base_image",
            "local_workstation": "Not applicable",
            EXPECTED_REGIONS[0]: base_image_digest,
            EXPECTED_REGIONS[1]: base_image_digest,
        },
        {
            "property": "final_container_image",
            "local_workstation": "Not applicable",
            EXPECTED_REGIONS[0]: final_image_uri,
            EXPECTED_REGIONS[1]: final_image_uri,
        },
        {
            "property": "locale",
            "local_workstation": (
                f"{local_locale['culture']} "
                f"(UI {local_locale['ui_culture']})"
            ),
            EXPECTED_REGIONS[0]: container_locale,
            EXPECTED_REGIONS[1]: container_locale,
        },
        {
            "property": "time_zone",
            "local_workstation": (
                f"{local_locale['time_zone']}; "
                f"UTC{local_locale['utc_offset'][:6]}"
            ),
            EXPECTED_REGIONS[0]: f"{cloud_time_zone}; UTC{cloud_utc_offset}",
            EXPECTED_REGIONS[1]: f"{cloud_time_zone}; UTC{cloud_utc_offset}",
        },
        {
            "property": "resource_or_power_mode",
            "local_workstation": str(local.get("active_power_scheme", "")),
            EXPECTED_REGIONS[0]: (
                "Cloud Run Jobs second-generation execution environment; "
                f"{EXPECTED_VCPU} vCPU; {EXPECTED_MEMORY_GIB} GiB; zero retries"
            ),
            EXPECTED_REGIONS[1]: (
                "Cloud Run Jobs second-generation execution environment; "
                f"{EXPECTED_VCPU} vCPU; {EXPECTED_MEMORY_GIB} GiB; zero retries"
            ),
        },
        {
            "property": "benchmark_configuration",
            "local_workstation": str(asia["config_path"]),
            EXPECTED_REGIONS[0]: str(asia["config_path"]),
            EXPECTED_REGIONS[1]: str(us["config_path"]),
        },
    ]

    columns = [
        "property",
        "local_workstation",
        EXPECTED_REGIONS[0],
        EXPECTED_REGIONS[1],
    ]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    comparison = {
        "schema": "replaybench-pg-cloud-environment-comparison-v2",
        "source_environment_manifest": {
            "path": ENVIRONMENT_MANIFEST.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(ENVIRONMENT_MANIFEST),
        },
        "source_metadata": {
            EXPECTED_REGIONS[0]: {
                "path": ASIA_METADATA.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(ASIA_METADATA),
                "region": asia["region"],
                "platform": asia["platform"],
                "python_version": asia["python_version"],
                "return_code": asia["return_code"],
            },
            EXPECTED_REGIONS[1]: {
                "path": US_METADATA.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(US_METADATA),
                "region": us["region"],
                "platform": us["platform"],
                "python_version": us["python_version"],
                "return_code": us["return_code"],
            },
        },
        "dependency_locks": {
            "local": {
                "path": local_lock_path.relative_to(ROOT).as_posix(),
                "sha256": local_lock_sha,
            },
            "cloud_container": {
                "path": CLOUD_LOCK_FILE.relative_to(ROOT).as_posix(),
                "sha256": cloud_lock_sha,
            },
            "identical": local_lock_sha == cloud_lock_sha,
        },
        "container": {
            "inspection_path": (
                CONTAINER_ENVIRONMENT_JSON.relative_to(ROOT).as_posix()
            ),
            "inspection_sha256": sha256_file(CONTAINER_ENVIRONMENT_JSON),
            "base_image_digest": base_image_digest,
            "final_image_uri": final_image_uri,
            "final_image_digest": final_image_digest,
            "os": container_os,
            "platform": archived_cloud_platform,
            "inspection_runtime_platform": inspection_platform,
            "architecture": str(container_environment["machine"]),
            "python_version": container_python,
            "locale": container_locale,
            "time_zone": cloud_time_zone,
            "utc_offset": cloud_utc_offset,
        },
        "local": {
            "os": str(local.get("os_caption")),
            "version": str(local.get("os_version")),
            "build": str(local.get("os_build_number")),
            "architecture": local_arch,
            "cpu": local_cpu,
            "physical_cores": local_physical,
            "logical_processors": local_logical,
            "ram_gib": local_ram,
            "python_version": local_python,
            "locale": local_locale["culture"],
            "ui_locale": local_locale["ui_culture"],
            "time_zone": local_locale["time_zone"],
            "utc_offset": local_locale["utc_offset"],
            "power_plan": str(local.get("active_power_scheme")),
        },
        "regional_resources": {
            EXPECTED_REGIONS[0]: {
                "vcpus": EXPECTED_VCPU,
                "memory_gib": EXPECTED_MEMORY_GIB,
                "execution_environment": "second-generation",
                "retries": 0,
            },
            EXPECTED_REGIONS[1]: {
                "vcpus": EXPECTED_VCPU,
                "memory_gib": EXPECTED_MEMORY_GIB,
                "execution_environment": "second-generation",
                "retries": 0,
            },
        },
        "comparison_csv": {
            "path": OUTPUT_CSV.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(OUTPUT_CSV),
            "row_count": len(rows),
            "columns": columns,
        },
        "interpretation": (
            "Portability and environment-consistency validation across a "
            "native Windows workstation and the same digest-pinned Linux "
            "container in two regions of one cloud provider."
        ),
        "claim_boundary": (
            "The comparison does not establish multi-cloud portability, "
            "distributed-systems performance, autoscaling behaviour, "
            "network-failure tolerance, or production reliability."
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("[PASS] Comment 13 cloud environment comparison generated")
    print(f"[OUT]  {CONTAINER_ENVIRONMENT_JSON}")
    print(f"[SHA]  {sha256_file(CONTAINER_ENVIRONMENT_JSON)}")
    print(f"[OUT]  {CLOUD_LOCK_FILE}")
    print(f"[SHA]  {cloud_lock_sha}")
    print(f"[OUT]  {OUTPUT_CSV}")
    print(f"[SHA]  {sha256_file(OUTPUT_CSV)}")
    print(f"[OUT]  {OUTPUT_JSON}")
    print(f"[SHA]  {sha256_file(OUTPUT_JSON)}")
    print(
        "[INFO] Local/cloud dependency locks identical: "
        f"{local_lock_sha == cloud_lock_sha}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
