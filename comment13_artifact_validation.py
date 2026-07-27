#!/usr/bin/env python3
"""Strict validation for Comment 13 cloud-environment comparison artifacts."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from final_artifact_utils import (
    ValidationError,
    evidence_file,
    read_csv_required,
    read_json_required,
    require_columns,
    sha256_file,
)


EXPECTED_REGIONS = ("asia-southeast1", "us-central1")
EXPECTED_PROPERTIES = {
    "execution_environment",
    "region",
    "operating_system",
    "cpu_architecture",
    "compute_resources",
    "memory",
    "python_version",
    "dependency_lock_sha256",
    "container_base_image",
    "final_container_image",
    "locale",
    "time_zone",
    "resource_or_power_mode",
    "benchmark_configuration",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PREFIXED_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValidationError(
            f"Comment 13 field {key!r} must be a JSON object"
        )
    return value


def _require_nonempty(parent: Mapping[str, Any], key: str, role: str) -> str:
    value = parent.get(key)
    if value is None or not str(value).strip():
        raise ValidationError(f"Missing Comment 13 {role}: {key!r}")
    return str(value).strip()


def _require_sha256(value: Any, role: str, *, prefixed: bool = False) -> str:
    token = str(value).strip().lower()
    pattern = PREFIXED_SHA256_PATTERN if prefixed else SHA256_PATTERN
    if pattern.fullmatch(token) is None:
        expected = "sha256:<64 hex>" if prefixed else "<64 hex>"
        raise ValidationError(f"Invalid {role}; expected {expected}")
    return token


def _path_from_record(
    project_dir: Path,
    record: Mapping[str, Any],
    role: str,
) -> Path:
    relative = _require_nonempty(record, "path", f"{role} path")
    path = Path(relative)
    if not path.is_absolute():
        path = project_dir / path
    if not path.is_file():
        raise ValidationError(f"Missing {role}: {path}")
    expected_sha = _require_sha256(
        _require_nonempty(record, "sha256", f"{role} SHA-256"),
        f"{role} SHA-256",
    )
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise ValidationError(
            f"{role} SHA-256 mismatch: expected={expected_sha}, "
            f"observed={observed_sha}"
        )
    return path


def _csv_value(frame: pd.DataFrame, property_name: str, column: str) -> str:
    row = frame.loc[frame["property"].eq(property_name)]
    if len(row) != 1:
        raise ValidationError(
            f"Comment 13 comparison CSV must contain exactly one "
            f"{property_name!r} row"
        )
    value = row.iloc[0][column]
    if pd.isna(value) or not str(value).strip():
        raise ValidationError(
            f"Comment 13 comparison CSV has an empty value for "
            f"{property_name!r}/{column!r}"
        )
    return str(value).strip()


def validate_comment13_artifacts(
    project_dir: Path,
) -> tuple[dict[str, Any], list[Any]]:
    """Validate local-versus-cloud comparison evidence and claim boundaries."""

    comparison_path = (
        project_dir
        / "paper_outputs"
        / "environment"
        / "cloud_environment_comparison.json"
    )
    comparison = read_json_required(
        comparison_path,
        "Comment 13 cloud-environment comparison JSON",
    )
    if not isinstance(comparison, Mapping):
        raise ValidationError(
            "Comment 13 cloud-environment comparison must be a JSON object"
        )
    if (
        comparison.get("schema")
        != "replaybench-pg-cloud-environment-comparison-v2"
    ):
        raise ValidationError(
            "Comment 13 comparison JSON has an unexpected schema"
        )

    environment_record = _require_mapping(
        comparison, "source_environment_manifest"
    )
    environment_path = _path_from_record(
        project_dir,
        environment_record,
        "Comment 12 execution-environment manifest",
    )
    environment = read_json_required(
        environment_path,
        "Comment 12 execution-environment manifest",
    )
    if not isinstance(environment, Mapping):
        raise ValidationError(
            "Comment 12 execution-environment manifest must be a JSON object"
        )
    environment_local = _require_mapping(environment, "local")
    environment_python = _require_mapping(environment, "python")
    environment_container = _require_mapping(environment, "container")

    source_metadata = _require_mapping(comparison, "source_metadata")
    source_metadata_paths: dict[str, Path] = {}
    source_metadata_payloads: dict[str, Mapping[str, Any]] = {}
    for region in EXPECTED_REGIONS:
        region_record = _require_mapping(source_metadata, region)
        path = _path_from_record(
            project_dir,
            region_record,
            f"{region} archived Cloud Run metadata",
        )
        payload = read_json_required(
            path,
            f"{region} archived Cloud Run metadata",
        )
        if not isinstance(payload, Mapping):
            raise ValidationError(
                f"{region} archived Cloud Run metadata must be a JSON object"
            )
        if payload.get("region") != region:
            raise ValidationError(
                f"{region} archived metadata reports the wrong region"
            )
        if int(payload.get("return_code", -1)) != 0:
            raise ValidationError(
                f"{region} archived Cloud Run job did not complete successfully"
            )
        if region_record.get("region") != region:
            raise ValidationError(
                f"Comment 13 source record has the wrong {region} label"
            )
        if region_record.get("platform") != payload.get("platform"):
            raise ValidationError(
                f"{region} platform differs between source record and metadata"
            )
        if region_record.get("python_version") != payload.get("python_version"):
            raise ValidationError(
                f"{region} Python build differs between source record and metadata"
            )
        if int(region_record.get("return_code", -1)) != 0:
            raise ValidationError(
                f"{region} source record reports a failed return code"
            )
        source_metadata_paths[region] = path
        source_metadata_payloads[region] = payload

    asia_payload = source_metadata_payloads[EXPECTED_REGIONS[0]]
    us_payload = source_metadata_payloads[EXPECTED_REGIONS[1]]
    if asia_payload.get("platform") != us_payload.get("platform"):
        raise ValidationError(
            "The two regional metadata files report different platforms"
        )
    if asia_payload.get("python_version") != us_payload.get("python_version"):
        raise ValidationError(
            "The two regional metadata files report different Python builds"
        )
    platform_token = str(asia_payload.get("platform", ""))
    if "Linux" not in platform_token or "x86_64" not in platform_token:
        raise ValidationError(
            "Archived regional platform is not the expected Linux x86_64 environment"
        )

    dependency_locks = _require_mapping(comparison, "dependency_locks")
    local_lock_record = _require_mapping(dependency_locks, "local")
    cloud_lock_record = _require_mapping(
        dependency_locks, "cloud_container"
    )
    local_lock_path = _path_from_record(
        project_dir, local_lock_record, "local dependency lock"
    )
    cloud_lock_path = _path_from_record(
        project_dir, cloud_lock_record, "cloud-container dependency lock"
    )
    local_lock_sha = sha256_file(local_lock_path)
    cloud_lock_sha = sha256_file(cloud_lock_path)

    manifest_lock_relative = _require_nonempty(
        environment_python,
        "pip_freeze_lock",
        "Comment 12 dependency-lock path",
    )
    manifest_lock_path = project_dir / manifest_lock_relative
    if manifest_lock_path.resolve() != local_lock_path.resolve():
        raise ValidationError(
            "Comment 13 local lock path differs from the Comment 12 manifest"
        )
    manifest_lock_sha = _require_sha256(
        _require_nonempty(
            environment_python,
            "pip_freeze_lock_sha256",
            "Comment 12 dependency-lock SHA-256",
        ),
        "Comment 12 dependency-lock SHA-256",
    )
    if local_lock_sha != manifest_lock_sha:
        raise ValidationError(
            "Comment 13 local dependency lock differs from Comment 12"
        )

    identical_value = dependency_locks.get("identical")
    if not isinstance(identical_value, bool):
        raise ValidationError(
            "dependency_locks.identical must be a Boolean"
        )
    if identical_value != (local_lock_sha == cloud_lock_sha):
        raise ValidationError(
            "dependency_locks.identical does not match the observed lock hashes"
        )

    container = _require_mapping(comparison, "container")
    inspection_record = {
        "path": _require_nonempty(
            container, "inspection_path", "container-inspection path"
        ),
        "sha256": _require_nonempty(
            container, "inspection_sha256", "container-inspection SHA-256"
        ),
    }
    inspection_path = _path_from_record(
        project_dir,
        inspection_record,
        "digest-pinned container inspection",
    )
    inspection = read_json_required(
        inspection_path,
        "digest-pinned container inspection",
    )
    if not isinstance(inspection, Mapping):
        raise ValidationError(
            "Digest-pinned container inspection must be a JSON object"
        )

    final_image_digest = _require_sha256(
        _require_nonempty(
            container, "final_image_digest", "final-image digest"
        ),
        "Comment 13 final-image digest",
        prefixed=True,
    )
    final_image_uri = _require_nonempty(
        container, "final_image_uri", "final-image URI"
    )
    if not final_image_uri.endswith("@" + final_image_digest):
        raise ValidationError(
            "Comment 13 final-image URI is not pinned to its digest"
        )

    comment12_final_digest = _require_sha256(
        _require_nonempty(
            environment_container,
            "final_image_digest",
            "Comment 12 final-image digest",
        ),
        "Comment 12 final-image digest",
        prefixed=True,
    )
    comment12_final_uri = _require_nonempty(
        environment_container,
        "final_image_uri",
        "Comment 12 final-image URI",
    )
    if final_image_digest != comment12_final_digest:
        raise ValidationError(
            "Comment 13 final-image digest differs from Comment 12"
        )
    if final_image_uri != comment12_final_uri:
        raise ValidationError(
            "Comment 13 final-image URI differs from Comment 12"
        )

    base_image_digest = _require_nonempty(
        container, "base_image_digest", "base-image digest"
    )
    comment12_base_digest = _require_nonempty(
        environment_container,
        "base_image_repo_digest",
        "Comment 12 base-image digest",
    )
    if base_image_digest != comment12_base_digest:
        raise ValidationError(
            "Comment 13 base-image digest differs from Comment 12"
        )

    inspection_os_release = inspection.get("os_release")
    if not isinstance(inspection_os_release, Mapping):
        raise ValidationError(
            "Container inspection is missing os_release"
        )
    if (
        inspection_os_release.get("PRETTY_NAME")
        != "Debian GNU/Linux 13 (trixie)"
    ):
        raise ValidationError(
            "Container inspection does not report Debian GNU/Linux 13"
        )
    if inspection.get("machine") != "x86_64":
        raise ValidationError(
            "Container inspection does not report x86_64"
        )
    if not str(inspection.get("python_version", "")).startswith("3.12.13"):
        raise ValidationError(
            "Container inspection does not report Python 3.12.13"
        )
    if not str(asia_payload.get("python_version", "")).startswith(
        str(inspection.get("python_version"))
    ):
        raise ValidationError(
            "Container-inspected Python version differs from archived cloud metadata"
        )

    for key in [
        "os",
        "platform",
        "inspection_runtime_platform",
        "architecture",
        "python_version",
        "locale",
        "time_zone",
        "utc_offset",
    ]:
        _require_nonempty(container, key, "container comparison field")
    if container.get("platform") != asia_payload.get("platform"):
        raise ValidationError(
            "Comment 13 cloud platform differs from archived metadata"
        )
    if container.get("inspection_runtime_platform") != inspection.get("platform"):
        raise ValidationError(
            "Comment 13 inspection platform differs from raw inspection"
        )
    if container.get("architecture") != inspection.get("machine"):
        raise ValidationError(
            "Container architecture differs from its inspection"
        )
    if container.get("python_version") != inspection.get("python_version"):
        raise ValidationError(
            "Container Python version differs from its inspection"
        )

    local = _require_mapping(comparison, "local")
    for key in [
        "os",
        "version",
        "build",
        "architecture",
        "cpu",
        "physical_cores",
        "logical_processors",
        "ram_gib",
        "python_version",
        "locale",
        "ui_locale",
        "time_zone",
        "utc_offset",
        "power_plan",
    ]:
        if key not in local or local.get(key) in (None, ""):
            raise ValidationError(
                f"Comment 13 local field {key!r} is missing"
            )

    if "Windows" not in str(local["os"]):
        raise ValidationError(
            "Comment 13 local environment is not identified as Windows"
        )
    if str(local["os"]) != str(environment_local.get("os_caption")):
        raise ValidationError(
            "Comment 13 local OS differs from Comment 12"
        )
    if str(local["version"]) != str(environment_local.get("os_version")):
        raise ValidationError(
            "Comment 13 local OS version differs from Comment 12"
        )
    if str(local["build"]) != str(
        environment_local.get("os_build_number")
    ):
        raise ValidationError(
            "Comment 13 local OS build differs from Comment 12"
        )
    if str(local["cpu"]) != str(environment_local.get("cpu_name")):
        raise ValidationError(
            "Comment 13 local CPU differs from Comment 12"
        )
    if int(local["logical_processors"]) != int(
        environment_local.get("logical_processors")
    ):
        raise ValidationError(
            "Comment 13 logical-processor count differs from Comment 12"
        )
    if not math.isclose(
        float(local["ram_gib"]),
        float(environment_local.get("installed_ram_gib")),
        rel_tol=0.0,
        abs_tol=0.05,
    ):
        raise ValidationError(
            "Comment 13 RAM amount differs from Comment 12"
        )
    if str(local["python_version"]) != str(
        environment_python.get("version")
    ):
        raise ValidationError(
            "Comment 13 local Python version differs from Comment 12"
        )

    regional_resources = _require_mapping(
        comparison, "regional_resources"
    )
    for region in EXPECTED_REGIONS:
        resources = _require_mapping(regional_resources, region)
        if int(resources.get("vcpus", -1)) != 2:
            raise ValidationError(
                f"{region} must report 2 allocated vCPU"
            )
        if int(resources.get("memory_gib", -1)) != 4:
            raise ValidationError(
                f"{region} must report 4 GiB allocated memory"
            )
        if resources.get("execution_environment") != "second-generation":
            raise ValidationError(
                f"{region} must report the second-generation environment"
            )
        if int(resources.get("retries", -1)) != 0:
            raise ValidationError(
                f"{region} must report zero retries"
            )
    if regional_resources[EXPECTED_REGIONS[0]] != regional_resources[
        EXPECTED_REGIONS[1]
    ]:
        raise ValidationError(
            "Regional resource configurations are not identical"
        )

    csv_record = _require_mapping(comparison, "comparison_csv")
    csv_path = _path_from_record(
        project_dir, csv_record, "Comment 13 comparison CSV"
    )
    frame = read_csv_required(csv_path, "Comment 13 comparison CSV")
    expected_columns = [
        "property",
        "local_workstation",
        EXPECTED_REGIONS[0],
        EXPECTED_REGIONS[1],
    ]
    require_columns(frame, expected_columns, "Comment 13 comparison CSV")
    if list(frame.columns) != expected_columns:
        raise ValidationError(
            "Comment 13 comparison CSV columns are not in the frozen order"
        )
    if int(csv_record.get("row_count", -1)) != len(frame):
        raise ValidationError(
            "Comment 13 comparison CSV row count differs from its manifest"
        )
    if int(csv_record.get("row_count", -1)) != len(EXPECTED_PROPERTIES):
        raise ValidationError(
            "Comment 13 comparison CSV has an unexpected row count"
        )
    if csv_record.get("columns") != expected_columns:
        raise ValidationError(
            "Comment 13 comparison JSON records unexpected CSV columns"
        )
    if frame["property"].duplicated().any():
        raise ValidationError(
            "Comment 13 comparison CSV contains duplicate properties"
        )
    observed_properties = set(frame["property"].astype(str))
    if observed_properties != EXPECTED_PROPERTIES:
        raise ValidationError(
            "Comment 13 comparison CSV property set differs from the protocol"
        )

    for column in expected_columns[1:]:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValidationError(
                f"Comment 13 comparison CSV contains an empty {column} value"
            )

    if "Windows" not in _csv_value(
        frame, "operating_system", "local_workstation"
    ):
        raise ValidationError(
            "Comparison CSV does not identify the local Windows environment"
        )
    for region in EXPECTED_REGIONS:
        if "Linux" not in _csv_value(frame, "operating_system", region):
            raise ValidationError(
                f"Comparison CSV does not identify {region} as Linux"
            )
        if _csv_value(frame, "region", region) != region:
            raise ValidationError(
                f"Comparison CSV has the wrong {region} label"
            )

    equal_cloud_properties = {
        "execution_environment",
        "operating_system",
        "cpu_architecture",
        "compute_resources",
        "memory",
        "dependency_lock_sha256",
        "container_base_image",
        "final_container_image",
        "locale",
        "time_zone",
        "resource_or_power_mode",
        "benchmark_configuration",
    }
    for property_name in equal_cloud_properties:
        if _csv_value(
            frame, property_name, EXPECTED_REGIONS[0]
        ) != _csv_value(frame, property_name, EXPECTED_REGIONS[1]):
            raise ValidationError(
                f"Cloud columns differ for {property_name!r}"
            )

    if _csv_value(
        frame, "dependency_lock_sha256", "local_workstation"
    ) != local_lock_sha:
        raise ValidationError(
            "Comparison CSV local dependency-lock hash is incorrect"
        )
    for region in EXPECTED_REGIONS:
        if _csv_value(
            frame, "dependency_lock_sha256", region
        ) != cloud_lock_sha:
            raise ValidationError(
                f"Comparison CSV {region} dependency-lock hash is incorrect"
            )
        if _csv_value(
            frame, "final_container_image", region
        ) != final_image_uri:
            raise ValidationError(
                f"Comparison CSV {region} final image is incorrect"
            )
        if _csv_value(
            frame, "container_base_image", region
        ) != base_image_digest:
            raise ValidationError(
                f"Comparison CSV {region} base image is incorrect"
            )

    interpretation = _require_nonempty(
        comparison, "interpretation", "interpretation"
    ).lower()
    claim_boundary = _require_nonempty(
        comparison, "claim_boundary", "claim boundary"
    ).lower()
    for token in ["portability", "environment-consistency"]:
        if token not in interpretation:
            raise ValidationError(
                f"Comment 13 interpretation is missing {token!r}"
            )
    for token in [
        "multi-cloud",
        "distributed-systems performance",
        "autoscaling",
        "production reliability",
    ]:
        if token not in claim_boundary:
            raise ValidationError(
                f"Comment 13 claim boundary is missing {token!r}"
            )

    inventory = [
        evidence_file(
            project_dir,
            "comment13_environment_comparison",
            "comparison_json",
            comparison_path,
        ),
        evidence_file(
            project_dir,
            "comment13_environment_comparison",
            "comparison_csv",
            csv_path,
        ),
        evidence_file(
            project_dir,
            "comment13_environment_comparison",
            "source_environment_manifest",
            environment_path,
        ),
        evidence_file(
            project_dir,
            "comment13_environment_comparison",
            "local_dependency_lock",
            local_lock_path,
        ),
        evidence_file(
            project_dir,
            "comment13_environment_comparison",
            "cloud_dependency_lock",
            cloud_lock_path,
        ),
        evidence_file(
            project_dir,
            "comment13_environment_comparison",
            "container_inspection",
            inspection_path,
        ),
    ]
    for region in EXPECTED_REGIONS:
        inventory.append(
            evidence_file(
                project_dir,
                "comment13_environment_comparison",
                f"{region}_metadata",
                source_metadata_paths[region],
            )
        )

    result = {
        "comparison_complete": True,
        "local_environment": {
            "os": str(local["os"]),
            "version": str(local["version"]),
            "build": str(local["build"]),
            "architecture": str(local["architecture"]),
            "python_version": str(local["python_version"]),
            "locale": str(local["locale"]),
            "time_zone": str(local["time_zone"]),
            "utc_offset": str(local["utc_offset"]),
        },
        "cloud_environment": {
            "regions": list(EXPECTED_REGIONS),
            "os": str(container["os"]),
            "platform": str(container["platform"]),
            "architecture": str(container["architecture"]),
            "python_version": str(container["python_version"]),
            "locale": str(container["locale"]),
            "time_zone": str(container["time_zone"]),
            "utc_offset": str(container["utc_offset"]),
            "vcpus_per_region": 2,
            "memory_gib_per_region": 4,
        },
        "same_cloud_platform": True,
        "same_cloud_python_build": True,
        "same_cloud_container_image": True,
        "same_cloud_resource_configuration": True,
        "local_dependency_lock_sha256": local_lock_sha,
        "cloud_dependency_lock_sha256": cloud_lock_sha,
        "local_cloud_dependency_locks_identical": bool(identical_value),
        "base_image_digest_matches_comment12": True,
        "final_image_digest_matches_comment12": True,
        "comparison_csv": csv_path.relative_to(project_dir).as_posix(),
        "comparison_csv_sha256": sha256_file(csv_path),
        "comparison_json_sha256": sha256_file(comparison_path),
        "portability_claim_boundary_present": True,
    }
    return result, inventory
