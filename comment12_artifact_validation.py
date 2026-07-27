#!/usr/bin/env python3
"""Strict validation for Comment 12 reproducibility and traceability artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from final_artifact_utils import (
    ValidationError,
    evidence_file,
    read_json_required,
    relative_posix,
    sha256_file,
)


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValidationError(f"Comment 12 field {key!r} must be an object")
    return value


def _require_value(parent: dict[str, Any], key: str, role: str) -> Any:
    value = parent.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValidationError(f"Missing Comment 12 {role}: {key!r}")
    return value


def _require_sha256(value: Any, role: str, *, prefixed: bool = False) -> str:
    token = str(value).strip().lower()
    pattern = r"^sha256:[0-9a-f]{64}$" if prefixed else r"^[0-9a-f]{64}$"
    if re.fullmatch(pattern, token) is None:
        expected = "sha256:<64 hex>" if prefixed else "<64 hex>"
        raise ValidationError(f"Invalid {role}; expected {expected}")
    return token


def validate_comment12_artifacts(
    project_dir: Path,
) -> tuple[dict[str, Any], list[Any]]:
    """Validate the environment, lock, and Figure 2 derivation artifacts."""

    environment_manifest_path = (
        project_dir
        / "paper_outputs"
        / "environment"
        / "execution_environment_manifest.json"
    )
    environment_manifest = read_json_required(
        environment_manifest_path,
        "Comment 12 execution-environment manifest",
    )
    if not isinstance(environment_manifest, dict):
        raise ValidationError("Execution-environment manifest must be a JSON object")

    local = _require_mapping(environment_manifest, "local")
    python_env = _require_mapping(environment_manifest, "python")
    threads = _require_mapping(environment_manifest, "threads")
    container = _require_mapping(environment_manifest, "container")

    local_fields = [
        "os_caption",
        "os_version",
        "os_build_number",
        "os_architecture",
        "cpu_name",
        "physical_cores",
        "logical_processors",
        "installed_ram_bytes",
        "installed_ram_gib",
        "storage_devices",
        "active_power_scheme",
    ]
    for key in local_fields:
        _require_value(local, key, "local-environment field")

    storage_devices = local["storage_devices"]
    if not isinstance(storage_devices, list) or not storage_devices:
        raise ValidationError("storage_devices must contain at least one device")
    for index, device in enumerate(storage_devices):
        if not isinstance(device, dict):
            raise ValidationError(f"storage_devices[{index}] must be an object")
        for key in ["friendly_name", "media_type", "bus_type", "size_bytes"]:
            _require_value(device, key, f"storage_devices[{index}] field")

    python_fields = [
        "version",
        "executable",
        "executable_sha256",
        "requirements_txt_sha256",
        "pip_freeze_lock",
        "pip_freeze_lock_sha256",
    ]
    for key in python_fields:
        _require_value(python_env, key, "Python-environment field")

    executable_sha = _require_sha256(
        python_env["executable_sha256"],
        "Python executable SHA-256",
    )
    requirements_sha = _require_sha256(
        python_env["requirements_txt_sha256"],
        "requirements.txt SHA-256",
    )
    lock_sha = _require_sha256(
        python_env["pip_freeze_lock_sha256"],
        "dependency-lock SHA-256",
    )

    requirements_path = project_dir / "requirements.txt"
    if not requirements_path.is_file():
        raise ValidationError(f"Missing requirements.txt: {requirements_path}")
    if sha256_file(requirements_path) != requirements_sha:
        raise ValidationError(
            "requirements.txt SHA-256 does not match the environment manifest"
        )

    lock_path = project_dir / str(python_env["pip_freeze_lock"])
    if not lock_path.is_file():
        raise ValidationError(f"Missing dependency lock: {lock_path}")
    if sha256_file(lock_path) != lock_sha:
        raise ValidationError(
            "Dependency-lock SHA-256 does not match the environment manifest"
        )

    executable_path = Path(str(python_env["executable"]))
    if executable_path.is_file() and sha256_file(executable_path) != executable_sha:
        raise ValidationError(
            "Current Python executable SHA-256 does not match the environment manifest"
        )

    thread_fields = [
        "logical_cpu_count",
        "environment",
        "torch_num_threads",
        "torch_num_interop_threads",
        "torch_version",
        "torch_cuda_available",
    ]
    for key in thread_fields:
        if key not in threads:
            raise ValidationError(f"Missing thread/process field: {key}")

    thread_environment = threads["environment"]
    if not isinstance(thread_environment, dict):
        raise ValidationError("threads.environment must be an object")
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PYTHONHASHSEED",
    ]:
        if key not in thread_environment:
            raise ValidationError(f"Missing thread environment field: {key}")

    base_image = str(
        _require_value(container, "dockerfile_base_image", "Docker base image")
    ).strip()
    base_digest = str(
        _require_value(container, "base_image_repo_digest", "base-image digest")
    ).strip().lower()
    final_image_uri = str(
        _require_value(container, "final_image_uri", "final-image URI")
    ).strip()
    final_image_digest = _require_sha256(
        _require_value(container, "final_image_digest", "final-image digest"),
        "final-image digest",
        prefixed=True,
    )

    if re.fullmatch(r".+@sha256:[0-9a-f]{64}", base_digest) is None:
        raise ValidationError(
            "base_image_repo_digest must use repository@sha256:<64 hex>"
        )
    expected_base_repo = base_image.split(":", 1)[0] + "@sha256:"
    if not base_digest.startswith(expected_base_repo):
        raise ValidationError(
            "Base-image digest repository does not match dockerfile_base_image"
        )
    if not final_image_uri.endswith("@" + final_image_digest):
        raise ValidationError(
            "final_image_uri is not pinned to final_image_digest"
        )

    figure_manifest_path = (
        project_dir / "figures" / "Figure_2_Runtime_Scaling_manifest.json"
    )
    figure_manifest = read_json_required(
        figure_manifest_path,
        "Figure 2 generation manifest",
    )
    if not isinstance(figure_manifest, dict):
        raise ValidationError("Figure 2 generation manifest must be a JSON object")

    source_path = project_dir / str(
        _require_value(figure_manifest, "input", "Figure 2 source path")
    )
    source_sha = _require_sha256(
        _require_value(
            figure_manifest,
            "input_sha256",
            "Figure 2 source SHA-256",
        ),
        "Figure 2 source SHA-256",
    )
    if not source_path.is_file():
        raise ValidationError(f"Missing Figure 2 source: {source_path}")
    if sha256_file(source_path) != source_sha:
        raise ValidationError(
            "Figure 2 source SHA-256 does not match its generation manifest"
        )

    outputs = figure_manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValidationError("Figure 2 outputs field must be an object")

    expected_output_suffixes = {".png", ".pdf"}
    observed_output_suffixes = {
        Path(str(relative)).suffix.lower() for relative in outputs
    }
    if observed_output_suffixes != expected_output_suffixes:
        raise ValidationError(
            "Figure 2 manifest must contain exactly one PNG and one PDF output"
        )

    output_paths: list[Path] = []
    for relative, expected_digest in outputs.items():
        output_path = project_dir / str(relative)
        expected_sha = _require_sha256(
            expected_digest,
            f"Figure 2 output SHA-256 for {relative}",
        )
        if not output_path.is_file():
            raise ValidationError(f"Missing Figure 2 output: {output_path}")
        if sha256_file(output_path) != expected_sha:
            raise ValidationError(
                f"Figure 2 output SHA-256 mismatch: {relative}"
            )
        output_paths.append(output_path)

    inventory = [
        evidence_file(
            project_dir,
            "comment12_traceability",
            "execution_environment_manifest",
            environment_manifest_path,
        ),
        evidence_file(
            project_dir,
            "comment12_traceability",
            "dependency_lock",
            lock_path,
        ),
        evidence_file(
            project_dir,
            "comment12_traceability",
            "requirements",
            requirements_path,
        ),
        evidence_file(
            project_dir,
            "comment12_traceability",
            "figure2_generation_manifest",
            figure_manifest_path,
        ),
        evidence_file(
            project_dir,
            "comment12_traceability",
            "figure2_source",
            source_path,
        ),
    ]
    for output_path in sorted(output_paths):
        inventory.append(
            evidence_file(
                project_dir,
                "comment12_traceability",
                "figure2_output",
                output_path,
            )
        )

    result = {
        "environment_manifest_complete": True,
        "environment_manifest_sha256": sha256_file(
            environment_manifest_path
        ),
        "dependency_lock_hash_matches": True,
        "requirements_hash_matches": True,
        "python_executable_sha256": executable_sha,
        "base_image": base_image,
        "base_image_digest": base_digest,
        "final_image_uri": final_image_uri,
        "final_image_digest": final_image_digest,
        "figure2_source": relative_posix(source_path, project_dir),
        "figure2_source_hash_matches": True,
        "figure2_outputs": [
            relative_posix(path, project_dir) for path in sorted(output_paths)
        ],
        "figure2_output_hashes_match": True,
    }
    return result, inventory
