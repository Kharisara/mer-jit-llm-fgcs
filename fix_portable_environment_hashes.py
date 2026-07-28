from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent

REQUIREMENTS = ROOT / "requirements.txt"
LOCK_FILE = ROOT / "paper_outputs/environment/pip_freeze_lock.txt"
EXECUTION_MANIFEST = (
    ROOT / "paper_outputs/environment/execution_environment_manifest.json"
)
CLOUD_COMPARISON = (
    ROOT / "paper_outputs/environment/cloud_environment_comparison.json"
)


def normalise_lf(path: Path) -> None:
    """Rewrite a UTF-8 text file using LF line endings."""
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    path.write_text(text, encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_execution_reference(value: Any, new_hash: str) -> int:
    """
    Recursively update SHA-256 fields belonging to references to
    execution_environment_manifest.json.
    """
    updated = 0

    if isinstance(value, dict):
        references_execution_manifest = any(
            isinstance(item, str)
            and item.replace("\\", "/").endswith(
                "execution_environment_manifest.json"
            )
            for item in value.values()
        )

        if references_execution_manifest:
            for key in list(value):
                if key.lower() in {"sha256", "hash", "file_sha256"}:
                    value[key] = new_hash
                    updated += 1

        for child in value.values():
            updated += update_execution_reference(child, new_hash)

    elif isinstance(value, list):
        for child in value:
            updated += update_execution_reference(child, new_hash)

    return updated


for required_file in (
    REQUIREMENTS,
    LOCK_FILE,
    EXECUTION_MANIFEST,
    CLOUD_COMPARISON,
):
    if not required_file.exists():
        raise FileNotFoundError(required_file)

# First normalise the files whose hashes are recorded.
normalise_lf(REQUIREMENTS)
normalise_lf(LOCK_FILE)

requirements_hash = sha256(REQUIREMENTS)
lock_hash = sha256(LOCK_FILE)

execution_data = json.loads(
    EXECUTION_MANIFEST.read_text(encoding="utf-8-sig")
)

python_data = execution_data.get("python")
if not isinstance(python_data, dict):
    raise KeyError(
        "The execution manifest does not contain a 'python' object."
    )

python_data["requirements_txt_sha256"] = requirements_hash
python_data["pip_freeze_lock_sha256"] = lock_hash

EXECUTION_MANIFEST.write_text(
    json.dumps(execution_data, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
    newline="\n",
)

execution_manifest_hash = sha256(EXECUTION_MANIFEST)

cloud_data = json.loads(
    CLOUD_COMPARISON.read_text(encoding="utf-8-sig")
)

updated_references = update_execution_reference(
    cloud_data,
    execution_manifest_hash,
)

if updated_references == 0:
    raise RuntimeError(
        "Could not find the execution-manifest SHA-256 reference "
        "inside cloud_environment_comparison.json."
    )

CLOUD_COMPARISON.write_text(
    json.dumps(cloud_data, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
    newline="\n",
)

print(f"requirements.txt:                    {requirements_hash}")
print(f"pip_freeze_lock.txt:                 {lock_hash}")
print(f"execution_environment_manifest.json: {execution_manifest_hash}")
print(f"cloud references updated:            {updated_references}")
print(f"cloud_environment_comparison.json:   {sha256(CLOUD_COMPARISON)}")