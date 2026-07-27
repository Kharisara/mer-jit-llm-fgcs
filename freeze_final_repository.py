#!/usr/bin/env python3
"""Freeze and package a validated ReplayBench-PG Git repository.

The script intentionally does not commit, push, or publish anything. It refuses
to package a dirty working tree, verifies the final validation SHA manifest,
requires all seven manuscript tables, and creates a deterministic ZIP from the
files tracked at the current Git commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from final_artifact_utils import (
    ValidationError,
    read_json_required,
    relative_posix,
    sha256_file,
    verify_sha256_manifest,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REQUIRED_TABLES = [
    "primary_benchmark_summary.csv",
    "timing_worker_scaling.csv",
    "ray_validation_summary.csv",
    "primary_fault_summary.csv",
    "metropt3_validation_summary.csv",
    "cloud_validation_summary.csv",
    "final_results_overview.csv",
]


def run_git(project_dir: Path, *args: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr if text else completed.stderr.decode("utf-8", errors="replace")
        raise ValidationError(f"git {' '.join(args)} failed: {stderr.strip()}")
    return completed.stdout


def zip_timestamp_from_git(project_dir: Path) -> tuple[int, int, int, int, int, int]:
    token = str(run_git(project_dir, "show", "-s", "--format=%ct", "HEAD")).strip()
    timestamp = int(token)
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    # ZIP timestamps cannot be earlier than 1980 and use two-second resolution.
    year = max(dt.year, 1980)
    second = dt.second - (dt.second % 2)
    return (year, dt.month, dt.day, dt.hour, dt.minute, second)


def tracked_files(project_dir: Path) -> list[Path]:
    output = run_git(project_dir, "ls-files", "-z", text=False)
    assert isinstance(output, bytes)
    paths = []
    for token in output.split(b"\0"):
        if not token:
            continue
        relative = token.decode("utf-8", errors="strict")
        path = project_dir / relative
        if not path.is_file():
            raise ValidationError(f"Tracked file is missing from the working tree: {relative}")
        paths.append(path)
    if not paths:
        raise ValidationError("Git returned no tracked files")
    return sorted(paths, key=lambda path: relative_posix(path, project_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and package the frozen ReplayBench-PG repository."
    )
    parser.add_argument("--project-dir", default=str(SCRIPT_DIR))
    parser.add_argument(
        "--validation-dir", default="paper_outputs/final_validation"
    )
    parser.add_argument(
        "--tables-dir", default="paper_outputs/final_manuscript_tables"
    )
    parser.add_argument("--output-dir", default="release")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Package a dirty tree. Not recommended for the final archival release.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    validation_dir = (project_dir / args.validation_dir).resolve()
    tables_dir = (project_dir / args.tables_dir).resolve()
    output_dir = (project_dir / args.output_dir).resolve()

    inside = str(run_git(project_dir, "rev-parse", "--is-inside-work-tree")).strip()
    if inside != "true":
        raise ValidationError(f"Not a Git working tree: {project_dir}")

    status = str(run_git(project_dir, "status", "--porcelain=v1")).strip()
    if status and not args.allow_dirty:
        raise ValidationError(
            "The working tree is not clean. Commit the finalized code and generated "
            "validation/table outputs before packaging.\n" + status
        )

    manifest_path = validation_dir / "final_validation_manifest.json"
    manifest = read_json_required(manifest_path, "final validation manifest")
    if not isinstance(manifest, dict) or manifest.get("status") != "passed":
        raise ValidationError("The final validation manifest does not have status='passed'")

    sha_manifest_path = validation_dir / "file_sha256_manifest.csv"
    hash_verification = verify_sha256_manifest(project_dir, sha_manifest_path)

    missing_tables = [name for name in REQUIRED_TABLES if not (tables_dir / name).is_file()]
    if missing_tables:
        raise ValidationError(f"Final manuscript tables are missing: {missing_tables}")

    commit = str(run_git(project_dir, "rev-parse", "HEAD")).strip()
    short_commit = commit[:12]
    branch = str(run_git(project_dir, "rev-parse", "--abbrev-ref", "HEAD")).strip()
    commit_iso = str(run_git(project_dir, "show", "-s", "--format=%cI", "HEAD")).strip()
    files = tracked_files(project_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"ReplayBench-PG-final-{short_commit}.zip"
    archive_sha_path = output_dir / f"ReplayBench-PG-final-{short_commit}.zip.sha256"
    date_time = zip_timestamp_from_git(project_dir)

    table_hashes = {
        name: sha256_file(tables_dir / name)
        for name in REQUIRED_TABLES
    }
    metadata: dict[str, Any] = {
        "archive_format": "ReplayBench-PG frozen Git working tree",
        "git_commit": commit,
        "git_short_commit": short_commit,
        "git_branch": branch,
        "git_commit_time": commit_iso,
        "working_tree_clean": not bool(status),
        "validation_manifest": relative_posix(manifest_path, project_dir),
        "validation_manifest_sha256": sha256_file(manifest_path),
        "file_sha256_manifest": relative_posix(sha_manifest_path, project_dir),
        "verified_hash_files": hash_verification["verified_files"],
        "final_manuscript_table_sha256": table_hashes,
        "tracked_files_packaged": len(files),
    }

    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(archive_path, "w", compression=compression, compresslevel=9) as archive:
        for path in files:
            relative = relative_posix(path, project_dir)
            info = zipfile.ZipInfo(relative, date_time=date_time)
            info.compress_type = compression
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes(), compress_type=compression, compresslevel=9)

        metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
        info = zipfile.ZipInfo("FREEZE_METADATA.json", date_time=date_time)
        info.compress_type = compression
        info.external_attr = 0o100644 << 16
        archive.writestr(info, metadata_bytes, compress_type=compression, compresslevel=9)

    archive_sha = sha256_file(archive_path)
    archive_sha_path.write_text(
        f"{archive_sha}  {archive_path.name}\n",
        encoding="utf-8",
        newline="\n",
    )

    print("[DONE] Frozen repository package created")
    print(f"[COMMIT] {commit}")
    print(f"[FILES] {len(files)} tracked files + FREEZE_METADATA.json")
    print(f"[OUT] {archive_path}")
    print(f"[OUT] {archive_sha_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
