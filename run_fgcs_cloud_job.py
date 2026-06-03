#!/usr/bin/env python3
"""
Cloud Run Jobs wrapper for the FGCS deterministic replay benchmark.

Runs the existing 360-condition benchmark and uploads selected raw outputs
to Google Cloud Storage under:

gs://<bucket>/fgcs_cloud_results/<run_id>/<region>/
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from google.cloud import storage


CORE_OUTPUTS = [
    "scaling_and_runtime_results.csv",
    "stage_latency_summary.csv",
    "determinism_hash_results.csv",
    "parallel_speedup_results.csv",
    "policy_ablation_costs.csv",
    "live_bc_predictions.csv",
]


def load_output_dir(config_path: str) -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Path(cfg.get("logging", {}).get("output_dir", "paper_outputs/fgcs_extended_benchmark"))


def upload_file(bucket, local_path: Path, blob_path: str) -> None:
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(local_path))
    print(f"[UPLOAD] {local_path} -> gs://{bucket.name}/{blob_path}", flush=True)


def main() -> None:
    bucket_name = os.environ["GCS_BUCKET"]
    region = os.environ.get("CLOUD_REGION", os.environ.get("CLOUD_RUN_REGION", "unknown-region"))
    run_id = os.environ.get("FGCS_RUN_ID", datetime.now(timezone.utc).strftime("fgcs_cloud_%Y%m%d_%H%M%S"))
    config_path = os.environ.get("CONFIG_PATH", "configs/fgcs_extended_benchmark.yaml")
    upload_traces = os.environ.get("UPLOAD_TRACES", "false").lower() == "true"

    output_dir = load_output_dir(config_path)

    print("[CLOUD] FGCS cloud benchmark starting", flush=True)
    print(f"[CLOUD] region={region}", flush=True)
    print(f"[CLOUD] run_id={run_id}", flush=True)
    print(f"[CLOUD] bucket={bucket_name}", flush=True)
    print(f"[CLOUD] config_path={config_path}", flush=True)
    print(f"[CLOUD] output_dir={output_dir}", flush=True)

    # Ensure clean outputs inside the container.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "run_fgcs_extended_benchmark.py", "--config", config_path],
        check=True,
    )
    end = time.perf_counter()

    metadata = {
        "run_id": run_id,
        "region": region,
        "bucket": bucket_name,
        "config_path": config_path,
        "output_dir": str(output_dir),
        "cloud_job_runtime_seconds": end - start,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "return_code": completed.returncode,
    }

    metadata_path = output_dir / "cloud_run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    prefix = f"fgcs_cloud_results/{run_id}/{region}"

    for fname in CORE_OUTPUTS:
        local_path = output_dir / fname
        if local_path.exists():
            upload_file(bucket, local_path, f"{prefix}/{fname}")
        else:
            print(f"[WARN] Missing expected output: {local_path}", flush=True)

    upload_file(bucket, metadata_path, f"{prefix}/cloud_run_metadata.json")

    if upload_traces:
        for trace_path in output_dir.glob("trace_*.csv"):
            upload_file(bucket, trace_path, f"{prefix}/traces/{trace_path.name}")

    print("[CLOUD] FGCS cloud benchmark complete", flush=True)


if __name__ == "__main__":
    main()