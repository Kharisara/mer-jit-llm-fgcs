#!/usr/bin/env python3
"""ReplayBench-PG R1: fresh cloud execution and label-independent full-model validation.

Do not replace or edit the frozen manuscript experiment. This wrapper runs the
repository's existing receipt-enabled execution, generic validation and scoring
scripts. Run locally for a small PILOT; run FULL inside a cloud container.

The collector and deterministic downstream stub remain in the same process;
cloud deployment does NOT constitute external-service attestation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NEEDED = [
    "configs/execution_integrity_validation.yaml",
    "paper_outputs/replay_input_v260.csv",
    "run_execution_integrity_validation.py",
    "run_phase1_label_independent_validation.py",
    "score_phase1_label_independent_validation.py",
    "run_fgcs_extended_benchmark.py",
    "replaybench/generic_validator.py",
    "replaybench/integrity.py",
    "replaybench/validation.py",
    "replaybench/workload.py",
]
RECEIPT_MODES = [
    "clean", "unlogged_downstream_call", "false_execution_log",
    "duplicate_downstream_call", "mismatched_correlation_id",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def preflight(repo: Path) -> dict[str, Any]:
    missing = [name for name in NEEDED if not (repo / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing repository files:\n" + "\n".join(missing))
    # Load the actual repo's config; never adjust the submitted artifact itself.
    import yaml
    config = yaml.safe_load((repo / NEEDED[0]).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config YAML is invalid")
    actual = config["benchmark"]
    expected = {"policy_modes": ["risk_proxy", "random", "always"], "seeds": [1, 2, 3], "workers": [1, 4]}
    for key, value in expected.items():
        if list(actual[key]) != value:
            raise RuntimeError(f"Expected frozen base {key}={value}; got {actual[key]}")
    if not bool(config.get("execution_receipts", {}).get("enabled", False)):
        raise RuntimeError("Receipt evidence is disabled in the input config")
    if config["dataset"].get("fractions") != [1.0]:
        raise RuntimeError("Full experiment requires dataset fraction [1.0]")
    code = {name: sha(repo / name) for name in NEEDED if name.endswith(".py")}
    code[NEEDED[0]] = sha(repo / NEEDED[0])
    commit = "unavailable"
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                                capture_output=True, check=True)
        commit = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    build_ref_path = repo / "array_r1_source_commit.txt"
    build_ref = build_ref_path.read_text(encoding="utf-8").strip() if build_ref_path.is_file() else None
    if build_ref and os.environ.get("ARRAY_R1_SOURCE_COMMIT") and build_ref != os.environ["ARRAY_R1_SOURCE_COMMIT"]:
        raise RuntimeError("Build source commit disagrees with ARRAY_R1_SOURCE_COMMIT")
    return {"input_csv_sha256": sha(repo / "paper_outputs/replay_input_v260.csv"),
            "source_files_sha256": code, "git_head_seen_inside_environment": commit,
            "source_commit_supplied_at_build": build_ref,
            "source_commit_env": os.environ.get("ARRAY_R1_SOURCE_COMMIT", ""),
            "python": sys.version, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "cloud_region": os.environ.get("CLOUD_REGION", ""),
            "container_image": os.environ.get("ARRAY_R1_CONTAINER_IMAGE", ""),
            "generated_at_utc": datetime.now(timezone.utc).isoformat()}


def run(args: list[str], repo: Path) -> None:
    print("[COMMAND]", " ".join(args), flush=True)
    subprocess.run(args, cwd=str(repo), check=True)


def make_config(repo: Path, output: Path, pilot: bool) -> Path:
    import yaml
    cfg = yaml.safe_load((repo / NEEDED[0]).read_text(encoding="utf-8"))
    cfg.setdefault("logging", {})["output_dir"] = str((output / "fresh_executions").resolve())
    if pilot:
        cfg["dataset"]["fractions"] = [0.1]
        cfg["benchmark"]["policy_modes"] = ["risk_proxy"]
        cfg["benchmark"]["seeds"] = [1]
        cfg["benchmark"]["workers"] = [1]
        cfg.setdefault("execution_integrity", {})["receipt_fault_probability"] = 0.08
    path = output / "effective_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def pilot_check(output: Path) -> dict[str, Any]:
    """Check that receipt evidence from fresh PILOT executions is observable."""
    import pandas as pd
    from replaybench.generic_validator import (prepare_reference_artifacts,
        sanitize_trace_for_validation, validate_observed_artifacts)
    raw = output / "fresh_executions"
    tok = "policy_risk_proxy_seed_1_workers_1"

    def load(mode: str):
        d = raw / mode
        trace = pd.read_csv(d / f"trace_{tok}.csv")
        receipts = pd.read_csv(d / f"execution_receipts_{tok}.csv")
        manifest = json.loads((d / f"trace_manifest_{tok}.json").read_text(encoding="utf-8"))
        return sanitize_trace_for_validation(trace), receipts, manifest["config_manifest"]

    reference_trace, reference_receipts, ref_config = load("clean")
    prepared = prepare_reference_artifacts(reference_trace=reference_trace,
        reference_receipts=reference_receipts, reference_config_manifest=ref_config)
    results: list[dict[str, Any]] = []
    for mode in RECEIPT_MODES:
        trace, receipts, cfg = load(mode)
        findings = validate_observed_artifacts(
            evidence_id=f"pilot-{mode}", observed_trace=trace,
            observed_receipts=receipts, observed_config_manifest=cfg,
            reference_trace=reference_trace, reference_receipts=reference_receipts,
            reference_config_manifest=ref_config, prepared_reference=prepared)
        if bool(findings.full_validator_triggered) != (mode != "clean"):
            raise RuntimeError(f"Pilot validator unexpected finding: {mode}: {findings.to_dict()}")
        results.append({"mode": mode, "rows": len(trace), "receipts": len(receipts),
                        "primary": int(findings.primary_validator_triggered),
                        "full": int(findings.full_validator_triggered),
                        "receipt_validation_passed": int(findings.receipt_validation_passed)})
    dump(output / "pilot_validator_results.json", results)
    print("[PASS] Pilot: 1 clean, 4 fresh receipt-fault executions; validator decisions correct", flush=True)
    return {"pilot_validator_results": results}


def full_check(output: Path) -> dict[str, Any]:
    import pandas as pd
    raw = output / "fresh_executions"
    score = output / "label_excluded_validation"
    ex = pd.read_csv(raw / "execution_receipt_validation_per_run.csv")
    scored = pd.read_csv(score / "per_evidence_scored_results.csv")
    if len(ex) != 90:
        raise RuntimeError(f"Expected 90 newly executed receipt runs, found {len(ex)}")
    if len(scored) != 270:
        raise RuntimeError(f"Expected 270 evidence units, found {len(scored)}")
    if ex["detected_as_expected"].astype(int).sum() != 90:
        raise RuntimeError("Execution integrity runner did not pass all 90 conditions")
    pos = scored[scored["positive_control"].astype(int) == 1]
    neg = scored[scored["positive_control"].astype(int) == 0]
    if (len(pos), len(neg)) != (228, 42):
        raise RuntimeError(f"Unexpected evidence units: positives={len(pos)} negatives={len(neg)}")
    if not pos["full_validator_triggered"].astype(int).eq(1).all():
        raise RuntimeError("Full validator missed at least one positive control")
    if not neg["full_validator_triggered"].astype(int).eq(0).all():
        raise RuntimeError("Full validator flagged at least one negative control")
    comparison = scored.groupby(["fault_mode", "source_kind", "positive_control"], as_index=False).agg(
        units=("evidence_id", "size"), primary_detected=("primary_validator_triggered", "sum"),
        full_detected=("full_validator_triggered", "sum"), event_tp=("event_true_positives", "sum"),
        event_fp=("event_false_positives", "sum"), event_fn=("event_false_negatives", "sum"))
    comparison.to_csv(output / "cloud_comparison_by_mode.csv", index=False)
    policy = scored.groupby(["policy_mode", "positive_control"], as_index=False).agg(
        units=("evidence_id", "size"), primary_detected=("primary_validator_triggered", "sum"),
        full_detected=("full_validator_triggered", "sum"))
    policy.to_csv(output / "cloud_comparison_by_policy.csv", index=False)
    summary = {"fresh_receipt_execution_instances": len(ex),
               "fresh_clean_execution_instances": int((ex["fault_mode"] == "clean").sum()),
               "fresh_receipt_fault_execution_instances": int((ex["fault_mode"] != "clean").sum()),
               "label_excluded_evidence_units": len(scored), "positive_units": len(pos), "negative_units": len(neg),
               "primary_positive_detections": int(pos["primary_validator_triggered"].astype(int).sum()),
               "full_positive_detections": int(pos["full_validator_triggered"].astype(int).sum()),
               "full_false_alarms": int(neg["full_validator_triggered"].astype(int).sum()),
               "localized_events_tp": int(scored["event_true_positives"].astype(int).sum()),
               "localized_events_fp": int(scored["event_false_positives"].astype(int).sum()),
               "localized_events_fn": int(scored["event_false_negatives"].astype(int).sum())}
    if summary["localized_events_fp"] or summary["localized_events_fn"]:
        raise RuntimeError("Event localization mismatch")
    print("[PASS] Full: fresh cloud receipt executions=90; label-excluded evidence units=270; "
          f"full positive={summary['full_positive_detections']}/228; false alarms={summary['full_false_alarms']}/42", flush=True)
    return summary


def manifest_files(output: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "file_integrity_manifest.json":
            result.append({"path": path.relative_to(output).as_posix(),
                           "bytes": path.stat().st_size, "sha256": sha(path)})
    return result


def archive_output(output: Path) -> tuple[Path, str]:
    zip_path = output.parent / (output.name + ".zip")
    if zip_path.exists():
        raise FileExistsError(f"Archive already exists: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=5, allowZip64=True) as z:
        for p in sorted(output.rglob("*")):
            if p.is_file():
                z.write(p, arcname=p.relative_to(output).as_posix())
    return zip_path, sha(zip_path)


def upload_gcs(bucket: str, prefix: str, filename: Path, checksum: str) -> str:
    from google.cloud import storage
    key = prefix.strip("/") + "/" + filename.name
    blob = storage.Client().bucket(bucket).blob(key)
    blob.metadata = {"sha256": checksum}
    blob.upload_from_filename(str(filename), timeout=1200)
    if blob.size != filename.stat().st_size:
        raise RuntimeError("GCS uploaded object size does not match local archive")
    return f"gs://{bucket}/{key}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["preflight", "pilot", "full"], required=True)
    ap.add_argument("--bucket", default=os.environ.get("GCS_BUCKET", ""))
    ap.add_argument("--gcs-prefix", default=os.environ.get("ARRAY_R1_GCS_PREFIX", "array_r1_cloud_full_model"))
    ns = ap.parse_args()
    repo = Path(ns.repo).resolve()
    out = Path(ns.out).resolve()
    protected = (repo / "paper_outputs").resolve()
    if out == protected or protected in out.parents:
        raise ValueError("Refusing to write inside paper_outputs; select a separate R1 output folder")
    if not repo.is_dir():
        raise NotADirectoryError(repo)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output folder is not empty; use a new --out: {out}")
    out.mkdir(parents=True, exist_ok=True)
    try:
        before = time.perf_counter()
        source = preflight(repo)
        dump(out / "environment_and_provenance.json", source)
        print("[PREFLIGHT] Base grid and required files validated; input SHA256="
              + source["input_csv_sha256"], flush=True)
        if ns.mode == "preflight":
            dump(out / "run_summary.json", {"mode": "preflight", "status": "passed"})
            print("[DONE] Cloud R1 preflight passed", flush=True)
            return
        cfg = make_config(repo, out, ns.mode == "pilot")
        run([sys.executable, "run_execution_integrity_validation.py", "--config", str(cfg)], repo)
        if ns.mode == "pilot":
            metrics = pilot_check(out)
        else:
            val = out / "label_excluded_validation"
            run([sys.executable, "run_phase1_label_independent_validation.py",
                 "--input-dir", str(out / "fresh_executions"), "--output-dir", str(val)], repo)
            run([sys.executable, "score_phase1_label_independent_validation.py", "--input-dir", str(val)], repo)
            metrics = full_check(out)
        summary = {"mode": ns.mode, "status": "passed", "metrics": metrics,
                   "duration_seconds": time.perf_counter() - before,
                   "scope": "fresh in-container replay with in-process stub receipts; not independent external attestation",
                   "trust_boundary": "in-process stub/receipt producer and trusted clean reference inside the container"}
        dump(out / "run_summary.json", summary)
        dump(out / "file_integrity_manifest.json", manifest_files(out))
        archive, check = archive_output(out)
        print(f"[ARCHIVE] {archive}", flush=True)
        print(f"[SHA256] {check}", flush=True)
        if ns.bucket:
            region = os.environ.get("CLOUD_REGION", "unspecified")
            run_id = os.environ.get("ARRAY_R1_RUN_ID", "manual")
            target = "/".join([ns.gcs_prefix.strip("/"), run_id, region])
            uri = upload_gcs(ns.bucket, target, archive, check)
            print(f"[GCS] {uri}", flush=True)
        print(f"[DONE] Cloud R1 {ns.mode} validation completed", flush=True)
    except BaseException:
        print(f"[FAIL] Cloud R1 {ns.mode} validation did not complete", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
