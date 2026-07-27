from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

INTEGRITY_SCHEMA_VERSION = "replaybench-pg-integrity-v1"
RECEIPT_SCHEMA_VERSION = "replaybench-pg-receipt-v1"


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for integrity hashing."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


@lru_cache(maxsize=64)
def file_sha256(path: str) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or "unavailable"
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _safe_file_hash(path_value: Any) -> str:
    if path_value in (None, ""):
        return ""
    path = Path(str(path_value))
    return file_sha256(str(path.resolve())) if path.is_file() else ""


def canonical_workload_hash(frame: pd.DataFrame) -> str:
    """Hash the ordered replay workload while avoiding platform-specific paths."""
    stable_columns = [
        column
        for column in (
            "source_record_id",
            "utterance_id",
            "timestamp",
            "label",
            "split",
            "Dialogue_ID",
            "Utterance_ID",
        )
        if column in frame.columns
    ]
    records: list[dict[str, Any]] = []
    for position, row in enumerate(frame.to_dict(orient="records")):
        record = {"trace_position": int(position)}
        for column in stable_columns:
            value = row.get(column, "")
            if pd.isna(value):
                value = ""
            record[column] = value
        records.append(record)
    return sha256_json(records)


def build_run_identity(
    *,
    frame: pd.DataFrame,
    cfg: Mapping[str, Any],
    policy_mode: str,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    dataset_cfg = cfg.get("dataset", {})
    policy_cfg = cfg.get("policy", {})
    generation_cfg = cfg.get("generation_stub", {})
    execution_cfg = cfg.get("execution_semantics", {})

    input_path = str(dataset_cfg.get("input_csv", ""))
    checkpoint_path = str(policy_cfg.get("bc_model_path", ""))
    bc_action_path = str(policy_cfg.get("bc_action_csv", ""))

    manifest = {
        "schema_version": INTEGRITY_SCHEMA_VERSION,
        "workload_hash": canonical_workload_hash(frame),
        "workload_rows": int(len(frame)),
        "input_file_sha256": _safe_file_hash(input_path),
        "policy_mode": str(policy_mode),
        "seed": int(seed),
        "workers": int(workers),
        "code_commit": git_commit(),
        "checkpoint_sha256": (
            _safe_file_hash(checkpoint_path)
            if str(policy_mode) == "bc_live"
            else ""
        ),
        "bc_action_sha256": (
            _safe_file_hash(bc_action_path)
            if str(policy_mode) == "bc"
            else ""
        ),
        "generation_mode": str(generation_cfg.get("mode", "affective_response")),
        "downstream_operation": str(
            generation_cfg.get("service_name", "generation_stub")
        ),
        "task_retry_enabled": bool(
            execution_cfg.get("task_retry_enabled", False)
        ),
        "task_failure_policy": str(
            execution_cfg.get("task_failure_policy", "fail_fast")
        ),
    }
    manifest_hash = sha256_json(manifest)
    trace_scope_manifest = {
        key: value for key, value in manifest.items() if key != "workers"
    }
    trace_scope_hash = sha256_json(trace_scope_manifest)
    return {
        "run_id": f"rbpg-{manifest_hash[:24]}",
        "trace_scope_id": f"rbpg-trace-{trace_scope_hash[:24]}",
        "trace_scope_hash": trace_scope_hash,
        "config_manifest_hash": manifest_hash,
        "config_manifest": manifest,
    }


def make_replay_point_id(row_index: int, row: Mapping[str, Any]) -> str:
    source_id = row.get(
        "source_record_id",
        row.get("utterance_id", row.get("Utterance_ID", row_index)),
    )
    return f"{int(row_index)}:{source_id}"


def make_correlation_id(run_id: str, replay_point_id: str) -> str:
    return sha256_text(f"{run_id}|{replay_point_id}")[:32]


@dataclass(frozen=True)
class ReceiptContext:
    run_id: str
    replay_point_id: str
    correlation_id: str
    downstream_operation: str


class DownstreamReceiptCollector:
    """Thread-safe receipt log owned by the downstream component."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event_index = 0
        self._attempts: dict[tuple[str, str], int] = {}
        self._rows: list[dict[str, Any]] = []

    def emit(self, context: ReceiptContext) -> dict[str, Any]:
        with self._lock:
            self._event_index += 1
            event_index = self._event_index
            attempt_key = (context.run_id, context.replay_point_id)
            attempt_index = self._attempts.get(attempt_key, 0) + 1
            self._attempts[attempt_key] = attempt_index

            semantic = {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "run_id": context.run_id,
                "replay_point_id": context.replay_point_id,
                "correlation_id": context.correlation_id,
                "downstream_operation": context.downstream_operation,
                "attempt_index": int(attempt_index),
                "execution_status": "executed",
            }
            row = {
                **semantic,
                "receipt_event_index": int(event_index),
                "execution_timestamp_ns": int(time.time_ns()),
                "execution_monotonic_ns": int(time.monotonic_ns()),
                "receipt_digest": sha256_json(semantic),
            }
            self._rows.append(row)
            return dict(row)

    def to_frame(self) -> pd.DataFrame:
        with self._lock:
            return pd.DataFrame([dict(row) for row in self._rows])


def reconcile_execution_receipts(
    trace_df: pd.DataFrame,
    receipt_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_trace = {
        "run_id",
        "replay_point_id",
        "correlation_id",
        "authorized_to_generate",
        "generation_invoked",
    }
    missing = required_trace - set(trace_df.columns)
    if missing:
        raise ValueError(f"Trace is missing receipt-validation columns: {sorted(missing)}")

    receipts = receipt_df.copy()
    if receipts.empty:
        receipts = pd.DataFrame(
            columns=[
                "run_id",
                "replay_point_id",
                "correlation_id",
                "receipt_digest",
            ]
        )

    details: list[dict[str, Any]] = []
    trace_point_ids = set(trace_df["replay_point_id"].astype(str))

    for trace_position, row in trace_df.reset_index(drop=True).iterrows():
        run_id = str(row["run_id"])
        replay_point_id = str(row["replay_point_id"])
        correlation_id = str(row["correlation_id"])
        point_receipts = receipts.loc[
            receipts["run_id"].astype(str).eq(run_id)
            & receipts["replay_point_id"].astype(str).eq(replay_point_id)
        ].copy()
        matching = point_receipts.loc[
            point_receipts["correlation_id"].astype(str).eq(correlation_id)
        ]

        receipt_count = int(len(point_receipts))
        matching_count = int(len(matching))
        authorized = int(row["authorized_to_generate"])
        primary_logged_execution = int(row["generation_invoked"])

        receipt_digests = sorted(
            point_receipts.get("receipt_digest", pd.Series(dtype=str))
            .astype(str)
            .tolist()
        )
        receipt_state_hashes = sorted(
            sha256_json(
                {
                    "replay_point_id": str(receipt.get("replay_point_id", "")),
                    "downstream_operation": str(
                        receipt.get("downstream_operation", "")
                    ),
                    "attempt_index": int(receipt.get("attempt_index", 0)),
                    "execution_status": str(
                        receipt.get("execution_status", "")
                    ),
                }
            )
            for receipt in point_receipts.to_dict(orient="records")
        )

        missing_receipt = int(primary_logged_execution == 1 and matching_count == 0)
        unlogged_downstream_call = int(primary_logged_execution == 0 and receipt_count > 0)
        duplicate_downstream_call = int(receipt_count > 1)
        mismatched_correlation_id = int(receipt_count > matching_count)
        unauthorized_downstream_call = int(authorized == 0 and receipt_count > 0)

        violation_count = sum(
            [
                missing_receipt,
                unlogged_downstream_call,
                duplicate_downstream_call,
                mismatched_correlation_id,
                unauthorized_downstream_call,
            ]
        )

        details.append(
            {
                "trace_position": int(trace_position),
                "run_id": run_id,
                "replay_point_id": replay_point_id,
                "correlation_id": correlation_id,
                "authorized_to_generate": authorized,
                "primary_logged_execution": primary_logged_execution,
                "receipt_count": receipt_count,
                "matching_receipt_count": matching_count,
                "receipt_digest_set": "|".join(receipt_digests),
                "receipt_state_hash_set": "|".join(receipt_state_hashes),
                "missing_receipt": missing_receipt,
                "unlogged_downstream_call": unlogged_downstream_call,
                "duplicate_downstream_call": duplicate_downstream_call,
                "mismatched_correlation_id": mismatched_correlation_id,
                "unauthorized_downstream_call": unauthorized_downstream_call,
                "receipt_consistent": int(violation_count == 0),
            }
        )

    detail_df = pd.DataFrame(details)
    orphan_receipts = receipts.loc[
        ~receipts["replay_point_id"].astype(str).isin(trace_point_ids)
    ]

    summary = {
        "trace_rows": int(len(trace_df)),
        "receipt_rows": int(len(receipts)),
        "orphan_receipts": int(len(orphan_receipts)),
        "missing_receipts": int(detail_df["missing_receipt"].sum()),
        "unlogged_downstream_calls": int(
            detail_df["unlogged_downstream_call"].sum()
        ),
        "duplicate_downstream_calls": int(
            detail_df["duplicate_downstream_call"].sum()
        ),
        "mismatched_correlation_ids": int(
            detail_df["mismatched_correlation_id"].sum()
        ),
        "unauthorized_downstream_calls": int(
            detail_df["unauthorized_downstream_call"].sum()
        ),
    }
    summary["receipt_validation_passed"] = int(
        summary["orphan_receipts"] == 0
        and summary["missing_receipts"] == 0
        and summary["unlogged_downstream_calls"] == 0
        and summary["duplicate_downstream_calls"] == 0
        and summary["mismatched_correlation_ids"] == 0
        and summary["unauthorized_downstream_calls"] == 0
    )
    summary["authorization_execution_consistent"] = int(
        summary["unauthorized_downstream_calls"] == 0
        and summary["missing_receipts"] == 0
        and summary["unlogged_downstream_calls"] == 0
        and summary["duplicate_downstream_calls"] == 0
        and summary["mismatched_correlation_ids"] == 0
        and summary["orphan_receipts"] == 0
    )
    return detail_df, summary


def record_bound_trace_hash(
    trace_df: pd.DataFrame,
    reconciliation_df: pd.DataFrame,
) -> str:
    """Hash ordered record identity and execution state, not just actions."""
    reconciliation = reconciliation_df.set_index("replay_point_id", drop=False)
    canonical_records: list[dict[str, Any]] = []

    for trace_position, row in trace_df.reset_index(drop=True).iterrows():
        replay_point_id = str(row["replay_point_id"])
        if replay_point_id not in reconciliation.index:
            receipt_values: Mapping[str, Any] = {}
        else:
            receipt_row = reconciliation.loc[replay_point_id]
            if isinstance(receipt_row, pd.DataFrame):
                receipt_row = receipt_row.iloc[0]
            receipt_values = receipt_row.to_dict()

        canonical_records.append(
            {
                "trace_position": int(trace_position),
                "replay_point_id": replay_point_id,
                "correlation_id": str(row.get("correlation_id", "")),
                "source_record_id": str(row.get("source_record_id", "")),
                "action": int(row.get("action", 0)),
                "authorized_to_generate": int(
                    row.get("authorized_to_generate", 0)
                ),
                "primary_logged_execution": int(row.get("generation_invoked", 0)),
                "unauthorized_invocation": int(
                    row.get("unauthorized_invocation", 0)
                ),
                "receipt_count": int(receipt_values.get("receipt_count", 0)),
                "matching_receipt_count": int(
                    receipt_values.get("matching_receipt_count", 0)
                ),
                "receipt_state_hash_set": str(
                    receipt_values.get("receipt_state_hash_set", "")
                ),
                "receipt_consistent": int(
                    receipt_values.get("receipt_consistent", 0)
                ),
            }
        )
    return sha256_json(canonical_records)


def config_bound_trace_hash(
    record_trace_hash: str,
    config_manifest_hash: str,
) -> str:
    return sha256_json(
        {
            "schema_version": INTEGRITY_SCHEMA_VERSION,
            "record_trace_hash": str(record_trace_hash),
            "config_manifest_hash": str(config_manifest_hash),
        }
    )


def validate_receipt_digest_rows(receipt_df: pd.DataFrame) -> bool:
    """Recompute every semantic receipt digest."""
    if receipt_df.empty:
        return True
    required = {
        "schema_version",
        "run_id",
        "replay_point_id",
        "correlation_id",
        "downstream_operation",
        "attempt_index",
        "execution_status",
        "receipt_digest",
    }
    if required - set(receipt_df.columns):
        return False
    for row in receipt_df.to_dict(orient="records"):
        semantic = {
            "schema_version": row["schema_version"],
            "run_id": row["run_id"],
            "replay_point_id": row["replay_point_id"],
            "correlation_id": row["correlation_id"],
            "downstream_operation": row["downstream_operation"],
            "attempt_index": int(row["attempt_index"]),
            "execution_status": row["execution_status"],
        }
        if sha256_json(semantic) != str(row["receipt_digest"]):
            return False
    return True
