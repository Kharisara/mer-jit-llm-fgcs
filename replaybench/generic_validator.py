from __future__ import annotations

"""Label-independent validation for ReplayBench-PG evidence.

The validator receives only observed execution artifacts and a trusted clean
reference. Fault labels, injection seeds, selected-index manifests, and
injection-marker columns are forbidden inputs. Ground-truth scoring belongs in
a separate post-validation stage.
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import pandas as pd

from replaybench.integrity import (
    config_bound_trace_hash,
    record_bound_trace_hash,
    sha256_json,
    validate_receipt_digest_rows,
)
from replaybench.validation import canonical_action_hash

SCHEMA_VERSION = "replaybench-pg-generic-validator-v1"

# These fields disclose how a fault was produced or directly mark injected rows.
FORBIDDEN_TRACE_COLUMNS = frozenset(
    {
        "action_before_fault",
        "action_flip_fault_injected",
        "unauthorized_invoke_fault_injected",
        "fault_injected",
        "receipt_fault_mode",
        "receipt_fault_injected",
    }
)

# These are derived findings written by the legacy executor. They are removed so
# the generic validator must recompute them from primary fields and receipts.
DERIVED_TRACE_COLUMNS = frozenset(
    {
        "unauthorized_invocation",
        "receipt_count",
        "matching_receipt_count",
        "receipt_digest_set",
        "receipt_state_hash_set",
        "missing_receipt",
        "unlogged_downstream_call",
        "duplicate_downstream_call",
        "mismatched_correlation_id",
        "unauthorized_downstream_call",
        "receipt_consistent",
    }
)

REQUIRED_TRACE_COLUMNS = frozenset(
    {
        "run_id",
        "replay_point_id",
        "correlation_id",
        "action",
        "authorized_to_generate",
        "generation_invoked",
    }
)

RECEIPT_ANOMALY_KEYS = (
    "orphan_receipts",
    "missing_receipts",
    "unlogged_downstream_calls",
    "duplicate_downstream_calls",
    "mismatched_correlation_ids",
    "unauthorized_downstream_calls",
)


def _reconcile_execution_receipts_fast(
    trace_df: pd.DataFrame, receipt_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Vectorized equivalent of the application receipt reconciliation."""
    trace = trace_df.copy(deep=True).reset_index(drop=True)
    receipts = receipt_df.copy(deep=True).reset_index(drop=True)
    if receipts.empty:
        receipts = pd.DataFrame(
            columns=[
                "run_id", "replay_point_id", "correlation_id",
                "receipt_digest", "downstream_operation",
                "attempt_index", "execution_status",
            ]
        )
    for column in ("run_id", "replay_point_id", "correlation_id"):
        trace[column] = trace[column].astype(str)
        receipts[column] = receipts[column].astype(str)

    if not receipts.empty:
        receipts = receipts.copy()
        receipts["_state_hash"] = [
            sha256_json(
                {
                    "replay_point_id": str(row.get("replay_point_id", "")),
                    "downstream_operation": str(row.get("downstream_operation", "")),
                    "attempt_index": int(row.get("attempt_index", 0)),
                    "execution_status": str(row.get("execution_status", "")),
                }
            )
            for row in receipts.to_dict(orient="records")
        ]
        point = (
            receipts.groupby(["run_id", "replay_point_id"], sort=False)
            .agg(
                receipt_count=("replay_point_id", "size"),
                receipt_digest_set=(
                    "receipt_digest",
                    lambda values: "|".join(sorted(map(str, values))),
                ),
                receipt_state_hash_set=(
                    "_state_hash",
                    lambda values: "|".join(sorted(map(str, values))),
                ),
            )
            .reset_index()
        )
        matching = (
            receipts.groupby(["run_id", "replay_point_id", "correlation_id"], sort=False)
            .size()
            .rename("matching_receipt_count")
            .reset_index()
        )
    else:
        point = pd.DataFrame(
            columns=[
                "run_id", "replay_point_id", "receipt_count",
                "receipt_digest_set", "receipt_state_hash_set",
            ]
        )
        matching = pd.DataFrame(
            columns=[
                "run_id", "replay_point_id", "correlation_id",
                "matching_receipt_count",
            ]
        )

    details = trace[[
        "run_id", "replay_point_id", "correlation_id",
        "authorized_to_generate", "generation_invoked",
    ]].copy()
    details.insert(0, "trace_position", range(len(details)))
    details = details.merge(point, on=["run_id", "replay_point_id"], how="left")
    details = details.merge(
        matching,
        on=["run_id", "replay_point_id", "correlation_id"],
        how="left",
    )
    for column in ("receipt_count", "matching_receipt_count"):
        details[column] = pd.to_numeric(details[column], errors="coerce").fillna(0).astype(int)
    for column in ("receipt_digest_set", "receipt_state_hash_set"):
        details[column] = details[column].fillna("").astype(str)
    details["authorized_to_generate"] = details["authorized_to_generate"].astype(int)
    details["primary_logged_execution"] = details.pop("generation_invoked").astype(int)
    details["missing_receipt"] = (
        details["primary_logged_execution"].eq(1)
        & details["matching_receipt_count"].eq(0)
    ).astype(int)
    details["unlogged_downstream_call"] = (
        details["primary_logged_execution"].eq(0)
        & details["receipt_count"].gt(0)
    ).astype(int)
    details["duplicate_downstream_call"] = details["receipt_count"].gt(1).astype(int)
    details["mismatched_correlation_id"] = (
        details["receipt_count"] > details["matching_receipt_count"]
    ).astype(int)
    details["unauthorized_downstream_call"] = (
        details["authorized_to_generate"].eq(0)
        & details["receipt_count"].gt(0)
    ).astype(int)
    anomaly_columns = [
        "missing_receipt", "unlogged_downstream_call",
        "duplicate_downstream_call", "mismatched_correlation_id",
        "unauthorized_downstream_call",
    ]
    details["receipt_consistent"] = details[anomaly_columns].sum(axis=1).eq(0).astype(int)

    trace_point_ids = set(trace["replay_point_id"].astype(str))
    orphan_receipts = int((~receipts["replay_point_id"].isin(trace_point_ids)).sum())
    summary = {
        "trace_rows": int(len(trace)),
        "receipt_rows": int(len(receipts)),
        "orphan_receipts": orphan_receipts,
        "missing_receipts": int(details["missing_receipt"].sum()),
        "unlogged_downstream_calls": int(details["unlogged_downstream_call"].sum()),
        "duplicate_downstream_calls": int(details["duplicate_downstream_call"].sum()),
        "mismatched_correlation_ids": int(details["mismatched_correlation_id"].sum()),
        "unauthorized_downstream_calls": int(details["unauthorized_downstream_call"].sum()),
    }
    summary["receipt_validation_passed"] = int(
        orphan_receipts == 0
        and all(summary[key] == 0 for key in RECEIPT_ANOMALY_KEYS[1:])
    )
    summary["authorization_execution_consistent"] = summary["receipt_validation_passed"]
    return details, summary


@dataclass(frozen=True)
class PreparedReference:
    trace: pd.DataFrame
    receipts: pd.DataFrame
    reconciliation: pd.DataFrame
    action_hash: str
    record_trace_hash: str
    config_manifest: dict[str, Any]
    config_manifest_hash: str
    config_bound_trace_hash: str


def prepare_reference_artifacts(
    *,
    reference_trace: pd.DataFrame,
    reference_receipts: pd.DataFrame,
    reference_config_manifest: Mapping[str, Any],
) -> PreparedReference:
    assert_label_independent_trace(reference_trace)
    trace = reference_trace.copy(deep=True).reset_index(drop=True)
    receipts = reference_receipts.copy(deep=True).reset_index(drop=True)
    reconciliation, summary = _reconcile_execution_receipts_fast(trace, receipts)
    if int(summary["receipt_validation_passed"]) != 1:
        raise ValueError("Trusted reference failed receipt reconciliation")
    action_hash = canonical_action_hash(trace["action"].astype(int).tolist())
    record_hash = record_bound_trace_hash(trace, reconciliation)
    config_manifest = dict(reference_config_manifest)
    config_hash = sha256_json(config_manifest)
    bound_hash = config_bound_trace_hash(record_hash, config_hash)
    return PreparedReference(
        trace=trace,
        receipts=receipts,
        reconciliation=reconciliation,
        action_hash=action_hash,
        record_trace_hash=record_hash,
        config_manifest=config_manifest,
        config_manifest_hash=config_hash,
        config_bound_trace_hash=bound_hash,
    )


@dataclass(frozen=True)
class GenericValidatorFindings:
    schema_version: str
    evidence_id: str
    observed_rows: int
    expected_rows: int
    row_count_match: int
    observed_action_hash: str
    expected_action_hash: str
    action_hash_match: int
    action_mismatch_count: int
    action_mismatch_replay_point_ids: tuple[str, ...]
    missing_replay_point_ids: tuple[str, ...]
    unexpected_replay_point_ids: tuple[str, ...]
    duplicate_replay_point_ids: tuple[str, ...]
    primary_unauthorized_count: int
    primary_unauthorized_replay_point_ids: tuple[str, ...]
    receipt_digest_rows_valid: int
    receipt_validation_passed: int
    orphan_receipts: int
    missing_receipts: int
    unlogged_downstream_calls: int
    duplicate_downstream_calls: int
    mismatched_correlation_ids: int
    unauthorized_downstream_calls: int
    receipt_missing_replay_point_ids: tuple[str, ...]
    receipt_unlogged_replay_point_ids: tuple[str, ...]
    receipt_duplicate_replay_point_ids: tuple[str, ...]
    receipt_mismatched_replay_point_ids: tuple[str, ...]
    receipt_unauthorized_replay_point_ids: tuple[str, ...]
    observed_record_trace_hash: str
    expected_record_trace_hash: str
    record_trace_hash_match: int
    observed_config_manifest_hash: str
    expected_config_manifest_hash: str
    config_manifest_hash_match: int
    observed_config_bound_trace_hash: str
    expected_config_bound_trace_hash: str
    config_bound_trace_hash_match: int
    primary_validator_triggered: int
    full_validator_triggered: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key, item in list(value.items()):
            if isinstance(item, tuple):
                value[key] = list(item)
        return value


def sanitize_trace_for_validation(trace: pd.DataFrame) -> pd.DataFrame:
    """Return a validator input with ground-truth and legacy finding fields removed."""
    if not isinstance(trace, pd.DataFrame):
        raise TypeError("trace must be a pandas DataFrame")
    return trace.drop(
        columns=[
            column
            for column in (*FORBIDDEN_TRACE_COLUMNS, *DERIVED_TRACE_COLUMNS)
            if column in trace.columns
        ],
        errors="ignore",
    ).copy(deep=True)


def assert_label_independent_trace(trace: pd.DataFrame) -> None:
    forbidden = sorted(FORBIDDEN_TRACE_COLUMNS.intersection(trace.columns))
    if forbidden:
        raise ValueError(
            "Generic validator input contains forbidden ground-truth columns: "
            f"{forbidden}"
        )
    missing = sorted(REQUIRED_TRACE_COLUMNS.difference(trace.columns))
    if missing:
        raise ValueError(f"Generic validator input is missing required columns: {missing}")


def _ids_where(frame: pd.DataFrame, column: str) -> tuple[str, ...]:
    if column not in frame.columns or frame.empty:
        return ()
    return tuple(
        sorted(
            frame.loc[frame[column].astype(int).eq(1), "replay_point_id"]
            .astype(str)
            .unique()
            .tolist()
        )
    )


def _action_comparison(
    observed: pd.DataFrame, reference: pd.DataFrame
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Compare actions by stable replay-point identity, independently of injection flags."""
    obs_ids = observed["replay_point_id"].astype(str)
    ref_ids = reference["replay_point_id"].astype(str)
    duplicate_ids = tuple(sorted(obs_ids.loc[obs_ids.duplicated(keep=False)].unique()))

    obs_unique = observed.assign(_id=obs_ids).drop_duplicates("_id", keep="first")
    ref_unique = reference.assign(_id=ref_ids).drop_duplicates("_id", keep="first")
    obs_map = obs_unique.set_index("_id")["action"].astype(int)
    ref_map = ref_unique.set_index("_id")["action"].astype(int)

    missing_ids = tuple(sorted(set(ref_map.index) - set(obs_map.index)))
    unexpected_ids = tuple(sorted(set(obs_map.index) - set(ref_map.index)))
    common = sorted(set(obs_map.index).intersection(ref_map.index))
    mismatches = tuple(
        replay_id for replay_id in common if int(obs_map[replay_id]) != int(ref_map[replay_id])
    )
    return mismatches, missing_ids, unexpected_ids, duplicate_ids


def _primary_unauthorized(trace: pd.DataFrame) -> tuple[int, tuple[str, ...]]:
    authorized = pd.to_numeric(trace["authorized_to_generate"], errors="raise").astype(int)
    executed = pd.to_numeric(trace["generation_invoked"], errors="raise").astype(int)
    mask = executed.eq(1) & authorized.eq(0)
    ids = tuple(sorted(trace.loc[mask, "replay_point_id"].astype(str).unique().tolist()))
    return int(mask.sum()), ids


def validate_observed_artifacts(
    *,
    evidence_id: str,
    observed_trace: pd.DataFrame,
    observed_receipts: pd.DataFrame,
    observed_config_manifest: Mapping[str, Any],
    reference_trace: pd.DataFrame,
    reference_receipts: pd.DataFrame,
    reference_config_manifest: Mapping[str, Any],
    prepared_reference: PreparedReference | None = None,
) -> GenericValidatorFindings:
    """Validate observed artifacts without fault labels or expected fault channels."""
    assert_label_independent_trace(observed_trace)
    assert_label_independent_trace(reference_trace)

    observed_trace = observed_trace.copy(deep=True).reset_index(drop=True)
    observed_receipts = observed_receipts.copy(deep=True).reset_index(drop=True)
    if prepared_reference is None:
        prepared_reference = prepare_reference_artifacts(
            reference_trace=reference_trace,
            reference_receipts=reference_receipts,
            reference_config_manifest=reference_config_manifest,
        )
    reference_trace = prepared_reference.trace

    observed_action_hash = canonical_action_hash(
        observed_trace["action"].astype(int).tolist()
    )
    expected_action_hash = prepared_reference.action_hash
    action_mismatches, missing_ids, unexpected_ids, duplicate_ids = _action_comparison(
        observed_trace, reference_trace
    )
    primary_unauthorized_count, primary_unauthorized_ids = _primary_unauthorized(
        observed_trace
    )

    observed_reconciliation, observed_receipt_summary = _reconcile_execution_receipts_fast(
        observed_trace, observed_receipts
    )
    observed_record_hash = record_bound_trace_hash(
        observed_trace, observed_reconciliation
    )
    expected_record_hash = prepared_reference.record_trace_hash
    observed_config_hash = sha256_json(dict(observed_config_manifest))
    expected_config_hash = prepared_reference.config_manifest_hash
    observed_bound_hash = config_bound_trace_hash(
        observed_record_hash, observed_config_hash
    )
    expected_bound_hash = prepared_reference.config_bound_trace_hash

    row_count_match = int(len(observed_trace) == len(reference_trace))
    action_hash_match = int(observed_action_hash == expected_action_hash)
    record_hash_match = int(observed_record_hash == expected_record_hash)
    config_hash_match = int(observed_config_hash == expected_config_hash)
    bound_hash_match = int(observed_bound_hash == expected_bound_hash)
    receipt_digest_valid = int(validate_receipt_digest_rows(observed_receipts))

    primary_triggered = int(
        row_count_match == 0
        or action_hash_match == 0
        or primary_unauthorized_count > 0
    )
    full_triggered = int(
        primary_triggered == 1
        or receipt_digest_valid == 0
        or int(observed_receipt_summary["receipt_validation_passed"]) == 0
        or record_hash_match == 0
        or config_hash_match == 0
        or bound_hash_match == 0
    )

    return GenericValidatorFindings(
        schema_version=SCHEMA_VERSION,
        evidence_id=str(evidence_id),
        observed_rows=int(len(observed_trace)),
        expected_rows=int(len(reference_trace)),
        row_count_match=row_count_match,
        observed_action_hash=observed_action_hash,
        expected_action_hash=expected_action_hash,
        action_hash_match=action_hash_match,
        action_mismatch_count=len(action_mismatches),
        action_mismatch_replay_point_ids=action_mismatches,
        missing_replay_point_ids=missing_ids,
        unexpected_replay_point_ids=unexpected_ids,
        duplicate_replay_point_ids=duplicate_ids,
        primary_unauthorized_count=primary_unauthorized_count,
        primary_unauthorized_replay_point_ids=primary_unauthorized_ids,
        receipt_digest_rows_valid=receipt_digest_valid,
        receipt_validation_passed=int(
            observed_receipt_summary["receipt_validation_passed"]
        ),
        orphan_receipts=int(observed_receipt_summary["orphan_receipts"]),
        missing_receipts=int(observed_receipt_summary["missing_receipts"]),
        unlogged_downstream_calls=int(
            observed_receipt_summary["unlogged_downstream_calls"]
        ),
        duplicate_downstream_calls=int(
            observed_receipt_summary["duplicate_downstream_calls"]
        ),
        mismatched_correlation_ids=int(
            observed_receipt_summary["mismatched_correlation_ids"]
        ),
        unauthorized_downstream_calls=int(
            observed_receipt_summary["unauthorized_downstream_calls"]
        ),
        receipt_missing_replay_point_ids=_ids_where(
            observed_reconciliation, "missing_receipt"
        ),
        receipt_unlogged_replay_point_ids=_ids_where(
            observed_reconciliation, "unlogged_downstream_call"
        ),
        receipt_duplicate_replay_point_ids=_ids_where(
            observed_reconciliation, "duplicate_downstream_call"
        ),
        receipt_mismatched_replay_point_ids=_ids_where(
            observed_reconciliation, "mismatched_correlation_id"
        ),
        receipt_unauthorized_replay_point_ids=_ids_where(
            observed_reconciliation, "unauthorized_downstream_call"
        ),
        observed_record_trace_hash=observed_record_hash,
        expected_record_trace_hash=expected_record_hash,
        record_trace_hash_match=record_hash_match,
        observed_config_manifest_hash=observed_config_hash,
        expected_config_manifest_hash=expected_config_hash,
        config_manifest_hash_match=config_hash_match,
        observed_config_bound_trace_hash=observed_bound_hash,
        expected_config_bound_trace_hash=expected_bound_hash,
        config_bound_trace_hash_match=bound_hash_match,
        primary_validator_triggered=primary_triggered,
        full_validator_triggered=full_triggered,
    )
