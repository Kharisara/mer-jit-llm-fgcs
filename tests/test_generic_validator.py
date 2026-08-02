from __future__ import annotations

import copy

import pandas as pd
import pytest

from replaybench.generic_validator import (
    FORBIDDEN_TRACE_COLUMNS,
    assert_label_independent_trace,
    sanitize_trace_for_validation,
    validate_observed_artifacts,
)
from replaybench.integrity import DownstreamReceiptCollector, ReceiptContext


def _trace() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_index": 0,
                "source_record_id": "r0",
                "run_id": "run",
                "replay_point_id": "0:r0",
                "correlation_id": "c0",
                "action": 0,
                "authorized_to_generate": 0,
                "generation_invoked": 0,
                "action_flip_fault_injected": 0,
                "receipt_fault_mode": "clean",
            },
            {
                "row_index": 1,
                "source_record_id": "r1",
                "run_id": "run",
                "replay_point_id": "1:r1",
                "correlation_id": "c1",
                "action": 1,
                "authorized_to_generate": 1,
                "generation_invoked": 1,
                "action_flip_fault_injected": 0,
                "receipt_fault_mode": "clean",
            },
        ]
    )


def _receipts() -> pd.DataFrame:
    collector = DownstreamReceiptCollector()
    collector.emit(
        ReceiptContext(
            run_id="run",
            replay_point_id="1:r1",
            correlation_id="c1",
            downstream_operation="stub",
        )
    )
    return collector.to_frame()


def _manifest() -> dict[str, object]:
    return {
        "schema_version": "replaybench-pg-integrity-v1",
        "policy_mode": "demo",
        "seed": 1,
        "workers": 1,
    }


def test_sanitizer_removes_ground_truth_and_legacy_findings() -> None:
    raw = _trace()
    raw["unauthorized_invocation"] = 0
    sanitized = sanitize_trace_for_validation(raw)
    assert not FORBIDDEN_TRACE_COLUMNS.intersection(sanitized.columns)
    assert "unauthorized_invocation" not in sanitized.columns
    assert_label_independent_trace(sanitized)


def test_validator_rejects_fault_labels() -> None:
    with pytest.raises(ValueError, match="forbidden ground-truth"):
        assert_label_independent_trace(_trace())


def test_action_localization_uses_clean_reference() -> None:
    reference = sanitize_trace_for_validation(_trace())
    observed = reference.copy(deep=True)
    observed.loc[observed["replay_point_id"].eq("0:r0"), "action"] = 1
    findings = validate_observed_artifacts(
        evidence_id="action-flip",
        observed_trace=observed,
        observed_receipts=_receipts(),
        observed_config_manifest=_manifest(),
        reference_trace=reference,
        reference_receipts=_receipts(),
        reference_config_manifest=_manifest(),
    )
    assert findings.action_mismatch_replay_point_ids == ("0:r0",)
    assert findings.primary_validator_triggered == 1


def test_primary_invariant_is_recomputed_not_trusted() -> None:
    reference = sanitize_trace_for_validation(_trace())
    observed = reference.copy(deep=True)
    observed.loc[observed["replay_point_id"].eq("0:r0"), "generation_invoked"] = 1
    findings = validate_observed_artifacts(
        evidence_id="unauthorized",
        observed_trace=observed,
        observed_receipts=_receipts(),
        observed_config_manifest=_manifest(),
        reference_trace=reference,
        reference_receipts=_receipts(),
        reference_config_manifest=_manifest(),
    )
    assert findings.primary_unauthorized_count == 1
    assert findings.primary_unauthorized_replay_point_ids == ("0:r0",)
    assert findings.primary_validator_triggered == 1


def test_config_change_is_full_only() -> None:
    reference = sanitize_trace_for_validation(_trace())
    observed_manifest = copy.deepcopy(_manifest())
    observed_manifest["policy_mode"] = "corrupted"
    findings = validate_observed_artifacts(
        evidence_id="config",
        observed_trace=reference,
        observed_receipts=_receipts(),
        observed_config_manifest=observed_manifest,
        reference_trace=reference,
        reference_receipts=_receipts(),
        reference_config_manifest=_manifest(),
    )
    assert findings.primary_validator_triggered == 0
    assert findings.config_manifest_hash_match == 0
    assert findings.full_validator_triggered == 1


def test_fast_receipt_reconciliation_matches_reference_implementation() -> None:
    """The vectorized validator path must preserve the archived semantics."""
    from replaybench.generic_validator import _reconcile_execution_receipts_fast
    from replaybench.integrity import reconcile_execution_receipts

    trace = sanitize_trace_for_validation(_trace())
    clean_receipts = _receipts()

    cases: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "clean": (trace.copy(deep=True), clean_receipts.copy(deep=True)),
        "missing": (trace.copy(deep=True), clean_receipts.iloc[0:0].copy()),
        "duplicate": (
            trace.copy(deep=True),
            pd.concat([clean_receipts, clean_receipts], ignore_index=True),
        ),
        "mismatched_correlation": (
            trace.copy(deep=True),
            clean_receipts.assign(correlation_id="wrong-correlation"),
        ),
        "unlogged_and_unauthorized": (
            trace.copy(deep=True),
            pd.concat(
                [
                    clean_receipts,
                    clean_receipts.assign(
                        replay_point_id="0:r0",
                        correlation_id="c0",
                    ),
                ],
                ignore_index=True,
            ),
        ),
        "orphan": (
            trace.copy(deep=True),
            pd.concat(
                [
                    clean_receipts,
                    clean_receipts.assign(
                        replay_point_id="99:orphan",
                        correlation_id="orphan-correlation",
                    ),
                ],
                ignore_index=True,
            ),
        ),
    }

    detail_columns = [
        "trace_position",
        "run_id",
        "replay_point_id",
        "correlation_id",
        "authorized_to_generate",
        "primary_logged_execution",
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
    ]

    for case_name, (case_trace, case_receipts) in cases.items():
        expected_details, expected_summary = reconcile_execution_receipts(
            case_trace, case_receipts
        )
        actual_details, actual_summary = _reconcile_execution_receipts_fast(
            case_trace, case_receipts
        )
        pd.testing.assert_frame_equal(
            actual_details[detail_columns].reset_index(drop=True),
            expected_details[detail_columns].reset_index(drop=True),
            check_dtype=False,
            obj=f"receipt reconciliation details for {case_name}",
        )
        assert actual_summary == expected_summary, case_name
