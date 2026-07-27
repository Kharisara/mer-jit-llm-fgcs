from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from replaybench.integrity import (  # noqa: E402
    config_bound_trace_hash,
    record_bound_trace_hash,
    reconcile_execution_receipts,
    validate_receipt_digest_rows,
)
from replaybench.validation import canonical_action_hash  # noqa: E402
from run_fgcs_extended_benchmark import run_replay  # noqa: E402
from run_validator_selectivity_validation import (  # noqa: E402
    _benign_transform,
    _posthoc_trace_corruption,
    _profile_targets,
    _record_corruption,
    _runtime_fault_config,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "utterance_id": [f"u{i}" for i in range(20)],
            "source_record_id": [f"s{i}" for i in range(20)],
            "label": ["sadness" if i % 2 == 0 else "joy" for i in range(20)],
            "state_path": [""] * 20,
        }
    )


def _base_config() -> dict[str, object]:
    return {
        "dataset": {"input_csv": "not-used.csv", "state_root": ""},
        "policy": {
            "random_intervention_probability": 0.5,
            "negative_labels": ["sadness"],
        },
        "execution_receipts": {"enabled": True},
        "fault_injection": {
            "enabled": False,
            "action_flip_probability": 0.0,
            "unauthorized_invoke_probability": 0.0,
            "receipt_fault_mode": "clean",
            "receipt_fault_probability": 0.0,
        },
        "generation_stub": {
            "mode": "generic_service",
            "service_name": "test_downstream",
            "record_id_column": "source_record_id",
        },
    }


def _run(cfg: dict[str, object], policy: str = "risk_proxy"):
    return run_replay(
        df=_frame(),
        cfg=cfg,
        policy_mode=policy,
        negative_labels={"sadness"},
        seed=1,
        workers=1,
        bc_actions=None,
    )


def test_exact_targeted_runtime_faults() -> None:
    targets = [0, 2]
    action_cfg = _runtime_fault_config(
        _base_config(),
        fault_mode="action_flip",
        target_indices=targets,
        policy_modes=["risk_proxy"],
    )
    trace, summary, _ = _run(action_cfg)
    assert trace.loc[trace["action_flip_fault_injected"].eq(1), "row_index"].tolist() == targets
    assert summary["receipt_validation_passed"] == 1

    unauthorized_cfg = _runtime_fault_config(
        _base_config(),
        fault_mode="unauthorized_invocation",
        target_indices=[1, 3],
        policy_modes=["risk_proxy"],
    )
    trace, summary, _ = _run(unauthorized_cfg)
    assert trace.loc[trace["unauthorized_invoke_fault_injected"].eq(1), "row_index"].tolist() == [1, 3]
    assert summary["receipt_validation_passed"] == 0


def test_benign_transforms_preserve_semantic_hashes() -> None:
    trace, summary, _ = _run(_base_config())
    receipts = trace.attrs["execution_receipts"]
    for index, mode in enumerate(
        (
            "timing_fields_changed",
            "metadata_column_order_changed",
            "permitted_logging_format_changed",
            "completion_order_changed_then_reconstructed",
        )
    ):
        transformed_trace, transformed_receipts = _benign_transform(
            trace, receipts, mode, seed=100 + index
        )
        assert validate_receipt_digest_rows(transformed_receipts)
        reconciliation, receipt_summary = reconcile_execution_receipts(
            transformed_trace, transformed_receipts
        )
        assert receipt_summary["receipt_validation_passed"] == 1
        assert canonical_action_hash(transformed_trace["action"].tolist()) == summary["trace_hash"]
        record_hash = record_bound_trace_hash(transformed_trace, reconciliation)
        assert record_hash == summary["record_trace_hash"]
        assert config_bound_trace_hash(record_hash, summary["config_manifest_hash"]) == summary["config_bound_trace_hash"]


def test_posthoc_and_record_faults_trigger_expected_hashes() -> None:
    trace, summary, _ = _run(_base_config())
    reconciliation = trace.attrs["receipt_reconciliation"]
    for mode in ("saved_action_corruption", "dropped_rows", "duplicated_rows"):
        corrupted = _posthoc_trace_corruption(trace, mode, [0])
        assert canonical_action_hash(corrupted["action"].tolist()) != summary["trace_hash"]
        if mode != "saved_action_corruption":
            assert len(corrupted) != len(trace)

    for mode in (
        "row_reordering",
        "replay_id_action_reassignment",
        "authorization_field_corruption",
        "execution_field_corruption",
    ):
        corrupted, _ = _record_corruption(trace, mode, [0])
        assert record_bound_trace_hash(corrupted, reconciliation) != summary["record_trace_hash"]


def test_profile_target_selection_is_exact_and_deterministic() -> None:
    eligible = list(range(1000))
    first = _profile_targets(eligible, {"name": "first", "kind": "first"}, injection_seed=7, namespace="x")
    last = _profile_targets(eligible, {"name": "last", "kind": "last"}, injection_seed=7, namespace="x")
    rate = _profile_targets(eligible, {"name": "rate", "kind": "rate", "rate": 0.001}, injection_seed=7, namespace="x")
    assert first == [0]
    assert last == [999]
    assert len(rate) == 1
    assert rate == _profile_targets(eligible, {"name": "rate", "kind": "rate", "rate": 0.001}, injection_seed=7, namespace="x")
