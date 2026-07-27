from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from replaybench.integrity import (  # noqa: E402
    config_bound_trace_hash,
    record_bound_trace_hash,
    sha256_json,
)
from run_fgcs_extended_benchmark import run_replay  # noqa: E402


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "utterance_id": ["u0", "u1", "u2", "u3"],
            "source_record_id": ["s0", "s1", "s2", "s3"],
            "label": ["sadness", "joy", "anger", "neutral"],
            "state_path": ["", "", "", ""],
        }
    )


def _config(mode: str = "clean") -> dict[str, object]:
    return {
        "dataset": {"input_csv": "not-required-for-unit-test.csv", "state_root": ""},
        "policy": {"random_intervention_probability": 0.5},
        "fault_injection": {
            "enabled": False,
            "action_flip_probability": 0.0,
            "unauthorized_invoke_probability": 0.0,
            "receipt_fault_mode": mode,
            "receipt_fault_probability": 0.0 if mode == "clean" else 1.0,
            "receipt_fault_allowed_policy_modes": ["always"],
        },
        "execution_receipts": {"enabled": True},
        "generation_stub": {
            "mode": "generic_service",
            "service_name": "test_downstream",
            "record_id_column": "source_record_id",
        },
    }


def _run(mode: str = "clean", workers: int = 1):
    return run_replay(
        df=_frame(),
        cfg=_config(mode),
        policy_mode="always",
        negative_labels=set(),
        seed=1,
        workers=workers,
        bc_actions=None,
    )


def test_clean_receipts_are_independently_reconciled() -> None:
    trace, summary, _ = _run("clean")
    receipts = trace.attrs["execution_receipts"]
    reconciliation = trace.attrs["receipt_reconciliation"]

    assert len(receipts) == len(trace)
    assert reconciliation["receipt_consistent"].eq(1).all()
    assert summary["receipt_validation_passed"] == 1
    assert summary["authorization_execution_consistent"] == 1
    assert summary["record_trace_hash"]
    assert summary["config_bound_trace_hash"]


@pytest.mark.parametrize(
    ("mode", "metric"),
    [
        ("unlogged_downstream_call", "unlogged_downstream_calls"),
        ("false_execution_log", "missing_receipts"),
        ("duplicate_downstream_call", "duplicate_downstream_calls"),
        ("mismatched_correlation_id", "mismatched_correlation_ids"),
    ],
)
def test_receipt_positive_controls_are_detected(mode: str, metric: str) -> None:
    _, summary, _ = _run(mode)
    assert summary["receipt_fault_injected_count"] == len(_frame())
    assert summary[metric] == len(_frame())
    assert summary["receipt_validation_passed"] == 0
    assert summary["validation_passed"] == 0


def test_record_hash_is_stable_across_worker_settings() -> None:
    _, one_worker, _ = _run("clean", workers=1)
    _, four_workers, _ = _run("clean", workers=4)
    assert one_worker["trace_hash"] == four_workers["trace_hash"]
    assert one_worker["record_trace_hash"] == four_workers["record_trace_hash"]
    assert one_worker["config_manifest_hash"] != four_workers["config_manifest_hash"]
    assert one_worker["config_bound_trace_hash"] != four_workers["config_bound_trace_hash"]


def test_record_bound_hash_detects_reordering_and_field_corruption() -> None:
    trace, summary, _ = _run("clean")
    reconciliation = trace.attrs["receipt_reconciliation"]
    clean_hash = summary["record_trace_hash"]

    reordered = trace.iloc[[1, 0, 2, 3]].reset_index(drop=True)
    assert record_bound_trace_hash(reordered, reconciliation) != clean_hash

    reassigned = trace.copy(deep=True)
    first_id = reassigned.loc[0, "replay_point_id"]
    reassigned.loc[0, "replay_point_id"] = reassigned.loc[1, "replay_point_id"]
    reassigned.loc[1, "replay_point_id"] = first_id
    assert record_bound_trace_hash(reassigned, reconciliation) != clean_hash

    auth_corrupt = trace.copy(deep=True)
    auth_corrupt.loc[0, "authorized_to_generate"] = 0
    assert record_bound_trace_hash(auth_corrupt, reconciliation) != clean_hash

    execution_corrupt = trace.copy(deep=True)
    execution_corrupt.loc[0, "generation_invoked"] = 0
    assert record_bound_trace_hash(execution_corrupt, reconciliation) != clean_hash


def test_config_bound_hash_detects_configuration_label_corruption() -> None:
    trace, summary, _ = _run("clean")
    run_identity = trace.attrs["run_identity"]
    manifest = copy.deepcopy(run_identity["config_manifest"])
    manifest["policy_mode"] = "corrupted::always"
    corrupted_manifest_hash = sha256_json(manifest)
    corrupted = config_bound_trace_hash(
        summary["record_trace_hash"], corrupted_manifest_hash
    )
    assert corrupted != summary["config_bound_trace_hash"]
