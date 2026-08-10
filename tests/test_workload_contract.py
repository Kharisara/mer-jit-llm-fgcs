from __future__ import annotations

import pandas as pd
import pytest

from replaybench.workload import (
    MELD_IDENTITY_SCHEME,
    MELD_PREQUALIFIED_IDENTITY_SCHEME,
    WorkloadContractError,
    prepare_replay_frame,
)
from run_fgcs_extended_benchmark import random_action, risk_proxy_action


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["train", "dev", "test"],
            "Dialogue_ID": [0, 0, 0],
            "Utterance_ID": [0, 0, 0],
            "label": ["neutral", "sadness", "joy"],
        }
    )


def config() -> dict[str, object]:
    return {
        "identity_scheme": MELD_IDENTITY_SCHEME,
        "require_portable_paths": True,
    }


def test_meld_identity_is_split_qualified_and_unique() -> None:
    prepared = prepare_replay_frame(sample_frame(), config())
    assert prepared["source_record_id"].tolist() == [
        "train:d0_u0",
        "dev:d0_u0",
        "test:d0_u0",
    ]
    assert prepared["source_record_id"].is_unique


def test_existing_wrong_identity_is_rejected() -> None:
    frame = sample_frame()
    frame["source_record_id"] = ["d0_u0", "d0_u0", "d0_u0"]
    with pytest.raises(WorkloadContractError, match="does not match"):
        prepare_replay_frame(frame, config())


def test_absolute_windows_path_is_rejected() -> None:
    frame = sample_frame()
    frame["audio_path"] = [
        r"D:\\MELD\\train\\a.wav",
        "relative/dev/a.wav",
        "relative/test/a.wav",
    ]
    with pytest.raises(WorkloadContractError, match="absolute paths"):
        prepare_replay_frame(frame, config())


def test_prequalified_meld_identity_requires_canonical_ids() -> None:
    frame = pd.DataFrame(
        {
            "source_record_id": ["train:d0_u0", "dev:d0_u0", "test:d0_u0"],
            "diagnostic_action": [0, 1, 0],
        }
    )
    prepared = prepare_replay_frame(
        frame,
        {"identity_scheme": MELD_PREQUALIFIED_IDENTITY_SCHEME},
    )
    assert prepared["source_record_id"].is_unique

    broken = frame.copy()
    broken.loc[0, "source_record_id"] = "d0_u0"
    with pytest.raises(WorkloadContractError, match="malformed"):
        prepare_replay_frame(
            broken,
            {"identity_scheme": MELD_PREQUALIFIED_IDENTITY_SCHEME},
        )


def test_risk_proxy_prefers_precomputed_diagnostic_action() -> None:
    assert risk_proxy_action(
        {"source_record_id": "train:d0_u0", "diagnostic_action": 1},
        set(),
        {"diagnostic_action_column": "diagnostic_action"},
    ) == 1
    assert risk_proxy_action(
        {"source_record_id": "train:d0_u1", "diagnostic_action": 0},
        set(),
        {"diagnostic_action_column": "diagnostic_action"},
    ) == 0
    with pytest.raises(ValueError, match="binary"):
        risk_proxy_action(
            {"source_record_id": "train:d0_u2", "diagnostic_action": 2},
            set(),
            {"diagnostic_action_column": "diagnostic_action"},
        )


def test_risk_proxy_rejects_legacy_label_only_rows() -> None:
    with pytest.raises(ValueError, match="diagnostic_action"):
        risk_proxy_action(
            {"source_record_id": "train:d0_u9", "label": "sadness"},
            {"sadness"},
            {"diagnostic_action_column": "diagnostic_action"},
        )


def test_random_policy_is_bound_to_canonical_record_identity() -> None:
    row = {"source_record_id": "train:d3_u7"}
    first = random_action(row, seed=3, row_index=19, p=0.5)
    second = random_action(row, seed=3, row_index=19, p=0.5)
    assert first == second
