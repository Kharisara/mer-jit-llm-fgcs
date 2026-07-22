from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from final_artifact_utils import (
    FAULT_ORDER,
    _infer_comparison_metrics,
    normalize_fault_summary,
)


def test_cloud_summary_metrics_from_compared_and_matched_columns() -> None:
    frame = pd.DataFrame(
        [
            {"policy": "a", "compared": 60, "matched": 60},
            {"policy": "b", "compared": 300, "matched": 300},
        ]
    )
    assert _infer_comparison_metrics(frame) == (360, 360)


def test_cloud_row_level_hash_match_metrics() -> None:
    frame = pd.DataFrame({"local_vs_cloud_hash_match": [1, True, "yes"]})
    assert _infer_comparison_metrics(frame) == (3, 3)


def test_fault_summary_normalization_preserves_five_classes() -> None:
    frame = pd.DataFrame(
        [
            {
                "fault_mode": "clean_replay",
                "runs": 54,
                "faults_or_corruptions_injected_total": 0,
                "detected_runs": 0,
                "false_positive_runs": 0,
                "detection_rate": None,
            },
            *[
                {
                    "fault_mode": mode,
                    "runs": 18,
                    "faults_or_corruptions_injected_total": 100,
                    "detected_runs": 18,
                    "false_positive_runs": 0,
                    "detection_rate": 1.0,
                }
                for mode in FAULT_ORDER
            ],
        ]
    )
    normalized = normalize_fault_summary(frame, "test")
    assert set(normalized["fault_mode"]) == {"clean_replay", *FAULT_ORDER}
    faults = normalized[normalized["fault_mode"].isin(FAULT_ORDER)]
    assert faults["detected_runs"].eq(18).all()
