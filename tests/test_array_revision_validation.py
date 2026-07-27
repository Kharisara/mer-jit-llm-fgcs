from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_fgcs_extended_benchmark import (  # noqa: E402
    authorization_execution_violation,
    canonical_action_hash,
    validate_trace,
)
from run_ray_comparison import (  # noqa: E402
    normalize_comparison_config,
    select_dropped_indices,
    validate_comparison_config,
)
from run_replaybench_timing_study import normalize_existing  # noqa: E402
from summarize_replaybench_timing_study import (  # noqa: E402
    calculate_worker_speedups,
    normalize_timing_frame,
    paired_bootstrap_ci,
    validate_timing_design,
)



def test_resume_normalization_populates_alias_columns() -> None:
    raw = pd.DataFrame(
        {
            "dataset_fraction": [1.0],
            "policy_mode": ["never"],
            "workers": [1],
            "repetition": [1],
            "total_runtime_seconds": [8.0],
            "trace_hash": ["abc"],
        }
    )
    normalized = normalize_existing(raw)
    assert normalized.loc[0, "policy"] == "never"
    assert normalized.loc[0, "runtime_seconds"] == pytest.approx(8.0)
    assert normalized.loc[0, "workload_fraction"] == pytest.approx(1.0)

def test_bootstrap_ci_contains_constant_value() -> None:
    values = np.array([1.1] * 15)
    estimate, lower, upper = paired_bootstrap_ci(values)

    assert estimate == pytest.approx(1.1)
    assert lower == pytest.approx(1.1)
    assert upper == pytest.approx(1.1)


def test_bootstrap_ci_is_ordered() -> None:
    values = np.array(
        [
            1.01,
            1.03,
            1.04,
            1.06,
            1.08,
            1.09,
            1.11,
            1.12,
            1.13,
            1.14,
            1.15,
            1.16,
            1.17,
            1.18,
            1.19,
        ]
    )
    estimate, lower, upper = paired_bootstrap_ci(values)
    assert lower <= estimate <= upper


def test_bootstrap_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array([1.0]))
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array([1.0, 0.0]))


def _complete_timing_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    policies = ["never", "random"]
    fractions = [0.5, 1.0]
    workers = [1, 4]

    for policy in policies:
        for fraction in fractions:
            active_workers = workers if fraction == 1.0 else [1]
            for worker in active_workers:
                repetitions = 15 if fraction == 1.0 else 7
                for repetition in range(1, repetitions + 1):
                    baseline = 10.0 + repetition * 0.01
                    runtime = baseline if worker == 1 else baseline / 1.10
                    rows.append(
                        {
                            "dataset_fraction": fraction,
                            "policy": policy,
                            "workers": worker,
                            "repetition": repetition,
                            "runtime_seconds": runtime,
                            "trace_hash": f"{policy}-{fraction}-{worker}",
                            "authorization_execution_consistent": 1,
                            "row_count_match": 1,
                            "validation_passed": 1,
                            "fault_injected_count": 0,
                            "unauthorized_invocations": 0,
                        }
                    )
    return pd.DataFrame(rows)


def test_timing_design_and_paired_speedup() -> None:
    raw = _complete_timing_frame()
    normalized = normalize_timing_frame(raw)
    manifest = validate_timing_design(
        normalized, workload_repetitions=7, worker_repetitions=15
    )
    assert manifest["workload_repetitions"] == 7
    assert manifest["worker_repetitions"] == 15

    speedups = calculate_worker_speedups(
        raw,
        workload_repetitions=7,
        worker_repetitions=15,
        bootstrap_samples=2_000,
    )
    four_workers = speedups.loc[speedups["workers"] == 4]
    assert len(four_workers) == 2
    assert np.allclose(four_workers["median_paired_speedup"], 1.10)

    one_worker = speedups.loc[speedups["workers"] == 1]
    assert np.allclose(one_worker["median_paired_speedup"], 1.0)
    assert np.allclose(one_worker["ci95_lower"], 1.0)
    assert np.allclose(one_worker["ci95_upper"], 1.0)


def test_timing_design_rejects_duplicate_repetition() -> None:
    raw = _complete_timing_frame()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate timing rows"):
        normalize_timing_frame(raw)


def test_timing_design_rejects_incomplete_configuration() -> None:
    raw = _complete_timing_frame().iloc[:-1].copy()
    normalized = normalize_timing_frame(raw)
    with pytest.raises(ValueError, match="incorrect measured-repetition counts"):
        validate_timing_design(
            normalized, workload_repetitions=7, worker_repetitions=15
        )


def test_authorization_execution_truth_table() -> None:
    assert authorization_execution_violation(False, False) == 0
    assert authorization_execution_violation(True, False) == 0
    assert authorization_execution_violation(True, True) == 0
    assert authorization_execution_violation(False, True) == 1


def test_trace_validation() -> None:
    actions = [0, 1, 0, 1]
    expected_hash = canonical_action_hash(actions)

    valid = validate_trace(
        actions=actions,
        expected_hash=expected_hash,
        observed_rows=4,
        expected_rows=4,
        unauthorized_invocations=0,
    )
    assert valid["validation_passed"] == 1

    invalid = validate_trace(
        actions=actions[:-1],
        expected_hash=expected_hash,
        observed_rows=3,
        expected_rows=4,
        unauthorized_invocations=0,
    )
    assert invalid["hash_match"] == 0
    assert invalid["row_count_match"] == 0
    assert invalid["validation_passed"] == 0


def test_dropped_row_selection_is_exact_and_deterministic() -> None:
    indices = list(range(11_351))
    first = select_dropped_indices(indices, "random", 1, 0.01)
    second = select_dropped_indices(indices, "random", 1, 0.01)
    assert first == second
    assert len(first) == int(len(indices) * 0.01)


def test_final_ray_config_validation(tmp_path: Path) -> None:
    input_csv = tmp_path / "replay.csv"
    pd.DataFrame(
        {
            "utterance_id": [1, 2],
            "label": ["sadness", "joy"],
        }
    ).to_csv(input_csv, index=False)

    raw = {
        "input": {
            "replay_csv": str(input_csv),
            "workload_fraction": 1.0,
        },
        "policies": ["risk_proxy", "random", "never"],
        "seeds": [1, 2, 3],
        "workers": [1, 4],
        "fault_modes": ["clean", "action_flip", "dropped_row"],
        "fault_rate": 0.01,
        "policy": {
            "negative_labels": ["sadness"],
            "random_intervention_probability": 0.5,
        },
        "reference": {
            "determinism_csv": str(tmp_path / "determinism.csv"),
            "require_reference_results": True,
        },
        "execution_semantics": {
            "task_retry_enabled": False,
            "task_failure_policy": "fail_fast",
            "duplicate_delivery_supported": False,
        },
        "output_dir": str(tmp_path / "output"),
    }

    normalized = normalize_comparison_config(raw)
    validate_comparison_config(normalized)


def test_ray_config_rejects_missing_risk_proxy_labels(tmp_path: Path) -> None:
    input_csv = tmp_path / "replay.csv"
    pd.DataFrame({"utterance_id": [1], "label": ["sadness"]}).to_csv(
        input_csv, index=False
    )

    raw = {
        "input": {
            "replay_csv": str(input_csv),
            "workload_fraction": 1.0,
        },
        "policies": ["risk_proxy", "random", "never"],
        "seeds": [1, 2, 3],
        "workers": [1, 4],
        "fault_modes": ["clean", "action_flip", "dropped_row"],
        "fault_rate": 0.01,
        "reference": {"require_reference_results": True},
        "execution_semantics": {
            "task_retry_enabled": False,
            "task_failure_policy": "fail_fast",
            "duplicate_delivery_supported": False,
        },
    }

    normalized = normalize_comparison_config(raw)
    with pytest.raises(ValueError, match="negative_labels"):
        validate_comparison_config(normalized)
