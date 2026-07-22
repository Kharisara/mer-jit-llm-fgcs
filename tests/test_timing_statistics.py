import numpy as np

from summarize_replaybench_timing_study import paired_bootstrap_ci


def test_bootstrap_ci_contains_constant_value() -> None:
    values = np.array([1.1] * 15)

    estimate, lower, upper = paired_bootstrap_ci(values)

    assert estimate == 1.1
    assert lower == 1.1
    assert upper == 1.1


def test_bootstrap_ci_is_ordered() -> None:
    values = np.array(
        [1.01, 1.03, 1.04, 1.06, 1.08, 1.09, 1.11]
    )

    estimate, lower, upper = paired_bootstrap_ci(values)

    assert lower <= estimate <= upper