from __future__ import annotations

import hashlib
from collections.abc import Sequence


def canonical_action_hash(actions: Sequence[int]) -> str:
    """Return SHA-256 for the ordered integer action sequence."""
    canonical = ",".join(str(int(action)) for action in actions)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_authorization_execution_violation(
    authorized: bool,
    executed: bool,
) -> bool:
    """True only when execution occurred without prior authorization."""
    return bool(executed and not authorized)


def validate_row_count(
    observed_rows: int,
    expected_rows: int,
) -> bool:
    return observed_rows == expected_rows


def validate_trace(
    actions: Sequence[int],
    expected_hash: str,
    observed_rows: int,
    expected_rows: int,
    unauthorized_invocations: int,
) -> dict[str, object]:
    actual_hash = canonical_action_hash(actions)

    return {
        "actual_hash": actual_hash,
        "expected_hash": expected_hash,
        "hash_match": actual_hash == expected_hash,
        "row_count_match": observed_rows == expected_rows,
        "authorization_execution_consistent": (
            unauthorized_invocations == 0
        ),
        "validation_passed": (
            actual_hash == expected_hash
            and observed_rows == expected_rows
            and unauthorized_invocations == 0
        ),
    }