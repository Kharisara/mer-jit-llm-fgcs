from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any


def canonical_action_hash(actions: Sequence[int]) -> str:
    """
    Compute SHA-256 over the canonical ordered integer action sequence.

    Example:
        [0, 1, 0] -> "0,1,0" -> SHA-256
    """
    canonical = ",".join(str(int(action)) for action in actions)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def authorization_execution_violation(
    authorized: bool | int,
    executed: bool | int,
) -> int:
    """
    Authorization-execution consistency invariant.

    Returns 1 only when downstream execution occurred without
    prior authorization.
    """
    return int(bool(executed) and not bool(authorized))


def validate_trace(
    *,
    actions: Sequence[int],
    expected_hash: str | None,
    observed_rows: int,
    expected_rows: int,
    unauthorized_invocations: int,
) -> dict[str, Any]:
    actual_hash = canonical_action_hash(actions)

    hash_match = (
        actual_hash == expected_hash
        if expected_hash is not None
        else None
    )

    row_count_match = observed_rows == expected_rows
    authorization_execution_consistent = unauthorized_invocations == 0

    validation_passed = (
        row_count_match
        and authorization_execution_consistent
        and (hash_match is not False)
    )

    return {
        "actual_hash": actual_hash,
        "expected_hash": expected_hash,
        "hash_match": hash_match,
        "observed_rows": int(observed_rows),
        "expected_rows": int(expected_rows),
        "row_count_match": int(row_count_match),
        "authorization_execution_consistent": int(
            authorization_execution_consistent
        ),
        "validation_passed": int(validation_passed),
    }