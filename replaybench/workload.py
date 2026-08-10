from __future__ import annotations

import ntpath
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

MELD_IDENTITY_SCHEME = "meld_split_dialogue_utterance"
MELD_PREQUALIFIED_IDENTITY_SCHEME = "meld_split_qualified_source_record_id"
EXISTING_IDENTITY_SCHEME = "existing_source_record_id"
CANONICAL_SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9_\-]+:d\d+_u\d+$")


class WorkloadContractError(ValueError):
    """Raised when a replay workload violates the active v2.6.0 contract."""


def _integer_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise WorkloadContractError(f"Missing required identity column: {column}")
    numeric = pd.to_numeric(frame[column], errors="raise")
    if numeric.isna().any():
        raise WorkloadContractError(f"Identity column {column} contains missing values")
    if not (numeric == numeric.astype("int64")).all():
        raise WorkloadContractError(f"Identity column {column} contains non-integer values")
    return numeric.astype("int64")


def canonical_meld_source_ids(frame: pd.DataFrame) -> pd.Series:
    """Build split-qualified MELD identities in source order as split:dN_uM."""
    if "split" not in frame.columns:
        raise WorkloadContractError("Missing required identity column: split")
    split = frame["split"].astype(str).str.strip().str.lower()
    if split.eq("").any() or split.isin({"nan", "none", "null"}).any():
        raise WorkloadContractError("Identity column split contains empty values")
    allowed_splits = {"train", "dev", "test"}
    unexpected = sorted(set(split.unique()) - allowed_splits)
    if unexpected:
        raise WorkloadContractError(
            f"MELD split values must be {sorted(allowed_splits)}; observed={unexpected}"
        )
    dialogue = _integer_series(frame, "Dialogue_ID")
    utterance = _integer_series(frame, "Utterance_ID")
    return split + ":d" + dialogue.astype(str) + "_u" + utterance.astype(str)


def _looks_absolute_path(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    return Path(text).is_absolute() or ntpath.isabs(text)


def assert_portable_path_columns(frame: pd.DataFrame) -> None:
    path_columns = [
        column
        for column in frame.columns
        if str(column).lower().endswith("_path")
        or str(column).lower() in {"path", "filepath", "file_path"}
    ]
    violations: list[str] = []
    for column in path_columns:
        count = int(frame[column].map(_looks_absolute_path).sum())
        if count:
            violations.append(f"{column}={count}")
    if violations:
        raise WorkloadContractError(
            "Replay input contains local absolute paths: " + ", ".join(violations)
        )


def prepare_replay_frame(
    frame: pd.DataFrame,
    dataset_config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Validate and normalize a replay frame under a configured identity scheme."""
    config = dataset_config or {}
    scheme = str(config.get("identity_scheme", EXISTING_IDENTITY_SCHEME)).strip().lower()
    require_portable_paths = bool(config.get("require_portable_paths", True))

    out = frame.copy().reset_index(drop=True)
    if out.empty:
        raise WorkloadContractError("Replay input contains no rows")

    if scheme == MELD_IDENTITY_SCHEME:
        expected = canonical_meld_source_ids(out)
        if "source_record_id" in out.columns:
            observed = out["source_record_id"].astype(str)
            mismatch = observed.ne(expected)
            if mismatch.any():
                first = int(mismatch[mismatch].index[0])
                raise WorkloadContractError(
                    "source_record_id does not match the split-qualified MELD "
                    f"identity at row {first}: observed={observed.iloc[first]!r}, "
                    f"expected={expected.iloc[first]!r}"
                )
        out["source_record_id"] = expected
    elif scheme in {MELD_PREQUALIFIED_IDENTITY_SCHEME, EXISTING_IDENTITY_SCHEME}:
        if "source_record_id" not in out.columns:
            raise WorkloadContractError(
                "Replay input must provide source_record_id for the configured identity_scheme"
            )
        out["source_record_id"] = out["source_record_id"].astype(str).str.strip()
    else:
        raise WorkloadContractError(f"Unsupported dataset.identity_scheme: {scheme!r}")

    if out["source_record_id"].eq("").any():
        raise WorkloadContractError("source_record_id contains empty values")
    duplicate_mask = out["source_record_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = out.loc[duplicate_mask, "source_record_id"].head(10).tolist()
        raise WorkloadContractError(
            "source_record_id must be one-to-one with replay rows; examples="
            f"{duplicates}"
        )

    if scheme in {MELD_IDENTITY_SCHEME, MELD_PREQUALIFIED_IDENTITY_SCHEME}:
        malformed = ~out["source_record_id"].str.match(CANONICAL_SOURCE_ID_PATTERN)
        if malformed.any():
            raise WorkloadContractError("A canonical MELD source_record_id is malformed")
        prefixes = out["source_record_id"].str.split(":", n=1).str[0]
        unexpected = sorted(set(prefixes.unique()) - {"train", "dev", "test"})
        if unexpected:
            raise WorkloadContractError(
                f"Canonical MELD identifiers contain unexpected splits: {unexpected}"
            )

    if require_portable_paths:
        assert_portable_path_columns(out)

    return out


def load_replay_frame(
    path: str | Path,
    dataset_config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Replay input CSV not found: {input_path}")
    return prepare_replay_frame(pd.read_csv(input_path), dataset_config)
