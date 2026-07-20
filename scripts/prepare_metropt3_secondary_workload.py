#!/usr/bin/env python3
"""Prepare a compact MetroPT-3 replay workload for ReplayBench-PG.

The script reads a verified MetroPT-3 CSV, selects a deterministic uniform
sample across the complete chronological source, normalizes column names, and
writes:

1. A compact replay-input CSV.
2. A machine-readable preparation manifest.

The source dataset does not need to be committed to the repository.

Dataset:
    MetroPT-3
    UCI dataset ID: 791
    DOI: 10.24432/C5VW3R
    License: CC BY 4.0

The generated workload is intended for execution-validation experiments. The
DV electric control signal is used as a deterministic rule-gate signal. It is
not interpreted as ground-truth equipment failure or maintenance necessity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


DATASET_NAME = "MetroPT-3"
DATASET_DOI = "10.24432/C5VW3R"
DATASET_URL = (
    "https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset"
)
DATASET_LICENSE = "CC BY 4.0"

# Verified number of data rows in the complete MetroPT-3 compressor CSV.
EXPECTED_ROWS = 15_169_480

DEFAULT_OUTPUT_CSV = (
    "paper_outputs/secondary_metropt3/replay_input_metropt3.csv"
)
DEFAULT_MANIFEST_JSON = (
    "paper_outputs/secondary_metropt3/preparation_manifest.json"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)

    return digest.hexdigest()


def detect_delimiter(path: Path) -> str:
    """Detect whether the CSV uses comma, semicolon, or tab separation."""

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as handle:
        sample = handle.read(8192)

    if not sample.strip():
        raise ValueError(f"The source file is empty: {path}")

    try:
        return csv.Sniffer().sniff(
            sample,
            delimiters=[",", ";", "\t"],
        ).delimiter
    except csv.Error:
        first_line = sample.splitlines()[0]
        counts = {
            delimiter: first_line.count(delimiter)
            for delimiter in [",", ";", "\t"]
        }

        selected = max(counts, key=counts.get)

        if counts[selected] == 0:
            raise ValueError(
                "Unable to detect a supported CSV delimiter. "
                "Expected comma, semicolon, or tab separation."
            )

        return selected


def count_data_rows(path: Path) -> int:
    """Count data rows while excluding the CSV header."""

    with path.open("rb") as handle:
        line_count = sum(1 for _ in handle)

    return max(0, line_count - 1)


def normalize_column(name: str) -> str:
    """Convert a source column name into a stable normalized form."""

    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")

    aliases = {
        "unnamed_0": "source_index",
        "index": "source_index",
        "dv_eletric": "dv_electric",
        "dv_electric": "dv_electric",
        "pressure_switch": "pressure_switch",
        "oil_level": "oil_level",
        "caudal_impulse": "caudal_impulses",
        "caudal_impulses": "caudal_impulses",
        "oil_temperature": "oil_temperature",
        "motor_current": "motor_current",
        "dv_pressure": "dv_pressure",
    }

    return aliases.get(text, text)


def deterministic_uniform_sample(
    source_csv: Path,
    delimiter: str,
    total_rows: int,
    sample_size: int,
    chunksize: int,
) -> pd.DataFrame:
    """Select uniformly spaced source rows without loading the full file."""

    if total_rows <= 0:
        raise ValueError("The MetroPT-3 source CSV contains no data rows")

    if sample_size <= 0:
        raise ValueError("The requested sample size must be positive")

    if chunksize <= 0:
        raise ValueError("The CSV chunk size must be positive")

    effective_sample_size = min(sample_size, total_rows)

    targets = np.unique(
        np.linspace(
            0,
            total_rows - 1,
            num=effective_sample_size,
            dtype=np.int64,
        )
    )

    selected: List[pd.DataFrame] = []
    target_position = 0
    source_offset = 0

    for chunk in pd.read_csv(
        source_csv,
        sep=delimiter,
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8-sig",
    ):
        chunk_end = source_offset + len(chunk)
        target_start = target_position

        while (
            target_position < len(targets)
            and targets[target_position] < chunk_end
        ):
            target_position += 1

        local_targets = (
            targets[target_start:target_position] - source_offset
        )

        if len(local_targets) > 0:
            selected.append(chunk.iloc[local_targets].copy())

        source_offset = chunk_end

    if target_position != len(targets):
        raise RuntimeError(
            f"Located only {target_position:,} of {len(targets):,} "
            "requested sample rows. The measured source-row count may not "
            "match the number of rows parsed by pandas."
        )

    if not selected:
        raise RuntimeError("No rows were selected from the source CSV")

    sampled = pd.concat(selected, ignore_index=True)

    if len(sampled) != len(targets):
        raise RuntimeError(
            f"Expected {len(targets):,} sampled rows but produced "
            f"{len(sampled):,}"
        )

    return sampled


def truthy_binary(series: pd.Series) -> pd.Series:
    """Convert positive numeric values to one and all other values to zero."""

    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return (numeric > 0).astype(np.int8)


def prepare_replay_frame(sample: pd.DataFrame) -> pd.DataFrame:
    """Normalize sampled MetroPT-3 rows into ReplayBench-PG input form."""

    sample = sample.rename(
        columns={column: normalize_column(column) for column in sample.columns}
    )

    duplicated_columns = sample.columns[
        sample.columns.duplicated()
    ].tolist()

    if duplicated_columns:
        raise ValueError(
            "Column normalization created duplicate columns: "
            f"{duplicated_columns}"
        )

    required_columns = {
        "timestamp",
        "dv_electric",
    }

    missing_columns = sorted(
        required_columns - set(sample.columns)
    )

    if missing_columns:
        raise KeyError(
            "MetroPT-3 source is missing required normalized columns "
            f"{missing_columns}. Available columns: "
            f"{sample.columns.tolist()}"
        )

    if "source_index" not in sample.columns:
        sample.insert(
            0,
            "source_index",
            np.arange(len(sample), dtype=np.int64),
        )
    else:
        source_index = pd.to_numeric(
            sample["source_index"],
            errors="coerce",
        )

        if source_index.isna().any():
            raise ValueError(
                "The source_index column contains non-numeric or missing values"
            )

        sample["source_index"] = source_index.astype(np.int64)

    # Normalize all available binary equipment/control signals.
    binary_columns = [
        "lps",
        "oil_level",
        "pressure_switch",
        "comp",
        "dv_electric",
        "towers",
        "mpg",
    ]

    for column in binary_columns:
        if column in sample.columns:
            sample[column] = truthy_binary(sample[column])

    # Replay policy signal:
    # DV_electric is a real binary equipment-control state. It is used only as
    # a deterministic gate input and is not treated as a fault label.
    default_gate = sample["dv_electric"].astype(np.int8)

    output = sample.copy()

    utterance_ids = [
        f"metropt3_{int(source_index):08d}"
        for source_index in output["source_index"]
    ]

    output.insert(0, "utterance_id", utterance_ids)
    output.insert(1, "source_record_id", output["utterance_id"])

    output["label"] = np.where(
        default_gate == 1,
        "control_signal_active",
        "control_signal_inactive",
    )

    output["split"] = "secondary_validation"
    output["text"] = ""
    output["state_path"] = ""
    output["has_text"] = False
    output["has_audio"] = False
    output["has_video"] = False
    output["state_source"] = "metropt3_sensor_row"
    output["dataset_name"] = DATASET_NAME

    front_columns = [
        "utterance_id",
        "source_record_id",
        "timestamp",
        "label",
        "split",
        "dataset_name",
        "text",
        "state_path",
        "has_text",
        "has_audio",
        "has_video",
        "state_source",
        "source_index",
    ]

    remaining_columns = [
        column
        for column in output.columns
        if column not in front_columns
    ]

    return output[front_columns + remaining_columns]


def summarize_binary_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
) -> Dict[str, Dict[str, float | int]]:
    """Summarize the prevalence of each available binary signal."""

    summary: Dict[str, Dict[str, float | int]] = {}

    for column in columns:
        if column not in dataframe.columns:
            continue

        values = truthy_binary(dataframe[column])

        summary[column] = {
            "positive_count": int(values.sum()),
            "positive_rate": float(values.mean()),
        }

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--source-csv",
        required=True,
        help="Path to the verified MetroPT-3 compressor CSV.",
    )

    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help="Destination path for the generated replay-input CSV.",
    )

    parser.add_argument(
        "--manifest-json",
        default=DEFAULT_MANIFEST_JSON,
        help="Destination path for the preparation manifest.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=20_000,
        help="Number of uniformly distributed source rows to select.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Number of source rows parsed per pandas chunk.",
    )

    parser.add_argument(
        "--allow-row-count-mismatch",
        action="store_true",
        help=(
            "Allow a source-row count different from the verified "
            f"MetroPT-3 count of {EXPECTED_ROWS:,} rows."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Prepare the secondary MetroPT-3 replay workload."""

    args = parse_args()

    source = Path(args.source_csv)
    output = Path(args.output_csv)
    manifest_path = Path(args.manifest_json)

    if not source.exists():
        raise FileNotFoundError(f"MetroPT-3 source CSV not found: {source}")

    if not source.is_file():
        raise ValueError(f"The source path is not a file: {source}")

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")

    delimiter = detect_delimiter(source)
    total_rows = count_data_rows(source)

    if (
        total_rows != EXPECTED_ROWS
        and not args.allow_row_count_mismatch
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS:,} MetroPT-3 data rows but found "
            f"{total_rows:,}. Use --allow-row-count-mismatch only after "
            "verifying the source file and documenting the difference."
        )

    sampled = deterministic_uniform_sample(
        source_csv=source,
        delimiter=delimiter,
        total_rows=total_rows,
        sample_size=args.sample_size,
        chunksize=args.chunksize,
    )

    replay = prepare_replay_frame(sampled)

    if replay.empty:
        raise RuntimeError("The prepared replay workload is empty")

    if replay["utterance_id"].duplicated().any():
        duplicate_count = int(
            replay["utterance_id"].duplicated().sum()
        )
        raise RuntimeError(
            f"The prepared workload contains {duplicate_count:,} duplicate "
            "utterance IDs"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    replay.to_csv(output, index=False)

    gate = truthy_binary(replay["dv_electric"])

    binary_columns = [
        "lps",
        "oil_level",
        "pressure_switch",
        "comp",
        "dv_electric",
        "towers",
        "mpg",
    ]

    manifest = {
        "dataset": DATASET_NAME,
        "dataset_doi": DATASET_DOI,
        "dataset_url": DATASET_URL,
        "dataset_license": DATASET_LICENSE,
        "source_filename": source.name,
        "source_sha256": sha256_file(source),
        "source_rows": int(total_rows),
        "expected_source_rows": int(EXPECTED_ROWS),
        "row_count_verified": bool(total_rows == EXPECTED_ROWS),
        "sample_strategy": "deterministic_uniform_across_source_order",
        "requested_sample_size": int(args.sample_size),
        "sample_rows": int(len(replay)),
        "chunksize": int(args.chunksize),
        "delimiter": delimiter,
        "first_timestamp": str(replay["timestamp"].iloc[0]),
        "last_timestamp": str(replay["timestamp"].iloc[-1]),
        "default_rule_gate": "dv_electric == 1",
        "default_rule_gate_semantics": (
            "Binary equipment-control signal used as a deterministic replay "
            "gate; not a ground-truth fault or maintenance label."
        ),
        "default_rule_gate_positive_count": int(gate.sum()),
        "default_rule_gate_negative_count": int((gate == 0).sum()),
        "default_rule_gate_positive_rate": float(gate.mean()),
        "label_mapping": {
            "1": "control_signal_active",
            "0": "control_signal_inactive",
        },
        "binary_signal_summary": summarize_binary_columns(
            replay,
            binary_columns,
        ),
        "output_csv": str(output),
        "output_sha256": sha256_file(output),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"[OUT] {output}")
    print(f"[OUT] {manifest_path}")
    print(f"[INFO] Source rows: {total_rows:,}")
    print(f"[INFO] Prepared rows: {len(replay):,}")
    print(
        "[INFO] Default rule-gate rate: "
        f"{gate.mean():.6f} "
        f"({int(gate.sum()):,}/{len(gate):,})"
    )
    print(f"[INFO] Source SHA-256: {manifest['source_sha256']}")
    print(f"[INFO] Output SHA-256: {manifest['output_sha256']}")


if __name__ == "__main__":
    main()