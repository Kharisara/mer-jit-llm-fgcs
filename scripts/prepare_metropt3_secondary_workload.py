#!/usr/bin/env python3
"""Prepare a canonical compact MetroPT-3 replay workload for ReplayBench-PG.

The script verifies the official UCI MetroPT-3 source file using its canonical
filename, data-row count, and SHA-256 digest. It then selects a deterministic
uniform sample across source-row order, normalizes the selected records, and
writes:

1. A compact ReplayBench-PG input CSV.
2. A machine-readable provenance and preparation manifest.

Dataset:
    MetroPT-3
    UCI dataset ID: 791
    DOI: 10.24432/C5VW3R
    License: CC BY 4.0

The generated workload is intended only for execution-validation experiments.
The DV electric control signal is used as a deterministic rule-gate signal. It
is not interpreted as a ground-truth equipment-failure or maintenance label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


DATASET_NAME = "MetroPT-3"
DATASET_ID = 791
DATASET_DOI = "10.24432/C5VW3R"
DATASET_URL = "https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset"
DATASET_LICENSE = "CC BY 4.0"

# Canonical UCI source verified from the official MetroPT-3 distribution.
EXPECTED_FILENAME = "MetroPT3(AirCompressor).csv"
EXPECTED_ROWS = 1_516_948
EXPECTED_SOURCE_SHA256 = (
    "db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24"
)

DEFAULT_OUTPUT_CSV = (
    "paper_outputs/secondary_metropt3/replay_input_metropt3.csv"
)
DEFAULT_MANIFEST_JSON = (
    "paper_outputs/secondary_metropt3/preparation_manifest.json"
)

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_REPORTED_TIMESTAMP_DELTAS = 20


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)

    return digest.hexdigest()


def sha256_int64_values(values: np.ndarray) -> str:
    """Hash values using a platform-independent little-endian int64 form."""

    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def weighted_median_from_counts(
    counts: Counter[int],
) -> float | None:
    """Return the exact median represented by an integer-value counter."""

    total = sum(counts.values())

    if total == 0:
        return None

    lower_position = (total - 1) // 2
    upper_position = total // 2
    cumulative = 0
    lower_value: int | None = None
    upper_value: int | None = None

    for value in sorted(counts):
        cumulative += counts[value]

        if lower_value is None and cumulative > lower_position:
            lower_value = value

        if cumulative > upper_position:
            upper_value = value
            break

    if lower_value is None or upper_value is None:
        raise RuntimeError(
            "Unable to calculate the timestamp-delta median"
        )

    return (float(lower_value) + float(upper_value)) / 2.0


def detect_delimiter(path: Path) -> str:
    """Detect comma, semicolon, or tab separation."""

    with path.open(
        "r",
        encoding="utf-8-sig",
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
                "Unable to detect a supported CSV delimiter. Expected "
                "comma, semicolon, or tab separation."
            )

        return selected


def inspect_source(path: Path, delimiter: str) -> dict:
    """Stream the complete source and collect provenance metadata."""

    rows = 0
    header: list[str] = []
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    previous_timestamp: datetime | None = None
    adjacent_duplicate_timestamps = 0
    backward_timestamp_steps = 0
    timestamp_delta_counts: Counter[int] = Counter()

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        header = list(reader.fieldnames or [])

        if "timestamp" not in header:
            raise KeyError(
                "The canonical MetroPT-3 source is missing the timestamp "
                f"column. Observed header: {header}"
            )

        for row in reader:
            raw_timestamp = row.get("timestamp")

            if raw_timestamp is None or not raw_timestamp.strip():
                raise ValueError(
                    f"Missing timestamp at source data row {rows + 1:,}"
                )

            try:
                current = datetime.strptime(
                    raw_timestamp,
                    TIMESTAMP_FORMAT,
                )
            except ValueError as exc:
                raise ValueError(
                    "Invalid timestamp at source data row "
                    f"{rows + 1:,}: {raw_timestamp!r}"
                ) from exc

            if first_timestamp is None:
                first_timestamp = raw_timestamp

            if previous_timestamp is not None:
                delta_seconds = int(
                    (current - previous_timestamp).total_seconds()
                )
                timestamp_delta_counts[delta_seconds] += 1

                if current == previous_timestamp:
                    adjacent_duplicate_timestamps += 1

                if current < previous_timestamp:
                    backward_timestamp_steps += 1

            previous_timestamp = current
            last_timestamp = raw_timestamp
            rows += 1

    timestamp_delta_median = weighted_median_from_counts(
        timestamp_delta_counts
    )

    most_common_timestamp_deltas = [
        {
            "delta_seconds": int(delta_seconds),
            "count": int(count),
        }
        for delta_seconds, count in timestamp_delta_counts.most_common(
            MAX_REPORTED_TIMESTAMP_DELTAS
        )
    ]

    return {
        "source_rows": rows,
        "header_names": header,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "timestamp_monotonic_non_decreasing": (
            backward_timestamp_steps == 0
        ),
        "adjacent_duplicate_timestamps": adjacent_duplicate_timestamps,
        "backward_timestamp_steps": backward_timestamp_steps,
        "timestamp_interval_count": int(
            sum(timestamp_delta_counts.values())
        ),
        "timestamp_unique_delta_count": int(
            len(timestamp_delta_counts)
        ),
        "timestamp_delta_min_seconds": (
            int(min(timestamp_delta_counts))
            if timestamp_delta_counts
            else None
        ),
        "timestamp_delta_median_seconds": timestamp_delta_median,
        "timestamp_delta_max_seconds": (
            int(max(timestamp_delta_counts))
            if timestamp_delta_counts
            else None
        ),
        "most_common_timestamp_deltas_seconds": (
            most_common_timestamp_deltas
        ),
    }


def normalize_column(name: str) -> str:
    """Convert a source column name into a stable normalized form."""

    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")

    aliases = {
        "": "source_index",
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


def deterministic_uniform_targets(
    total_rows: int,
    sample_size: int,
) -> np.ndarray:
    """Return deterministic uniformly spaced zero-based source positions."""

    if total_rows <= 0:
        raise ValueError("The source CSV contains no data rows")

    if sample_size <= 0:
        raise ValueError("The requested sample size must be positive")

    effective_sample_size = min(sample_size, total_rows)

    if effective_sample_size == 1:
        return np.asarray([0], dtype=np.int64)

    # Exact integer arithmetic avoids floating-point and NumPy-version
    # differences while matching floor-based integer linspace for this range.
    numerators = (
        np.arange(effective_sample_size, dtype=np.int64)
        * np.int64(total_rows - 1)
    )
    targets = numerators // np.int64(effective_sample_size - 1)

    if len(targets) != effective_sample_size:
        raise RuntimeError(
            "The deterministic sampler produced an unexpected target count"
        )

    if len(np.unique(targets)) != effective_sample_size:
        raise RuntimeError(
            "The deterministic sampler produced duplicate target positions"
        )

    return targets


def deterministic_uniform_sample(
    source_csv: Path,
    delimiter: str,
    targets: np.ndarray,
    chunksize: int,
) -> pd.DataFrame:
    """Select target source rows without loading the full CSV into memory."""

    if chunksize <= 0:
        raise ValueError("The CSV chunk size must be positive")

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
            "requested sample rows"
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
    """Convert positive numeric values to one and other values to zero."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)

    return (numeric > 0).astype(np.int8)


def prepare_replay_frame(sample: pd.DataFrame) -> pd.DataFrame:
    """Normalize sampled MetroPT-3 rows into ReplayBench-PG input form."""

    sample = sample.rename(
        columns={
            column: normalize_column(column)
            for column in sample.columns
        }
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
        "source_index",
    }
    missing_columns = sorted(
        required_columns - set(sample.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required normalized columns "
            f"{missing_columns}. Available columns: "
            f"{sample.columns.tolist()}"
        )

    source_index = pd.to_numeric(
        sample["source_index"],
        errors="coerce",
    )

    if source_index.isna().any():
        raise ValueError(
            "The source_index column contains non-numeric or missing values"
        )

    if not np.equal(
        source_index,
        np.floor(source_index),
    ).all():
        raise ValueError(
            "The source_index column contains non-integer values"
        )

    sample["source_index"] = source_index.astype(np.int64)

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

    default_gate = sample["dv_electric"].astype(np.int8)
    output = sample.copy()

    utterance_ids = [
        f"metropt3_{int(source_index_value):08d}"
        for source_index_value in output["source_index"]
    ]

    output.insert(
        0,
        "utterance_id",
        utterance_ids,
    )
    output.insert(
        1,
        "source_record_id",
        output["utterance_id"],
    )

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

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--source-csv",
        required=True,
        help="Path to the canonical UCI MetroPT-3 CSV",
    )

    parser.add_argument(
        "--archive-path",
        default=None,
        help=(
            "Optional path to the downloaded UCI ZIP. Its name, size, and "
            "SHA-256 are recorded in the manifest."
        ),
    )

    parser.add_argument(
        "--download-date",
        default=None,
        help="Optional archive download date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help="Destination for the prepared replay-input CSV",
    )

    parser.add_argument(
        "--manifest-json",
        default=DEFAULT_MANIFEST_JSON,
        help="Destination for the preparation manifest",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=20_000,
        help="Number of uniformly distributed source rows to select",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Number of source rows parsed per pandas chunk",
    )

    return parser.parse_args()


def main() -> None:
    """Prepare the canonical secondary MetroPT-3 replay workload."""

    args = parse_args()

    source = Path(args.source_csv)
    output = Path(args.output_csv)
    manifest_path = Path(args.manifest_json)
    archive = (
        Path(args.archive_path)
        if args.archive_path
        else None
    )

    if not source.is_file():
        raise FileNotFoundError(
            f"MetroPT-3 source CSV not found: {source}"
        )

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")

    if args.download_date is not None:
        try:
            datetime.strptime(
                args.download_date,
                "%Y-%m-%d",
            )
        except ValueError as exc:
            raise ValueError(
                "--download-date must use YYYY-MM-DD format"
            ) from exc

    if archive is not None and not archive.is_file():
        raise FileNotFoundError(
            f"MetroPT-3 archive not found: {archive}"
        )

    if archive is not None and args.download_date is None:
        raise ValueError(
            "--download-date is required when --archive-path is supplied"
        )

    source_resolved = source.resolve()
    output_resolved = output.resolve()
    manifest_resolved = manifest_path.resolve()

    if source_resolved in {
        output_resolved,
        manifest_resolved,
    }:
        raise ValueError(
            "The source CSV cannot also be used as an output path"
        )

    if output_resolved == manifest_resolved:
        raise ValueError(
            "--output-csv and --manifest-json must be different paths"
        )

    delimiter = detect_delimiter(source)
    source_sha256 = sha256_file(source)
    source_metadata = inspect_source(
        source,
        delimiter,
    )
    total_rows = int(
        source_metadata["source_rows"]
    )

    provenance_errors: list[str] = []

    if source.name != EXPECTED_FILENAME:
        provenance_errors.append(
            f"filename {source.name!r} != {EXPECTED_FILENAME!r}"
        )

    if total_rows != EXPECTED_ROWS:
        provenance_errors.append(
            f"row count {total_rows:,} != {EXPECTED_ROWS:,}"
        )

    if source_sha256 != EXPECTED_SOURCE_SHA256:
        provenance_errors.append(
            "SHA-256 does not match the verified canonical UCI file"
        )

    if provenance_errors:
        raise RuntimeError(
            "MetroPT-3 provenance verification failed: "
            + "; ".join(provenance_errors)
        )

    targets = deterministic_uniform_targets(
        total_rows,
        args.sample_size,
    )

    sampled = deterministic_uniform_sample(
        source,
        delimiter,
        targets,
        args.chunksize,
    )

    replay = prepare_replay_frame(sampled)

    if replay.empty:
        raise RuntimeError(
            "The prepared replay workload is empty"
        )

    if replay["utterance_id"].duplicated().any():
        duplicate_count = int(
            replay["utterance_id"].duplicated().sum()
        )
        raise RuntimeError(
            "The prepared workload contains "
            f"{duplicate_count:,} duplicate utterance IDs"
        )

    sampled_source_indices = replay[
        "source_index"
    ].to_numpy(
        dtype=np.int64,
        copy=True,
    )

    # The first CSV column is an identifier exported with the canonical UCI
    # dataset. It is preserved for record-level traceability, but it is not
    # assumed to equal the physical zero-based row position in the CSV.
    source_index_matches_sampling_targets = bool(
        np.array_equal(
            sampled_source_indices,
            targets,
        )
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    replay.to_csv(
        output,
        index=False,
    )

    gate = truthy_binary(
        replay["dv_electric"]
    )

    binary_columns = [
        "lps",
        "oil_level",
        "pressure_switch",
        "comp",
        "dv_electric",
        "towers",
        "mpg",
    ]

    archive_metadata = None

    if archive is not None:
        archive_metadata = {
            "archive_filename": archive.name,
            "archive_file_size_bytes": archive.stat().st_size,
            "archive_sha256": sha256_file(archive),
        }

    sampling_target_sha256 = sha256_int64_values(
        targets
    )
    sampled_source_index_sha256 = sha256_int64_values(
        sampled_source_indices
    )

    manifest = {
        "dataset": DATASET_NAME,
        "dataset_id": DATASET_ID,
        "dataset_doi": DATASET_DOI,
        "dataset_url": DATASET_URL,
        "dataset_license": DATASET_LICENSE,
        "download_date": args.download_date,
        "archive": archive_metadata,
        "source_filename": source.name,
        "expected_source_filename": EXPECTED_FILENAME,
        "source_file_size_bytes": source.stat().st_size,
        "source_sha256": source_sha256,
        "expected_source_sha256": EXPECTED_SOURCE_SHA256,
        "source_rows": total_rows,
        "expected_source_rows": EXPECTED_ROWS,
        "filename_verified": (
            source.name == EXPECTED_FILENAME
        ),
        "row_count_verified": (
            total_rows == EXPECTED_ROWS
        ),
        "hash_verified": (
            source_sha256 == EXPECTED_SOURCE_SHA256
        ),
        "provenance_verified": True,
        "delimiter": delimiter,
        "header_names": source_metadata[
            "header_names"
        ],
        "first_source_timestamp": source_metadata[
            "first_timestamp"
        ],
        "last_source_timestamp": source_metadata[
            "last_timestamp"
        ],
        "timestamp_monotonic_non_decreasing": source_metadata[
            "timestamp_monotonic_non_decreasing"
        ],
        "adjacent_duplicate_timestamps": source_metadata[
            "adjacent_duplicate_timestamps"
        ],
        "backward_timestamp_steps": source_metadata[
            "backward_timestamp_steps"
        ],
        "timestamp_interval_count": source_metadata[
            "timestamp_interval_count"
        ],
        "timestamp_unique_delta_count": source_metadata[
            "timestamp_unique_delta_count"
        ],
        "timestamp_delta_min_seconds": source_metadata[
            "timestamp_delta_min_seconds"
        ],
        "timestamp_delta_median_seconds": source_metadata[
            "timestamp_delta_median_seconds"
        ],
        "timestamp_delta_max_seconds": source_metadata[
            "timestamp_delta_max_seconds"
        ],
        "most_common_timestamp_deltas_seconds": source_metadata[
            "most_common_timestamp_deltas_seconds"
        ],
        "timestamp_cadence_interpretation": (
            "Descriptive interval statistics only; no fixed 1 Hz cadence "
            "is assumed."
        ),
        "sample_strategy": (
            "deterministic_uniform_across_zero_based_source_row_order"
        ),
        "sampling_formula": (
            "target[i] = floor(i * (source_rows - 1) / "
            "(effective_sample_size - 1)); special case target[0] = 0 "
            "when effective_sample_size = 1"
        ),
        "requested_sample_size": int(
            args.sample_size
        ),
        "effective_sample_size": int(
            len(targets)
        ),
        "sample_rows": int(
            len(replay)
        ),
        "selected_source_row_position_min": int(
            targets.min()
        ),
        "selected_source_row_position_max": int(
            targets.max()
        ),
        "selected_source_row_positions_sha256": (
            sampling_target_sha256
        ),
        "exported_source_index_min": int(
            sampled_source_indices.min()
        ),
        "exported_source_index_max": int(
            sampled_source_indices.max()
        ),
        "exported_source_index_sha256": (
            sampled_source_index_sha256
        ),
        "exported_source_index_matches_selected_row_positions": (
            source_index_matches_sampling_targets
        ),
        "exported_source_index_semantics": (
            "Identifier supplied in the canonical UCI CSV first column; "
            "preserved for record-level traceability and not assumed to "
            "equal the physical zero-based source-row position."
        ),
        "chunksize": int(
            args.chunksize
        ),
        "first_sample_timestamp": str(
            replay["timestamp"].iloc[0]
        ),
        "last_sample_timestamp": str(
            replay["timestamp"].iloc[-1]
        ),
        "default_rule_gate": (
            "dv_electric == 1"
        ),
        "default_rule_gate_semantics": (
            "Binary equipment-control signal used as a deterministic replay "
            "gate; not a ground-truth fault or maintenance label."
        ),
        "default_rule_gate_positive_count": int(
            gate.sum()
        ),
        "default_rule_gate_negative_count": int(
            (gate == 0).sum()
        ),
        "default_rule_gate_positive_rate": float(
            gate.mean()
        ),
        "label_mapping": {
            "1": "control_signal_active",
            "0": "control_signal_inactive",
        },
        "binary_signal_summary": summarize_binary_columns(
            replay,
            binary_columns,
        ),
        "output_csv": str(
            output
        ),
        "output_file_size_bytes": (
            output.stat().st_size
        ),
        "output_sha256": sha256_file(
            output
        ),
        "preparation_software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OUT] {output}")
    print(f"[OUT] {manifest_path}")
    print(f"[INFO] Source rows: {total_rows:,}")
    print(f"[INFO] Prepared rows: {len(replay):,}")
    print(
        "[INFO] Observed timestamp median delta: "
        f"{source_metadata['timestamp_delta_median_seconds']} seconds"
    )
    print(
        "[INFO] Default rule-gate rate: "
        f"{gate.mean():.6f} "
        f"({int(gate.sum()):,}/{len(gate):,})"
    )
    print(
        f"[INFO] Source SHA-256: {source_sha256}"
    )
    print(
        "[INFO] Selected row-position SHA-256: "
        f"{sampling_target_sha256}"
    )
    print(
        "[INFO] Exported source-index SHA-256: "
        f"{sampled_source_index_sha256}"
    )
    print(
        "[INFO] Exported source index matches selected row positions: "
        f"{source_index_matches_sampling_targets}"
    )
    print(
        f"[INFO] Output SHA-256: "
        f"{manifest['output_sha256']}"
    )
    print(
        "[INFO] Canonical UCI provenance verification: PASS"
    )


if __name__ == "__main__":
    main()