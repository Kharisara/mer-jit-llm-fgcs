#!/usr/bin/env python3
"""Prepare the minimal split-qualified primary replay input for v2.6.0.

The source v2.5.9 replay table is treated as historical preparation input only.
The generated active table contains exactly two fields:
``source_record_id`` and ``diagnostic_action``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from replaybench.workload import (  # noqa: E402
    MELD_IDENTITY_SCHEME,
    MELD_PREQUALIFIED_IDENTITY_SCHEME,
    prepare_replay_frame,
)

DEFAULT_INPUT = Path("paper_outputs/replay_input_clean.csv")
DEFAULT_OUTPUT = Path("paper_outputs/replay_input_v260.csv")
DEFAULT_MANIFEST = Path("paper_outputs/replay_input_v260_manifest.json")
OUTPUT_COLUMNS = ["source_record_id", "diagnostic_action"]
NEGATIVE_LABELS = {
    "anger",
    "angry",
    "disgust",
    "disgusted",
    "fear",
    "fearful",
    "sadness",
    "sad",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the split-qualified four-policy primary replay input."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-rows", type=int, default=11_351)
    parser.add_argument("--expected-positive", type=int, default=2_609)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Source replay CSV not found: {args.input}")

    source = pd.read_csv(args.input)
    if len(source) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} source rows; found {len(source)}")
    required = {"split", "Dialogue_ID", "Utterance_ID", "label"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Source replay CSV is missing required columns: {missing}")

    prepared = prepare_replay_frame(
        source,
        {
            "identity_scheme": MELD_IDENTITY_SCHEME,
            "require_portable_paths": False,
        },
    )
    output = pd.DataFrame(
        {
            "source_record_id": prepared["source_record_id"].astype(str),
            "diagnostic_action": (
                prepared["label"].astype(str).str.strip().str.lower().isin(NEGATIVE_LABELS)
            ).astype(int),
        }
    )[OUTPUT_COLUMNS]
    output = prepare_replay_frame(
        output,
        {
            "identity_scheme": MELD_PREQUALIFIED_IDENTITY_SCHEME,
            "require_portable_paths": True,
        },
    )

    positives = int(output["diagnostic_action"].sum())
    if len(output) != args.expected_rows:
        raise ValueError("Prepared output row count changed unexpectedly")
    if output["source_record_id"].nunique() != args.expected_rows:
        raise ValueError("Prepared source_record_id values are not one-to-one with rows")
    if not set(output["diagnostic_action"].unique()).issubset({0, 1}):
        raise ValueError("diagnostic_action must be binary")
    if positives != args.expected_positive:
        raise ValueError(
            f"Expected {args.expected_positive} diagnostic positives; found {positives}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, lineterminator="\n")

    manifest = {
        "schema_version": "replaybench-pg-primary-input-v2.6.0",
        "source_csv": args.input.as_posix(),
        "source_sha256": sha256_file(args.input),
        "output_csv": args.output.as_posix(),
        "output_sha256": sha256_file(args.output),
        "source_rows": int(len(source)),
        "output_rows": int(len(output)),
        "unique_source_record_ids": int(output["source_record_id"].nunique()),
        "identity_scheme": MELD_PREQUALIFIED_IDENTITY_SCHEME,
        "identity_provenance": MELD_IDENTITY_SCHEME,
        "output_columns": OUTPUT_COLUMNS,
        "diagnostic_action_positive_count": positives,
        "diagnostic_action_positive_rate": float(output["diagnostic_action"].mean()),
        "diagnostic_rule_source": "historical MELD emotion label mapped to a frozen binary diagnostic control",
        "diagnostic_rule_positive_labels": sorted(NEGATIVE_LABELS),
        "utterance_text_redistributed_in_output": False,
        "emotion_labels_redistributed_in_output": False,
        "state_embeddings_redistributed_in_output": False,
        "media_paths_redistributed_in_output": False,
        "data_terms_review_required_before_public_release": True,
    }
    write_json(args.manifest, manifest)

    print("[PASS] Prepared v2.6.0 primary replay input")
    print(f"[OUT] {args.output}")
    print(f"[OUT] {args.manifest}")
    print(f"[ROWS] {len(output)}")
    print(f"[UNIQUE IDS] {output['source_record_id'].nunique()}")
    print(f"[DIAGNOSTIC POSITIVES] {positives}")


if __name__ == "__main__":
    main()
