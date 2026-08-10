#!/usr/bin/env python3
"""Reconstruct the canonical ReplayBench-PG v2.6.0 MELD replay workload.

MELD source content is obtained separately by the user from the original
provider and is not redistributed by ReplayBench-PG.

The public ReplayBench-PG artifact contains only:
  * an ordered ID-only selection manifest; and
  * this deterministic reconstruction script.

The generated replay table contains exactly:
    source_record_id,diagnostic_action
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_ROWS = 11_351
EXPECTED_POSITIVES = 2_609

EXPECTED_SELECTION_SHA256 = (
    "3ecd5826976393d0c44ae3d59d5d7e7a8b8b6ccd416571dd96b564504f261646"
)

EXPECTED_OUTPUT_SHA256 = (
    "2b46fd5e6887305d2eb43b1f4383e23ed3c1062a4826e1b7a428a10bd8f84f44"
)

NEGATIVE_EMOTIONS = {
    "anger",
    "disgust",
    "fear",
    "sadness",
}

EXPECTED_MELD_EMOTIONS = {
    "neutral",
    "surprise",
    "fear",
    "sadness",
    "joy",
    "disgust",
    "anger",
}

OUTPUT_COLUMNS = [
    "source_record_id",
    "diagnostic_action",
]

DEFAULT_SELECTION = Path("data_provenance/meld_v260_record_ids.csv")
DEFAULT_OUTPUT = Path("paper_outputs/replay_input_v260.csv")
DEFAULT_MANIFEST = Path("paper_outputs/replay_input_v260_manifest.json")


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


def find_source_csv(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.is_file():
        return direct

    matches = sorted(
        path for path in root.rglob(filename)
        if path.is_file()
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} under MELD root: {root}"
        )

    if len(matches) > 1:
        formatted = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            f"Multiple copies of {filename!r} were found. "
            f"Use a MELD root containing one authoritative copy:\n{formatted}"
        )

    return matches[0]


def load_meld_split(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)

    required = {
        "Emotion",
        "Dialogue_ID",
        "Utterance_ID",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{path} is missing required MELD columns: {missing}"
        )

    dialogue = pd.to_numeric(
        frame["Dialogue_ID"],
        errors="raise",
    ).astype("int64")

    utterance = pd.to_numeric(
        frame["Utterance_ID"],
        errors="raise",
    ).astype("int64")

    emotion = (
        frame["Emotion"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    unexpected_emotions = sorted(
        set(emotion.unique()) - EXPECTED_MELD_EMOTIONS
    )
    if unexpected_emotions:
        raise ValueError(
            f"{path} contains unexpected MELD emotion labels: "
            f"{unexpected_emotions}"
        )

    source_record_id = (
        split
        + ":d"
        + dialogue.astype(str)
        + "_u"
        + utterance.astype(str)
    )

    out = pd.DataFrame(
        {
            "source_record_id": source_record_id,
            "emotion": emotion,
        }
    )

    if out["source_record_id"].duplicated().any():
        duplicates = out.loc[
            out["source_record_id"].duplicated(keep=False),
            "source_record_id",
        ].head(10).tolist()

        raise ValueError(
            f"{path} contains duplicate MELD identities: {duplicates}"
        )

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the canonical ReplayBench-PG v2.6.0 "
            "primary workload from provider-obtained MELD annotations."
        )
    )

    parser.add_argument(
        "--meld-root",
        type=Path,
        required=True,
        help=(
            "Directory containing provider-obtained train_sent_emo.csv, "
            "dev_sent_emo.csv and test_sent_emo.csv. Files may also be "
            "inside subdirectories of this root."
        ),
    )

    parser.add_argument(
        "--selection",
        type=Path,
        default=DEFAULT_SELECTION,
        help="Ordered ID-only ReplayBench-PG record-selection manifest.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Generated canonical two-column replay CSV.",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Generated provenance/reconstruction manifest.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    meld_root = args.meld_root.resolve()

    if not meld_root.is_dir():
        raise NotADirectoryError(
            f"MELD root is not a directory: {meld_root}"
        )

    if not args.selection.is_file():
        raise FileNotFoundError(
            f"Selection manifest not found: {args.selection}"
        )

    selection_hash = sha256_file(args.selection)
    if selection_hash != EXPECTED_SELECTION_SHA256:
        raise ValueError(
            "Selection-manifest SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_SELECTION_SHA256}\n"
            f"Observed: {selection_hash}"
        )

    selection = pd.read_csv(args.selection)

    if list(selection.columns) != ["source_record_id"]:
        raise ValueError(
            "Selection manifest must contain exactly one column: "
            "source_record_id"
        )

    if len(selection) != EXPECTED_ROWS:
        raise ValueError(
            f"Selection manifest must contain {EXPECTED_ROWS} rows; "
            f"found {len(selection)}"
        )

    selection["source_record_id"] = (
        selection["source_record_id"]
        .astype(str)
        .str.strip()
    )

    if selection["source_record_id"].duplicated().any():
        raise ValueError(
            "Selection manifest contains duplicate source_record_id values"
        )

    train_csv = find_source_csv(
        meld_root,
        "train_sent_emo.csv",
    )
    dev_csv = find_source_csv(
        meld_root,
        "dev_sent_emo.csv",
    )
    test_csv = find_source_csv(
        meld_root,
        "test_sent_emo.csv",
    )

    train = load_meld_split(train_csv, "train")
    dev = load_meld_split(dev_csv, "dev")
    test = load_meld_split(test_csv, "test")

    provider = pd.concat(
        [train, dev, test],
        ignore_index=True,
    )

    if provider["source_record_id"].duplicated().any():
        raise ValueError(
            "Provider MELD data produced duplicate split-qualified identities"
        )

    indexed = provider.set_index(
        "source_record_id",
        verify_integrity=True,
    )

    requested = selection["source_record_id"].tolist()

    missing = [
        record_id
        for record_id in requested
        if record_id not in indexed.index
    ]

    if missing:
        preview = missing[:20]
        raise ValueError(
            f"{len(missing)} selected ReplayBench-PG identities "
            f"were not found in provider MELD data. Examples: {preview}"
        )

    # Selection order is authoritative and is preserved exactly.
    selected = indexed.loc[requested].reset_index()

    output = pd.DataFrame(
        {
            "source_record_id": selected[
                "source_record_id"
            ].astype(str),
            "diagnostic_action": (
                selected["emotion"]
                .isin(NEGATIVE_EMOTIONS)
                .astype("int64")
            ),
        }
    )[OUTPUT_COLUMNS]

    if len(output) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} reconstructed rows; "
            f"found {len(output)}"
        )

    unique_ids = int(output["source_record_id"].nunique())
    if unique_ids != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} unique identities; "
            f"found {unique_ids}"
        )

    positives = int(output["diagnostic_action"].sum())
    if positives != EXPECTED_POSITIVES:
        raise ValueError(
            f"Expected {EXPECTED_POSITIVES} diagnostic-positive rows; "
            f"found {positives}"
        )

    if set(output["diagnostic_action"].unique()) - {0, 1}:
        raise ValueError(
            "diagnostic_action is not strictly binary"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        args.output,
        index=False,
        lineterminator="\n",
    )

    output_hash = sha256_file(args.output)

    if output_hash != EXPECTED_OUTPUT_SHA256:
        raise ValueError(
            "Canonical replay-input SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_OUTPUT_SHA256}\n"
            f"Observed: {output_hash}\n"
            "The reconstructed provider data does not exactly reproduce "
            "the v2.6.0 experiment input."
        )

    manifest = {
        "schema_version":
            "replaybench-pg-primary-input-reconstruction-v2.6.0",

        "source_dataset":
            "MELD",

        "source_dataset_redistributed":
            False,

        "source_content_in_release":
            False,

        "selection_manifest":
            args.selection.as_posix(),

        "selection_manifest_sha256":
            selection_hash,

        "selection_manifest_contains_source_content":
            False,

        "selection_manifest_contains_labels":
            False,

        "selection_manifest_rows":
            int(len(selection)),

        "provider_files": {
            "train": {
                "filename": train_csv.name,
                "sha256": sha256_file(train_csv),
                "rows": int(len(train)),
            },
            "dev": {
                "filename": dev_csv.name,
                "sha256": sha256_file(dev_csv),
                "rows": int(len(dev)),
            },
            "test": {
                "filename": test_csv.name,
                "sha256": sha256_file(test_csv),
                "rows": int(len(test)),
            },
        },

        "identity_rule":
            "<split>:d<Dialogue_ID>_u<Utterance_ID>",

        "selection_order_preserved":
            True,

        "diagnostic_action_source_field":
            "Emotion",

        "diagnostic_action_positive_labels":
            sorted(NEGATIVE_EMOTIONS),

        "output_columns":
            OUTPUT_COLUMNS,

        "output_rows":
            int(len(output)),

        "unique_source_record_ids":
            unique_ids,

        "diagnostic_action_positive_count":
            positives,

        "output_csv":
            args.output.as_posix(),

        "output_sha256":
            output_hash,

        "expected_output_sha256":
            EXPECTED_OUTPUT_SHA256,

        "canonical_output_verified":
            True,

        "utterance_text_redistributed":
            False,

        "emotion_labels_redistributed":
            False,

        "audio_or_video_redistributed":
            False,

        "state_embeddings_redistributed":
            False,
    }

    write_json(
        args.manifest,
        manifest,
    )

    print()
    print("[PASS] MELD -> ReplayBench-PG v2.6.0 reconstruction")
    print(f"provider_rows= {len(provider)}")
    print(f"selected_rows= {len(output)}")
    print(f"unique_ids= {unique_ids}")
    print(f"diagnostic_positive= {positives}")
    print(f"selection_sha256= {selection_hash}")
    print(f"output_sha256= {output_hash}")
    print(f"canonical_hash_match= {output_hash == EXPECTED_OUTPUT_SHA256}")
    print(f"[OUT] {args.output}")
    print(f"[OUT] {args.manifest}")


if __name__ == "__main__":
    main()