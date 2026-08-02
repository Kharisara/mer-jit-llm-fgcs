#!/usr/bin/env python3
from __future__ import annotations

"""Post-hoc scoring for frozen label-independent validator findings."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from replaybench.generic_validator import FORBIDDEN_TRACE_COLUMNS, GenericValidatorFindings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="paper_outputs/phase1_label_independent_validation",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_findings(row: dict[str, Any]) -> GenericValidatorFindings:
    values: dict[str, Any] = {}
    for field in GenericValidatorFindings.__dataclass_fields__:
        value = row[field]
        if isinstance(value, list):
            value = tuple(map(str, value))
        values[field] = value
    return GenericValidatorFindings(**values)


def _localized_ids(findings: GenericValidatorFindings, mode: str) -> tuple[bool, set[str]]:
    mapping = {
        "saved_action_corruption": findings.action_mismatch_replay_point_ids,
        "dropped_rows": findings.missing_replay_point_ids,
        "duplicated_rows": findings.duplicate_replay_point_ids,
        "logged_unauthorized_invocation": findings.primary_unauthorized_replay_point_ids,
        "unlogged_downstream_call": findings.receipt_unlogged_replay_point_ids,
        "false_execution_log": findings.receipt_missing_replay_point_ids,
        "duplicate_downstream_call": findings.receipt_duplicate_replay_point_ids,
        "mismatched_correlation_id": findings.receipt_mismatched_replay_point_ids,
    }
    if mode not in mapping:
        return False, set()
    return True, set(map(str, mapping[mode]))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.input_dir)
    findings_rows = _read_jsonl(output_dir / "generic_validator_findings.jsonl")
    truth_rows = _read_jsonl(output_dir / "ground_truth_manifest.jsonl")
    audit_df = pd.read_csv(output_dir / "validator_input_separation_audit.csv")
    findings_by_id = {
        str(row["evidence_id"]): _as_findings(row) for row in findings_rows
    }
    findings_df = pd.DataFrame(findings_rows)
    truth_df = pd.DataFrame(truth_rows)
    if len(findings_df) != len(truth_df):
        raise RuntimeError("Findings and ground-truth counts differ")
    if findings_df["evidence_id"].duplicated().any():
        raise RuntimeError("Duplicate finding evidence IDs")
    if truth_df["evidence_id"].duplicated().any():
        raise RuntimeError("Duplicate ground-truth evidence IDs")

    join_columns = [
        "evidence_id",
        "evidence_unit_type",
        "policy_mode",
        "seed",
        "workers",
        "source_kind",
    ]
    scored = findings_df.merge(
        truth_df,
        on=join_columns,
        validate="one_to_one",
        suffixes=("_finding", "_truth"),
    )
    scored["expected_trigger"] = scored["positive_control"].astype(int)
    scored["primary_correct"] = (
        scored["primary_validator_triggered"].astype(int)
        == scored["expected_trigger"].astype(int)
    ).astype(int)
    scored["full_correct"] = (
        scored["full_validator_triggered"].astype(int)
        == scored["expected_trigger"].astype(int)
    ).astype(int)

    localization_rows: list[dict[str, Any]] = []
    for row in scored.to_dict(orient="records"):
        evidence_id = str(row["evidence_id"])
        findings = findings_by_id[evidence_id]
        mode = str(row["fault_mode"])
        supported, localized = _localized_ids(findings, mode)
        targets = set(map(str, row["target_replay_point_ids"]))
        tp = len(targets & localized) if supported else 0
        fp = len(localized - targets) if supported else 0
        fn = len(targets - localized) if supported else 0
        localization_rows.append(
            {
                "evidence_id": evidence_id,
                "fault_mode": mode,
                "event_localization_supported": int(supported),
                "target_events": len(targets),
                "localized_events": len(localized),
                "event_true_positives": tp,
                "event_false_positives": fp,
                "event_false_negatives": fn,
                "target_replay_point_ids_json": json.dumps(sorted(targets)),
                "localized_replay_point_ids_json": json.dumps(sorted(localized)),
            }
        )
    localization_df = pd.DataFrame(localization_rows)
    scored = scored.merge(
        localization_df.drop(columns=["fault_mode"]),
        on="evidence_id",
        validate="one_to_one",
    )

    positive = scored.loc[scored["positive_control"].astype(int).eq(1)].copy()
    benign = scored.loc[scored["positive_control"].astype(int).eq(0)].copy()
    comparison = (
        scored.groupby(
            ["fault_mode", "evidence_unit_type", "positive_control"],
            as_index=False,
        )
        .agg(
            evidence_units=("evidence_id", "nunique"),
            target_events=("target_count", "sum"),
            primary_detected=("primary_validator_triggered", "sum"),
            full_detected=("full_validator_triggered", "sum"),
            primary_correct=("primary_correct", "sum"),
            full_correct=("full_correct", "sum"),
            localization_supported_units=("event_localization_supported", "sum"),
            event_true_positives=("event_true_positives", "sum"),
            event_false_positives=("event_false_positives", "sum"),
            event_false_negatives=("event_false_negatives", "sum"),
        )
        .sort_values(["positive_control", "fault_mode"], ascending=[False, True])
        .reset_index(drop=True)
    )

    if not audit_df["validator_input_label_independent"].astype(int).eq(1).all():
        raise RuntimeError("A validator input retained forbidden fault-label fields")
    if not positive["full_validator_triggered"].astype(int).eq(1).all():
        raise RuntimeError("The full validator missed at least one positive control")
    if not benign["full_validator_triggered"].astype(int).eq(0).all():
        raise RuntimeError("The full validator flagged at least one benign control")
    localized = localization_df.loc[
        localization_df["event_localization_supported"].astype(int).eq(1)
    ]
    if int(localized["event_false_positives"].sum()) != 0:
        raise RuntimeError("Event localization produced false-positive identifiers")
    if int(localized["event_false_negatives"].sum()) != 0:
        raise RuntimeError("Event localization missed injected identifiers")

    scored.to_csv(output_dir / "per_evidence_scored_results.csv", index=False)
    localization_df.to_csv(output_dir / "event_localization_results.csv", index=False)
    comparison.to_csv(output_dir / "baseline_comparison_by_fault_class.csv", index=False)

    clean_units = int(truth_df["fault_mode"].astype(str).eq("clean").sum())
    benign_control_units = int(
        truth_df["source_kind"].astype(str).eq("post_execution_benign_control").sum()
    )
    summary: dict[str, Any] = {
        "schema_version": "replaybench-pg-phase1-validation-v1",
        "status": "passed",
        "method": (
            "Generic findings were frozen before a one-to-one post-hoc join "
            "with a separate ground-truth manifest."
        ),
        "generic_validator_findings": int(len(findings_df)),
        "ground_truth_records": int(len(truth_df)),
        "receipt_execution_instances": int(
            (truth_df["source_kind"] == "frozen_receipt_execution").sum()
        ),
        "clean_reference_units": clean_units,
        "benign_control_units": benign_control_units,
        "negative_control_units": int(len(benign)),
        "positive_control_units": int(len(positive)),
        "primary_detected_positive_units": int(
            positive["primary_validator_triggered"].sum()
        ),
        "full_detected_positive_units": int(
            positive["full_validator_triggered"].sum()
        ),
        "full_false_positive_units": int(
            benign["full_validator_triggered"].sum()
        ),
        "label_independent_localized_events": int(
            localized["event_true_positives"].sum()
        ),
        "localization_false_positives": int(
            localized["event_false_positives"].sum()
        ),
        "localization_false_negatives": int(
            localized["event_false_negatives"].sum()
        ),
        "forbidden_trace_columns": sorted(FORBIDDEN_TRACE_COLUMNS),
        "files": {},
    }
    manifest_path = output_dir / "phase1_validation_manifest.json"
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path != manifest_path:
            summary["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    manifest_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
