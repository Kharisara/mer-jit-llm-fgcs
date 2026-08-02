#!/usr/bin/env python3
from __future__ import annotations

"""Phase-1 label-independent validation and fair evidence-model comparison.

This runner re-analyses the frozen receipt-enabled raw artifacts and applies a
small, deterministic set of post-execution controls to their clean references.
The generic validator never receives fault labels, injection flags, injection
seeds, target-index manifests, or expected anomaly channels. Those values are
written separately and joined only after findings are frozen.
"""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from replaybench.generic_validator import (
    FORBIDDEN_TRACE_COLUMNS,
    GenericValidatorFindings,
    sanitize_trace_for_validation,
    prepare_reference_artifacts,
    validate_observed_artifacts,
)
from replaybench.integrity import sha256_json

RECEIPT_MODES = (
    "clean",
    "unlogged_downstream_call",
    "false_execution_log",
    "duplicate_downstream_call",
    "mismatched_correlation_id",
)
RECORD_MODES = (
    "row_reordering",
    "replay_id_action_reassignment",
    "authorization_field_corruption",
    "execution_field_corruption",
    "configuration_label_corruption",
)
PRIMARY_CONTROL_MODES = (
    "saved_action_corruption",
    "dropped_rows",
    "duplicated_rows",
    "logged_unauthorized_invocation",
)
BENIGN_MODES = (
    "timing_fields_changed",
    "metadata_column_order_changed",
    "permitted_logging_format_changed",
    "completion_order_changed_then_reconstructed",
)
TOKEN_RE = re.compile(r"policy_(?P<policy>.+)_seed_(?P<seed>\d+)_workers_(?P<workers>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="paper_outputs/execution_integrity_validation",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_outputs/phase1_label_independent_validation",
    )
    return parser.parse_args()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _jsonl_dump(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_id(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:24]


def _token_metadata(path: Path) -> tuple[str, int, int, str]:
    token = path.stem.removeprefix("trace_")
    match = TOKEN_RE.fullmatch(token)
    if match is None:
        raise ValueError(f"Unrecognized trace token: {path.name}")
    return (
        match.group("policy"),
        int(match.group("seed")),
        int(match.group("workers")),
        token,
    )


def _load_bundle(base: Path, mode: str, token: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mode_dir = base / mode
    trace = pd.read_csv(mode_dir / f"trace_{token}.csv")
    receipts = pd.read_csv(mode_dir / f"execution_receipts_{token}.csv")
    manifest = json.loads((mode_dir / f"trace_manifest_{token}.json").read_text(encoding="utf-8"))
    return trace, receipts, manifest


def _config_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("config_manifest")
    if not isinstance(value, dict):
        raise ValueError("Trace manifest is missing config_manifest")
    return copy.deepcopy(value)


def _ground_truth_ids(raw_trace: pd.DataFrame, mode: str) -> list[str]:
    if mode == "clean":
        return []
    if "receipt_fault_injected" not in raw_trace.columns:
        raise ValueError("Receipt ground truth is missing from raw generation artifact")
    return sorted(
        raw_trace.loc[
            raw_trace["receipt_fault_injected"].astype(int).eq(1),
            "replay_point_id",
        ].astype(str).unique().tolist()
    )


def _first_ids(trace: pd.DataFrame, count: int = 1) -> list[str]:
    return trace["replay_point_id"].astype(str).iloc[:count].tolist()


def _primary_control(trace: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, list[str]]:
    observed = trace.copy(deep=True).reset_index(drop=True)
    if mode == "saved_action_corruption":
        target = 0
        observed.loc[target, "action"] = 1 - int(observed.loc[target, "action"])
        return observed, [str(observed.loc[target, "replay_point_id"])]
    if mode == "dropped_rows":
        target_id = str(observed.loc[0, "replay_point_id"])
        return observed.iloc[1:].reset_index(drop=True), [target_id]
    if mode == "duplicated_rows":
        target_id = str(observed.loc[0, "replay_point_id"])
        return pd.concat([observed.iloc[:1], observed.iloc[:1], observed.iloc[1:]], ignore_index=True), [target_id]
    if mode == "logged_unauthorized_invocation":
        eligible = observed.index[observed["authorized_to_generate"].astype(int).eq(0)].tolist()
        if not eligible:
            return observed, []
        target = int(eligible[0])
        observed.loc[target, "generation_invoked"] = 1
        return observed, [str(observed.loc[target, "replay_point_id"])]
    raise ValueError(mode)


def _record_control(trace: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, list[str]]:
    observed = trace.copy(deep=True).reset_index(drop=True)
    if len(observed) < 2:
        raise ValueError("At least two rows are required")
    targets = _first_ids(observed, 2)
    if mode == "row_reordering":
        order = list(range(len(observed)))
        order[0], order[1] = order[1], order[0]
        return observed.iloc[order].reset_index(drop=True), targets
    if mode == "replay_id_action_reassignment":
        values = observed.loc[[0, 1], ["replay_point_id", "correlation_id"]].copy()
        observed.loc[0, ["replay_point_id", "correlation_id"]] = values.iloc[1].tolist()
        observed.loc[1, ["replay_point_id", "correlation_id"]] = values.iloc[0].tolist()
        return observed, targets
    if mode == "authorization_field_corruption":
        observed.loc[0, "authorized_to_generate"] = 1 - int(observed.loc[0, "authorized_to_generate"])
        return observed, [targets[0]]
    if mode == "execution_field_corruption":
        observed.loc[0, "generation_invoked"] = 1 - int(observed.loc[0, "generation_invoked"])
        return observed, [targets[0]]
    raise ValueError(mode)


def _benign_transform(trace: pd.DataFrame, receipts: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_trace = trace.copy(deep=True)
    observed_receipts = receipts.copy(deep=True)
    if mode == "timing_fields_changed":
        for column in (
            "state_loading_ms",
            "policy_inference_ms",
            "gating_ms",
            "generation_stub_ms",
            "logging_ms",
            "trace_logging_ms",
            "total_latency_ms",
        ):
            if column in observed_trace.columns:
                observed_trace[column] = pd.to_numeric(observed_trace[column], errors="coerce").fillna(0.0) + 7.125
        for column, delta in (
            ("receipt_event_index", 1_000_000),
            ("execution_timestamp_ns", 1_000_000_000),
            ("execution_monotonic_ns", 1_000_000_000),
        ):
            if column in observed_receipts.columns:
                observed_receipts[column] = pd.to_numeric(observed_receipts[column], errors="coerce").fillna(0).astype("int64") + delta
        return observed_trace, observed_receipts
    if mode == "metadata_column_order_changed":
        return (
            observed_trace.loc[:, list(reversed(observed_trace.columns))],
            observed_receipts.loc[:, list(reversed(observed_receipts.columns))],
        )
    if mode == "permitted_logging_format_changed":
        observed_trace["nonsemantic_log_format"] = "pretty-v2"
        observed_receipts["nonsemantic_log_format"] = "pretty-v2"
        return observed_trace, observed_receipts
    if mode == "completion_order_changed_then_reconstructed":
        observed_trace = observed_trace.sample(frac=1.0, random_state=20260724).sort_values("row_index").reset_index(drop=True)
        observed_receipts = observed_receipts.sample(frac=1.0, random_state=20260724).reset_index(drop=True)
        return observed_trace, observed_receipts
    raise ValueError(mode)


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
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    findings_rows: list[dict[str, Any]] = []
    findings_by_id: dict[str, GenericValidatorFindings] = {}
    truth_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    clean_paths = sorted((input_dir / "clean").glob("trace_*.csv"))
    if len(clean_paths) != 18:
        raise RuntimeError(f"Expected 18 clean references, found {len(clean_paths)}")

    clean_bundles: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Any, str, int, int]] = {}
    for path in clean_paths:
        policy, seed, workers, token = _token_metadata(path)
        raw_trace, receipts, manifest = _load_bundle(input_dir, "clean", token)
        sanitized = sanitize_trace_for_validation(raw_trace)
        prepared = prepare_reference_artifacts(
            reference_trace=sanitized,
            reference_receipts=receipts,
            reference_config_manifest=_config_manifest(manifest),
        )
        clean_bundles[token] = (
            raw_trace, receipts, manifest, prepared, policy, seed, workers
        )

    def evaluate(
        *,
        evidence_id: str,
        evidence_unit_type: str,
        mode: str,
        policy: str,
        seed: int,
        workers: int,
        observed_raw_trace: pd.DataFrame,
        observed_receipts: pd.DataFrame,
        observed_manifest: dict[str, Any],
        reference_raw_trace: pd.DataFrame,
        reference_receipts: pd.DataFrame,
        reference_manifest: dict[str, Any],
        prepared_reference: Any,
        target_ids: list[str],
        positive_control: bool,
        source_kind: str,
    ) -> None:
        observed_trace = sanitize_trace_for_validation(observed_raw_trace)
        reference_trace = sanitize_trace_for_validation(reference_raw_trace)
        forbidden_present = sorted(FORBIDDEN_TRACE_COLUMNS.intersection(observed_trace.columns))
        if forbidden_present:
            raise RuntimeError(f"Sanitization failure: {forbidden_present}")
        findings = validate_observed_artifacts(
            evidence_id=evidence_id,
            observed_trace=observed_trace,
            observed_receipts=observed_receipts,
            observed_config_manifest=_config_manifest(observed_manifest),
            reference_trace=reference_trace,
            reference_receipts=reference_receipts,
            reference_config_manifest=_config_manifest(reference_manifest),
            prepared_reference=prepared_reference,
        )
        row = findings.to_dict()
        row.update({
            "policy_mode": policy,
            "seed": seed,
            "workers": workers,
            "evidence_unit_type": evidence_unit_type,
            "source_kind": source_kind,
        })
        findings_rows.append(row)
        findings_by_id[evidence_id] = findings
        truth_rows.append({
            "schema_version": "replaybench-pg-ground-truth-v1",
            "evidence_id": evidence_id,
            "evidence_unit_type": evidence_unit_type,
            "fault_mode": mode,
            "positive_control": int(positive_control),
            "policy_mode": policy,
            "seed": seed,
            "workers": workers,
            "target_replay_point_ids": sorted(map(str, target_ids)),
            "target_count": len(target_ids),
            "source_kind": source_kind,
        })
        audit_rows.append({
            "evidence_id": evidence_id,
            "raw_trace_columns": len(observed_raw_trace.columns),
            "validator_trace_columns": len(observed_trace.columns),
            "removed_forbidden_columns": "|".join(sorted(FORBIDDEN_TRACE_COLUMNS.intersection(observed_raw_trace.columns))),
            "forbidden_columns_in_validator_input": "|".join(forbidden_present),
            "validator_input_label_independent": int(not forbidden_present),
        })

    # Frozen receipt-enabled execution instances (90 = 18 clean + 72 controls).
    for mode in RECEIPT_MODES:
        for path in sorted((input_dir / mode).glob("trace_*.csv")):
            policy, seed, workers, token = _token_metadata(path)
            observed_trace, observed_receipts, observed_manifest = _load_bundle(input_dir, mode, token)
            clean_trace, clean_receipts, clean_manifest, prepared, _, _, _ = clean_bundles[token]
            evidence_id = _evidence_id("receipt-execution", mode, token)
            evaluate(
                evidence_id=evidence_id,
                evidence_unit_type="execution_instance",
                mode=mode,
                policy=policy,
                seed=seed,
                workers=workers,
                observed_raw_trace=observed_trace,
                observed_receipts=observed_receipts,
                observed_manifest=observed_manifest,
                reference_raw_trace=clean_trace,
                reference_receipts=clean_receipts,
                reference_manifest=clean_manifest,
                prepared_reference=prepared,
                target_ids=_ground_truth_ids(observed_trace, mode),
                positive_control=mode != "clean",
                source_kind="frozen_receipt_execution",
            )

    # Deterministic post-execution primary controls and record/config controls.
    for token, (clean_trace, clean_receipts, clean_manifest, prepared, policy, seed, workers) in sorted(clean_bundles.items()):
        for mode in PRIMARY_CONTROL_MODES:
            observed_trace, targets = _primary_control(clean_trace, mode)
            if mode == "logged_unauthorized_invocation" and not targets:
                continue
            evaluate(
                evidence_id=_evidence_id("primary-application", mode, token),
                evidence_unit_type="validator_application",
                mode=mode,
                policy=policy,
                seed=seed,
                workers=workers,
                observed_raw_trace=observed_trace,
                observed_receipts=clean_receipts,
                observed_manifest=clean_manifest,
                reference_raw_trace=clean_trace,
                reference_receipts=clean_receipts,
                reference_manifest=clean_manifest,
                prepared_reference=prepared,
                target_ids=targets,
                positive_control=True,
                source_kind="post_execution_primary_control",
            )
        for mode in RECORD_MODES:
            observed_manifest = copy.deepcopy(clean_manifest)
            if mode == "configuration_label_corruption":
                observed_trace = clean_trace.copy(deep=True)
                observed_manifest = copy.deepcopy(clean_manifest)
                observed_manifest["config_manifest"] = _config_manifest(clean_manifest)
                observed_manifest["config_manifest"]["policy_mode"] = (
                    "corrupted::" + str(observed_manifest["config_manifest"].get("policy_mode", ""))
                )
                targets: list[str] = []
            else:
                observed_trace, targets = _record_control(clean_trace, mode)
            evaluate(
                evidence_id=_evidence_id("record-application", mode, token),
                evidence_unit_type="validator_application",
                mode=mode,
                policy=policy,
                seed=seed,
                workers=workers,
                observed_raw_trace=observed_trace,
                observed_receipts=clean_receipts,
                observed_manifest=observed_manifest,
                reference_raw_trace=clean_trace,
                reference_receipts=clean_receipts,
                reference_manifest=clean_manifest,
                prepared_reference=prepared,
                target_ids=targets,
                positive_control=True,
                source_kind="post_execution_record_config_control",
            )

    # 24 prespecified benign applications: four modes x three policies x two workers, seed 1.
    for token, (clean_trace, clean_receipts, clean_manifest, prepared, policy, seed, workers) in sorted(clean_bundles.items()):
        if seed != 1 or policy not in {"risk_proxy", "random", "always"} or workers not in {1, 4}:
            continue
        for mode in BENIGN_MODES:
            observed_trace, observed_receipts = _benign_transform(clean_trace, clean_receipts, mode)
            evaluate(
                evidence_id=_evidence_id("benign-application", mode, token),
                evidence_unit_type="benign_validator_application",
                mode=mode,
                policy=policy,
                seed=seed,
                workers=workers,
                observed_raw_trace=observed_trace,
                observed_receipts=observed_receipts,
                observed_manifest=clean_manifest,
                reference_raw_trace=clean_trace,
                reference_receipts=clean_receipts,
                reference_manifest=clean_manifest,
                prepared_reference=prepared,
                target_ids=[],
                positive_control=False,
                source_kind="post_execution_benign_control",
            )

    findings_df = pd.DataFrame(findings_rows)
    truth_df = pd.DataFrame(truth_rows)
    audit_df = pd.DataFrame(audit_rows)
    if findings_df["evidence_id"].duplicated().any() or truth_df["evidence_id"].duplicated().any():
        raise RuntimeError("Evidence IDs are not unique")

    # Freeze findings first; only then join ground truth for scoring.
    findings_path = output_dir / "generic_validator_findings.jsonl"
    truth_path = output_dir / "ground_truth_manifest.jsonl"
    _jsonl_dump(findings_path, findings_rows)
    _jsonl_dump(truth_path, truth_rows)
    audit_df.to_csv(output_dir / "validator_input_separation_audit.csv", index=False, lineterminator="\n")

    scored = findings_df.merge(
        truth_df,
        on=[
            "evidence_id",
            "evidence_unit_type",
            "policy_mode",
            "seed",
            "workers",
            "source_kind",
        ],
        validate="one_to_one",
        suffixes=("_finding", "_truth"),
    )
    scored["expected_trigger"] = scored["positive_control"].astype(int)
    scored["primary_correct"] = (
        scored["primary_validator_triggered"].astype(int).eq(scored["expected_trigger"])
    ).astype(int)
    scored["full_correct"] = (
        scored["full_validator_triggered"].astype(int).eq(scored["expected_trigger"])
    ).astype(int)

    localization_rows: list[dict[str, Any]] = []
    for row in scored.to_dict(orient="records"):
        finding = findings_by_id[str(row["evidence_id"])]
        supported, localized = _localized_ids(finding, str(row["fault_mode"]))
        targets = set(map(str, row["target_replay_point_ids"]))
        tp = len(targets & localized) if supported else 0
        fp = len(localized - targets) if supported else 0
        fn = len(targets - localized) if supported else 0
        localization_rows.append({
            "evidence_id": row["evidence_id"],
            "fault_mode": row["fault_mode"],
            "event_localization_supported": int(supported),
            "target_events": len(targets),
            "localized_events": len(localized),
            "event_true_positives": tp,
            "event_false_positives": fp,
            "event_false_negatives": fn,
            "target_replay_point_ids_json": json.dumps(sorted(targets)),
            "localized_replay_point_ids_json": json.dumps(sorted(localized)),
        })
    localization_df = pd.DataFrame(localization_rows)
    scored = scored.merge(localization_df.drop(columns=["fault_mode"]), on="evidence_id", validate="one_to_one")

    scored = scored.reindex(sorted(scored.columns), axis=1)
    scored.to_csv(output_dir / "per_evidence_scored_results.csv", index=False, lineterminator="\n")
    localization_df.to_csv(output_dir / "event_localization_results.csv", index=False, lineterminator="\n")

    positive = scored.loc[scored["positive_control"].astype(int).eq(1)].copy()
    benign = scored.loc[scored["positive_control"].astype(int).eq(0)].copy()
    comparison = (
        scored.groupby(["fault_mode", "evidence_unit_type", "positive_control"], as_index=False)
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
    comparison.to_csv(output_dir / "baseline_comparison_by_fault_class.csv", index=False, lineterminator="\n")

    if not audit_df["validator_input_label_independent"].astype(int).eq(1).all():
        raise RuntimeError("A validator input retained forbidden fault-label fields")
    if not positive["full_validator_triggered"].astype(int).eq(1).all():
        failures = positive.loc[positive["full_validator_triggered"].astype(int).ne(1), ["fault_mode", "evidence_id"]]
        raise RuntimeError(f"Full validator missed positive controls:\n{failures}")
    if not benign["full_validator_triggered"].astype(int).eq(0).all():
        failures = benign.loc[benign["full_validator_triggered"].astype(int).ne(0), ["fault_mode", "evidence_id"]]
        raise RuntimeError(f"Full validator flagged benign controls:\n{failures}")
    localized = localization_df.loc[localization_df["event_localization_supported"].astype(int).eq(1)]
    if int(localized["event_false_positives"].sum()) != 0 or int(localized["event_false_negatives"].sum()) != 0:
        raise RuntimeError("Label-independent event localization did not match ground truth")

    clean_units = int(truth_df["fault_mode"].astype(str).eq("clean").sum())
    benign_control_units = int(
        truth_df["source_kind"].astype(str).eq("post_execution_benign_control").sum()
    )
    summary = {
        "schema_version": "replaybench-pg-phase1-validation-v1",
        "status": "passed",
        "method": (
            "Generic findings were frozen before a one-to-one post-hoc join "
            "with a separate ground-truth manifest."
        ),
        "generic_validator_findings": int(len(findings_df)),
        "ground_truth_records": int(len(truth_df)),
        "receipt_execution_instances": int((truth_df["source_kind"] == "frozen_receipt_execution").sum()),
        "clean_reference_units": clean_units,
        "benign_control_units": benign_control_units,
        "negative_control_units": int(len(benign)),
        "positive_control_units": int(len(positive)),
        "primary_detected_positive_units": int(positive["primary_validator_triggered"].sum()),
        "full_detected_positive_units": int(positive["full_validator_triggered"].sum()),
        "full_false_positive_units": int(benign["full_validator_triggered"].sum()),
        "label_independent_localized_events": int(localized["event_true_positives"].sum()),
        "localization_false_positives": int(localized["event_false_positives"].sum()),
        "localization_false_negatives": int(localized["event_false_negatives"].sum()),
        "forbidden_trace_columns": sorted(FORBIDDEN_TRACE_COLUMNS),
        "files": {},
    }
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "phase1_validation_manifest.json":
            summary["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    _json_dump(output_dir / "phase1_validation_manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
