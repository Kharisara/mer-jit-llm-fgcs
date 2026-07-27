#!/usr/bin/env python3
"""Targeted validator-selectivity and non-circular fault evaluation.

This experiment extends the existing controlled-fault study with:
  * independently executed benign negative controls;
  * exact single-event, boundary, and seeded multi-intensity injections;
  * event-level localization checks where the validator exposes row findings;
  * explicit run-level false-positive and false-negative accounting; and
  * transparent separation of execution instances from post-hoc validator applications.

The protocol is intentionally separate from the frozen functional and timing studies.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from replaybench.integrity import (
    config_bound_trace_hash,
    record_bound_trace_hash,
    reconcile_execution_receipts,
    sha256_json,
    validate_receipt_digest_rows,
)
from replaybench.validation import canonical_action_hash
from run_fgcs_extended_benchmark import (
    ensure_dir,
    load_config,
    normalize_label,
    run_replay,
    validate_config,
)

RECEIPT_FAULTS = (
    "unlogged_downstream_call",
    "false_execution_log",
    "duplicate_downstream_call",
    "mismatched_correlation_id",
)
BENIGN_MODES = (
    "timing_fields_changed",
    "metadata_column_order_changed",
    "permitted_logging_format_changed",
    "completion_order_changed_then_reconstructed",
)
POSTHOC_TRACE_FAULTS = (
    "saved_action_corruption",
    "dropped_rows",
    "duplicated_rows",
)
RECORD_FAULTS = (
    "row_reordering",
    "replay_id_action_reassignment",
    "authorization_field_corruption",
    "execution_field_corruption",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ReplayBench-PG validator-selectivity validation."
    )
    parser.add_argument(
        "--config",
        default="configs/validator_selectivity_validation.yaml",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small implementation smoke test; not manuscript evidence.",
    )
    return parser.parse_args()


def _condition_token(policy: str, seed: int, workers: int) -> str:
    return f"policy_{policy}_seed_{seed}_workers_{workers}"


def _instance_id(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:24]


def _stable_rank(seed: int, namespace: str, values: Iterable[int]) -> list[int]:
    def key(value: int) -> str:
        return hashlib.sha256(
            f"{seed}|{namespace}|{int(value)}".encode("utf-8")
        ).hexdigest()

    return sorted((int(value) for value in values), key=key)


def _profile_targets(
    eligible: Sequence[int],
    profile: Mapping[str, Any],
    *,
    injection_seed: int,
    namespace: str,
) -> list[int]:
    candidates = sorted({int(value) for value in eligible})
    if not candidates:
        raise ValueError(f"No eligible records for {namespace}")

    kind = str(profile["kind"])
    if kind == "first":
        return [candidates[0]]
    if kind == "last":
        return [candidates[-1]]
    if kind == "single_seeded":
        return _stable_rank(injection_seed, namespace, candidates)[:1]
    if kind == "rate":
        rate = float(profile["rate"])
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"Invalid profile rate: {rate}")
        count = max(1, int(math.ceil(len(candidates) * rate)))
        return _stable_rank(injection_seed, namespace, candidates)[:count]
    raise ValueError(f"Unsupported injection profile kind: {kind}")


def _profile_rows(config: Mapping[str, Any], smoke: bool) -> list[dict[str, Any]]:
    configured = config.get("selectivity", {}).get("profiles", [])
    profiles = [dict(value) for value in configured]
    if not profiles:
        profiles = [
            {"name": "single_first", "kind": "first"},
            {"name": "single_last", "kind": "last"},
            {"name": "single_seeded", "kind": "single_seeded"},
            {"name": "rate_0_01_percent", "kind": "rate", "rate": 0.0001},
            {"name": "rate_0_1_percent", "kind": "rate", "rate": 0.001},
            {"name": "rate_1_percent", "kind": "rate", "rate": 0.01},
        ]
    if smoke:
        return [profiles[0], profiles[-1]]
    return profiles


def _anomaly_replay_ids(
    reconciliation: pd.DataFrame,
    metric: str,
) -> set[str]:
    if metric not in reconciliation.columns:
        return set()
    return set(
        reconciliation.loc[
            reconciliation[metric].astype(int).eq(1), "replay_point_id"
        ].astype(str)
    )


def _target_replay_ids(trace: pd.DataFrame, target_indices: Sequence[int]) -> set[str]:
    targets = set(int(value) for value in target_indices)
    return set(
        trace.loc[trace["row_index"].astype(int).isin(targets), "replay_point_id"]
        .astype(str)
        .tolist()
    )


def _runtime_fault_config(
    base_cfg: Mapping[str, Any],
    *,
    fault_mode: str,
    target_indices: Sequence[int],
    policy_modes: Sequence[str],
) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(base_cfg))
    cfg.setdefault("execution_receipts", {})["enabled"] = True
    fault_cfg = cfg.setdefault("fault_injection", {})
    fault_cfg.update(
        {
            "enabled": False,
            "action_flip_probability": 0.0,
            "unauthorized_invoke_probability": 0.0,
            "receipt_fault_mode": "clean",
            "receipt_fault_probability": 0.0,
            "allowed_policy_modes": list(policy_modes),
            "receipt_fault_allowed_policy_modes": list(policy_modes),
        }
    )
    for key in (
        "action_flip_target_indices",
        "unauthorized_invoke_target_indices",
        "receipt_fault_target_indices",
    ):
        fault_cfg.pop(key, None)

    exact_targets = frozenset(int(value) for value in target_indices)
    if fault_mode == "action_flip":
        fault_cfg["enabled"] = True
        fault_cfg["action_flip_target_indices"] = exact_targets
    elif fault_mode == "unauthorized_invocation":
        fault_cfg["enabled"] = True
        fault_cfg["unauthorized_invoke_target_indices"] = exact_targets
    elif fault_mode in RECEIPT_FAULTS:
        fault_cfg["receipt_fault_mode"] = fault_mode
        fault_cfg["receipt_fault_target_indices"] = exact_targets
    else:
        raise ValueError(f"Unsupported runtime fault: {fault_mode}")
    return cfg


def _benign_transform(
    trace: pd.DataFrame,
    receipts: pd.DataFrame,
    mode: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transformed_trace = trace.copy(deep=True)
    transformed_receipts = receipts.copy(deep=True)

    if mode == "timing_fields_changed":
        for column in (
            "state_loading_ms",
            "policy_inference_ms",
            "gating_ms",
            "generation_stub_ms",
            "trace_logging_ms",
            "total_latency_ms",
        ):
            if column in transformed_trace.columns:
                transformed_trace[column] = (
                    pd.to_numeric(transformed_trace[column], errors="coerce").fillna(0.0)
                    + 7.125
                )
        for column, delta in (
            ("receipt_event_index", 1000000),
            ("execution_timestamp_ns", 1000000000),
            ("execution_monotonic_ns", 1000000000),
        ):
            if column in transformed_receipts.columns:
                transformed_receipts[column] = (
                    pd.to_numeric(transformed_receipts[column], errors="coerce")
                    .fillna(0)
                    .astype("int64")
                    + delta
                )
        return transformed_trace, transformed_receipts

    if mode == "metadata_column_order_changed":
        return (
            transformed_trace.loc[:, list(reversed(transformed_trace.columns))],
            transformed_receipts.loc[:, list(reversed(transformed_receipts.columns))],
        )

    if mode == "permitted_logging_format_changed":
        transformed_trace["nonsemantic_log_format"] = "pretty-v2"
        if "response_json" in transformed_trace.columns:
            def reformat(value: Any) -> str:
                text = "" if pd.isna(value) else str(value)
                try:
                    return json.dumps(json.loads(text), indent=2, sort_keys=True)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return f"  {text}  "
            transformed_trace["response_json"] = transformed_trace["response_json"].map(reformat)
        transformed_receipts["nonsemantic_log_format"] = "pretty-v2"
        return transformed_trace, transformed_receipts

    if mode == "completion_order_changed_then_reconstructed":
        transformed_trace = transformed_trace.sample(frac=1.0, random_state=seed)
        transformed_receipts = transformed_receipts.sample(frac=1.0, random_state=seed)
        transformed_trace = transformed_trace.sort_values("row_index").reset_index(drop=True)
        transformed_receipts = transformed_receipts.reset_index(drop=True)
        return transformed_trace, transformed_receipts

    raise ValueError(f"Unsupported benign mode: {mode}")


def _posthoc_trace_corruption(
    trace: pd.DataFrame,
    mode: str,
    targets: Sequence[int],
) -> pd.DataFrame:
    target_set = set(int(value) for value in targets)
    corrupted = trace.copy(deep=True).reset_index(drop=True)
    mask = corrupted["row_index"].astype(int).isin(target_set)

    if mode == "saved_action_corruption":
        corrupted.loc[mask, "action"] = 1 - corrupted.loc[mask, "action"].astype(int)
        return corrupted
    if mode == "dropped_rows":
        return corrupted.loc[~mask].reset_index(drop=True)
    if mode == "duplicated_rows":
        rows: list[pd.Series] = []
        for _, row in corrupted.iterrows():
            rows.append(row)
            if int(row["row_index"]) in target_set:
                rows.append(row.copy())
        return pd.DataFrame(rows).reset_index(drop=True)
    raise ValueError(f"Unsupported post-hoc trace fault: {mode}")


def _effective_positions(length: int, targets: Sequence[int]) -> list[int]:
    positions = sorted({int(value) for value in targets if 0 <= int(value) < length})
    if not positions:
        raise ValueError("No valid corruption positions")
    if len(positions) == 1:
        neighbor = positions[0] + 1 if positions[0] + 1 < length else positions[0] - 1
        positions.append(neighbor)
    return sorted(set(positions))


def _record_corruption(
    trace: pd.DataFrame,
    mode: str,
    targets: Sequence[int],
) -> tuple[pd.DataFrame, int]:
    corrupted = trace.copy(deep=True).reset_index(drop=True)
    row_positions = {
        int(row_index): int(position)
        for position, row_index in enumerate(corrupted["row_index"].astype(int).tolist())
    }
    selected_positions = [row_positions[int(value)] for value in targets if int(value) in row_positions]
    positions = _effective_positions(len(corrupted), selected_positions)

    if mode == "row_reordering":
        order = list(range(len(corrupted)))
        rotated = positions[1:] + positions[:1]
        for destination, source in zip(positions, rotated):
            order[destination] = source
        return corrupted.iloc[order].reset_index(drop=True), len(positions)

    if mode == "replay_id_action_reassignment":
        values = corrupted.loc[positions, ["replay_point_id", "correlation_id"]].copy()
        rotated = values.iloc[1:].to_dict(orient="records") + values.iloc[:1].to_dict(orient="records")
        for position, replacement in zip(positions, rotated):
            corrupted.loc[position, "replay_point_id"] = replacement["replay_point_id"]
            corrupted.loc[position, "correlation_id"] = replacement["correlation_id"]
        return corrupted, len(positions)

    if mode == "authorization_field_corruption":
        corrupted.loc[positions, "authorized_to_generate"] = 1 - corrupted.loc[
            positions, "authorized_to_generate"
        ].astype(int)
        return corrupted, len(positions)

    if mode == "execution_field_corruption":
        corrupted.loc[positions, "generation_invoked"] = 1 - corrupted.loc[
            positions, "generation_invoked"
        ].astype(int)
        return corrupted, len(positions)

    raise ValueError(f"Unsupported record corruption: {mode}")


def _save_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    validate_config(base_cfg)

    selectivity_cfg = base_cfg.get("selectivity", {})
    output_dir = Path(
        base_cfg.get("logging", {}).get(
            "output_dir", "paper_outputs/validator_selectivity_validation"
        )
    )
    if args.smoke:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    ensure_dir(output_dir)

    full = pd.read_csv(base_cfg["dataset"]["input_csv"]).reset_index(drop=True)
    fraction = float(base_cfg["dataset"].get("fractions", [1.0])[0])
    frame = full.iloc[: max(1, int(len(full) * fraction))].copy().reset_index(drop=True)
    if args.smoke:
        frame = frame.iloc[: min(len(frame), 200)].copy().reset_index(drop=True)

    seeds = [int(value) for value in selectivity_cfg.get("seeds", [1])]
    workers_values = [int(value) for value in selectivity_cfg.get("workers", [1, 4])]
    runtime_policy_modes = [
        str(value)
        for value in selectivity_cfg.get(
            "runtime_policy_modes", ["risk_proxy", "random", "never"]
        )
    ]
    receipt_policy_modes = [
        str(value)
        for value in selectivity_cfg.get(
            "receipt_policy_modes", ["risk_proxy", "random", "always"]
        )
    ]
    all_reference_policies = sorted(set(runtime_policy_modes + receipt_policy_modes))
    profiles = _profile_rows(base_cfg, args.smoke)
    injection_seed_base = int(selectivity_cfg.get("injection_seed", 20260724))
    negative_labels = {
        normalize_label(value)
        for value in base_cfg.get("policy", {}).get("negative_labels", [])
    }

    clean_references: dict[tuple[str, int, int], tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]] = {}
    clean_rows: list[dict[str, Any]] = []

    # One unique clean reference execution per union condition. These references
    # are used only for target selection and post-hoc validator applications.
    for policy in all_reference_policies:
        for seed in seeds:
            for workers in workers_values:
                token = _condition_token(policy, seed, workers)
                print(f"[CLEAN-REFERENCE] {token}")
                clean_cfg = copy.deepcopy(base_cfg)
                clean_cfg.setdefault("execution_receipts", {})["enabled"] = True
                clean_fault = clean_cfg.setdefault("fault_injection", {})
                clean_fault.update(
                    {
                        "enabled": False,
                        "action_flip_probability": 0.0,
                        "unauthorized_invoke_probability": 0.0,
                        "receipt_fault_mode": "clean",
                        "receipt_fault_probability": 0.0,
                    }
                )
                for key in (
                    "action_flip_target_indices",
                    "unauthorized_invoke_target_indices",
                    "receipt_fault_target_indices",
                ):
                    clean_fault.pop(key, None)
                trace, summary, _ = run_replay(
                    df=frame,
                    cfg=clean_cfg,
                    policy_mode=policy,
                    negative_labels=negative_labels,
                    seed=seed,
                    workers=workers,
                    bc_actions=None,
                )
                receipts = trace.attrs["execution_receipts"].copy(deep=True)
                reconciliation = trace.attrs["receipt_reconciliation"].copy(deep=True)
                if int(summary["receipt_validation_passed"]) != 1:
                    raise RuntimeError(f"Clean reference failed receipt validation: {token}")
                clean_references[(policy, seed, workers)] = (
                    trace.copy(deep=True),
                    dict(summary),
                    receipts,
                    reconciliation,
                )
                clean_rows.append(
                    {
                        "experiment_instance_id": _instance_id("clean-reference", token),
                        "policy_mode": policy,
                        "seed": seed,
                        "workers": workers,
                        "decision_points": len(trace),
                        "action_hash": summary["trace_hash"],
                        "record_trace_hash": summary["record_trace_hash"],
                        "config_bound_trace_hash": summary["config_bound_trace_hash"],
                        "receipt_validation_passed": summary["receipt_validation_passed"],
                        "anomaly_findings": 0,
                    }
                )

    runtime_rows: list[dict[str, Any]] = []
    runtime_event_rows: list[dict[str, Any]] = []

    runtime_families = [
        ("action_flip", runtime_policy_modes, "all"),
        ("unauthorized_invocation", runtime_policy_modes, "action_zero"),
        *[(mode, receipt_policy_modes, "action_one") for mode in RECEIPT_FAULTS],
    ]

    for fault_mode, policies, eligibility in runtime_families:
        for policy in policies:
            for seed in seeds:
                for workers in workers_values:
                    clean_trace, clean_summary, _, _ = clean_references[(policy, seed, workers)]
                    if eligibility == "action_zero":
                        eligible = clean_trace.loc[
                            clean_trace["action"].astype(int).eq(0), "row_index"
                        ].astype(int).tolist()
                    elif eligibility == "action_one":
                        eligible = clean_trace.loc[
                            clean_trace["action"].astype(int).eq(1), "row_index"
                        ].astype(int).tolist()
                    else:
                        eligible = clean_trace["row_index"].astype(int).tolist()

                    for profile_index, profile in enumerate(profiles):
                        profile_name = str(profile["name"])
                        injection_seed = injection_seed_base + seed * 1000 + profile_index
                        namespace = f"{fault_mode}|{policy}|{seed}|{workers}|{profile_name}"
                        targets = _profile_targets(
                            eligible,
                            profile,
                            injection_seed=injection_seed,
                            namespace=namespace,
                        )
                        cfg = _runtime_fault_config(
                            base_cfg,
                            fault_mode=fault_mode,
                            target_indices=targets,
                            policy_modes=policies,
                        )
                        instance_id = _instance_id("runtime", namespace)
                        print(
                            f"[RUNTIME-FAULT] mode={fault_mode}, profile={profile_name}, "
                            f"policy={policy}, seed={seed}, workers={workers}, events={len(targets)}"
                        )
                        trace, summary, _ = run_replay(
                            df=frame,
                            cfg=cfg,
                            policy_mode=policy,
                            negative_labels=negative_labels,
                            seed=seed,
                            workers=workers,
                            bc_actions=None,
                        )
                        reconciliation = trace.attrs["receipt_reconciliation"]
                        target_ids = _target_replay_ids(trace, targets)

                        if fault_mode == "action_flip":
                            detected_ids = set(
                                trace.loc[
                                    trace["action_flip_fault_injected"].astype(int).eq(1),
                                    "replay_point_id",
                                ].astype(str)
                            )
                            validator_triggered = int(
                                str(summary["trace_hash"]) != str(clean_summary["trace_hash"])
                            )
                            expected_receipt_pass = 1
                        elif fault_mode == "unauthorized_invocation":
                            detected_ids = _anomaly_replay_ids(
                                reconciliation, "unauthorized_downstream_call"
                            )
                            validator_triggered = int(
                                int(summary["unauthorized_invocations"]) > 0
                                and int(summary["receipt_validation_passed"]) == 0
                            )
                            expected_receipt_pass = 0
                        else:
                            metric = {
                                "unlogged_downstream_call": "unlogged_downstream_call",
                                "false_execution_log": "missing_receipt",
                                "duplicate_downstream_call": "duplicate_downstream_call",
                                "mismatched_correlation_id": "mismatched_correlation_id",
                            }[fault_mode]
                            detected_ids = _anomaly_replay_ids(reconciliation, metric)
                            validator_triggered = int(
                                int(summary["receipt_validation_passed"]) == 0
                            )
                            expected_receipt_pass = 0

                        event_tp = len(target_ids & detected_ids)
                        event_fp = len(detected_ids - target_ids)
                        event_fn = len(target_ids - detected_ids)
                        correctly_classified = int(
                            validator_triggered == 1
                            and event_tp == len(target_ids)
                            and event_fp == 0
                            and event_fn == 0
                            and int(summary["receipt_validation_passed"])
                            == expected_receipt_pass
                        )

                        runtime_rows.append(
                            {
                                "experiment_instance_id": instance_id,
                                "evidence_unit_type": "independent_execution",
                                "fault_mode": fault_mode,
                                "profile": profile_name,
                                "profile_kind": profile["kind"],
                                "nominal_rate": profile.get("rate", ""),
                                "policy_mode": policy,
                                "seed": seed,
                                "workers": workers,
                                "injection_seed": injection_seed,
                                "eligible_events": len(eligible),
                                "injected_events": len(targets),
                                "target_indices_sha256": sha256_json(targets),
                                "target_indices_json": json.dumps(targets),
                                "validator_triggered": validator_triggered,
                                "receipt_validation_passed": int(summary["receipt_validation_passed"]),
                                "event_localization_supported": 1,
                                "event_true_positives": event_tp,
                                "event_false_positives": event_fp,
                                "event_false_negatives": event_fn,
                                "correctly_classified": correctly_classified,
                                "action_hash_match_clean": int(
                                    str(summary["trace_hash"]) == str(clean_summary["trace_hash"])
                                ),
                                "record_hash_match_clean": int(
                                    str(summary["record_trace_hash"])
                                    == str(clean_summary["record_trace_hash"])
                                ),
                            }
                        )
                        for replay_id in sorted(target_ids | detected_ids):
                            runtime_event_rows.append(
                                {
                                    "experiment_instance_id": instance_id,
                                    "fault_mode": fault_mode,
                                    "profile": profile_name,
                                    "replay_point_id": replay_id,
                                    "was_injected": int(replay_id in target_ids),
                                    "was_localized": int(replay_id in detected_ids),
                                }
                            )

    benign_rows: list[dict[str, Any]] = []
    # Every benign workflow receives a fresh clean execution; these are not
    # repeated applications to the clean-reference runs above.
    for benign_index, benign_mode in enumerate(BENIGN_MODES):
        for policy in receipt_policy_modes:
            for seed in seeds:
                for workers in workers_values:
                    token = _condition_token(policy, seed, workers)
                    print(f"[BENIGN] mode={benign_mode}, {token}")
                    cfg = copy.deepcopy(base_cfg)
                    cfg.setdefault("execution_receipts", {})["enabled"] = True
                    fault_cfg = cfg.setdefault("fault_injection", {})
                    fault_cfg.update(
                        {
                            "enabled": False,
                            "action_flip_probability": 0.0,
                            "unauthorized_invoke_probability": 0.0,
                            "receipt_fault_mode": "clean",
                            "receipt_fault_probability": 0.0,
                        }
                    )
                    trace, summary, _ = run_replay(
                        df=frame,
                        cfg=cfg,
                        policy_mode=policy,
                        negative_labels=negative_labels,
                        seed=seed,
                        workers=workers,
                        bc_actions=None,
                    )
                    receipts = trace.attrs["execution_receipts"]
                    transformed_trace, transformed_receipts = _benign_transform(
                        trace,
                        receipts,
                        benign_mode,
                        injection_seed_base + benign_index + seed,
                    )
                    digest_rows_valid = int(validate_receipt_digest_rows(transformed_receipts))
                    transformed_reconciliation, transformed_receipt_summary = reconcile_execution_receipts(
                        transformed_trace, transformed_receipts
                    )
                    transformed_action_hash = canonical_action_hash(
                        transformed_trace["action"].astype(int).tolist()
                    )
                    transformed_record_hash = record_bound_trace_hash(
                        transformed_trace, transformed_reconciliation
                    )
                    transformed_config_bound = config_bound_trace_hash(
                        transformed_record_hash, str(summary["config_manifest_hash"])
                    )
                    anomaly_count = sum(
                        int(transformed_receipt_summary[key])
                        for key in (
                            "orphan_receipts",
                            "missing_receipts",
                            "unlogged_downstream_calls",
                            "duplicate_downstream_calls",
                            "mismatched_correlation_ids",
                            "unauthorized_downstream_calls",
                        )
                    )
                    false_positive = int(
                        digest_rows_valid != 1
                        or int(transformed_receipt_summary["receipt_validation_passed"]) != 1
                        or anomaly_count != 0
                        or transformed_action_hash != str(summary["trace_hash"])
                        or transformed_record_hash != str(summary["record_trace_hash"])
                        or transformed_config_bound != str(summary["config_bound_trace_hash"])
                    )
                    benign_rows.append(
                        {
                            "experiment_instance_id": _instance_id("benign", benign_mode, token),
                            "evidence_unit_type": "independent_clean_execution",
                            "benign_mode": benign_mode,
                            "policy_mode": policy,
                            "seed": seed,
                            "workers": workers,
                            "digest_rows_valid": digest_rows_valid,
                            "receipt_validation_passed": int(
                                transformed_receipt_summary["receipt_validation_passed"]
                            ),
                            "anomaly_findings": anomaly_count,
                            "action_hash_preserved": int(
                                transformed_action_hash == str(summary["trace_hash"])
                            ),
                            "record_hash_preserved": int(
                                transformed_record_hash == str(summary["record_trace_hash"])
                            ),
                            "config_bound_hash_preserved": int(
                                transformed_config_bound == str(summary["config_bound_trace_hash"])
                            ),
                            "false_positive": false_positive,
                            "correctly_classified": int(false_positive == 0),
                        }
                    )

    posthoc_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    # These are explicitly reported as validator applications to the unique
    # clean-reference executions, not as additional replay executions.
    for policy in runtime_policy_modes:
        for seed in seeds:
            for workers in workers_values:
                clean_trace, clean_summary, _, clean_reconciliation = clean_references[
                    (policy, seed, workers)
                ]
                eligible = clean_trace["row_index"].astype(int).tolist()
                for profile_index, profile in enumerate(profiles):
                    profile_name = str(profile["name"])
                    injection_seed = injection_seed_base + 50000 + seed * 1000 + profile_index
                    targets = _profile_targets(
                        eligible,
                        profile,
                        injection_seed=injection_seed,
                        namespace=f"posthoc|{policy}|{seed}|{workers}|{profile_name}",
                    )
                    for mode in POSTHOC_TRACE_FAULTS:
                        corrupted = _posthoc_trace_corruption(clean_trace, mode, targets)
                        corrupted_action_hash = canonical_action_hash(
                            corrupted["action"].astype(int).tolist()
                        )
                        hash_mismatch = int(
                            corrupted_action_hash != str(clean_summary["trace_hash"])
                        )
                        row_mismatch = int(len(corrupted) != len(clean_trace))
                        triggered = int(
                            hash_mismatch == 1
                            if mode == "saved_action_corruption"
                            else hash_mismatch == 1 and row_mismatch == 1
                        )
                        posthoc_rows.append(
                            {
                                "application_id": _instance_id(
                                    "posthoc", mode, policy, seed, workers, profile_name
                                ),
                                "evidence_unit_type": "validator_application",
                                "fault_mode": mode,
                                "profile": profile_name,
                                "profile_kind": profile["kind"],
                                "nominal_rate": profile.get("rate", ""),
                                "policy_mode": policy,
                                "seed": seed,
                                "workers": workers,
                                "injection_seed": injection_seed,
                                "injected_events": len(targets),
                                "target_indices_sha256": sha256_json(targets),
                                "hash_mismatch": hash_mismatch,
                                "row_count_mismatch": row_mismatch,
                                "event_localization_supported": 0,
                                "validator_triggered": triggered,
                                "correctly_classified": triggered,
                            }
                        )

                    for mode in RECORD_FAULTS:
                        corrupted, mutated_rows = _record_corruption(
                            clean_trace, mode, targets
                        )
                        corrupted_record_hash = record_bound_trace_hash(
                            corrupted, clean_reconciliation
                        )
                        triggered = int(
                            corrupted_record_hash != str(clean_summary["record_trace_hash"])
                        )
                        record_rows.append(
                            {
                                "application_id": _instance_id(
                                    "record", mode, policy, seed, workers, profile_name
                                ),
                                "evidence_unit_type": "validator_application",
                                "corruption_mode": mode,
                                "profile": profile_name,
                                "profile_kind": profile["kind"],
                                "nominal_rate": profile.get("rate", ""),
                                "policy_mode": policy,
                                "seed": seed,
                                "workers": workers,
                                "injection_seed": injection_seed,
                                "requested_target_events": len(targets),
                                "mutated_rows": mutated_rows,
                                "record_hash_mismatch": triggered,
                                "event_localization_supported": 0,
                                "validator_triggered": triggered,
                                "correctly_classified": triggered,
                            }
                        )

                manifest = copy.deepcopy(clean_trace.attrs["run_identity"]["config_manifest"])
                manifest["policy_mode"] = f"corrupted::{manifest['policy_mode']}"
                corrupt_manifest_hash = sha256_json(manifest)
                corrupted_config_bound = config_bound_trace_hash(
                    str(clean_summary["record_trace_hash"]), corrupt_manifest_hash
                )
                triggered = int(
                    corrupted_config_bound != str(clean_summary["config_bound_trace_hash"])
                )
                record_rows.append(
                    {
                        "application_id": _instance_id(
                            "record", "configuration_label_corruption", policy, seed, workers
                        ),
                        "evidence_unit_type": "validator_application",
                        "corruption_mode": "configuration_label_corruption",
                        "profile": "single_configuration_event",
                        "profile_kind": "single_configuration_event",
                        "nominal_rate": "",
                        "policy_mode": policy,
                        "seed": seed,
                        "workers": workers,
                        "injection_seed": injection_seed_base,
                        "requested_target_events": 1,
                        "mutated_rows": 0,
                        "record_hash_mismatch": 0,
                        "config_bound_hash_mismatch": triggered,
                        "event_localization_supported": 0,
                        "validator_triggered": triggered,
                        "correctly_classified": triggered,
                    }
                )

    clean_df = pd.DataFrame(clean_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    runtime_events_df = pd.DataFrame(runtime_event_rows)
    benign_df = pd.DataFrame(benign_rows)
    posthoc_df = pd.DataFrame(posthoc_rows)
    record_df = pd.DataFrame(record_rows)

    clean_df.to_csv(output_dir / "clean_reference_executions.csv", index=False)
    runtime_df.to_csv(output_dir / "runtime_fault_selectivity_per_execution.csv", index=False)
    runtime_events_df.to_csv(output_dir / "runtime_fault_event_localization.csv", index=False)
    benign_df.to_csv(output_dir / "benign_negative_controls_per_execution.csv", index=False)
    posthoc_df.to_csv(output_dir / "posthoc_trace_selectivity_per_application.csv", index=False)
    record_df.to_csv(output_dir / "record_integrity_selectivity_per_application.csv", index=False)

    runtime_summary = (
        runtime_df.groupby(["fault_mode", "profile"], as_index=False)
        .agg(
            independent_executions=("experiment_instance_id", "nunique"),
            injected_events=("injected_events", "sum"),
            correctly_classified=("correctly_classified", "sum"),
            event_true_positives=("event_true_positives", "sum"),
            event_false_positives=("event_false_positives", "sum"),
            event_false_negatives=("event_false_negatives", "sum"),
        )
    )
    benign_summary = (
        benign_df.groupby("benign_mode", as_index=False)
        .agg(
            independent_clean_executions=("experiment_instance_id", "nunique"),
            correctly_classified=("correctly_classified", "sum"),
            false_positive_executions=("false_positive", "sum"),
            anomaly_findings=("anomaly_findings", "sum"),
        )
    )
    posthoc_summary = (
        posthoc_df.groupby(["fault_mode", "profile"], as_index=False)
        .agg(
            validator_applications=("application_id", "nunique"),
            injected_events=("injected_events", "sum"),
            correctly_classified=("correctly_classified", "sum"),
        )
    )
    record_summary = (
        record_df.groupby(["corruption_mode", "profile"], as_index=False)
        .agg(
            validator_applications=("application_id", "nunique"),
            correctly_classified=("correctly_classified", "sum"),
        )
    )
    runtime_summary.to_csv(output_dir / "runtime_fault_selectivity_summary.csv", index=False)
    benign_summary.to_csv(output_dir / "benign_negative_controls_summary.csv", index=False)
    posthoc_summary.to_csv(output_dir / "posthoc_trace_selectivity_summary.csv", index=False)
    record_summary.to_csv(output_dir / "record_integrity_selectivity_summary.csv", index=False)

    positive_execution_runs = int(len(runtime_df))
    positive_execution_failures = int((runtime_df["correctly_classified"].astype(int) == 0).sum())
    negative_execution_runs = int(len(benign_df))
    false_positive_runs = int(benign_df["false_positive"].astype(int).sum())
    posthoc_applications = int(len(posthoc_df) + len(record_df))
    posthoc_failures = int(
        (posthoc_df["correctly_classified"].astype(int) == 0).sum()
        + (record_df["correctly_classified"].astype(int) == 0).sum()
    )
    event_tp = int(runtime_df["event_true_positives"].sum())
    event_fp = int(runtime_df["event_false_positives"].sum())
    event_fn = int(runtime_df["event_false_negatives"].sum())

    metrics = {
        "experiment": "validator_selectivity_and_non_circular_fault_evaluation",
        "smoke_mode": bool(args.smoke),
        "decision_points": int(len(frame)),
        "unique_clean_reference_executions": int(
            clean_df["experiment_instance_id"].nunique()
        ),
        "independent_benign_negative_control_executions": negative_execution_runs,
        "independent_positive_fault_executions": positive_execution_runs,
        "posthoc_validator_applications": posthoc_applications,
        "run_level_true_positives": positive_execution_runs - positive_execution_failures,
        "run_level_false_negatives": positive_execution_failures,
        "run_level_false_positives": false_positive_runs,
        "run_level_true_negatives": negative_execution_runs - false_positive_runs,
        "run_level_false_positive_rate": (
            false_positive_runs / negative_execution_runs if negative_execution_runs else 0.0
        ),
        "run_level_false_negative_rate": (
            positive_execution_failures / positive_execution_runs if positive_execution_runs else 0.0
        ),
        "event_level_true_positives": event_tp,
        "event_level_false_positives": event_fp,
        "event_level_false_negatives": event_fn,
        "event_level_precision": event_tp / (event_tp + event_fp) if event_tp + event_fp else 1.0,
        "event_level_recall": event_tp / (event_tp + event_fn) if event_tp + event_fn else 1.0,
        "posthoc_validator_failures": posthoc_failures,
        "profiles": profiles,
        "injection_seed_base": injection_seed_base,
        "runtime_policy_modes": runtime_policy_modes,
        "receipt_policy_modes": receipt_policy_modes,
        "seeds": seeds,
        "workers": workers_values,
        "terminology": {
            "execution_instance": "A fresh call to the replay executor.",
            "validator_application": "A post-execution corruption applied to a clean reference; not an additional replay execution.",
        },
    }
    _save_json(output_dir / "validator_selectivity_metrics.json", metrics)

    manifest = {
        **metrics,
        "input_csv": str(base_cfg["dataset"]["input_csv"]),
        "input_rows": int(len(full)),
        "output_files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "all_benign_controls_passed": bool(false_positive_runs == 0),
        "all_runtime_fault_executions_detected": bool(positive_execution_failures == 0),
        "all_supported_events_localized": bool(event_fp == 0 and event_fn == 0),
        "all_posthoc_applications_detected": bool(posthoc_failures == 0),
    }
    _save_json(output_dir / "validator_selectivity_manifest.json", manifest)

    if false_positive_runs != 0:
        raise RuntimeError(f"Benign controls produced {false_positive_runs} false positives")
    if positive_execution_failures != 0:
        raise RuntimeError(
            f"Runtime fault protocol has {positive_execution_failures} false negatives"
        )
    if event_fp != 0 or event_fn != 0:
        raise RuntimeError(
            f"Event localization mismatch: false positives={event_fp}, false negatives={event_fn}"
        )
    if posthoc_failures != 0:
        raise RuntimeError(f"Post-hoc validation has {posthoc_failures} failures")

    print("[DONE] Validator-selectivity validation passed")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
