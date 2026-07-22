#!/usr/bin/env python3
"""Build the finalized 720-row local-to-cloud hash comparison.

This script does not rerun any experiment. It joins the finalized local
ReplayBench-PG determinism results against the two finalized regional cloud
determinism outputs and writes one auditable row per local/cloud comparison.

Expected result:
    360 local conditions x 2 cloud regions = 720 comparisons
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_LOCAL = (
    "paper_outputs/fgcs_extended_benchmark/determinism_hash_results.csv"
)
DEFAULT_CLOUD_ROOT = "cloud_results/cloud360_riskproxy_20260702"
DEFAULT_REGIONS = ("asia-southeast1", "us-central1")
DEFAULT_OUTPUT = (
    "cloud_results/cloud360_riskproxy_20260702/"
    "local_to_cloud_hash_comparison.csv"
)


def first_existing(frame: pd.DataFrame, candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"Could not find {label}. Expected one of {candidates}; "
        f"available columns={list(frame.columns)}"
    )


def normalize(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"Input is empty: {source}")

    policy_col = first_existing(
        frame, ["policy_mode", "policy"], "policy column"
    )
    fraction_col = first_existing(
        frame,
        ["dataset_fraction", "workload_fraction", "fraction"],
        "dataset-fraction column",
    )
    seed_col = first_existing(frame, ["seed", "random_seed"], "seed column")
    workers_col = first_existing(
        frame, ["workers", "worker_count", "num_workers"], "workers column"
    )
    hash_col = first_existing(
        frame,
        ["trace_hash", "action_sequence_hash", "sequence_hash", "hash"],
        "trace-hash column",
    )

    rename_map = {
        policy_col: "policy_mode",
        fraction_col: "dataset_fraction",
        seed_col: "seed",
        workers_col: "workers",
        hash_col: "trace_hash",
    }

    unauthorized_col = None
    for candidate in [
        "unauthorized_invocations",
        "unauthorized_invocation_count",
        "authorization_violations",
        "authorization_contradictions",
    ]:
        if candidate in frame.columns:
            unauthorized_col = candidate
            rename_map[candidate] = "unauthorized_invocations"
            break

    out = frame.rename(columns=rename_map).copy()

    required = [
        "policy_mode",
        "dataset_fraction",
        "seed",
        "workers",
        "trace_hash",
    ]
    out = out[required + (
        ["unauthorized_invocations"]
        if "unauthorized_invocations" in out.columns
        else []
    )].copy()

    out["policy_mode"] = out["policy_mode"].astype(str).str.strip()
    out["dataset_fraction"] = pd.to_numeric(
        out["dataset_fraction"], errors="raise"
    ).astype(float)
    out["seed"] = pd.to_numeric(out["seed"], errors="raise").astype(int)
    out["workers"] = pd.to_numeric(out["workers"], errors="raise").astype(int)
    out["trace_hash"] = out["trace_hash"].astype(str).str.strip()

    if "unauthorized_invocations" not in out.columns:
        out["unauthorized_invocations"] = 0
    else:
        out["unauthorized_invocations"] = pd.to_numeric(
            out["unauthorized_invocations"], errors="raise"
        ).astype(int)

    if out["trace_hash"].eq("").any() or out["trace_hash"].str.lower().eq("nan").any():
        raise ValueError(f"Missing trace hashes in {source}")

    keys = ["dataset_fraction", "policy_mode", "seed", "workers"]
    duplicates = out.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError(
            f"Duplicate determinism keys in {source}:\n"
            + out.loc[duplicates, keys].to_string(index=False)
        )

    return out.sort_values(keys).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the finalized 720-row local-to-cloud hash comparison."
    )
    parser.add_argument("--local", default=DEFAULT_LOCAL)
    parser.add_argument("--cloud-root", default=DEFAULT_CLOUD_ROOT)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(DEFAULT_REGIONS),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    local_path = Path(args.local)
    cloud_root = Path(args.cloud_root)
    output_path = Path(args.output)

    if not local_path.is_file():
        raise FileNotFoundError(f"Local determinism file not found: {local_path}")
    if not cloud_root.is_dir():
        raise FileNotFoundError(f"Cloud root not found: {cloud_root}")

    local = normalize(pd.read_csv(local_path), local_path)
    if len(local) != 360:
        raise ValueError(
            f"Expected 360 finalized local conditions, observed {len(local)}"
        )

    keys = ["dataset_fraction", "policy_mode", "seed", "workers"]
    comparisons: list[pd.DataFrame] = []

    for region in args.regions:
        cloud_path = cloud_root / region / "determinism_hash_results.csv"
        if not cloud_path.is_file():
            raise FileNotFoundError(
                f"Cloud determinism file not found for {region}: {cloud_path}"
            )

        cloud = normalize(pd.read_csv(cloud_path), cloud_path)
        if len(cloud) != 360:
            raise ValueError(
                f"Expected 360 cloud conditions for {region}, observed {len(cloud)}"
            )

        merged = local.merge(
            cloud,
            on=keys,
            how="outer",
            suffixes=("_local", "_cloud"),
            indicator=True,
            validate="one_to_one",
        )

        unmatched = merged.loc[merged["_merge"] != "both"]
        if not unmatched.empty:
            raise ValueError(
                f"Local/cloud condition mismatch for {region}:\n"
                + unmatched[keys + ["_merge"]].to_string(index=False)
            )

        merged["region"] = region
        merged["comparison_scope"] = "local_to_cloud"
        merged["hash_match"] = (
            merged["trace_hash_local"] == merged["trace_hash_cloud"]
        ).astype(int)
        merged["unauthorized_invocations"] = merged[
            "unauthorized_invocations_cloud"
        ].astype(int)

        comparisons.append(
            merged[
                [
                    "comparison_scope",
                    "region",
                    *keys,
                    "trace_hash_local",
                    "trace_hash_cloud",
                    "hash_match",
                    "unauthorized_invocations",
                ]
            ]
        )

    result = pd.concat(comparisons, ignore_index=True)

    expected_rows = 360 * len(args.regions)
    if len(result) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} local-to-cloud comparisons, "
            f"observed {len(result)}"
        )
    if not result["hash_match"].eq(1).all():
        failures = result.loc[
            result["hash_match"].ne(1),
            ["region", *keys, "trace_hash_local", "trace_hash_cloud"],
        ]
        raise ValueError(
            "Local-to-cloud hash mismatches detected:\n"
            + failures.to_string(index=False)
        )
    if not result["unauthorized_invocations"].eq(0).all():
        failures = result.loc[
            result["unauthorized_invocations"].ne(0),
            ["region", *keys, "unauthorized_invocations"],
        ]
        raise ValueError(
            "Cloud authorization contradictions detected:\n"
            + failures.to_string(index=False)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("[DONE] Local-to-cloud comparison complete")
    print(f"[OUT] {output_path}")
    print(f"[CHECK] comparisons={len(result)}/{expected_rows}")
    print(f"[CHECK] hash_matches={int(result['hash_match'].sum())}/{expected_rows}")
    print(
        "[CHECK] unauthorized_invocations="
        f"{int(result['unauthorized_invocations'].sum())}"
    )


if __name__ == "__main__":
    main()