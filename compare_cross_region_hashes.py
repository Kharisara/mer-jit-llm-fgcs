#!/usr/bin/env python3
"""
Compare FGCS cloud benchmark outputs across two regions.

Expected directory after download:

cloud_results/<RUN_ID>/
  asia-southeast1/
    determinism_hash_results.csv
    scaling_and_runtime_results.csv
    policy_ablation_costs.csv
  us-central1/
    determinism_hash_results.csv
    scaling_and_runtime_results.csv
    policy_ablation_costs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MERGE_KEYS = ["dataset_fraction", "workload_name", "policy_mode", "seed", "workers"]


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def hash_summary(base_dir: Path, region_a: str, region_b: str) -> pd.DataFrame:
    a = read_csv_required(base_dir / region_a / "determinism_hash_results.csv")
    b = read_csv_required(base_dir / region_b / "determinism_hash_results.csv")

    merged = a.merge(
        b,
        on=MERGE_KEYS,
        suffixes=(f"_{region_a}", f"_{region_b}"),
        how="inner",
    )

    expected_rows = min(len(a), len(b))
    if len(merged) != expected_rows:
        print(f"[WARN] merged rows={len(merged)}, expected approximately={expected_rows}")

    merged["cross_region_match"] = (
        merged[f"trace_hash_{region_a}"] == merged[f"trace_hash_{region_b}"]
    )

    merged["unauthorized_max"] = merged[
        [f"unauthorized_invocations_{region_a}", f"unauthorized_invocations_{region_b}"]
    ].max(axis=1)

    return merged


def make_table8(merged: pd.DataFrame, region_a: str, region_b: str) -> pd.DataFrame:
    rows = []

    for policy, group in merged.groupby("policy_mode"):
        full = group[
            (group["dataset_fraction"].astype(float) == 1.0)
            & (group["seed"].astype(int) == 1)
            & (group["workers"].astype(int) == 1)
        ]

        if full.empty:
            sample = group.iloc[0]
        else:
            sample = full.iloc[0]

        hash_a = str(sample[f"trace_hash_{region_a}"])
        hash_b = str(sample[f"trace_hash_{region_b}"])

        rows.append({
            "policy_mode": policy,
            f"{region_a}_hash_sample": hash_a[:16] + "...",
            f"{region_b}_hash_sample": hash_b[:16] + "...",
            "conditions_compared": int(len(group)),
            "matching_conditions": int(group["cross_region_match"].sum()),
            "all_match": bool(group["cross_region_match"].all()),
            "max_unauthorized_invocations": int(group["unauthorized_max"].max()),
        })

    return pd.DataFrame(rows).sort_values("policy_mode")


def best_full_workload(df: pd.DataFrame) -> pd.DataFrame:
    full = df[df["dataset_fraction"].astype(float) == 1.0].copy()
    idx = full.groupby("policy_mode")["throughput_points_per_second"].idxmax()
    out = full.loc[idx, ["policy_mode", "workers", "total_runtime_seconds", "throughput_points_per_second"]].copy()
    out = out.rename(columns={
        "workers": "best_workers",
        "total_runtime_seconds": "runtime_seconds",
        "throughput_points_per_second": "throughput_points_per_second",
    })
    return out.reset_index(drop=True)


def make_table9(base_dir: Path, region_a: str, region_b: str, local_dir: Path) -> pd.DataFrame:
    local = best_full_workload(read_csv_required(local_dir / "scaling_and_runtime_results.csv"))
    a = best_full_workload(read_csv_required(base_dir / region_a / "scaling_and_runtime_results.csv"))
    b = best_full_workload(read_csv_required(base_dir / region_b / "scaling_and_runtime_results.csv"))

    local = local.rename(columns={
        "best_workers": "local_best_workers",
        "runtime_seconds": "local_runtime_seconds",
        "throughput_points_per_second": "local_throughput_points_per_second",
    })
    a = a.rename(columns={
        "best_workers": f"{region_a}_best_workers",
        "runtime_seconds": f"{region_a}_runtime_seconds",
        "throughput_points_per_second": f"{region_a}_throughput_points_per_second",
    })
    b = b.rename(columns={
        "best_workers": f"{region_b}_best_workers",
        "runtime_seconds": f"{region_b}_runtime_seconds",
        "throughput_points_per_second": f"{region_b}_throughput_points_per_second",
    })

    out = local.merge(a, on="policy_mode").merge(b, on="policy_mode")

    for region in [region_a, region_b]:
        out[f"{region}_overhead_ms_per_point_vs_local"] = (
            (1.0 / out[f"{region}_throughput_points_per_second"])
            - (1.0 / out["local_throughput_points_per_second"])
        ) * 1000.0

    return out.sort_values("policy_mode")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, help="Downloaded cloud result directory for one run_id.")
    parser.add_argument("--region-a", default="asia-southeast1")
    parser.add_argument("--region-b", default="us-central1")
    parser.add_argument("--local-dir", default="paper_outputs/fgcs_extended_benchmark")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    local_dir = Path(args.local_dir)

    merged = hash_summary(base_dir, args.region_a, args.region_b)
    table8 = make_table8(merged, args.region_a, args.region_b)
    table9 = make_table9(base_dir, args.region_a, args.region_b, local_dir)

    merged_path = base_dir / "cross_region_comparison.csv"
    table8_path = base_dir / "fgcs_table8_cross_region_hash_summary.csv"
    table9_path = base_dir / "fgcs_table9_local_vs_cloud_throughput.csv"

    merged.to_csv(merged_path, index=False)
    table8.to_csv(table8_path, index=False)
    table9.to_csv(table9_path, index=False)

    print("\nCross-region match summary:")
    print(table8.to_string(index=False))

    print("\nLocal vs cloud throughput summary:")
    print(table9.to_string(index=False))

    print(f"\nSaved: {merged_path}")
    print(f"Saved: {table8_path}")
    print(f"Saved: {table9_path}")

    if not merged["cross_region_match"].all():
        failed = merged[~merged["cross_region_match"]]
        print("\n[WARN] Some cross-region hashes did not match:")
        print(failed[MERGE_KEYS + ["cross_region_match"]].head(20).to_string(index=False))
    else:
        print("\n[OK] All cross-region hashes matched for identical workload/policy/seed/worker conditions.")

    if merged["unauthorized_max"].max() == 0:
        print("[OK] Zero unauthorized invocations across both cloud regions.")
    else:
        print("[WARN] Unauthorized invocations detected in cloud output.")


if __name__ == "__main__":
    main()