#!/usr/bin/env python3
"""Regenerate ReplayBench-PG Figure 2 from the frozen mixed-repetition summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


POLICIES = ["always", "bc", "bc_live", "never", "random", "risk_proxy"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="paper_outputs/replaybench_timing_study/"
                "timing_runtime_summary_mixed_7_15.csv",
    )
    parser.add_argument(
        "--png",
        default="figures/Figure_2_Runtime_Scaling.png",
    )
    parser.add_argument(
        "--pdf",
        default="figures/Figure_2_Runtime_Scaling.pdf",
    )
    parser.add_argument(
        "--manifest",
        default="figures/Figure_2_Runtime_Scaling_manifest.json",
    )
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_file():
        raise FileNotFoundError(source)

    frame = pd.read_csv(source)
    required = {
        "dataset_fraction",
        "policy",
        "workers",
        "repetitions",
        "runtime_median_seconds",
        "runtime_q1_seconds",
        "runtime_q3_seconds",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    frame = frame.loc[frame["workers"].eq(1)].copy()
    expected_rows = len(POLICIES) * len(FRACTIONS)
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} one-worker rows; found {len(frame)}")

    observed_policies = sorted(frame["policy"].astype(str).unique())
    if observed_policies != sorted(POLICIES):
        raise ValueError(f"Unexpected policies: {observed_policies}")

    observed_fractions = sorted(frame["dataset_fraction"].astype(float).unique())
    if observed_fractions != FRACTIONS:
        raise ValueError(f"Unexpected fractions: {observed_fractions}")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for policy, marker in zip(POLICIES, MARKERS):
        group = frame.loc[frame["policy"].eq(policy)].sort_values("dataset_fraction")
        median = group["runtime_median_seconds"].to_numpy()
        lower = median - group["runtime_q1_seconds"].to_numpy()
        upper = group["runtime_q3_seconds"].to_numpy() - median
        ax.errorbar(
            group["dataset_fraction"].to_numpy(),
            median,
            yerr=[lower, upper],
            marker=marker,
            capsize=3,
            label=policy,
        )

    ax.set_xlabel("Workload fraction")
    ax.set_ylabel("Timed-execution runtime (s)")
    ax.set_xticks(FRACTIONS)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()

    png = Path(args.png)
    pdf = Path(args.pdf)
    manifest = Path(args.manifest)
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "input": source.as_posix(),
        "input_sha256": sha256_file(source),
        "outputs": {
            png.as_posix(): sha256_file(png),
            pdf.as_posix(): sha256_file(pdf),
        },
        "filter": "workers == 1",
        "expected_rows": expected_rows,
        "policies": POLICIES,
        "fractions": FRACTIONS,
        "error_bars": "Q1 and Q3 around the median",
        "repetition_rule": "7 repetitions for fractions 0.10-0.75; 15 for fraction 1.00",
    }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OUT] {png}")
    print(f"[OUT] {pdf}")
    print(f"[OUT] {manifest}")


if __name__ == "__main__":
    main()
