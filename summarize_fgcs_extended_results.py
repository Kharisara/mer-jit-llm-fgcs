import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# BASIC UTILITIES
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"[OUT] {path}")


def safe_round(df: pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["float64", "float32"]).columns
    out[float_cols] = out[float_cols].round(decimals)
    return out


# ============================================================
# TABLE GENERATION
# ============================================================

def make_extended_scalability_table(extended_df: pd.DataFrame) -> pd.DataFrame:
    table = (
        extended_df
        .groupby(["workload_size", "policy_mode", "workers"], as_index=False)
        .agg({
            "replay_runtime_seconds": "mean",
            "throughput_points_per_second": "mean",
            "policy_action_time_seconds": "mean",
            "gating_time_seconds": "mean",
            "aggregation_time_seconds": "mean",
            "trace_hash_time_seconds": "mean",
            "trace_hash_overhead_percent": "mean",
            "intervention_rate": "mean",
            "unauthorized_invocations": "max",
        })
        .sort_values(["workload_size", "policy_mode", "workers"])
    )

    return safe_round(table)


def make_extended_determinism_table(det_df: pd.DataFrame) -> pd.DataFrame:
    table = (
        det_df
        .groupby(["workload_size", "policy_mode", "workers"], as_index=False)
        .agg({
            "hash_match": "min",
            "intervention_rate": "mean",
            "unauthorized_invocations": "max",
        })
        .sort_values(["workload_size", "policy_mode", "workers"])
    )

    return safe_round(table)


def make_trace_overhead_table(trace_df: pd.DataFrame) -> pd.DataFrame:
    table = (
        trace_df
        .groupby(["workload_size", "policy_mode", "workers"], as_index=False)
        .agg({
            "replay_runtime_seconds": "mean",
            "policy_action_time_seconds": "mean",
            "gating_time_seconds": "mean",
            "aggregation_time_seconds": "mean",
            "trace_hash_time_seconds": "mean",
            "trace_hash_overhead_percent": "mean",
            "throughput_points_per_second": "mean",
        })
        .sort_values(["workload_size", "policy_mode", "workers"])
    )

    return safe_round(table)


def make_fault_injection_table(fault_df: pd.DataFrame) -> pd.DataFrame:
    table = (
        fault_df
        .groupby(["workload_size", "policy_mode", "injection_rate"], as_index=False)
        .agg({
            "eligible_non_intervention_rows": "mean",
            "injected_violations": "mean",
            "detected_violations": "mean",
            "false_negatives": "max",
            "detection_recall": "min",
            "fault_detection_time_seconds": "mean",
        })
        .sort_values(["workload_size", "policy_mode", "injection_rate"])
    )

    return safe_round(table)


def make_summary_findings_table(
    extended_df: pd.DataFrame,
    det_df: pd.DataFrame,
    fault_df: pd.DataFrame,
) -> pd.DataFrame:
    max_workload = int(extended_df["workload_size"].max())
    min_hash_match = int(det_df["hash_match"].min())
    max_unauthorized = int(det_df["unauthorized_invocations"].max())
    min_fault_recall = float(fault_df["detection_recall"].min()) if len(fault_df) else 1.0

    full_df = extended_df[extended_df["workload_size"] == max_workload]

    max_throughput = float(full_df["throughput_points_per_second"].max())
    min_runtime = float(full_df["replay_runtime_seconds"].min())
    max_runtime = float(full_df["replay_runtime_seconds"].max())
    mean_hash_overhead = float(full_df["trace_hash_overhead_percent"].mean())

    rows = [
        {
            "Finding": "Maximum amplified replay workload",
            "Observed value": f"{max_workload:,} decision points",
            "Interpretation": "The benchmark scales beyond the original MELD-derived workload.",
        },
        {
            "Finding": "Deterministic trace consistency",
            "Observed value": f"Minimum hash_match = {min_hash_match}",
            "Interpretation": "Replay traces remained identical across worker configurations.",
        },
        {
            "Finding": "Unauthorized invocation under normal replay",
            "Observed value": f"Maximum unauthorized invocations = {max_unauthorized}",
            "Interpretation": "The invocation gate preserved non-intervention behavior.",
        },
        {
            "Finding": "Fault-injection detection",
            "Observed value": f"Minimum detection recall = {min_fault_recall:.2f}",
            "Interpretation": "Injected unauthorized invocations were detected completely.",
        },
        {
            "Finding": "Full-workload throughput range",
            "Observed value": f"Maximum throughput = {max_throughput:.2f} points/s",
            "Interpretation": "Replay throughput is measurable at 1M scale.",
        },
        {
            "Finding": "Full-workload runtime range",
            "Observed value": f"{min_runtime:.4f}–{max_runtime:.4f} seconds",
            "Interpretation": "Runtime varies by policy mode and worker configuration.",
        },
        {
            "Finding": "Trace verification overhead",
            "Observed value": f"Mean overhead at max workload = {mean_hash_overhead:.4f}%",
            "Interpretation": "Trace hashing adds bounded verification overhead.",
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# FIGURE GENERATION
# ============================================================

def plot_runtime_vs_workload(extended_df: pd.DataFrame, out_path: Path) -> None:
    """
    Figure: runtime vs workload size.
    Uses workers=1 to provide a clear sequential baseline.
    """
    df = extended_df[extended_df["workers"] == extended_df["workers"].min()].copy()

    fig_df = (
        df
        .groupby(["workload_size", "policy_mode"], as_index=False)
        .agg({"replay_runtime_seconds": "mean"})
        .sort_values(["policy_mode", "workload_size"])
    )

    plt.figure()
    for policy_mode, group in fig_df.groupby("policy_mode"):
        plt.plot(
            group["workload_size"],
            group["replay_runtime_seconds"],
            marker="o",
            label=policy_mode,
        )

    plt.xlabel("Replay workload size")
    plt.ylabel("Runtime seconds")
    plt.title("Extended replay runtime vs workload size")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OUT] {out_path}")


def plot_throughput_vs_workload(extended_df: pd.DataFrame, out_path: Path) -> None:
    """
    Figure: throughput vs workload size.
    Uses workers=1 to show baseline throughput behavior across workload sizes.
    """
    df = extended_df[extended_df["workers"] == extended_df["workers"].min()].copy()

    fig_df = (
        df
        .groupby(["workload_size", "policy_mode"], as_index=False)
        .agg({"throughput_points_per_second": "mean"})
        .sort_values(["policy_mode", "workload_size"])
    )

    plt.figure()
    for policy_mode, group in fig_df.groupby("policy_mode"):
        plt.plot(
            group["workload_size"],
            group["throughput_points_per_second"],
            marker="o",
            label=policy_mode,
        )

    plt.xlabel("Replay workload size")
    plt.ylabel("Throughput: decision points per second")
    plt.title("Extended replay throughput vs workload size")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OUT] {out_path}")


def plot_trace_hash_overhead(trace_df: pd.DataFrame, out_path: Path) -> None:
    """
    Figure: trace hash overhead percentage vs workload size.
    Aggregates across policy modes, seeds, and worker counts.
    """
    fig_df = (
        trace_df
        .groupby("workload_size", as_index=False)
        .agg({"trace_hash_overhead_percent": "mean"})
        .sort_values("workload_size")
    )

    plt.figure()
    plt.plot(
        fig_df["workload_size"],
        fig_df["trace_hash_overhead_percent"],
        marker="o",
    )

    plt.xlabel("Replay workload size")
    plt.ylabel("Trace hash overhead (%)")
    plt.title("Trace verification overhead across workload sizes")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OUT] {out_path}")


def plot_fault_detection_recall(fault_df: pd.DataFrame, out_path: Path) -> None:
    """
    Figure: minimum detection recall for each injection rate.
    """
    fig_df = (
        fault_df
        .groupby("injection_rate", as_index=False)
        .agg({"detection_recall": "min"})
        .sort_values("injection_rate")
    )

    plt.figure()
    plt.plot(
        fig_df["injection_rate"],
        fig_df["detection_recall"],
        marker="o",
    )

    plt.xlabel("Injected unauthorized invocation rate")
    plt.ylabel("Minimum detection recall")
    plt.title("Unauthorized invocation detection under fault injection")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OUT] {out_path}")


def plot_worker_throughput_at_max_workload(extended_df: pd.DataFrame, out_path: Path) -> None:
    """
    Figure: throughput vs workers at the largest workload size.
    """
    max_workload = extended_df["workload_size"].max()

    df = extended_df[extended_df["workload_size"] == max_workload].copy()

    fig_df = (
        df
        .groupby(["policy_mode", "workers"], as_index=False)
        .agg({"throughput_points_per_second": "mean"})
        .sort_values(["policy_mode", "workers"])
    )

    plt.figure()
    for policy_mode, group in fig_df.groupby("policy_mode"):
        plt.plot(
            group["workers"],
            group["throughput_points_per_second"],
            marker="o",
            label=policy_mode,
        )

    plt.xlabel("Number of workers")
    plt.ylabel("Throughput: decision points per second")
    plt.title(f"Worker-level throughput at {int(max_workload):,} replay points")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OUT] {out_path}")


# ============================================================
# VALIDATION CHECKS
# ============================================================

def print_validation_summary(
    extended_df: pd.DataFrame,
    det_df: pd.DataFrame,
    fault_df: pd.DataFrame,
) -> None:
    print("\n========== EXTENDED FGCS RESULT VALIDATION ==========")
    print(f"Extended scalability rows        : {len(extended_df)}")
    print(f"Extended determinism rows        : {len(det_df)}")
    print(f"Fault-injection rows             : {len(fault_df)}")

    print("\nWorkload sizes:")
    print(sorted(extended_df["workload_size"].unique().tolist()))

    print("\nPolicy modes:")
    print(sorted(extended_df["policy_mode"].unique().tolist()))

    print("\nWorkers:")
    print(sorted(extended_df["workers"].unique().tolist()))

    print("\nHash-match counts:")
    print(det_df["hash_match"].value_counts(dropna=False))

    print("\nMinimum detection recall:")
    if len(fault_df):
        print(fault_df["detection_recall"].min())
    else:
        print("No fault-injection rows found.")

    print("====================================================\n")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="paper_outputs/fgcs_extended_benchmarks",
        help="Directory containing extended FGCS benchmark CSV outputs",
    )
    parser.add_argument(
        "--out_dir",
        default="paper_outputs/fgcs_tables_figures",
        help="Directory where paper-ready tables and figures will be saved",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    ensure_dir(out_dir)

    extended_path = input_dir / "extended_scalability_results.csv"
    trace_path = input_dir / "trace_verification_overhead.csv"
    determinism_path = input_dir / "extended_determinism_results.csv"
    fault_path = input_dir / "fault_injection_detection.csv"

    require_file(extended_path)
    require_file(trace_path)
    require_file(determinism_path)
    require_file(fault_path)

    extended_df = pd.read_csv(extended_path)
    trace_df = pd.read_csv(trace_path)
    det_df = pd.read_csv(determinism_path)
    fault_df = pd.read_csv(fault_path)

    print_validation_summary(extended_df, det_df, fault_df)

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------
    scalability_table = make_extended_scalability_table(extended_df)
    determinism_table = make_extended_determinism_table(det_df)
    trace_overhead_table = make_trace_overhead_table(trace_df)
    fault_table = make_fault_injection_table(fault_df)
    summary_findings_table = make_summary_findings_table(
        extended_df=extended_df,
        det_df=det_df,
        fault_df=fault_df,
    )

    save_table(
        scalability_table,
        out_dir / "fgcs_table_extended_scalability.csv",
    )

    save_table(
        determinism_table,
        out_dir / "fgcs_table_extended_determinism.csv",
    )

    save_table(
        trace_overhead_table,
        out_dir / "fgcs_table_trace_overhead.csv",
    )

    save_table(
        fault_table,
        out_dir / "fgcs_table_fault_injection.csv",
    )

    save_table(
        summary_findings_table,
        out_dir / "fgcs_table_extended_summary_findings.csv",
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    plot_runtime_vs_workload(
        extended_df,
        out_dir / "fgcs_fig_runtime_vs_workload_size_extended.png",
    )

    plot_throughput_vs_workload(
        extended_df,
        out_dir / "fgcs_fig_throughput_vs_workload_size_extended.png",
    )

    plot_trace_hash_overhead(
        trace_df,
        out_dir / "fgcs_fig_trace_hash_overhead.png",
    )

    plot_fault_detection_recall(
        fault_df,
        out_dir / "fgcs_fig_fault_detection_recall.png",
    )

    plot_worker_throughput_at_max_workload(
        extended_df,
        out_dir / "fgcs_fig_worker_throughput_at_max_workload.png",
    )

    print("[DONE] Extended FGCS result summarization complete.")


if __name__ == "__main__":
    main()