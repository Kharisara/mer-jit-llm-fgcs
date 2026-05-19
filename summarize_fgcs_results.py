# summarize_fgcs_results.py

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"[OUT] {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="paper_outputs/fgcs_benchmarks",
        help="Directory containing FGCS benchmark CSV outputs",
    )
    parser.add_argument(
        "--out_dir",
        default="paper_outputs/fgcs_tables_figures",
        help="Directory to save paper-ready tables and figures",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    runtime_path = input_dir / "scaling_and_runtime_results.csv"
    stage_path = input_dir / "stage_latency_summary.csv"
    speedup_path = input_dir / "parallel_speedup_results.csv"
    determinism_path = input_dir / "determinism_hash_results.csv"
    policy_cost_path = input_dir / "policy_ablation_costs.csv"

    runtime_df = pd.read_csv(runtime_path)
    stage_df = pd.read_csv(stage_path)
    speedup_df = pd.read_csv(speedup_path)
    determinism_df = pd.read_csv(determinism_path)
    policy_cost_df = pd.read_csv(policy_cost_path)

    # ---------------------------------------------------------
    # Table 1: Dataset scaling and runtime summary
    # ---------------------------------------------------------
    scaling_table = (
        runtime_df
        .groupby(["dataset_fraction", "decision_points", "policy_mode", "workers"], as_index=False)
        .agg({
            "total_runtime_seconds": "mean",
            "throughput_points_per_second": "mean",
            "mean_latency_ms": "mean",
            "p95_latency_ms": "mean",
            "memory_delta_mb": "mean",
            "intervention_rate": "mean",
            "unauthorized_invocations": "mean",
        })
        .sort_values(["dataset_fraction", "policy_mode", "workers"])
    )

    save_table(scaling_table, out_dir / "fgcs_table_scalability_runtime.csv")

    # ---------------------------------------------------------
    # Table 2: Stage latency summary
    # ---------------------------------------------------------
    stage_cols = [
        "state_loading_ms_mean",
        "policy_inference_ms_mean",
        "gating_ms_mean",
        "generation_stub_ms_mean",
        "logging_ms_mean",
        "total_latency_ms_mean",
    ]

    available_stage_cols = [c for c in stage_cols if c in stage_df.columns]

    stage_table = (
        stage_df
        .groupby(["dataset_fraction", "policy_mode", "workers"], as_index=False)[available_stage_cols]
        .mean()
        .sort_values(["dataset_fraction", "policy_mode", "workers"])
    )

    save_table(stage_table, out_dir / "fgcs_table_stage_latency.csv")

    # ---------------------------------------------------------
    # Table 3: Parallel speedup summary
    # ---------------------------------------------------------
    speedup_table = (
        speedup_df
        .groupby(["dataset_fraction", "policy_mode", "workers"], as_index=False)
        .agg({
            "runtime_seconds": "mean",
            "throughput_points_per_second": "mean",
            "speedup_vs_single_worker": "mean",
        })
        .sort_values(["dataset_fraction", "policy_mode", "workers"])
    )

    save_table(speedup_table, out_dir / "fgcs_table_parallel_speedup.csv")

    # ---------------------------------------------------------
    # Table 4: Deterministic replay consistency
    # ---------------------------------------------------------
    determinism_table = (
        determinism_df
        .groupby(["dataset_fraction", "policy_mode", "workers"], as_index=False)
        .agg({
            "hash_match": "min",
            "intervention_rate": "mean",
            "unauthorized_invocations": "mean",
        })
        .sort_values(["dataset_fraction", "policy_mode", "workers"])
    )

    save_table(determinism_table, out_dir / "fgcs_table_determinism.csv")

    # ---------------------------------------------------------
    # Table 5: Policy ablation execution cost
    # ---------------------------------------------------------
    save_table(policy_cost_df, out_dir / "fgcs_table_policy_ablation_costs.csv")

    # ---------------------------------------------------------
    # Figure 1: Throughput vs workers
    # ---------------------------------------------------------
    full_runtime = runtime_df[runtime_df["dataset_fraction"] == runtime_df["dataset_fraction"].max()]
    fig_df = (
        full_runtime
        .groupby(["policy_mode", "workers"], as_index=False)
        .agg({"throughput_points_per_second": "mean"})
    )

    plt.figure()
    for policy_mode, group in fig_df.groupby("policy_mode"):
        group = group.sort_values("workers")
        plt.plot(
            group["workers"],
            group["throughput_points_per_second"],
            marker="o",
            label=policy_mode,
        )

    plt.xlabel("Number of workers")
    plt.ylabel("Throughput: decision points per second")
    plt.title("Replay throughput under parallel execution")
    plt.legend()
    plt.tight_layout()
    fig_path = out_dir / "fgcs_fig_throughput_vs_workers.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[OUT] {fig_path}")

    # ---------------------------------------------------------
    # Figure 2: Runtime vs dataset size
    # ---------------------------------------------------------
    one_worker_df = runtime_df[runtime_df["workers"] == runtime_df["workers"].min()]
    fig_df = (
        one_worker_df
        .groupby(["policy_mode", "dataset_fraction"], as_index=False)
        .agg({"total_runtime_seconds": "mean", "decision_points": "mean"})
    )

    plt.figure()
    for policy_mode, group in fig_df.groupby("policy_mode"):
        group = group.sort_values("dataset_fraction")
        plt.plot(
            group["decision_points"],
            group["total_runtime_seconds"],
            marker="o",
            label=policy_mode,
        )

    plt.xlabel("Decision points")
    plt.ylabel("Runtime seconds")
    plt.title("Replay runtime under dataset scaling")
    plt.legend()
    plt.tight_layout()
    fig_path = out_dir / "fgcs_fig_runtime_vs_dataset_size.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[OUT] {fig_path}")

    # ---------------------------------------------------------
    # Figure 3: Mean latency by pipeline stage
    # ---------------------------------------------------------
    stage_full = stage_df[
        (stage_df["dataset_fraction"] == stage_df["dataset_fraction"].max())
        & (stage_df["workers"] == stage_df["workers"].min())
    ]

    stage_means = {
        "State loading": stage_full["state_loading_ms_mean"].mean(),
        "Policy inference": stage_full["policy_inference_ms_mean"].mean(),
        "Gating": stage_full["gating_ms_mean"].mean(),
        "Generation stub": stage_full["generation_stub_ms_mean"].mean(),
        "Logging": stage_full["logging_ms_mean"].mean(),
    }

    plt.figure()
    plt.bar(stage_means.keys(), stage_means.values())
    plt.ylabel("Mean latency ms")
    plt.title("Mean latency by replay pipeline stage")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig_path = out_dir / "fgcs_fig_latency_by_stage.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[OUT] {fig_path}")

    print("[DONE] FGCS result summarization complete.")


if __name__ == "__main__":
    main()