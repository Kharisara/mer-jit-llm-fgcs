from pathlib import Path
import pandas as pd
import numpy as np

IN_DIR = Path("paper_outputs/fgcs_extended_benchmark")
OUT_DIR = Path("paper_outputs/fgcs_tables_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

runtime_path = IN_DIR / "scaling_and_runtime_results.csv"
speedup_path = IN_DIR / "parallel_speedup_results.csv"

runtime_df = pd.read_csv(runtime_path)
speedup_df = pd.read_csv(speedup_path)

print("Runtime columns:", runtime_df.columns.tolist())
print("Speedup columns:", speedup_df.columns.tolist())


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find any of these columns: {candidates}")


policy_col = find_col(runtime_df, ["policy_mode", "policy"])
worker_col = find_col(runtime_df, ["workers", "num_workers", "worker_count"])
seed_col = find_col(runtime_df, ["seed", "random_seed"])
runtime_col = find_col(runtime_df, ["runtime_seconds", "runtime_s", "total_runtime_seconds", "replay_runtime_seconds"])
throughput_col = find_col(runtime_df, [
    "throughput_decision_points_per_second",
    "throughput_points_per_second",
    "throughput",
    "decision_points_per_second"
])

if "decision_points" in runtime_df.columns:
    workload_col = "decision_points"
elif "workload_size" in runtime_df.columns:
    workload_col = "workload_size"
elif "dataset_fraction" in runtime_df.columns:
    workload_col = "dataset_fraction"
else:
    raise KeyError("Could not find workload column.")

# Full workload = maximum workload size/fraction
full_workload = runtime_df[workload_col].max()
runtime_full = runtime_df[runtime_df[workload_col] == full_workload].copy()

# Runtime/throughput mean ± SD across seeds for each policy-worker pair
runtime_summary = (
    runtime_full
    .groupby([policy_col, worker_col], as_index=False)
    .agg(
        runtime_mean=(runtime_col, "mean"),
        runtime_sd=(runtime_col, "std"),
        throughput_mean=(throughput_col, "mean"),
        throughput_sd=(throughput_col, "std"),
        n_seeds=(seed_col, "nunique")
    )
)

runtime_summary["runtime_sd"] = runtime_summary["runtime_sd"].fillna(0)
runtime_summary["throughput_sd"] = runtime_summary["throughput_sd"].fillna(0)

# Select best worker per policy by highest mean throughput
best_runtime = (
    runtime_summary
    .sort_values([policy_col, "throughput_mean"], ascending=[True, False])
    .groupby(policy_col, as_index=False)
    .head(1)
)

policy_order = ["always", "bc", "bc_live", "never", "proxy", "random"]
best_runtime[policy_col] = pd.Categorical(best_runtime[policy_col], categories=policy_order, ordered=True)
best_runtime = best_runtime.sort_values(policy_col)

def fmt_mean_sd(mean, sd, decimals=2):
    return f"{mean:.{decimals}f} $\\pm$ {sd:.{decimals}f}"

best_runtime["Runtime (s)"] = best_runtime.apply(
    lambda r: fmt_mean_sd(r["runtime_mean"], r["runtime_sd"], 2), axis=1
)
best_runtime["Throughput (points/s)"] = best_runtime.apply(
    lambda r: fmt_mean_sd(r["throughput_mean"], r["throughput_sd"], 2), axis=1
)

table5 = best_runtime[[policy_col, worker_col, "Runtime (s)", "Throughput (points/s)"]].copy()
table5.columns = ["Policy", "Best workers", "Runtime (s)", "Throughput (points/s)"]

table5.to_csv(OUT_DIR / "fgcs_table_runtime_full_workload_mean_sd.csv", index=False)

# Speedup table
sp_policy_col = find_col(speedup_df, ["policy_mode", "policy"])
sp_worker_col = find_col(speedup_df, ["workers", "num_workers", "worker_count"])
sp_seed_col = find_col(speedup_df, ["seed", "random_seed"])
speedup_col = find_col(speedup_df, ["speedup", "speedup_vs_single_worker", "parallel_speedup"])

if "decision_points" in speedup_df.columns:
    sp_workload_col = "decision_points"
elif "workload_size" in speedup_df.columns:
    sp_workload_col = "workload_size"
elif "dataset_fraction" in speedup_df.columns:
    sp_workload_col = "dataset_fraction"
else:
    raise KeyError("Could not find speedup workload column.")

sp_full_workload = speedup_df[sp_workload_col].max()
speedup_full = speedup_df[speedup_df[sp_workload_col] == sp_full_workload].copy()

speedup_summary = (
    speedup_full
    .groupby([sp_policy_col, sp_worker_col], as_index=False)
    .agg(
        speedup_mean=(speedup_col, "mean"),
        speedup_sd=(speedup_col, "std"),
        n_seeds=(sp_seed_col, "nunique")
    )
)

speedup_summary["speedup_sd"] = speedup_summary["speedup_sd"].fillna(0)

best_speedup = (
    speedup_summary
    .sort_values([sp_policy_col, "speedup_mean"], ascending=[True, False])
    .groupby(sp_policy_col, as_index=False)
    .head(1)
)

best_speedup[sp_policy_col] = pd.Categorical(best_speedup[sp_policy_col], categories=policy_order, ordered=True)
best_speedup = best_speedup.sort_values(sp_policy_col)

best_speedup["Mean speedup"] = best_speedup.apply(
    lambda r: fmt_mean_sd(r["speedup_mean"], r["speedup_sd"], 2) + "$\\times$", axis=1
)

def interp(row):
    if row["speedup_mean"] < 1.05:
        return "No meaningful acceleration."
    if row["speedup_mean"] < 1.5:
        return "Modest acceleration."
    return "Strongest full-workload acceleration."

best_speedup["Interpretation"] = best_speedup.apply(interp, axis=1)

table6 = best_speedup[[sp_policy_col, sp_worker_col, "Mean speedup", "Interpretation"]].copy()
table6.columns = ["Policy", "Best workers", "Mean speedup", "Interpretation"]

table6.to_csv(OUT_DIR / "fgcs_table_parallel_speedup_full_workload_mean_sd.csv", index=False)

# Write LaTeX versions manually, clean and compact
def tex_escape_policy(x):
    return "\\texttt{" + str(x).replace("_", "\\_") + "}"

with open(OUT_DIR / "fgcs_table_runtime_full_workload_mean_sd.tex", "w", encoding="utf-8") as f:
    f.write("\\begin{table}[H]\n")
    f.write("\\centering\n")
    f.write("\\caption{Best full-workload runtime and throughput by policy. Values are mean $\\pm$ SD across three seeds.}\n")
    f.write("\\label{tab:runtime_full_workload}\n")
    f.write("\\footnotesize\n")
    f.write("\\setlength{\\tabcolsep}{3pt}\n")
    f.write("\\begin{tabular}{lrlr}\n")
    f.write("\\toprule\n")
    f.write("Policy & Best workers & Runtime (s) & Throughput (points/s) \\\\\n")
    f.write("\\midrule\n")
    for _, r in table5.iterrows():
        f.write(f"{tex_escape_policy(r['Policy'])} & {int(r['Best workers'])} & {r['Runtime (s)']} & {r['Throughput (points/s)']} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

with open(OUT_DIR / "fgcs_table_parallel_speedup_full_workload_mean_sd.tex", "w", encoding="utf-8") as f:
    f.write("\\begin{table}[H]\n")
    f.write("\\centering\n")
    f.write("\\caption{Best mean full-workload speedup by policy. Values are mean $\\pm$ SD across three seeds.}\n")
    f.write("\\label{tab:parallel_speedup_full}\n")
    f.write("\\footnotesize\n")
    f.write("\\setlength{\\tabcolsep}{4pt}\n")
    f.write("\\begin{tabularx}{\\columnwidth}{lrlY}\n")
    f.write("\\toprule\n")
    f.write("Policy & Best workers & Mean speedup & Interpretation \\\\\n")
    f.write("\\midrule\n")
    for _, r in table6.iterrows():
        f.write(f"{tex_escape_policy(r['Policy'])} & {int(r['Best workers'])} & {r['Mean speedup']} & {r['Interpretation']} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabularx}\n")
    f.write("\\end{table}\n")

print("[OUT]", OUT_DIR / "fgcs_table_runtime_full_workload_mean_sd.tex")
print("[OUT]", OUT_DIR / "fgcs_table_parallel_speedup_full_workload_mean_sd.tex")
print("\nTable 5:")
print(table5.to_string(index=False))
print("\nTable 6:")
print(table6.to_string(index=False))