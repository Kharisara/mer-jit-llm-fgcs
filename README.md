Deterministic Replay and Benchmarking Infrastructure for Policy-Gated Multimodal AI Systems

This repository contains the reproducibility pipeline for the paper:

A Deterministic Replay and Benchmarking Infrastructure for Reproducible Validation of Policy-Gated Multimodal AI Systems

The repository reproduces the FGCS deterministic replay benchmark using prepared MELD-derived replay inputs, an offline behavioral-cloning trace, and a live behavioral-cloning replay mode.

Zenodo archive:

https://doi.org/10.5281/zenodo.20394431
Requirements
Python 3.10+
Recommended: virtual environment

Create a virtual environment:

python -m venv .venv

Activate on Windows:

.venv\Scripts\activate

Activate on Linux/Mac:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Running the FGCS Reproduction Pipeline

Run the full FGCS benchmark:

python run_fgcs_extended_benchmark.py --config configs/fgcs_extended_benchmark.yaml

Then generate manuscript-ready tables, figures, benchmark summaries, and mean-SD runtime/speedup tables:

python summarize_fgcs_extended_results.py

If run_full_reproduction.py is available and has been updated to call both commands above, the full pipeline can also be executed using:

python run_full_reproduction.py
Final Benchmark Design

The final FGCS benchmark evaluates:

5 workload fractions × 6 policy modes × 3 seeds × 4 worker configurations = 360 benchmark conditions

The workload fractions are:

0.10, 0.25, 0.50, 0.75, 1.00

These correspond to:

1,135, 2,837, 5,675, 8,513, and 11,351 replay decision points

The six policy modes are:

proxy
bc
bc_live
random
always
never

The replay seeds are:

1, 2, 3

The worker configurations are:

1, 2, 4, 8

No synthetic million-scale workload is reported as part of the final benchmark.

Required Prepared Inputs

The FGCS benchmark expects the following prepared inputs:

configs/fgcs_extended_benchmark.yaml
paper_outputs/replay_input_clean.csv
paper_outputs/policy_first_outputs_bc.csv

The file replay_input_clean.csv contains the prepared MELD-derived replay decision points.

The file policy_first_outputs_bc.csv contains the prepared offline behavioral-cloning action trace used by the bc replay mode.

The live behavioral-cloning mode, bc_live, executes the behavioral-cloning model during replay using stored state representations where available. This mode is used to evaluate live policy-inference integration and latency measurement, not policy optimality.

Important Notes on Policy Modes

The benchmark includes both offline and live behavioral-cloning modes:

bc: replays a prepared behavioral-cloning action trace.
bc_live: executes the behavioral-cloning policy during replay and records state-loading and policy-inference latency.

The bc_live results should be interpreted as execution-level integration and latency results. They should not be interpreted as evidence of policy optimality, causal intervention effectiveness, clinical effectiveness, or learned-policy quality.

Pipeline Overview

The reproduction pipeline performs the following steps:

Load prepared MELD-derived replay inputs.
Select deterministic workload fractions.
Execute replay across policy modes, seeds, and worker configurations.
Apply policy-gated invocation control.
Reconstruct action traces in replay order.
Verify deterministic replay using SHA-256 action hashing.
Record runtime, throughput, intervention rates, speedup, unauthorized invocation counts, and live BC latency metrics.
Generate manuscript-ready tables, figures, benchmark summaries, and mean-SD runtime/speedup tables.
Output Structure
Raw Benchmark Outputs

Generated in:

paper_outputs/fgcs_extended_benchmark/

Main files:

scaling_and_runtime_results.csv
Runtime, throughput, workload fraction, policy, seed, and worker-level execution results.
determinism_hash_results.csv
Trace-hash and hash-match results for replay reproducibility analysis.
parallel_speedup_results.csv
Speedup and throughput-gain summaries relative to single-worker execution.
policy_ablation_costs.csv
Policy-level intervention and cost summaries.
stage_latency_summary.csv
Stage-level latency summaries for replay execution.
live_bc_predictions.csv
Live behavioral-cloning predictions with state-loading and inference latencies.
Manuscript Tables and Figures

Generated in:

paper_outputs/fgcs_tables_figures/

Important generated outputs include:

fgcs_table_benchmark_design.csv
fgcs_table_benchmark_design.tex
fgcs_table_runtime_full_workload.csv
fgcs_table_runtime_full_workload.tex
fgcs_table_runtime_full_workload_mean_sd.csv
fgcs_table_runtime_full_workload_mean_sd.tex
fgcs_table_parallel_speedup_full_workload.csv
fgcs_table_parallel_speedup_full_workload.tex
fgcs_table_parallel_speedup_full_workload_mean_sd.csv
fgcs_table_parallel_speedup_full_workload_mean_sd.tex
fgcs_table_determinism_compact.csv
fgcs_table_determinism_compact.tex
fgcs_table_policy_ablation_full_workload.csv
fgcs_table_policy_ablation_full_workload.tex
fgcs_table_live_bc_compact.csv
fgcs_table_live_bc_compact.tex
fgcs_extended_benchmark_summary.md

Generated figures include runtime, throughput, worker-level throughput, speedup, intervention-rate, policy-runtime, and live-BC diagnostic figures. The manuscript may use only a subset of these figures; the remaining figures are retained as reproducibility artifacts.

Deterministic Reproducibility

The benchmark uses:

fixed random seeds,
deterministic workload fractions,
deterministic replay ordering,
fixed prepared replay trajectories,
replay-order trace reconstruction,
SHA-256 action-sequence hashing.

Under identical software and prepared-input conditions, deterministic policy modes are expected to produce stable action traces for each workload fraction. The seed-controlled random policy is expected to vary across seeds.

Dataset

Experiments use prepared replay inputs derived from the MELD dataset:

https://affective-meld.github.io/

The raw MELD dataset is not redistributed in this repository due to licensing and storage constraints.

MELD is used here as a public multimodal replay workload. It is not treated as a real intervention dataset, clinical dataset, or real user-state dataset.

Dataset Preparation

This repository does not include the full raw MELD preprocessing pipeline.

To reproduce the benchmark results directly, use the prepared replay input files included in this artifact:

paper_outputs/replay_input_clean.csv
paper_outputs/policy_first_outputs_bc.csv

If these files are not present, users must obtain MELD from the original providers and recreate equivalent replay inputs using their own preprocessing pipeline.

Scope

This repository reproduces execution-level replay validation experiments under deterministic offline replay.

It evaluates:

deterministic replay consistency,
policy-gated invocation behavior,
intervention-frequency measurement,
worker-level replay behavior,
runtime and throughput under a deterministic generation stub,
live behavioral-cloning integration and latency measurement,
reproducibility artifacts.

It does not provide:

live open-ended LLM serving evaluation,
end-to-end conversational response-generation throughput,
conversational quality evaluation,
clinical evaluation,
causal intervention-effectiveness evaluation,
behavioral-cloning policy training.
Citation

If you use this repository, cite the Zenodo archive:

@software{kharisara_mer_jit_llm_fgcs_2026,
  author       = {Kharisara},
  title        = {Kharisara/mer-jit-llm-fgcs: FGCS reproducibility package v1.0.2},
  year         = {2026},
  version      = {v1.0.2},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20394431},
  url          = {https://doi.org/10.5281/zenodo.20394431}
}
License

Released for research reproducibility purposes.