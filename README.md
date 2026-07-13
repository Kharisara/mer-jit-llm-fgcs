# ReplayBench-PG: A Fault-Aware Deterministic Replay Benchmarking Framework for Policy-Gated Multimodal AI Pipelines

## Overview

This repository contains the reproducibility artifact accompanying the paper:

> **ReplayBench-PG: A Fault-Aware Deterministic Replay Benchmarking Framework for Policy-Gated Multimodal AI Pipelines**

ReplayBench-PG is a deterministic replay benchmarking framework for reproducible validation of execution-level properties in policy-gated multimodal AI pipelines. The framework provides controlled replay execution, policy-ablation benchmarking, invocation-boundary verification, controlled fault validation, cross-region cloud-job validation, and reproducibility artifact generation.

The benchmark operates on a MELD-derived replay workload containing **11,351 decision points** and evaluates replay behavior across **5 workload fractions**, **6 policy modes**, **3 random seeds**, and **4 worker configurations**, producing **360 benchmark conditions**.

ReplayBench-PG evaluates execution-level properties including:

- Deterministic replay consistency
- Policy-gated invocation behavior
- Invocation-boundary enforcement
- Trace reproducibility
- Worker-level replay consistency
- Replay scalability
- Cross-region cloud-job consistency
- Controlled execution-fault detection

ReplayBench-PG does **not** evaluate policy quality, policy optimality, intervention effectiveness, conversational quality, personalization, clinical effectiveness, content-level safety, or production deployment performance.

---

# Repository Structure

```text
.
├── configs/
│   ├── fgcs_extended_benchmark.yaml
│   ├── fgcs_fault_action_flip.yaml
│   └── fgcs_fault_unauthorized_invoke.yaml
├── checkpoints/
│   └── jitai_policy_bc.pt
├── paper_outputs/
│   ├── fgcs_extended_benchmark/
│   ├── fgcs_fault_action_flip/
│   ├── fgcs_fault_unauthorized_invoke/
│   └── fgcs_tables_figures/
├── cloud_results/
├── run_fgcs_extended_benchmark.py
├── run_fgcs_cloud_job.py
├── compare_cross_region_hashes.py
├── summarize_fgcs_fault_action_flip.py
├── summarize_fgcs_fault_unauthorized_invoke.py
├── summarize_fgcs_fault_trace_corruption.py
├── summarize_fgcs_rq7_fault_validation.py
├── fgcs_fault_validation_framework.py
├── Dockerfile
├── requirements.txt
├── requirements_cloud.txt
└── README.md
```

*Note:* Internal filenames retain the original **fgcs** prefix for compatibility with the released reproducibility package. The accompanying manuscript refers to the framework as **ReplayBench-PG**.

---

# Replay Workload

ReplayBench-PG uses a replay workload containing **11,351 decision points** derived from the publicly available MELD dataset.

Each replay state includes:

- Multimodal availability metadata
- Emotion label metadata
- Replay identifiers
- State embeddings

The replay workload is used exclusively for deterministic replay benchmarking.

---

# Benchmark Design

```text
5 workload fractions
× 6 policy modes
× 3 random seeds
× 4 worker configurations
= 360 benchmark conditions
```

## Workload Fractions

```text
0.10
0.25
0.50
0.75
1.00
```

## Policy Modes

```text
risk_proxy
bc
bc_live
random
always
never
```

### Policy Descriptions

- **risk_proxy** — deterministic diagnostic policy providing action-diverse replay for infrastructure validation.
- **bc** — offline behavioural-cloning replay using stored actions.
- **bc_live** — live behavioural-cloning policy executed during replay.
- **random** — deterministic seed-controlled stochastic replay.
- **always** — always intervene.
- **never** — never intervene.

The included policies are intended to exercise ReplayBench-PG under different execution characteristics and are **not** intended to compare policy quality.

---

# Running the Main Benchmark

```bash
python run_fgcs_extended_benchmark.py --config configs/fgcs_extended_benchmark.yaml
```

Expected outputs:

```text
scaling_and_runtime_results.csv
stage_latency_summary.csv
determinism_hash_results.csv
parallel_speedup_results.csv
policy_ablation_costs.csv
live_bc_predictions.csv
```

---

# Controlled Fault Validation (RQ7)

ReplayBench-PG includes compact controlled fault-validation workflows that evaluate whether execution anomalies are correctly detected.

Supported fault categories include:

- Action-flip
- Unauthorized invocation
- Trace-action corruption
- Dropped replay rows
- Duplicated replay rows

## Action-Flip Validation

```bash
python run_fgcs_extended_benchmark.py --config configs/fgcs_fault_action_flip.yaml
python summarize_fgcs_fault_action_flip.py
```

## Unauthorized Invocation Validation

```bash
python run_fgcs_extended_benchmark.py --config configs/fgcs_fault_unauthorized_invoke.yaml
python summarize_fgcs_fault_unauthorized_invoke.py
```

## Trace Integrity Validation

```bash
python summarize_fgcs_fault_trace_corruption.py
```

## Combined RQ7 Summary

```bash
python summarize_fgcs_rq7_fault_validation.py
```

Generated outputs include:

```text
fgcs_table_fault_action_flip_detection_summary.csv
fgcs_table_fault_unauthorized_invoke_detection_summary.csv
fgcs_table_fault_trace_corruption_detection_summary.csv
fgcs_table_rq7_fault_detection_combined.csv
fgcs_table_validation_ablation_matrix.csv
```

---

# Docker and Cloud Validation

Build:

```bash
docker build -t fgcs-replay-cloud:v1 .
```

ReplayBench-PG was validated using Google Cloud Run Jobs in:

```text
asia-southeast1
us-central1
```

Observed results:

- 360/360 completed benchmark conditions
- 360/360 matching SHA-256 replay hashes
- Zero unauthorized invocations during clean replay

---

# Generated Outputs

## Local Benchmark

```text
paper_outputs/fgcs_extended_benchmark/
```

## Controlled Fault Validation

```text
paper_outputs/fgcs_fault_action_flip/
paper_outputs/fgcs_fault_unauthorized_invoke/
paper_outputs/fgcs_tables_figures/
```

## Cloud Validation

```text
cloud_results/
```

---

# Reproducibility Scope

ReplayBench-PG validates:

- Deterministic replay
- Replay trace reproducibility
- Invocation-boundary enforcement
- Replay scalability
- Cross-region cloud-job consistency
- Controlled execution-fault detection

ReplayBench-PG does **not** validate:

- Policy quality
- Policy optimality
- Clinical effectiveness
- Conversational quality
- Personalization
- Reinforcement learning performance
- Content-level safety

---

# Reproducibility Manifest

The repository includes:

```text
fgcs_extended_reproducibility_manifest.json
```

covering:

- Benchmark configuration
- Replay workload
- Policy modes
- Controlled fault-validation configuration
- Experimental design
- Generated artifacts
- Validation outputs

---

# Key Results

## Main Benchmark

- 11,351 replay decision points
- 360 benchmark conditions
- Stable SHA-256 replay hashes for deterministic policy modes
- Approximately 23% intervention rate for `risk_proxy`
- Zero unauthorized invocations during clean replay
- More than 2,200 replay decisions/s for `bc_live`

## Controlled Fault Validation

- 100% detection of injected action-flip faults
- 100% detection of unauthorized invocations
- 100% detection of trace-action corruption
- 100% detection of dropped replay rows
- 100% detection of duplicated replay rows
- No false positives during clean replay

## Cross-Region Cloud Validation

- Successful execution in two Google Cloud regions
- Matching SHA-256 replay hashes across regions
- Reproducible execution under identical benchmark configurations

---

# License

This repository is provided for academic research and reproducibility purposes.