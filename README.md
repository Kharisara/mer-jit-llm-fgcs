# A Cloud-Deployable Deterministic Replay and Benchmarking Infrastructure for Reproducible Validation of Policy-Gated Multimodal AI Systems

## Overview

This repository contains the reproducibility artifact accompanying the
paper:

> **A Cloud-Deployable Deterministic Replay and Benchmarking
> Infrastructure for Reproducible Validation of Policy-Gated Multimodal
> AI Systems**

The artifact implements a deterministic replay and benchmarking
infrastructure for reproducible validation of policy-gated multimodal AI
systems under deterministic execution conditions.

The benchmark operates on a MELD-derived replay workload containing
**11,351 decision points** and evaluates replay behavior across **5
workload fractions**, **6 policy modes**, **3 random seeds**, and **4
worker configurations** (360 benchmark conditions).

The infrastructure focuses on **execution-identifiable** properties:

-   Deterministic replay consistency
-   Policy-gated invocation behavior
-   Invocation-boundary enforcement
-   Auditability and trace reproducibility
-   Worker-level replay consistency
-   Benchmark scalability
-   Cross-region cloud reproducibility

It **does not** evaluate policy optimality, intervention effectiveness,
personalization quality, or real-world deployment outcomes.

------------------------------------------------------------------------

# Repository Structure

``` text
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

------------------------------------------------------------------------

# Replay Workload

The replay benchmark contains **11,351 decision points** derived from
MELD replay traces.

Each replay state includes:

-   Multimodal availability metadata
-   Emotion label metadata
-   Replay identifiers
-   State embeddings

The workload is used **only for deterministic replay benchmarking**.

------------------------------------------------------------------------

# Final Benchmark Design

``` text
5 workload fractions
× 6 policy modes
× 3 random seeds
× 4 worker configurations
= 360 benchmark conditions
```

## Workload Fractions

``` text
0.10
0.25
0.50
0.75
1.00
```

## Policy Modes

``` text
risk_proxy
bc
bc_live
random
always
never
```

### Policy Descriptions

-   **risk_proxy** -- deterministic affective-risk proxy providing
    selective intervention for diagnostic replay.
-   **bc** -- offline behavioural-cloning replay policy using stored
    actions.
-   **bc_live** -- live behavioural-cloning neural policy executed
    during replay.
-   **random** -- deterministic seed-controlled stochastic baseline.
-   **always** -- always intervene.
-   **never** -- never intervene.

------------------------------------------------------------------------

# Running the Main Benchmark

``` bash
python run_fgcs_extended_benchmark.py --config configs/fgcs_extended_benchmark.yaml
```

Expected outputs:

``` text
scaling_and_runtime_results.csv
stage_latency_summary.csv
determinism_hash_results.csv
parallel_speedup_results.csv
policy_ablation_costs.csv
live_bc_predictions.csv
```

------------------------------------------------------------------------

# Compact Fault Validation (RQ7)

The repository also includes compact fault-validation experiments that
evaluate the infrastructure's ability to detect injected execution
faults.

Supported fault categories:

-   Action-flip (policy-output corruption)
-   Unauthorized invocation
-   Trace-action corruption
-   Trace-row deletion
-   Trace-row duplication

## Action-Flip Validation

``` bash
python run_fgcs_extended_benchmark.py --config configs/fgcs_fault_action_flip.yaml
python summarize_fgcs_fault_action_flip.py
```

## Unauthorized Invocation Validation

``` bash
python run_fgcs_extended_benchmark.py --config configs/fgcs_fault_unauthorized_invoke.yaml
python summarize_fgcs_fault_unauthorized_invoke.py
```

## Trace-Integrity Validation

``` bash
python summarize_fgcs_fault_trace_corruption.py
```

## Combined RQ7 Tables

``` bash
python summarize_fgcs_rq7_fault_validation.py
```

Generated outputs include:

``` text
fgcs_table_fault_action_flip_detection_summary.csv
fgcs_table_fault_unauthorized_invoke_detection_summary.csv
fgcs_table_fault_trace_corruption_detection_summary.csv
fgcs_table_rq7_fault_detection_combined.csv
fgcs_table_validation_ablation_matrix.csv
```

------------------------------------------------------------------------

# Docker and Cloud Validation

Build:

``` bash
docker build -t fgcs-replay-cloud:v1 .
```

Cloud validation was executed using Google Cloud Run Jobs in:

``` text
asia-southeast1
us-central1
```

Results:

-   360/360 matched benchmark conditions
-   360/360 matching SHA-256 action-trace hashes
-   Zero unauthorized invocations during clean replay

------------------------------------------------------------------------

# Generated Outputs

## Local Benchmark

``` text
paper_outputs/fgcs_extended_benchmark/
```

## Compact Fault Validation

``` text
paper_outputs/fgcs_fault_action_flip/
paper_outputs/fgcs_fault_unauthorized_invoke/
paper_outputs/fgcs_tables_figures/
```

## Cloud Validation

``` text
cloud_results/
```

------------------------------------------------------------------------

# Reproducibility Scope

The infrastructure validates:

-   Deterministic replay
-   Trace reproducibility
-   Invocation-boundary enforcement
-   Replay scalability
-   Cross-region reproducibility
-   Injected execution-fault detection

It does **not** evaluate:

-   Policy quality
-   Clinical effectiveness
-   Reinforcement learning performance
-   Personalization
-   Safety effectiveness

------------------------------------------------------------------------

# Reproducibility Manifest

The repository includes:

``` text
fgcs_extended_reproducibility_manifest.json
```

covering:

-   Benchmark configuration
-   Replay workload
-   Policy modes
-   Fault-validation configuration
-   Experimental design
-   Generated artifacts
-   Validation outputs

------------------------------------------------------------------------

# Key Results

## Main Benchmark

-   11,351 replay decision points
-   360 benchmark conditions
-   Deterministic policies produced one stable trace hash per workload
    fraction
-   `risk_proxy` produced an intermediate intervention profile (\~23%)
-   Zero unauthorized invocations during clean replay
-   Best full-workload throughput exceeded 2,200 decision points/s for
    `bc_live`

## Compact Fault Validation

-   100% detection of injected action-flip faults
-   100% detection of injected unauthorized-invocation faults
-   100% detection of injected trace-action corruption
-   100% detection of dropped trace rows
-   100% detection of duplicated trace rows
-   No false positives during clean replay

## Cross-Region Cloud Validation

-   Successful execution in two cloud regions
-   Matching SHA-256 action-trace hashes across regions
-   Reproducible execution under identical configurations

------------------------------------------------------------------------

# License

This repository is provided for academic research and reproducibility
purposes.
