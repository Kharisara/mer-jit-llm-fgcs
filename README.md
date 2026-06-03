# A Cloud-Deployable Deterministic Replay and Benchmarking Infrastructure for Reproducible Validation of Policy-Gated Multimodal AI Systems

## Overview

This repository contains the reproducibility artifact accompanying the paper:

> **A Cloud-Deployable Deterministic Replay and Benchmarking Infrastructure for Reproducible Validation of Policy-Gated Multimodal AI Systems**

The artifact implements a deterministic replay and benchmarking framework for evaluating policy-gated multimodal AI systems under reproducible execution conditions.

The benchmark operates on a MELD-derived replay workload containing **11,351 decision points** and evaluates replay behavior across multiple workload fractions, policy modes, random seeds, and worker configurations.

The framework focuses on execution-identifiable properties including:

- Intervention controllability
- Deterministic replay consistency
- Policy agreement behavior
- Invocation-boundary enforcement
- Auditability and trace reproducibility
- Cross-region cloud deployment consistency

The framework does **not** attempt to evaluate causal effectiveness, policy optimality, personalization quality, or real-world intervention outcomes.

---

## Repository Structure

```text
.
├── configs/
│   └── fgcs_extended_benchmark.yaml
│
├── checkpoints/
│   └── jitai_policy_bc.pt
│
├── paper_outputs/
│   ├── fgcs_extended_benchmark/
│   └── fgcs_tables_figures/
│
├── cloud_results/
│   └── cloud360_20260603/
│
├── run_fgcs_extended_benchmark.py
├── run_fgcs_cloud_job.py
├── compare_cross_region_hashes.py
├── check_fgcs_paths.py
├── Dockerfile
├── requirements.txt
├── requirements_cloud.txt
└── README.md
```

---

## Replay Workload

The benchmark uses a MELD-derived replay workload consisting of:

```text
11,351 replay decision points
```

Each replay state contains:

- Text modality availability
- Audio modality availability
- Video modality availability
- Emotion label metadata
- State embeddings
- Replay identifiers

The workload is used exclusively for deterministic replay benchmarking and not for causal intervention evaluation.

---

## Final Benchmark Design

The final benchmark evaluates:

```text
5 workload fractions
× 6 policy modes
× 3 random seeds
× 4 worker configurations
= 360 benchmark conditions
```

### Workload Fractions

```text
0.10
0.25
0.50
0.75
1.00
```

### Policy Modes

```text
proxy
bc
bc_live
random
always
never
```

### Random Seeds

```text
1
2
3
```

### Worker Configurations

```text
1
2
4
8
```

---

## Policy Modes

### proxy

Rule-based proxy intervention policy using emotion labels.

### bc

Behavioral-cloning replay policy using pre-generated actions.

### bc_live

Live neural behavioral-cloning policy executed during replay.

### random

Seed-controlled pseudo-random intervention policy.

### always

Always intervene.

### never

Never intervene.

---

## Local Benchmark Execution

Run the complete benchmark:

```bash
python run_fgcs_extended_benchmark.py \
  --config configs/fgcs_extended_benchmark.yaml
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

## Docker Build

Build the cloud-validation container:

```bash
docker build -t fgcs-replay-cloud:v1 .
```

The generated image can be executed locally or deployed to cloud environments.

---

## Cross-Region Cloud Validation

The benchmark infrastructure was validated using containerized deployments executed as Google Cloud Run Jobs in:

```text
asia-southeast1
us-central1
```

Both regions executed the same benchmark configuration:

```text
360 benchmark conditions
```

### Cross-Region Validation Results

```text
360 / 360 matched conditions
360 / 360 matching SHA-256 action-trace hashes
0 unauthorized invocations
```

This validation demonstrates that identical replay configurations preserve trace-level reproducibility across geographically separated cloud environments.

---

## Generated Outputs

### Local Benchmark Outputs

```text
paper_outputs/
└── fgcs_extended_benchmark/
    ├── scaling_and_runtime_results.csv
    ├── stage_latency_summary.csv
    ├── determinism_hash_results.csv
    ├── parallel_speedup_results.csv
    ├── policy_ablation_costs.csv
    └── live_bc_predictions.csv
```

### Cloud Validation Outputs

```text
cloud_results/
└── cloud360_20260603/
    ├── asia-southeast1/
    ├── us-central1/
    ├── cross_region_comparison.csv
    ├── fgcs_table8_cross_region_hash_summary.csv
    ├── fgcs_table9_local_vs_cloud_throughput.csv
    └── cross_region_console_output.txt
```

---

## Reproducibility Scope

The framework is intended to evaluate:

- Deterministic replay consistency
- Policy-gated invocation behavior
- Execution stability
- Trace reproducibility
- Benchmark scalability
- Cross-region cloud deployment consistency
- Invocation-boundary enforcement

The framework is **not** intended to evaluate:

- Policy optimality
- Reinforcement learning performance
- Intervention effectiveness
- Personalization quality
- Safety effectiveness
- Real-world outcome improvement

---

## Reproducibility Manifest

The repository includes:

```text
fgcs_extended_reproducibility_manifest.json
```

which documents:

- Benchmark configuration
- Workload definition
- Policy modes
- Experimental design
- Generated artifacts
- Reproducibility outputs

---

## Key Results

### Local Benchmark

- 11,351 replay decision points
- 360 benchmark conditions
- Deterministic policies produced stable trace hashes
- Zero unauthorized invocations
- Best full-workload throughput exceeded 2,200 decision points/s for `bc_live`

### Cloud Validation

- Google Cloud Run deployment across two regions
- 360 matched cross-region conditions
- 360/360 matching SHA-256 action-trace hashes
- Zero unauthorized invocations
- Cross-region reproducibility successfully demonstrated

---

## License

This repository is provided for academic research and reproducibility purposes.