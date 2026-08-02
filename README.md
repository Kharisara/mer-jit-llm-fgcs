# ReplayBench-PG: Fault-Detecting Deterministic Action-Trace Replay for Execution Validation of Policy-Gated AI Pipelines

## Overview

This repository contains the reproducibility artifact accompanying the paper:

> **ReplayBench-PG: Fault-Detecting Deterministic Action-Trace Replay for Execution Validation of Policy-Gated AI Pipelines**

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

# v2.5.9 Label-Independent Evaluation Correction

Version 2.5.9 corrects the fault-evaluation methodology by separating generic validation from fault injection and post-hoc scoring. It supersedes the earlier ground-truth-aware event-localization interpretation while retaining the historical outputs for reproducibility.

## Evidence models

The corrected comparison uses the same observed artifacts and trusted clean references for both models:

- **Primary evidence model, `V_primary`:** ordered action-hash equality, row-cardinality equality, and a primary-log authorization contradiction recomputed directly from `authorized_to_generate` and `generation_invoked`.
- **Full evidence model, `V_full`:** `V_primary` plus receipt-digest validation, receipt reconciliation, record-bound digest `H_R`, configuration digest `H_C`, and configuration-bound digest `H_RC`.

The generic validator rejects trace inputs containing fault labels or injection-marker columns. Ground-truth labels, selected event identifiers, and expected outcomes are retained in a separate manifest and are joined only after generic findings have been frozen.

## Corrected evaluation results

The corrected corpus contains 270 evidence units: 228 positive controls and 42 negative controls. The negative set comprises 18 clean receipt-enabled execution instances and 24 prespecified benign post-execution applications.

- `V_primary` detected **84/228** positive-control units.
- `V_full` detected **228/228** positive-control units.
- `V_full` flagged **0/42** negative-control units.
- Independent comparison with clean references localized **4,906/4,906** supported injected events, with zero off-target or missed identifiers.

These counts mix distinct evidence-unit types by design: receipt faults are execution instances, whereas saved-trace, record/configuration, primary-control, and benign cases are post-execution validator applications. The machine-readable outputs retain that distinction.

## Reproduce the corrected analysis

From the repository root:

```bash
python run_phase1_label_independent_validation.py
python score_phase1_label_independent_validation.py
python -m pytest -q
python run_final_artifact_validation.py
```

The corrected outputs are written to:

```text
paper_outputs/phase1_label_independent_validation/
```

Key files include:

```text
generic_validator_findings.jsonl
ground_truth_manifest.jsonl
validator_input_separation_audit.csv
per_evidence_scored_results.csv
event_localization_results.csv
baseline_comparison_by_fault_class.csv
phase1_validation_manifest.json
```

The authoritative implementation is `replaybench/generic_validator.py`. The historical mode-aware evaluation scripts remain available but contain explicit notices directing label-independent claims to the corrected pipeline.

---

# Repository Structure

```text
.
â”œâ”€â”€ configs/
â”‚   â”œâ”€â”€ fgcs_extended_benchmark.yaml
â”‚   â”œâ”€â”€ fgcs_fault_action_flip.yaml
â”‚   â””â”€â”€ fgcs_fault_unauthorized_invoke.yaml
â”œâ”€â”€ checkpoints/
â”‚   â””â”€â”€ jitai_policy_bc.pt
â”œâ”€â”€ paper_outputs/
â”‚   â”œâ”€â”€ fgcs_extended_benchmark/
â”‚   â”œâ”€â”€ fgcs_fault_action_flip/
â”‚   â”œâ”€â”€ fgcs_fault_unauthorized_invoke/
â”‚   â””â”€â”€ fgcs_tables_figures/
â”œâ”€â”€ cloud_results/
â”œâ”€â”€ run_fgcs_extended_benchmark.py
â”œâ”€â”€ run_fgcs_cloud_job.py
â”œâ”€â”€ compare_cross_region_hashes.py
â”œâ”€â”€ summarize_fgcs_fault_action_flip.py
â”œâ”€â”€ summarize_fgcs_fault_unauthorized_invoke.py
â”œâ”€â”€ summarize_fgcs_fault_trace_corruption.py
â”œâ”€â”€ summarize_fgcs_rq7_fault_validation.py
â”œâ”€â”€ fgcs_fault_validation_framework.py
â”œâ”€â”€ Dockerfile
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ requirements_cloud.txt
â””â”€â”€ README.md
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
Ã— 6 policy modes
Ã— 3 random seeds
Ã— 4 worker configurations
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

- **risk_proxy** â€” deterministic diagnostic policy providing action-diverse replay for infrastructure validation.
- **bc** â€” offline behavioural-cloning replay using stored actions.
- **bc_live** â€” live behavioural-cloning policy executed during replay.
- **random** â€” deterministic seed-controlled stochastic replay.
- **always** â€” always intervene.
- **never** â€” never intervene.

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

# Historical Controlled Fault Validation (RQ7)

The following workflows are retained to reproduce the originally archived mode-aware summaries. They must not be used as the authoritative source for label-independent detection or localization claims. Use the v2.5.9 corrected label-independent pipeline above for those claims.

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

## Corrected Label-Independent Fault Validation

- 270 total evidence units: 228 positive and 42 negative controls
- `V_primary`: 84/228 positive units detected
- `V_full`: 228/228 positive units detected
- `V_full`: 0/42 negative units flagged
- 4,906/4,906 supported injected-event identifiers localized independently of injection-marker columns
- Zero localization false positives and zero localization false negatives

## Cross-Region Cloud Validation

- Successful execution in two Google Cloud regions
- Matching SHA-256 replay hashes across regions
- Reproducible execution under identical benchmark configurations

---

# License

This repository is provided for academic research and reproducibility purposes.
