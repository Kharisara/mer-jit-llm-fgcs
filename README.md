# Deterministic Replay and Benchmarking Infrastructure for Policy-Gated Multimodal AI Systems

This repository contains the reproducibility pipeline for the paper:

**A Deterministic Replay and Benchmarking Infrastructure for Reproducible Validation of Policy-Gated Multimodal AI Systems**

The repository reproduces the deterministic replay and benchmarking experiments reported in the paper using prepared MELD-derived replay inputs.

---

# Requirements

* Python 3.10+
* Recommended: virtual environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Activate on Linux/Mac:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the FGCS Reproduction Pipeline

Run the full FGCS reproduction pipeline:

```bash
python run_full_reproduction.py
```

This command runs the extended deterministic replay benchmark and then generates the paper-ready tables and figures.

Optional benchmark-only execution:

```bash
python run_fgcs_extended_benchmark.py --config configs/fgcs_extended_benchmark.yaml
```

Optional summarization-only execution:

```bash
python summarize_fgcs_extended_results.py
```

---

# Required Prepared Inputs

The FGCS benchmark expects the following prepared inputs:

```text
configs/fgcs_extended_benchmark.yaml
paper_outputs/replay_input_clean.csv
paper_outputs/policy_first_outputs_bc.csv
```

The file `replay_input_clean.csv` contains the prepared MELD-derived replay decision points.

The file `policy_first_outputs_bc.csv` contains the prepared learned-policy action trace used by the `bc` replay mode.

---

# Important Note on Policy Training

This FGCS reproduction package does **not** retrain the behavioral-cloning policy.

The learned-policy mode in this benchmark uses a prepared behavioral-cloning action trace:

```text
paper_outputs/policy_first_outputs_bc.csv
```

Therefore, the `bc` policy mode should be interpreted as:

```text
replay under a prepared learned-policy action trace
```

It should **not** be interpreted as live BC neural inference latency or as policy training inside the benchmark loop.

---

# Pipeline Overview

The reproduction pipeline performs the following steps:

1. Load prepared deterministic replay inputs
2. Construct amplified replay workloads
3. Execute replay scheduling across worker configurations
4. Run policy-ablation replay modes
5. Apply policy-gated invocation control
6. Run unauthorized invocation fault injection
7. Verify deterministic replay using SHA-256 action hashing
8. Generate manuscript-ready tables and figures
9. Produce reproducibility manifests and benchmark summaries

---

# Output Structure

## Raw Benchmark Outputs

Generated in:

```text
paper_outputs/fgcs_extended_benchmarks/
```

Main files:

* `extended_scalability_results.csv`

  * Runtime and throughput measurements across workload sizes

* `extended_determinism_results.csv`

  * Deterministic replay consistency results

* `trace_verification_overhead.csv`

  * Trace-hashing overhead measurements

* `fault_injection_detection.csv`

  * Unauthorized invocation fault-detection results

* `fgcs_reproducibility_manifest.json`

  * Benchmark environment and execution metadata

---

## Manuscript Tables and Figures

Generated in:

```text
paper_outputs/fgcs_tables_figures/
```

Generated manuscript-ready tables:

* `fgcs_table_extended_scalability.csv`
* `fgcs_table_extended_determinism.csv`
* `fgcs_table_trace_overhead.csv`
* `fgcs_table_fault_injection.csv`
* `fgcs_table_extended_summary_findings.csv`

Generated manuscript-ready figures:

* `fgcs_fig_runtime_vs_workload_size_extended.png`
* `fgcs_fig_throughput_vs_workload_size_extended.png`
* `fgcs_fig_trace_hash_overhead.png`
* `fgcs_fig_fault_detection_recall.png`
* `fgcs_fig_worker_throughput_at_max_workload.png`

---

# Deterministic Reproducibility

All experiments use:

* fixed random seeds,
* deterministic replay ordering,
* fixed prepared replay trajectories,
* deterministic workload amplification,
* replay-order trace reconstruction.

Under identical software and prepared-input conditions, the replay infrastructure is expected to produce identical action traces across runs.

---

# Dataset

Experiments use prepared replay inputs derived from the MELD dataset:

https://affective-meld.github.io/

The raw MELD dataset is not redistributed in this repository due to licensing and storage constraints.

This artifact reproduces the reported FGCS benchmark results from prepared MELD-derived replay inputs, not from the full raw MELD preprocessing pipeline.

---

# Dataset Preparation

This repository does not include the full raw MELD preprocessing pipeline.

To reproduce the benchmark results directly, use the prepared replay input files included in this artifact:

```text
paper_outputs/replay_input_clean.csv
paper_outputs/policy_first_outputs_bc.csv
```

If these files are not present, users must obtain MELD from the original providers and recreate equivalent replay inputs using their own preprocessing pipeline.

---

# Notes

* The repository reproduces the execution-level FGCS benchmark from prepared replay inputs.
* The repository does not redistribute the raw MELD dataset.
* The repository does not include full raw MELD preprocessing or behavioral-cloning policy training.
* The learned-policy trace is treated as a fixed replay input and does not measure live BC model inference latency.
* Amplified replay workloads are generated through deterministic repetition of the prepared replay rows.
* The repository reproduces execution-level replay validation experiments under deterministic offline replay.
* The repository does not provide conversational quality evaluation, clinical evaluation, or live LLM serving evaluation.
* The current implementation evaluates local replay behavior and is not intended as a production distributed serving system.

---

# License

Released for research reproducibility purposes.