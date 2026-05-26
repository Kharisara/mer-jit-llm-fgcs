# Deterministic Replay and Benchmarking Infrastructure for Policy-Gated Multimodal AI Systems

This repository contains the reproducibility pipeline for the paper:

**A Deterministic Replay and Benchmarking Infrastructure for Reproducible Validation of Policy-Gated Multimodal AI Systems**

The repository reproduces the deterministic replay and benchmarking experiments reported in the paper using a MELD-derived replay workload.

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

# Running the Full Reproduction Pipeline

Run the full reproduction pipeline:

```bash
python run_full_reproduction.py
```

Optional quick test:

```bash
python run_full_reproduction.py --max_rows 500
```

Optional benchmark-only execution:

```bash
python run_fgcs_extended_benchmark.py --config configs/fgcs_extended_benchmark.yaml
```

Optional summarization-only execution:

```bash
python summarize_fgcs_extended_results.py
```

---

# Pipeline Overview

The reproduction pipeline performs the following steps:

1. Load deterministic replay inputs
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
* fixed replay trajectories,
* deterministic workload amplification,
* replay-order trace reconstruction.

Under identical software and dataset conditions, the replay infrastructure is expected to produce identical action traces across runs.

---

# Dataset

Experiments use a replay workload derived from the MELD dataset (Multimodal EmotionLines Dataset):

https://affective-meld.github.io/

The MELD dataset itself is not redistributed in this repository due to licensing and storage constraints.

---

# Dataset Preparation

Download MELD from:

https://affective-meld.github.io/

Place the raw dataset in:

```text
data/raw/
```

Prepare the replay inputs using the preprocessing scripts:

```bash
python reorganize_meld_frames.py
python extract_tav_context_states.py
python precompute_meld_video_embeddings.py
```

These scripts generate the multimodal replay representations required for deterministic replay benchmarking.

Refer to the individual scripts for configuration details.

---

# Notes

* Full reproduction requires access to the MELD dataset.
* Ensure dataset preprocessing is completed before running the pipeline.
* Amplified replay workloads are generated through deterministic repetition of the original replay rows.
* The repository reproduces execution-level replay validation experiments under deterministic offline replay.
* The repository does not provide conversational quality evaluation, clinical evaluation, or live LLM serving evaluation.
* The current implementation evaluates local replay behavior and is not intended as a production distributed serving system.

---

# License

Released for research reproducibility purposes.