# Policy-First Architecture for Offline Validation of Emotion-Aware Conversational AI

This repository contains the reproducibility pipeline for the paper:

**A Policy-First Architecture for Offline Systems Validation of Emotion-Aware Conversational AI**

The repository reproduces the deterministic offline replay experiments reported in the paper using the MELD dataset.

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

Optional skip of BC training:

```bash
python run_full_reproduction.py --skip_train
```

---

# Pipeline Overview

The reproduction pipeline performs the following steps:

1. Train the behavioral cloning (BC) policy
2. Run learned BC policy replay
3. Run deterministic proxy replay
4. Generate random baseline outputs
5. Generate deployment-time gated outputs
6. Run optional unconstrained end-to-end baseline
7. Run deterministic replay robustness assessment
8. Generate manuscript-ready Tables II–IX
9. Produce summary statistics and execution manifests

---

# Output Structure

## Raw Replay Outputs

Generated in:

```text
paper_outputs/
```

Main files:

* `policy_first_outputs_bc.csv`

  * Learned behavioral cloning replay output
  * Used for intervention-policy learnability analysis

* `policy_first_outputs_proxy.csv`

  * Deterministic proxy replay output
  * Used for architectural validation and robustness analysis

* `random_outputs.csv`

  * Deterministic random baseline output

* `gated_outputs.csv`

  * Deployment-time gated policy output

* `e2e_outputs.csv`

  * Optional unconstrained end-to-end baseline output

---

## Manuscript Tables

Generated in:

```text
paper_tables/
```

Generated manuscript-ready tables:

* `table_ii_rl_diagnostic.csv`
* `table_iii_proxy_execution.csv`
* `table_iv_aggregate_behavior.csv`
* `table_v_representation_ablation.csv`
* `table_vi_emotion_conditioned.csv`
* `table_vii_robustness.csv`
* `table_viii_architectural_comparison.csv`
* `table_ix_summary.csv`

---

## Robustness Outputs

Generated in:

```text
robustness_results/
```

Contains deterministic replay robustness assessment results.

---

# Deterministic Reproducibility

All experiments use:

* fixed random seeds,
* deterministic replay ordering,
* fixed logged conversational trajectories.

Under identical software and dataset conditions, the replay pipeline is expected to produce identical execution behavior across runs.

---

# Dataset

Experiments use the MELD dataset (Multimodal EmotionLines Dataset):

https://affective-meld.github.io/

The dataset and processed representations are not included in this repository due to licensing and storage constraints.

---

# Dataset Preparation

Download MELD from:

https://affective-meld.github.io/

Place the raw dataset in:

```text
data/raw/
```

Prepare the dataset using:

```bash
python reorganize_meld_frames.py
python extract_tav_context_states.py
python precompute_meld_video_embeddings.py
```

These scripts generate the multimodal state representations required for training and deterministic offline replay.

Refer to the individual scripts for configuration details.

---

# Notes

* Full reproduction requires access to the MELD dataset.
* Ensure dataset preprocessing is completed before running the pipeline.
* The repository reproduces execution-level architectural validation experiments under deterministic offline replay.
* The repository does not provide real clinical intervention evaluation or human-subject assessment.

---

# License

Released for research reproducibility purposes.
