# Phase 1 Evaluation Correction Report

## Scope

This report documents the v2.5.9 correction of the ReplayBench-PG fault-evaluation pipeline.

## Completion status

### 1. Separate generic validator — completed

Implemented `replaybench/generic_validator.py`. The validator consumes only an observed trace, observed receipts, observed configuration manifest, and a trusted clean reference. It emits generic findings without receiving a fault mode, expected anomaly channel, injection seed, or selected-index manifest.

The receipt-reconciliation path was vectorized for tractable reanalysis. A parity test compares the vectorized implementation with the archived reference implementation across clean, missing-receipt, duplicate-receipt, mismatched-correlation, unlogged/unauthorized, and orphan-receipt cases.

### 2. Remove fault labels and injection flags from validator inputs — completed

The validator rejects these columns if they reach its input boundary:

- `action_before_fault`
- `action_flip_fault_injected`
- `unauthorized_invoke_fault_injected`
- `fault_injected`
- `receipt_fault_mode`
- `receipt_fault_injected`

Legacy derived-finding columns are also removed before validation so the corrected validator must recompute its own findings.

`validator_input_separation_audit.csv` records the separation check for all 270 evidence units.

### 3. Store ground truth separately — completed

Generic findings and ground truth are written to different files:

- `generic_validator_findings.jsonl`
- `ground_truth_manifest.jsonl`

The scoring script freezes and loads the generic findings first, then performs a one-to-one post-hoc join by `evidence_id`. The ground-truth manifest contains the fault labels and target identifiers used only for scoring and supported localization evaluation.

### 4. Clean-reference row-level action localization — completed

Action changes are localized by joining the observed and trusted clean traces on stable `replay_point_id` values and comparing their actions. The corrected localization does not inspect `action_flip_fault_injected` or another injection-marker column.

The same generic findings identify missing, unexpected, and duplicate replay identifiers. Receipt-event localization comes from recomputed reconciliation findings rather than the archived injection markers.

### 5. Recompute primary-log invariants on corrupted traces — completed

For every evidence unit, the corrected validator recomputes:

- ordered action hash;
- observed row count;
- primary authorization contradiction, defined by `generation_invoked = 1` and `authorized_to_generate = 0`.

It does not trust archived `unauthorized_invocation` counters or expected-fault fields.

### 6. Fair `V_primary` versus `V_full` comparison — completed

The compared evidence models are:

- `V_primary`: action-hash equality, row-cardinality equality, and recomputed primary-log authorization consistency;
- `V_full`: `V_primary` plus receipt-digest validation, receipt reconciliation, record-bound digest `H_R`, configuration digest `H_C`, and configuration-bound digest `H_RC`.

Both models evaluate the same observed artifacts and trusted references.

### 7. Machine-readable results and manifest — completed

The corrected evidence directory contains:

- `generic_validator_findings.jsonl`
- `ground_truth_manifest.jsonl`
- `validator_input_separation_audit.csv`
- `per_evidence_scored_results.csv`
- `event_localization_results.csv`
- `baseline_comparison_by_fault_class.csv`
- `phase1_validation_manifest.json`

The manifest records counts and SHA-256 values for the retained output files.

### 8. Tests and final artifact validator — completed

The test suite contains explicit checks that:

- ground-truth columns are removed or rejected;
- action localization uses the clean reference;
- the primary authorization invariant is recomputed;
- a configuration-only change is visible only to the full evidence model;
- the vectorized receipt reconciliation preserves the archived reconciliation semantics.

Final verification result:

- Python files compiled by the master validator: **114**
- Pytest result: **36 passed**
- Master artifact validation: **PASS**

### 9. Release alignment — prepared for v2.5.9

The code and retained outputs changed, so this correction is aligned to ReplayBench-PG v2.5.9. Publishing the corresponding GitHub and Zenodo versions remains a user-controlled external step.

## Corrected quantitative results

The corrected corpus contains **270 evidence units**:

- **228 positive-control units**;
- **42 negative-control units**;
- the negative set comprises **18 clean receipt-enabled execution instances** and **24 benign post-execution applications**.

Results:

- `V_primary` detected **84/228** positive-control units;
- `V_full` detected **228/228** positive-control units;
- `V_full` flagged **0/42** negative-control units;
- **4,906/4,906** supported injected-event identifiers were localized;
- localization false positives: **0**;
- localization false negatives: **0**.

These results replace the suggested but unsupported `0/72` and “seven of nine” formulations. In particular, the primary model detects some authorization- and execution-field corruptions after the primary invariant is correctly recomputed:

| Fault/control class | Evidence units | `V_primary` | `V_full` |
|---|---:|---:|---:|
| Saved-action corruption | 18 | 18 | 18 |
| Dropped rows | 18 | 18 | 18 |
| Duplicated rows | 18 | 18 | 18 |
| Logged unauthorized invocation | 12 | 12 | 12 |
| Authorization-field corruption | 18 | 8 | 18 |
| Execution-field corruption | 18 | 10 | 18 |
| Unlogged downstream call | 18 | 0 | 18 |
| False execution log | 18 | 0 | 18 |
| Duplicate downstream call | 18 | 0 | 18 |
| Mismatched correlation identifier | 18 | 0 | 18 |
| Replay-ID/action reassignment | 18 | 0 | 18 |
| Row reordering | 18 | 0 | 18 |
| Configuration-label corruption | 18 | 0 | 18 |

## Evidence-unit boundaries and limitations

The 270 units are not 270 fresh replay executions. They comprise retained receipt-enabled execution instances and deterministic post-execution validator applications to trusted clean references. The corrected 24 benign units reproduce the four prescribed benign transformations as post-execution applications; they are not replacements for, or additional claims about, the earlier 24 freshly executed benign controls.

The comparison is an internal evidence-layer ablation, not an empirical benchmark against MLflow, OpenTelemetry, or another independently deployed third-party product. It quantifies the incremental coverage of the study's full evidence model over its own primary action/cardinality/log model.

The archived 3,240-event selectivity output was generated by a ground-truth-aware historical scorer. It is retained for reproducibility but is superseded for label-independent localization claims by the corrected 4,906-event analysis.
