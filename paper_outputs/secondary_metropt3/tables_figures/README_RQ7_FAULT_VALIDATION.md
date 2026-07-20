# Compact Fault Validation (RQ7)

This section documents the additional compact fault-validation experiments used for RQ7. These experiments are separate from the main clean 360-condition benchmark and are intended to test whether the replay infrastructure detects deliberately injected execution and trace-integrity faults.

## Clean benchmark prerequisite

Run the main clean benchmark first:

```bash
python run_fgcs_extended_benchmark.py --config configs/fgcs_extended_benchmark.yaml
python summarize_fgcs_extended_results.py
```

The clean benchmark writes its outputs to:

```text
paper_outputs/fgcs_extended_benchmark/
paper_outputs/fgcs_tables_figures/
```

## Runtime fault experiments

Two compact runtime fault experiments are used. Both use the full replay workload only, three policy modes (`risk_proxy`, `random`, `never`), three seeds, and two worker settings (`1`, `4`), giving 18 runs per runtime fault mode.

### 1. Action-flip fault

This fault simulates policy-output corruption by flipping a deterministic 1% subset of actions before invocation gating. It should be detected through SHA-256 action-trace mismatch against the clean reference trace.

```bash
python run_fgcs_extended_benchmark.py --config configs/fgcs_fault_action_flip.yaml
```

Output directory:

```text
paper_outputs/fgcs_fault_action_flip/
```

### 2. Unauthorized-invocation fault

This fault simulates an invocation-boundary violation by forcing generation after `action = 0`. The policy action remains unchanged, so the fault should be detected by the unauthorized-invocation counter.

```bash
python run_fgcs_extended_benchmark.py --config configs/fgcs_fault_unauthorized_invoke.yaml
```

Output directory:

```text
paper_outputs/fgcs_fault_unauthorized_invoke/
```

## Unified RQ7 validation framework

After the clean benchmark and the two runtime fault experiments have been run, execute:

```bash
python fgcs_fault_validation_framework.py --all
```

This script consolidates all RQ7 outputs using a common fault taxonomy:

```text
FaultType.ACTION_FLIP
FaultType.UNAUTHORIZED_INVOKE
FaultType.TRACE_ACTION_CORRUPTION
FaultType.TRACE_ROW_DROP
FaultType.TRACE_ROW_DUPLICATE
```

It also performs post-hoc trace corruption by modifying clean trace artifacts:

1. flipping 1% of saved `action` values;
2. dropping 1% of trace rows;
3. duplicating 1% of trace rows.

These post-hoc faults are evaluated using row-count checks and SHA-256 action-trace hashes.

## Main RQ7 outputs

The unified framework generates the following paper-facing outputs:

```text
paper_outputs/fgcs_tables_figures/fgcs_table_fault_action_flip_detection_summary.csv
paper_outputs/fgcs_tables_figures/fgcs_table_fault_unauthorized_invoke_detection_summary.csv
paper_outputs/fgcs_tables_figures/fgcs_table_fault_trace_corruption_detection_summary.csv
paper_outputs/fgcs_tables_figures/fgcs_table_rq7_fault_detection_combined.csv
paper_outputs/fgcs_tables_figures/fgcs_table_validation_ablation_matrix.csv
paper_outputs/fgcs_tables_figures/fgcs_fig_rq7_fault_detection_rate.png
paper_outputs/fgcs_tables_figures/fgcs_fig_validation_ablation_matrix.png
```

## Paper interpretation

Use cautious wording:

> The infrastructure detects deliberately injected policy-output corruption, invocation-boundary violations, and trace-integrity corruption through SHA-256 trace mismatches, unauthorized-invocation counters, and row-count invariants.

Do not claim that the system prevents all execution faults. These experiments evaluate deterministic fault detection under controlled replay perturbations.

## Optional cloud validation

The compact RQ7 experiments can also be run through the existing Cloud Run Jobs wrapper by using the same two fault configs and separate cloud output prefixes. This is optional. It is useful only if the manuscript needs the stronger claim that local and cloud replay detect identical injected faults. If time or page budget is limited, local RQ7 validation is sufficient for the current manuscript revision.
