# MetroPT-3 Secondary-Workload Validation

## Purpose

This extension evaluates whether ReplayBench-PG's execution-validation mechanisms transfer from the inherited MELD-derived conversational workload to an independent industrial telemetry workload. It does **not** evaluate predictive-maintenance accuracy, fault diagnosis, remaining useful life, or maintenance-policy quality.

## Prepared workload

The preparation script is:

```text
scripts/prepare_metropt3_secondary_workload.py
```

Example command:

```cmd
python scripts\prepare_metropt3_secondary_workload.py --source-csv "data\MetroPT3(CompressorDatase).csv" --sample-size 20000
```

The script creates:

```text
paper_outputs/secondary_metropt3/replay_input_metropt3.csv
paper_outputs/secondary_metropt3/preparation_manifest.json
```

The manifest records the observed source row count, deterministic sampling strategy, source and output SHA-256 hashes, gate definition, class counts, and sample statistics. The raw source CSV is not redistributed.

## Deterministic rule gate

The final transparent execution-control gate is:

```text
dv_electric == 1
```

The generated labels are:

```text
control_signal_active
control_signal_inactive
```

`dv_electric` is used only to produce deterministic action diversity for replay validation. It is not treated as a failure label or maintenance recommendation.

Final 20,000-point sample:

- Active: 3,244
- Inactive: 16,756
- Gate rate: 0.16220

## Clean benchmark

Configuration:

- Fractions: 0.25, 0.50, 1.00
- Policies: `rule_gate`, `random`, `always`, `never`
- Seeds: 1, 2, 3
- Workers: 1, 4
- Conditions: 3 x 4 x 3 x 2 = 72

Run:

```cmd
python run_secondary_metropt3_validation.py --skip-preparation
```

Final clean results:

| Policy | Conditions | Unique hashes | All worker matches | Max unauthorized invocations | Full-workload intervention rate |
|---|---:|---:|:---:|---:|---:|
| always | 18 | 3 | Yes | 0 | 1.00000 |
| never | 18 | 3 | Yes | 0 | 0.00000 |
| random | 18 | 9 | Yes | 0 | 0.50205 |
| rule_gate | 18 | 3 | Yes | 0 | 0.16220 |

## Controlled fault validation

The same five fault classes used in the primary study are evaluated on 18 full-workload configurations per class:

| Fault class | Injected events | Flagged runs | Run-level detection | False-positive runs |
|---|---:|---:|---:|---:|
| Action flip (1%) | 3,522 | 18/18 | 100% | 0 |
| Unauthorized invocation (1%) | 2,762 | 18/18 | 100% | 0 |
| Saved-action corruption (1%) | 3,600 | 18/18 | 100% | 0 |
| Dropped trace rows (1%) | 3,600 | 18/18 | 100% | 0 |
| Duplicated trace rows (1%) | 3,600 | 18/18 | 100% | 0 |

These are run-level detection results for controlled perturbations. They do not establish event localization, root-cause diagnosis, adversarial security, or live industrial reliability.

## Final supported claim

ReplayBench-PG preserved deterministic execution and detected controlled policy-, invocation-, and trace-level faults on an independent industrial telemetry workload, providing cross-domain evidence that the validation infrastructure is not specific to the original conversational replay setting.
