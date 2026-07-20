# MetroPT-3 Secondary-Workload Validation

## Purpose

This implementation adds a compact, independent workload path for testing
whether ReplayBench-PG's execution invariants are tied to the MELD-derived
conversational workload. It does not modify or replace the original 360-condition
MELD experiment.

The selected secondary source is **MetroPT-3**, a real industrial multivariate
time-series dataset collected from a metro-train air-production unit. The UCI
record identifies 1,516,948 records, 15 sensor signals, DOI
`10.24432/C5VW3R`, and a CC BY 4.0 license.

## New implementation components

- `rule_gate` policy mode in `run_fgcs_extended_benchmark.py`
  - categorical equality rules;
  - numeric threshold rules;
  - `any` or `all` combination semantics.
- Dataset-independent deterministic downstream service stub.
- Generic trace identifiers (`source_record_id` and `timestamp`).
- `scripts/prepare_metropt3_secondary_workload.py`
  - validates the source row count;
  - uniformly samples the complete chronological file;
  - normalizes sensor-column names;
  - writes a replay CSV and SHA-256 preparation manifest.
- Three secondary configurations:
  - 72-condition clean benchmark;
  - 18-condition action-flip study;
  - 18-condition unauthorized-invocation study.
- Existing post-hoc saved-action, dropped-row, and duplicated-row validation is
  reused through the unified fault-validation framework.
- `summarize_secondary_workload_results.py` performs fail-fast checks over the
  clean and fault outputs.
- `run_secondary_metropt3_validation.py` runs the complete workflow.

## Diagnostic gate

The initial transparent maintenance gate authorizes the deterministic downstream
maintenance-diagnostic stub when either of these real sensor signals is active:

```text
LPS == 1 OR Oil_level == 1
```

This gate is used only to provide deterministic, domain-relevant action diversity
for execution validation. It is not claimed to be an optimal predictive-
maintenance policy. The preparation manifest reports the resulting action rate.
The rule must be inspected after preparing the real workload; it should be
changed only if the real data produce no meaningful action diversity.

## Obtain the source dataset

Download `MetroPT3(AirCompressor).csv` from the official UCI MetroPT-3 record:

```text
https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset
```

The raw 208 MB source file is intentionally not bundled in this development
package.

## Run the complete implementation

From the repository root:

```bash
python run_secondary_metropt3_validation.py \
  --source-csv "/path/to/MetroPT3(AirCompressor).csv" \
  --sample-size 20000
```

The workflow executes:

```text
3 fractions x 4 policies x 3 seeds x 2 workers = 72 clean conditions
1 fraction x 3 policies x 3 seeds x 2 workers = 18 runs per runtime fault class
```

Policies:

```text
rule_gate, random, always, never
```

The compact fault subset uses:

```text
rule_gate, random, never
```

## Important pre-paper checks

Before using the results in the manuscript, verify all of the following:

1. The preparation manifest reports exactly 1,516,948 source rows unless the
   official source has changed and that change has been independently verified.
2. The `rule_gate` intervention rate is neither zero nor one and provides enough
   non-intervention rows for unauthorized-invocation injection.
3. All 72 clean conditions are present.
4. Deterministic policies produce one action hash per fraction.
5. The random policy produces one action hash per fraction and seed.
6. Four-worker traces match their one-worker references.
7. Clean authorization-execution contradictions are zero.
8. Every injected fault class is detected in all 18 affected runs.
9. Raw data provenance, source SHA-256, output SHA-256, and configuration files
   are retained in the artifact.

Do not edit the manuscript before the real dataset has been prepared, the gate
rate has been inspected, and the complete output checks have passed.
