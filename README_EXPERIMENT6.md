# ReplayBench-PG / Array R1 — Experiment 6: OpenTelemetry comparator

**Purpose:** Reviewer 1's most important unresolved objection: whether a deliberately instrumented existing tracing implementation can reproduce the proposed policy-gated evidence checks. The comparison MUST NOT present OpenTelemetry as inherently incapable of such checks.

## Status / test boundary

- **Locally tested:** real OpenTelemetry Python SDK + `BatchSpanProcessor` + OTLP/HTTP protobuf exporter + local HTTP protobuf receiver; independent policy-aware audit. 342 synthetic cases (18 base configurations × 19 conditions), 2 unit tests, and 240 warm-up-excluded timing repetitions (12 selected configuration/scenario pairs × 20). OTel-only audit identified **234/234 deliberately observable positive applications**, flagged **0/72** designed negative applications, and flagged **0/36** observational-boundary cases.
- **NOT YET TESTED:** the head-to-head numerical result against *your actual `replaybench.generic_validator`*. This runs only after GitHub Actions executes the workflow on your repository with `--otel-only` **not** set. The local tool's output has `p_alarms: null`; never cite it as a verified ReplayBench comparison.
- **Synthetic only:** each case has 12 synthetic records. No inference service, MELD, MetroPT-3, independent host, distributed ledger, real adversarial forgery, or matched cloud/local timing.
- **Not a new scientific discovery:** observed-positive perfect detection is constructed because the policy-aware auditor is given all required inputs and a trusted reference. That is precisely the fair comparator Reviewer 1 requested.

## Packages to add to your repo

Extract this ZIP at the **root** of `D:\Project final\mer-jit-llm-fgcs`; preserve nested folders:

```
.github/workflows/r1-otel-comparator.yml
scripts/array_r1_otel_comparator.py
tests/test_array_r1_otel_comparator.py
requirements-otel-r1.txt
README_EXPERIMENT6.md
```

You are not replacing any existing file. This is an additive branch- or commit-scoped experiment. It does not modify `replaybench/`.

### PowerShell: verify, stage and push only these five files

```powershell
cd 'D:\Project final\mer-jit-llm-fgcs'
git branch --show-current
git status --short

@(
 '.github/workflows/r1-otel-comparator.yml',
 'scripts/array_r1_otel_comparator.py',
 'tests/test_array_r1_otel_comparator.py',
 'requirements-otel-r1.txt',
 'README_EXPERIMENT6.md'
) | ForEach-Object { "$_ : $(Test-Path $_)" }

git add .github/workflows/r1-otel-comparator.yml scripts/array_r1_otel_comparator.py tests/test_array_r1_otel_comparator.py requirements-otel-r1.txt README_EXPERIMENT6.md
git diff --cached --name-only
# Expect only the five specified files.
git commit -m "Add independent OpenTelemetry audit comparator for Array R1"
git push origin master
```

Do **not** use `git add .` because this repo has other untracked research outputs. If `git diff --cached --name-only` contains unrelated files, unstage them before committing.

### Launch

1. Navigate to `https://github.com/Kharisara/mer-jit-llm-fgcs/actions/workflows/r1-otel-comparator.yml`.
2. Click **Run workflow**, select `master`, and run a **new** workflow.
3. Upload the `r1-otel-comparator-<run_id>.zip` artifact here, **even if the workflow fails**. I will audit every result, compare P and B1, and only then write the manuscript interpretation.

## Experimental design: exact comparison

- **B0:** ordinary OTel span status/error indication with no policy auditor. This is a descriptive operational tracing reference, not a strong competitor.
- **B1:** independent bespoke policy-aware auditor over actual SDK-exported OTel spans. It has all point, execution, correlation, record, and configuration fields and the **same clean reference** as ReplayBench. It independently checks order, row cardinality, semantic fields, authorization, receipt multiplicity, correlation, stable identity, semantic digest and configuration equality. It does **not** consume ReplayBench validator findings.
- **P:** unmodified real `replaybench.generic_validator.validate_observed_artifacts` from the current repository. It receives `pandas` artifacts derived from the *same producer* and the same clean reference and manifest.

**Inputs**: three policies (`risk_proxy`, `random`, `always`) × seeds 1–3 × workers 1 and 4 = 18 dependent base configurations, each with 12 synthetic records. Each base has 4 benign scenarios, 13 designed positive applications, and 2 observational-boundary cases = 342 scored cases. **The 13 applications are not 13 independently identifiable failure mechanisms.** E.g., `lost_receipt` and `false_execution_log` generate the same evidence signature with the current synthetic reference; this cannot identify which causal event occurred.

**OTel details:** `ALWAYS_ON` sampling, explicit case spans, per-record primary spans and downstream-attempt spans, full application evidence attributes. The exporter sends real serialized OTLP trace protobuf via HTTP to a local collector. It uses `BatchSpanProcessor`; `force_flush()` is checked and every decoded point and receipt is compared byte/field-equivalently with the producer's normalized record data. One duplicate telemetry span is deliberately deduplicated by attempt identity/digest; two *distinct* attempts use different attempt indices. A controlled out-of-order case reverses span arrival order while preserving source positions. Timestamp drift is an excluded auxiliary attribute.

**Label exclusion:** The emitter sees the injection mode; both auditors receive only generic opaque unit IDs, observed artifacts, and the trusted reference. Ground-truth labels are written separately to `ground_truth_post_scoring.json`. The protocol is reference-based validation, not unsupervised discovery.

**Observability-limit cases:** `unobserved_external_effect` records no extra observable evidence; `compromised_reference` changes both trusted reference and observed source identity consistently. Neither case has an independent truth anchor. Do not calculate ordinary detector sensitivity from them; report that they are observationally non-identifiable under the test contract.

**Predeclared overhead:** timing only for two conditions (`clean` and `lost_receipt`) across 3 policies × 2 worker counts × seed 1; 1 warmup + 20 measured calls per condition → 240 measured paired repetitions. Timing columns separate OTel SDK emission/export, OTel decode-and-audit, and P dataframe adaptation-plus-validation. **These are different stage boundaries, so they are descriptive and not valid as a total-cost speed comparison.** RSS is whole-process not per-algorithm peak memory. OTLP bytes are wire bytes not equivalent to P retained file bytes. Do not claim a CPU/memory advantage from these columns.

**Reproducibility outputs:** `per_case.csv`, `timing_raw.csv`, `experiment_summary.json`, `ground_truth_post_scoring.json`, `observed_evidence.jsonl`, `exported_otel_spans.jsonl`, `file_hashes.json`, `commit_sha.txt`, `pip_freeze.txt`, and Python environment details. Only synthetic records are exported.

**Interpretation before observing results:**

- **B1=P:** no demonstrated new detection capability versus sufficiently instrumented OpenTelemetry plus custom checks; the contribution is the specified contract, reproducible benchmark and audit process.
- **B1<P:** inspect data and check fairness before inferring any superiority; a misconfigured B1 is not a valid baseline.
- **B1>P:** report missing checks and reduce claims or fix implementation independently.
- **Both clean on hidden-only:** documents the observability/trust limit, not robustness to adversarial falsification.

## Validation commands (optional)

The workflow runs automatically. Locally, in a Python 3.12 environment with pinned dependencies:

```
python -m pip install -r requirements-otel-r1.txt
python -m unittest discover -s tests -p 'test_array_r1_otel_*.py' -v
python -m scripts.array_r1_otel_comparator --otel-only --output local_otel_smoke --timing-reps 0
```

`--otel-only` DOES NOT test ReplayBench-PG. For the full test, omit that flag and run from the repo root.

## Trust and quality limitations

1. An OTel manual-instrumentation adapter is necessary; OTel does not natively implement the study's application-specific invariants. This is fair because an existing tracing system can be extended.
2. B1 compares reference fields directly; P uses layer-specific hashes and receipts. The equivalent reference information is available to both, but the checking algorithms differ.
3. Transport re-emission and timestamp drift are controlled benign changes, not an independent network chaos experiment.
4. Both systems receive reference-generated synthetic inputs. Results cannot establish handling of adversarial coordinated evidence forgery, compromised trust roots, arbitrary unknown faults, or real committed effects.
5. The above method does not satisfy Reviewer 1's independent multi-machine, cross-host network-failure, or matched cloud/local overhead requests; those require separate evidence or explicit limitation statements.
6. Reviewer 2 and Reviewer 4 experimental counts remain separate; DO NOT add these 342 cases to the existing 270 evidence-unit corpus.
