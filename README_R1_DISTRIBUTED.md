# ReplayBench-PG — Reviewer 1 isolated-service experiment (pilot)

**Status:** proposed experiment, not yet run on GitHub with the repository's real validator. Do **not** cite pilot outcomes until its GitHub Actions artifact passes verification.

## Scientific scope

This is a **single-host, multi-container** test. On GitHub Actions, two independently running HTTP worker services communicate across Docker's bridge network with an HTTP receipt collector service. An orchestrator running on the Linux host creates a primary trace and invokes the repository's **unchanged** `replaybench.generic_validator.validate_observed_artifacts` and `prepare_reference_artifacts` functions. Receipts originate in worker service processes, not in the host coordinator; they are delivered to a collector in a third process.

It tests **networked service-process boundaries**, but not separate physical hosts or independent clouds. Synthetic 6-point replay input is used (4 authorized, 2 unauthorized/never), so no MELD annotations or other proprietary user data are downloaded or uploaded. The expected clean reference is deterministically generated from the declared execution contract; this is comparison-based, reference-dependent validation, **not independent ground truth for downstream side effects**.

The receipt's semantic digest is generated inside the downstream worker using the same receipt schema, canonical JSON serialization, and SHA-256 rule as `replaybench/integrity.py`. The generic validator itself is imported unchanged from the repository. The collector deduplicates the same receipt transmitted twice using the transport envelope's `delivery_id` but does **not** deduplicate separate execution attempts.

## Predeclared eight scenarios

| Case | Observation | Expected primary validator | Expected full validator |
|---|---|---|---|
| clean | Workers A and B both produce receipts | Pass | Pass |
| reordered | Asynchronous receipts arrive out of issue order; await quiescence | Pass | Pass |
| delayed | 700–1,000 ms network-transport delivery delay; await quiescence | Pass | Pass |
| transport_redelivery | One receipt is transmitted twice with same delivery UUID; collector deduplicates | Pass | Pass |
| lost_receipt | Worker executes but controlled transport adapter **omits** one receipt | Pass | Alarm: missing receipt |
| changed_correlation | One real execution receipt uses a wrong correlation ID and valid digest | Pass | Alarm: correlation mismatch |
| non_idempotent_retry | First successful execution returns an intentionally lost-ack response (HTTP 503); retry re-executes | Pass | Alarm: duplicate downstream operation |
| worker_crash_after_receipt | Worker sends a receipt, then actually terminates without acknowledgement and restarts | Pass | Alarm: unlogged downstream operation |

**Nuances:**
- A delayed receipt is not inherently an integrity fault. Validation occurs **after** a quiescence barrier (4 receipts collected), not at an arbitrary intermediate time.
- Message loss is injected as deliberate non-delivery at the worker's receipt transport adapter, not as actual kernel packet loss or a network partition.
- Retry is deliberately **non-idempotent**; this shows duplicate detection, not automatic exactly-once execution or recovery.
- A worker crash is an actual process exit, followed by Docker's restart policy. The collector survives and retains the receipt in memory.
- This does not evaluate adversarial tampering, reference compromise, TLS/mTLS, independently verified committed effects, nondeterministic model outputs, durable event logs, multi-machine failure, or matched local/cloud overhead.
- No population sensitivity or production false-positive-rate inference follows from 4/4 controls.

## Add these five files to the existing repository

The ZIP paths are already relative to the repository root, and it contains exactly:

```
.github/workflows/r1-distributed-services.yml
scripts/r1_dist_service.py
scripts/r1_dist_runner.py
docker-compose.yml
README_R1_DISTRIBUTED.md
```

Extract the ZIP directly into the root of `Kharisara/mer-jit-llm-fgcs`, **not** into a new nested directory. In PowerShell:

```powershell
cd 'D:\Project final\mer-jit-llm-fgcs'
Expand-Archive -LiteralPath 'PATH_TO_DOWNLOADED_ZIP' -DestinationPath . -Force

git status --short
git add .github/workflows/r1-distributed-services.yml scripts/r1_dist_service.py scripts/r1_dist_runner.py docker-compose.yml README_R1_DISTRIBUTED.md
git diff --cached --stat
git commit -m "Add controlled isolated-service fault pilot for Array R1"
git push origin master
```

**Branch check:** Only perform the final `git push origin master` if `git branch --show-current` prints `master`, and the commit contains only these five files. Otherwise switch/synchronize `master` or ask for instructions before pushing. No history rewriting is needed.

## Launch on GitHub Actions

Once the five files are on `master`, open:

https://github.com/Kharisara/mer-jit-llm-fgcs/actions/workflows/r1-distributed-services.yml

Select **Run workflow** → branch `master` → **Run workflow**. The workflow uses GitHub's `ubuntu-24.04` hosted runner, Python 3.12, three separate Docker service containers and the actual repository generic validator.

After success (or failure), download the ZIP artifact named `r1-isolated-service-pilot-...`. It contains `experiment_summary.json`, `scenario_results.json`, image information, service logs, and commit ID. Upload that ZIP to ChatGPT for numerical and provenance audit. The workflow uploads logs even if validation fails to aid debugging. This does **not** require a Google Cloud billing account. GitHub's public-repository Actions policies and quotas are subject to change.

## Validation protocol

The runner checks 8/8 predeclared scenario contracts:
- 4 negative controls not flagged by primary or full validation;
- 4 positive controls not flagged by primary but flagged by the full validator in the expected receipt channel;
- acceptable receipt counts, including one omitted receipt and two distinct attempts for a non-idempotent retry;
- worker restart confirmed by a changed service boot identifier.

Scientific reporting requires recording the source commit, all container image identifiers, protocol parameters, per-case channel flags, and limitations. Do not count four positive cases as independent samples of real-world fault conditions.

## Further work to satisfy Reviewer 1 fully

This pilot only strengthens the **separate-service** aspect of Reviewer 1's request. It does **not** meet the requirement for independent machines, actual network disruption, distributed fault-tolerance or cross-host failure recovery. For those claims, deploy the same services on separate nodes/cloud instances with trusted receipts, inject real network disruptions, and compare under a prespecified protocol. If that cannot be performed within budget, clearly disclose the remaining limitation in the manuscript/reviewer response rather than overstate this pilot.
