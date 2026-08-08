# FGCS Extended Benchmark Summary

## Benchmark design

| item                    | value                             |
|:------------------------|:----------------------------------|
| Observed benchmark rows | 240                               |
| Expected benchmark rows | 240                               |
| Workload settings       | 5                                 |
| Dataset fractions       | 0.1, 0.25, 0.5, 0.75, 1.0         |
| Policy modes            | risk_proxy, random, always, never |
| Seeds                   | 3                                 |
| Worker settings         | 1, 2, 4, 8                        |

## Main findings

| finding                               | observed_value                                                            | paper_interpretation                                                                                                                                    |
|:--------------------------------------|:--------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
| Benchmark coverage                    | 240 runs                                                                  | The benchmark covers 5 workload fractions, 4 policies, and 4 worker settings.                                                                           |
| Policy modes                          | risk_proxy, random, always, never                                         | The table reports the policy modes present in the supplied benchmark evidence; the active v2.6.0 evaluation uses risk_proxy, random, always, and never. |
| Maximum replay workload               | 11,351 decision points                                                    | The benchmark reaches the full MELD-derived replay workload.                                                                                            |
| Full-workload runtime range           | 0.096197–0.646154 seconds                                                 | Runtime varies by policy mode and worker configuration under deterministic replay.                                                                      |
| Maximum full-workload throughput      | 117997.810735 decision points/s                                           | This value is descriptive benchmark throughput; repeated timing measurements are authoritative for worker-performance comparisons.                      |
| Deterministic trace hashes            | {'always': 5, 'never': 5, 'random': 15, 'risk_proxy': 5}                  | Deterministic policies should produce one stable trace per workload; the random baseline is expected to vary across seeds.                              |
| Unauthorized invocations              | 0                                                                         | The policy-first gate prevented unauthorized generator invocation in normal replay.                                                                     |
| Full-workload mean intervention rates | {'always': 1.0, 'never': 0.0, 'random': 0.499662, 'risk_proxy': 0.229848} | Intervention frequency is policy-dependent and execution-identifiable under replay.                                                                     |

## Determinism compact summary

| policy_mode   |   unique_trace_hashes_across_all_workloads |   minimum_hash_match |   max_unauthorized_invocations |   max_fault_injected_count |   max_intervention_rate_delta | interpretation                                    |
|:--------------|-------------------------------------------:|---------------------:|-------------------------------:|---------------------------:|------------------------------:|:--------------------------------------------------|
| always        |                                          5 |                    1 |                              0 |                          0 |                             0 | one stable hash per workload                      |
| never         |                                          5 |                    1 |                              0 |                          0 |                             0 | one stable hash per workload                      |
| random        |                                         15 |                    1 |                              0 |                          0 |                             0 | stochastic policy varies across seeds as expected |
| risk_proxy    |                                          5 |                    1 |                              0 |                          0 |                             0 | one stable hash per workload                      |

## Safe paper wording

The active v2.6.0 benchmark executed offline replay across five workload fractions, four policy modes (risk_proxy, random, always, and never), three seeds, and four worker settings. Deterministic policies produced one stable action-trace hash per workload fraction, whereas the random baseline varied across seeds as expected. No unauthorized invocations were observed during clean replay. Worker-performance claims are based on the separate repeated timing study, which showed concurrency overhead rather than speedup for the evaluated lightweight workload.
