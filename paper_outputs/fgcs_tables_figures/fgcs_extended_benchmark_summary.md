# FGCS Extended Benchmark Summary

## Benchmark design

| item                    | value                                          |
|:------------------------|:-----------------------------------------------|
| Observed benchmark rows | 360                                            |
| Expected benchmark rows | 360                                            |
| Workload settings       | 5                                              |
| Dataset fractions       | 0.1, 0.25, 0.5, 0.75, 1.0                      |
| Policy modes            | never, risk_proxy, bc, bc_live, random, always |
| Seeds                   | 3                                              |
| Worker settings         | 1, 2, 4, 8                                     |

## Main findings

| finding                               | observed_value                                                                                       | paper_interpretation                                                                                                       |
|:--------------------------------------|:-----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|
| Benchmark coverage                    | 360 runs                                                                                             | The benchmark covers 5 workload fractions, 6 policies, and 4 worker settings.                                              |
| Policy modes                          | never, risk_proxy, bc, bc_live, random, always                                                       | The evaluation includes fixed, stochastic, proxy, offline BC, and live BC policy modes.                                    |
| Maximum replay workload               | 11,351 decision points                                                                               | The benchmark reaches the full MELD-derived replay workload.                                                               |
| Full-workload runtime range           | 5.553629–30.310232 seconds                                                                           | Runtime varies by policy mode and worker configuration under deterministic replay.                                         |
| Maximum full-workload throughput      | 2043.888669 decision points/s                                                                        | Replay throughput is directly measurable across workload and worker settings.                                              |
| Maximum observed speedup              | 24.725832x                                                                                           | Parallel workers provide measurable acceleration, though gains depend on policy and workload.                              |
| Deterministic trace hashes            | {'always': 5, 'bc': 5, 'bc_live': 5, 'never': 5, 'random': 15, 'risk_proxy': 5}                      | Deterministic policies should produce one stable trace per workload; the random baseline is expected to vary across seeds. |
| Unauthorized invocations              | 0                                                                                                    | The policy-first gate prevented unauthorized generator invocation in normal replay.                                        |
| Live BC action rate                   | 1.000000                                                                                             | The live BC policy executed end-to-end; action diversity should be reported conservatively.                                |
| Full-workload mean intervention rates | {'always': 1.0, 'bc': 1.0, 'bc_live': 1.0, 'never': 0.0, 'random': 0.499604, 'risk_proxy': 0.229848} | Intervention frequency is policy-dependent and execution-identifiable under replay.                                        |

## Determinism compact summary

| policy_mode   |   unique_trace_hashes_across_all_workloads |   minimum_hash_match |   max_unauthorized_invocations |   max_fault_injected_count |   max_intervention_rate_delta | interpretation                                    |
|:--------------|-------------------------------------------:|---------------------:|-------------------------------:|---------------------------:|------------------------------:|:--------------------------------------------------|
| always        |                                          5 |                    1 |                              0 |                          0 |                             0 | one stable hash per workload                      |
| bc            |                                          5 |                    1 |                              0 |                          0 |                             0 | one stable hash per workload                      |
| bc_live       |                                          5 |                    1 |                              0 |                          0 |                             0 | one stable hash per workload                      |
| never         |                                          5 |                    1 |                              0 |                          0 |                             0 | one stable hash per workload                      |
| random        |                                         15 |                    1 |                              0 |                          0 |                             0 | stochastic policy varies across seeds as expected |
| risk_proxy    |                                          5 |                    1 |                              0 |                          0 |                             0 | one stable hash per workload                      |

## Live BC compact summary

| metric                    |         value |
|:--------------------------|--------------:|
| Live BC prediction rows   | 354132        |
| Live BC intervention rate |      1        |
| Live BC action=0 count    |      0        |
| Live BC action=1 count    | 354132        |
| state_loading_ms mean     |      0.471163 |
| state_loading_ms max      |     44.8126   |
| policy_inference_ms mean  |      0.005216 |
| policy_inference_ms max   |      0.343525 |

## Safe paper wording

The extended benchmark executed deterministic offline replay across multiple workload fractions, policy modes, random seeds, and worker settings. Trace-level hashes provide an audit mechanism for reproducibility. Deterministic policies produced stable trace behavior, whereas the random baseline varied across seeds as expected. The live BC policy was evaluated as an execution-level policy mode; its observed action distribution should be interpreted as checkpoint-dependent behavior rather than evidence of policy optimality.
