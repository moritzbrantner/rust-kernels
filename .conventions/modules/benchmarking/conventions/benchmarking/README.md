# Benchmarking conventions

## BENCH-001 — Benchmark named representative scenarios

- Define a named workload, measured unit, optimization direction, sampling method, and environment fingerprint.
- Use deterministic inputs and keep setup outside the measured region unless setup itself is the subject of the benchmark.
- Record throughput when it makes differently sized workloads easier to compare.

## BENCH-002 — Compare candidates against versioned baselines

- Compare equivalent harness runs on equivalent infrastructure.
- Fail only regressions beyond committed relative and absolute-noise thresholds.
- Treat incompatible environment fingerprints as incomparable rather than silently accepting or rejecting the candidate.

## BENCH-003 — Separate blocking regression signals from noisy wall-clock timing

- Do not make raw wall-clock deltas from ordinary shared CI runners a blocking gate.
- Use a deterministic or sufficiently low-noise proxy for blocking regression thresholds when that proxy represents the intended workload.
- Keep the metric identity explicit: instruction counts, allocations, cache events, operation counts, and wall-clock latency are different evidence and must not be presented as interchangeable.
- Use controlled hardware when wall-clock latency itself is a blocking contract.

## BENCH-004 — Benchmark enough workload shapes to expose scaling regressions

- Performance-sensitive algorithms should cover multiple representative input sizes or shapes when one fixed case could hide an asymptotic, cache, allocation, or branch-behavior regression.
- Include common cases and a bounded stress or adversarial case when they exercise materially different behavior.
- Prefer a small stable matrix over an exhaustive benchmark suite that is too expensive to run or review routinely.

## BENCH-005 — Keep benchmark references outside production boundaries

- Reference implementations and comparison libraries should remain development-only or explicitly feature-gated unless production behavior deliberately depends on them.
- Benchmark-only dependencies must not silently become part of public APIs or runtime selection paths.
- Competitive benchmarks are evidence about implementation quality, not a requirement to copy another library's architecture.
