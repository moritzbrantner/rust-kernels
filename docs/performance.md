# Performance contract

`rust-kernels` treats execution cost as an explicit engineering contract for performance-sensitive kernels.

## Evidence layers

- Correctness remains owned by tests and property/reference-model checks.
- New developer-facing Rust microbenchmarks use Divan. Existing Criterion suites remain valid and are not migrated solely for uniformity.
- A deliberately small Iai-Callgrind suite provides deterministic PR regression evidence through the `benchmark:smoke` capability.
- Runtime-profiler scenarios remain broader runtime evidence and are not replaced by the microbenchmark layer.
- Raw wall-clock timing from ordinary shared CI runners is informational only.

## Search-kernel microbenchmarks

Run the representative scaling matrix with:

```sh
cargo bench -p search-kernels --bench search
```

The suite measures radix sort, middle quickselect, and top-k selection at 256, 4,096, and 65,536 items. Input generation is deterministic and excluded from the measured region.

## Deterministic smoke gate

Run the bounded Callgrind sentinel with:

```sh
bash scripts/benchmark-smoke.sh
```

For a pull-request comparison, provide the base commit:

```sh
PERF_BASE_SHA=<base-commit> bash scripts/benchmark-smoke.sh
```

When the base commit already contains the smoke benchmark, the script records that revision as an Iai-Callgrind baseline and compares the candidate against it. A regression greater than 5% in Callgrind instruction reads fails the benchmark. When no compatible base benchmark exists, the run seeds the contract instead of inventing a historical comparison.

The script writes the candidate/base logs and an environment fingerprint to `.artifacts/performance-smoke/`. The fingerprint records the revisions, Rust/Cargo versions, `RUSTFLAGS`, Valgrind/Iai-Callgrind runner versions, and host architecture so incompatible evidence is not treated as directly comparable.

The GitHub Actions smoke job uses `ubuntu-24.04`, Iai-Callgrind 0.16.1, and the distribution Valgrind package. Compiler/toolchain changes remain visible in the fingerprint and should be treated as an environment change when interpreting a baseline shift.
