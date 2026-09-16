# Performance contract

`rust-kernels` treats execution cost as an explicit engineering contract for performance-sensitive kernels.

## Machine-readable contract

`.performance/contract.json` is the repository-owned machine-readable summary of representative performance scenarios and their deterministic budgets. It follows coding-tooling performance-contract schema v1. The repository owns scenario dimensions, correctness evidence, metrics, and budgets; coding-tooling owns the shared schema and discovery contract; runtime-profiler owns runtime captures; Moonlight/evaluators own cross-revision verdicts.

The contract intentionally records only evidence that can be interpreted deterministically. Shared-runner wall-clock measurements are not blocking performance evidence.

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

## Collection-kernel microbenchmarks

Run the Fenwick construction comparison with:

```sh
cargo bench -p collection-kernels --bench fenwick_tree
```

The Divan matrix compares ordered `FenwickTree::from_slice` construction with the explicitly regrouped O(n) `FenwickTree::from_slice_linear` path at 256, 4,096, and 65,536 items. Input generation is excluded from the measured region. The ordinary constructor preserves repeated-point-update addition order for non-associative arithmetic; the linear constructor is a separate API because its regrouping can change floating-point results.

A deterministic unit-level work check complements the timing benchmark by counting `AddAssign` operations on a 64-item fixture. It locks the linear constructor to 127 additions and verifies that it performs less addition work than the ordered constructor without using a wall-clock threshold.

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
