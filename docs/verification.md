# Verification contract

Issue #52 makes correctness and performance evidence explicit per public kernel without treating one metric as a substitute for correctness.

The machine-readable matrix is [`verification-matrix.json`](../verification-matrix.json). Each row records the owning crate/module plus the current state of:

- independent oracle or differential testing;
- property or metamorphic testing;
- `cargo llvm-cov` coverage reporting;
- mutation testing (`mutationPolicy`: every kernel is mutated by the advisory baseline, so every row is `partial` until survivors are triaged or a threshold exists);
- deterministic performance/complexity invariants (Iai-Callgrind sentinels and Rust work/allocation gates);
- benchmarks (`benchmark`: any `cargo bench` target, whether Divan, Criterion or Iai-Callgrind; wall-clock time is informational); and
- runtime-profiler scenarios.

Every row fills every column. States are `present`, `partial`, `missing`, `planned`, or `n/a`. `missing` and `partial` record a real gap and are preferred over an unsupported `present`. `planned` is not evidence: it identifies work that has not landed on `main`. `n/a` means the evidence form does not fit the kernel's contract, not that it is unbuilt; `n/a` and `planned` each require a row note `<column>: <reason>` of at least 20 characters.

## CI semantics

The existing blocking Rust checks remain unchanged: formatting, strict Clippy, and the full workspace test suite still decide Rust correctness.

Production code may not call `unwrap()` or `expect()`: the workspace denies `clippy::unwrap_used` and `clippy::expect_used` (`[workspace.lints.clippy]`, inherited by every member through `[lints] workspace = true`), so ordinary strict Clippy enforces the shared fast tier's RUST-002 check (`cargo clippy --workspace --all-features -- -D clippy::unwrap_used -D clippy::expect_used`). Encode invariants in types or control flow, or return the existing error type when the failure is reachable; a documented caller-contract violation may still `assert!`/`panic!` under a `# Panics` doc section. A panic that is genuinely unavoidable uses one narrowly scoped `#[allow(clippy::expect_used, reason = "...")]` that states the invariant. Unit tests are exempt through `clippy.toml` (`allow-unwrap-in-tests`, `allow-expect-in-tests`); integration-test, benchmark and example target roots carry a crate-level `#![allow]` because they are not production code.

`python3 scripts/validate_verification_matrix.py` runs as the blocking `verification-matrix` CI job. It checks matrix structure and inventory only. Every `src/**/*.rs` implementation source in every Cargo workspace member must be owned by at least one kernel row or explicitly listed as a facade. It rejects a row with an empty or absent evidence column, an unknown field, or an `n/a`/`planned` cell without its reason note. Evidence states such as `missing` and `partial` do **not** fail this validation.

Coverage runs in the blocking `coverage-baseline` job. It runs `cargo llvm-cov` across the workspace with all features, then `python3 scripts/coverage_gate.py check target/llvm-cov-baseline/summary.json`, and always uploads `summary.json` (per-file summary data), `summary.txt` (the human-readable table), `lcov.info` (raw LCOV) and `gate.md` (the per-crate comparison) as the `llvm-cov-baseline-<sha>` artifact.

The gate sums line coverage per workspace member and compares it with [`coverage-baseline.json`](../coverage-baseline.json). It fails when a crate falls more than `tolerancePercentagePoints` below its recorded percentage, or when a crate is measured but not recorded (or recorded but no longer measured). Coverage above the baseline never fails; the table only suggests updating it. Coverage is visibility, not proof: raising the percentage is not a goal, and oracle, differential and property tests remain the correctness signal.

The tolerance is 1.0 percentage point. Coverage is deterministic for unchanged code (seeded tests; two `main` runs reproduce each untouched file exactly), so the tolerance does not absorb noise between runs. It absorbs line-mapping drift from the floating `stable` toolchain and lets a change leave a line or two uncovered in the smallest crate (`statistics-kernels`, about 0.5 pp per line), while dropping a meaningful untested path in any crate still fails.

### Updating the coverage baseline

When coverage changes legitimately (code moved between crates, dead code removed, a deliberately untestable path added, a new crate), regenerate the baseline in the same PR and say why in the description:

```sh
cargo llvm-cov --workspace --all-features --json --summary-only --output-path target/llvm-cov-baseline/summary.json
python3 scripts/coverage_gate.py update target/llvm-cov-baseline/summary.json
```

Without a local `cargo-llvm-cov`, download `summary.json` from the PR's `llvm-cov-baseline-<sha>` CI artifact (`gh run download <run-id> -n llvm-cov-baseline-<sha>`) and pass it to `update`; report paths are matched by member directory, so a CI report works on any checkout. `update` keeps the recorded tolerance.

## Updating the matrix

When adding a public kernel:

1. add or update its row in `verification-matrix.json`;
2. point the row at the implementation source file(s);
3. record evidence conservatively (`missing` is preferable to an unsupported `present`);
4. fill every evidence column, documenting `planned` and `n/a` states in `<column>: <reason>` row notes; and
5. run `python3 scripts/validate_verification_matrix.py`.

Multiple public kernels may share one implementation file and therefore the same source path. Facade-only `src/lib.rs` files are tracked separately so adding a new implementation module cannot silently bypass the matrix.

A blocking mutation policy and broader benchmark/runtime-profiler coverage remain later slices of issue #52.
