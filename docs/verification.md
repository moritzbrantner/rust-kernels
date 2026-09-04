# Verification contract

Issue #52 makes correctness and performance evidence explicit per public kernel without treating one metric as a substitute for correctness.

The machine-readable matrix is [`verification-matrix.json`](../verification-matrix.json). Each row records the owning crate/module plus the current state of:

- independent oracle or differential testing;
- property or metamorphic testing;
- `cargo llvm-cov` coverage reporting;
- mutation testing;
- deterministic performance/complexity invariants;
- Criterion benchmarks; and
- runtime-profiler scenarios.

States are `present`, `partial`, `missing`, `planned`, or `n/a`. `planned` is not evidence: it identifies work such as open PR #48 that has not landed on `main`. `n/a` requires a row note explaining why that evidence form does not fit the kernel's contract.

## CI semantics

The existing blocking Rust checks remain unchanged: formatting, strict Clippy, and the full workspace test suite still decide Rust correctness. Registry/source-contract checks also remain blocking.

`python3 scripts/validate_verification_matrix.py` is added to the existing registry job. It checks matrix structure and inventory only. Every `src/**/*.rs` implementation source in every Cargo workspace member must be owned by at least one kernel row or explicitly listed as a facade. Evidence states such as `missing` and `partial` do **not** fail this validation.

Coverage runs in a separate advisory job. It uses `cargo llvm-cov` across the workspace with all features and publishes:

- `summary.json`: per-file summary data;
- `summary.txt`: the human-readable baseline table; and
- `lcov.info`: the raw LCOV baseline.

The job has `continue-on-error: true`, and there are deliberately no `--fail-under-*` flags or percentage thresholds in this slice. Existing blocking jobs therefore keep their semantics while coverage/tooling failures and low-covered files remain visible.

## Updating the matrix

When adding a public kernel:

1. add or update its row in `verification-matrix.json`;
2. point the row at the implementation source file(s);
3. record evidence conservatively (`missing` is preferable to an unsupported `present`);
4. document `planned` and `n/a` states in row notes; and
5. run `python3 scripts/validate_verification_matrix.py`.

Multiple public kernels may share one implementation file and therefore the same source path. Facade-only `src/lib.rs` files are tracked separately so adding a new implementation module cannot silently bypass the matrix.

Mutation testing, complexity gates, coverage-regression thresholds, and broader benchmark/runtime-profiler coverage remain later slices of issue #52.
