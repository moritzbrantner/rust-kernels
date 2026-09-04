# Advisory mutation testing

`rust-kernels` uses mutation testing as a deeper test-quality signal after correctness, structural verification, coverage observation, and deterministic Callgrind smoke.

The first mutation-testing slice is intentionally **advisory**. It does not define a mutation score, surviving-mutant threshold, or merge requirement.

## Pull requests

Pull requests run a mutation catalog smoke with exact `cargo-mutants 27.1.0`. This verifies that the workspace can be discovered and mutation candidates can be enumerated under the checked-in `.cargo/mutants.toml` configuration without paying the cost of executing the full mutation set on every change.

The catalog is uploaded as immutable workflow evidence for the exact pull-request head.

## Scheduled and manual baseline

The `mutation-baseline` workflow runs the full workspace mutation suite weekly and on manual dispatch. Its `mutants.out` evidence is uploaded even when cargo-mutants reports surviving, timed-out, or otherwise unsuccessful mutants.

Those outcomes describe where tests may be weak. They are inputs for targeted follow-up tests, not a newly invented policy threshold. Existing strict Clippy/workspace tests, source-contract checks, and deterministic Callgrind smoke remain the blocking correctness/performance gates.

## Graduation path

A later PR may propose a blocking mutation-regression policy only after the repository has accumulated enough stable baseline evidence to distinguish meaningful regressions from noisy or structurally unviable mutants. Any such policy change must be explicit and independently reviewed; this baseline does not imply one.
