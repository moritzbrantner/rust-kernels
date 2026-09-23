# Repository agent guidance

This file contains repository-specific guidance for coding agents working in `rust-kernels`. Shared policy belongs in managed conventions; keep this file focused on repository-owned boundaries and validation.

## Repository boundaries

- Keep kernels domain-neutral, deterministic, dependency-light, and independently testable.
- Consumers own product rules, ranking/query semantics, application orchestration, and visualization.
- Reference implementations, benchmarks, registries, and provenance tooling may verify or distribute kernels but must not redefine consumer-owned semantics.

## Authority boundaries

- Owns: `rust-kernels/algorithm-semantics`, `rust-kernels/kernel-public-contracts`, `rust-kernels/source-registry-provenance`
- Adapts: `coding-agent-conventions/policy`
- Non-authoritative: `consumer-domain-semantics`, `consumer-ranking-policy`, `application-visualization`
- Prohibited write-back: benchmark, registry, copied-source, and reporting paths must not silently change consumer-owned semantic decisions

## Validation

- Start with the narrowest affected crate or script check and expand only after it passes.
- Before integration, run the repository-owned registry/source tests plus `cargo fmt --all --check`, workspace Clippy with warnings denied, and the full workspace test suite when the environment supports them.
- Preserve deterministic output and differential/oracle evidence when changing algorithm behavior.
- Keep kernel correctness and performance evidence Cargo-native whenever practical: use Rust `#[test]`/integration tests for correctness, work and allocation ratchets, Divan for developer-facing microbenchmarks, and Iai-Callgrind for deterministic blocking performance sentinels.
- Invoke benchmark suites directly with `cargo bench`; do not make Python parsing of benchmark output the test oracle when a deterministic Rust assertion can express the contract.
- Reserve Python validation for the repository's registry/provenance/source-distribution tooling or cases where the evidence genuinely cannot be expressed through Cargo/Rust.
