# rust-kernels

Reusable, independently testable and benchmarkable Rust algorithms and data structures.

The repository starts with spatial/collision kernels and is intended to grow through evidence-driven reuse across projects. Algorithms stay small and composable; applications and visualization belong in consumers such as `collision-lab`.

## MVP

The first crate is `spatial-kernels` and contains:

- `Aabb` and stable `ColliderId`/`Pair` primitives
- `NaiveBroadPhase`, an intentionally O(n²) correctness oracle
- `UniformGridBroadPhase`, a sparse 3D grid that avoids testing objects which cannot share space
- deterministic sorted pair output and instrumentation for AABB-test counts
- differential tests proving the grid produces the same overlapping pairs as the naive oracle across several cell sizes

The broad phase currently treats AABB overlap as the candidate boundary. SAT, GJK/EPA, sweep-and-prune, BVHs, dynamic AABB trees, rays, and CCD are intentionally future kernels rather than hidden inside the MVP.

## Development

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```
