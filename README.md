# rust-kernels

Reusable, independently testable and benchmarkable Rust algorithms and data structures.

The repository starts with spatial/collision kernels and expands into small general-purpose kernels that can be reused across projects. Algorithms stay composable and dependency-light; applications and visualization belong in consumers such as `collision-lab`.

## Current kernels

### Spatial

`spatial-kernels` contains:

- `Aabb` and stable `ColliderId`/`Pair` primitives
- `NaiveBroadPhase`, an intentionally O(n²) correctness oracle
- `UniformGridBroadPhase`, a sparse 3D grid that avoids testing objects which cannot share space
- deterministic sorted pair output and instrumentation for AABB-test counts
- differential tests proving the grid produces the same overlapping pairs as the naive oracle across several cell sizes

`bvh-kernels` adds:

- `StaticBvh`, an immutable median-split bounding-volume hierarchy
- deterministic AABB queries
- `StaticBvhBroadPhase`, differential-tested against `NaiveBroadPhase`

SAT, GJK/EPA, sweep-and-prune, dynamic AABB trees, rays, and CCD remain future kernels rather than being hidden inside the current broad-phase implementations.

### Collections

`collection-kernels` currently contains `UnionFind`, using path compression and union by size with deterministic tie-breaking. Its connectivity results are tested against a trivial traversal oracle.

### Graph search

`graph-kernels` currently contains Dijkstra and A* search. Callers retain their own graph representation and provide neighbor callbacks. A* is tested against Dijkstra as the minimum-cost oracle.

## Registry

`registry.json` is a shadcn-inspired machine-readable catalog for an additional source-level integration path. It does not imply a `rust-kernels` CLI and does not replace normal Cargo dependencies.

Each registry item describes its crate/source files, registry dependencies, tags, and the algorithms or data structures it provides. Agents or external tooling can use that metadata to discover and vendor a kernel while Cargo consumers can continue depending on the workspace crates normally.

## Development

```bash
python3 scripts/validate_registry.py
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```
