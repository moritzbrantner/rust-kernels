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

## Registry

`registry.json` is a machine-readable catalog for source-level integration, inspired by shadcn registries. It is metadata, not a `rust-kernels` CLI.

Normal Cargo dependencies remain the default integration path when a consumer wants shared library semantics. The registry provides an additional path for agents or external tooling that want to discover a kernel, copy/vendor its source, and then own that code locally. Registry items describe their source files, suggested target paths, registry dependencies, Cargo dependencies, and the algorithms/data structures they provide.

The initial registry exposes `spatial-kernels` as one source bundle because its MVP implementation currently lives in a single Rust module. Individual algorithms such as `aabb`, `naive-broad-phase`, and `uniform-grid-broad-phase` are still discoverable through the item's `provides` metadata. They can become independently vendorable registry items once the source layout supports that without duplicating code or inventing artificial package boundaries.

`registry.schema.json` defines the registry contract. `scripts/validate_registry.py` keeps the catalog deterministic by checking its syntax, item uniqueness, dependency references, and referenced source files.

## Development

```bash
python3 scripts/validate_registry.py
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```
