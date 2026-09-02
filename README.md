# rust-kernels

Reusable, independently testable and benchmarkable Rust algorithms and data structures.

The repository starts with spatial/collision kernels and expands into small general-purpose kernels that can be reused across projects. Algorithms stay composable and dependency-light; applications and visualization belong in consumers such as `collision-lab`.

## Current kernels

### Spatial

`spatial-kernels` contains:

- `Aabb` and stable `ColliderId`/`Pair` primitives
- `SpatialHash3D` and `CellCoord3` for deterministic world-to-cell mapping and stable compact cell hashes
- 2D and 3D Morton/Z-order encode/decode kernels for compact integer spatial keys
- `NaiveBroadPhase`, an intentionally O(n²) correctness oracle
- `UniformGridBroadPhase`, a sparse 3D grid built on the same public spatial-hash cell mapping
- deterministic sorted pair output and instrumentation for AABB-test counts
- differential tests proving the grid produces the same overlapping pairs as the naive oracle across several cell sizes

`bvh-kernels` adds:

- `StaticBvh`, an immutable median-split bounding-volume hierarchy
- deterministic AABB queries
- `StaticBvhBroadPhase`, differential-tested against `NaiveBroadPhase`

SAT, GJK/EPA, sweep-and-prune, dynamic AABB trees, rays, and CCD remain future kernels rather than being hidden inside the current broad-phase implementations.

### Collections

`collection-kernels` contains:

- `UnionFind`, using path compression and union by size with deterministic tie-breaking and connectivity checks against a traversal oracle
- `RingBuffer`, a fixed-capacity FIFO buffer that does not reallocate after construction
- `SparseSet`, the sparse/dense integer-key set primitive commonly used for fast membership and ECS-style storage
- `BitSet`, a packed fixed-universe set with efficient set-bit iteration
- `GenerationalArena`, O(1) slot storage with stale-handle rejection and deterministic live-entry iteration

The collection kernels deliberately stop before lock-free queues or a full ECS framework.

### Graph algorithms

`graph-kernels` keeps graph storage caller-owned through neighbor callbacks or explicit edge lists and currently contains:

- Dijkstra and A* minimum-cost path search, with A* checked against Dijkstra
- breadth-first and depth-first traversal with deterministic neighbor-order semantics
- topological sort with cycle detection
- Tarjan strongly connected components with deterministic normalized output
- Kruskal minimum spanning forests, reusing `UnionFind` for connectivity

### Search and selection

`search-kernels` contains:

- `quickselect`, an in-place deterministic three-way selection algorithm
- `top_k_smallest`, which combines selection with sorting only the requested result set
- `BloomFilter`, deterministic probabilistic membership over byte-oriented keys using the shared `BitSet` primitive

Selection results are checked against fully sorted reference outputs. Bloom-filter tests enforce the no-false-negative contract while keeping false-positive behavior explicit.

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
