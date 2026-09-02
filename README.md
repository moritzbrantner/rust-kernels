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
- `FenwickTree`, compact O(log n) additive point updates with half-open prefix/range-sum queries
- `LruCache`, a fixed-capacity O(1)-expected least-recently-used cache with deterministic recency order
- `AddressablePriorityQueue`, a stable deterministic indexed min-heap with opaque stale-handle rejection and O(log n) priority updates/removals

The collection kernels deliberately stop before lock-free queues or a full ECS framework.

### Graph algorithms

`graph-kernels` keeps graph storage caller-owned through neighbor callbacks or explicit edge lists and currently contains:

- Dijkstra and A* minimum-cost path search, with A* checked against Dijkstra
- breadth-first and depth-first traversal with deterministic neighbor-order semantics
- topological sort with cycle detection
- Tarjan strongly connected components with deterministic normalized output
- Kruskal minimum spanning forests, reusing `UnionFind` for connectivity

### Search, selection, and sorting

`search-kernels` contains:

- `quickselect`, an in-place deterministic three-way selection algorithm
- `top_k_smallest`, which combines selection with sorting only the requested result set
- `BloomFilter`, deterministic probabilistic membership over byte-oriented keys using the shared `BitSet` primitive
- stable LSD radix-sort kernels for `u32` and `u64`, checked against Rust's stable sort

Selection and sorting results are checked against simple standard-library oracles. Bloom-filter tests enforce the no-false-negative contract while keeping false-positive behavior explicit.

### Statistics

`statistics-kernels` contains `RunningStats`, a Welford-style streaming accumulator for count, mean, population/sample variance, and standard deviation. Accumulators can be merged without replaying observations, with tests against batch calculations and large-offset fixtures.

## Registry

`registry.json` is a shadcn-inspired machine-readable catalog for an additional source-level integration path. It does not imply a `rust-kernels` CLI and does not replace normal Cargo dependencies.

The registry has two first-class source lanes:

- **crate items** describe coherent crate/source sets; and
- **standalone-module items** describe one independently compilable algorithm or data-structure source file that lands at `src/kernels/<module>.rs`.

Standalone status is verified, not assumed: CI invokes `rustc` directly on every registered standalone source file. Modules with hidden crate or external dependencies therefore stay crate-level until a dependency-aware source-set contract exists.

For example, consumers can choose the integration boundary they actually want:

```bash
# Copy the coherent search crate source set.
python3 scripts/source_registry.py install search-kernels --root ../consumer

# Or own only one algorithm module.
python3 scripts/source_registry.py install radix-sort --root ../consumer
python3 scripts/source_registry.py install selection --root ../consumer
```

The granular install lands source under `src/kernels/`; the consumer remains responsible for wiring that module into its own `lib.rs`, `main.rs`, or public API. See [`docs/granular-registry.md`](docs/granular-registry.md) for the standalone-module contract and the initial granular catalog.

Each registry item also describes tags and the algorithms or data structures it provides. Agents or external tooling can use that metadata to discover and copy a kernel while Cargo consumers can continue depending on the workspace crates normally.

Copied source has an explicit provenance contract. A consumer records the exact upstream Git revision and SHA-256 hashes in `.rust-kernels.lock.json`, using [`provenance.schema.json`](provenance.schema.json). Local modification after copying is supported: the lock preserves the upstream base so tooling can distinguish local changes from upstream changes and reconstruct a real three-way update.

The optional standard-library reference helpers exercise the contract without turning this repository into a CLI product:

```bash
# Copy source and create provenance.
python3 scripts/source_registry.py install radix-sort --root ../consumer

# See whether copied files have diverged locally.
python3 scripts/source_registry.py status --root ../consumer

# Compare recorded base, local source, and this checkout as the candidate upstream.
python3 scripts/source_update.py plan --root ../consumer

# Apply only a plan that is entirely safe; conflicts are never overwritten.
python3 scripts/source_update.py apply --root ../consumer

# Materialize exact base / ours / theirs evidence for real conflicts.
python3 scripts/source_resolution.py export \
  --root ../consumer \
  --out ../consumer/.rust-kernels-merge

# After resolving and testing a conflict, explicitly advance its provenance.
python3 scripts/source_resolution.py accept radix-sort \
  --root ../consumer \
  --bundle ../consumer/.rust-kernels-merge/merge-bundle.json
```

See [`docs/provenance.md`](docs/provenance.md) for the lock and update contract and [`docs/source-resolution.md`](docs/source-resolution.md) for the human/coding-agent conflict handoff. The JSON registry and schemas remain the actual integration surface; agents and other tooling can implement the same protocol directly.

## Development

```bash
python3 scripts/validate_registry.py
python3 scripts/test_standalone_registry.py
python3 scripts/test_source_registry.py
python3 scripts/test_source_update.py
python3 scripts/test_source_resolution.py
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```
