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
- `WideBvh`, a four-wide copy of a `DynamicAabbTree` (O(n)) for leaves that rarely change: one node visit is a four-lane slab or overlap test, so segment and box queries visit about `log4 n` nodes

`geometry-kernels` supplies analytical relations, OBB/SAT, GJK, planar/3D EPA, distance-normalized ray queries, bounded sphere time-of-impact, and reusable f64 sphere/box/capsule/wedge contact + fixed-orientation sweep kernels. `octree-kernels` provides conservative hierarchical broad-phase detection; `spatial-kernels` also supplies sweep-and-prune and `bvh-kernels` has dynamic trees. Triangle/mesh rays and general convex continuous collision remain separate future work. See [the ray and CCD guide](docs/ray-and-ccd.md) [uniform primitive volume products](docs/primitive-volume.md), [capsule intersection normals](docs/capsule-contact-normals.md), the [experimental ellipsoid primitive](docs/ellipsoid-primitive.md), and the [experimental tetrahedron/octahedron swept SAT](docs/platonic-swept-sat.md). The [capsule row acceptance](docs/capsule-row-acceptance.md) records independent reference and bounded-work coverage. [Retained f64 heightfields](docs/heightfield-surface.md) provide bounded vertical surface queries with stable triangle identity and prepared normals. [2D polygon triangulation and convex decomposition](docs/polygon2.md) preprocess simple collision polygons with explicit winding, duplicate and collinearity contracts.

### Collections

`collection-kernels` contains:

- `UnionFind`, using path compression and union by size with deterministic tie-breaking and connectivity checks against a traversal oracle
- `RingBuffer`, a fixed-capacity FIFO buffer that does not reallocate after construction
- `SparseSet`, the sparse/dense integer-key set primitive commonly used for fast membership and ECS-style storage
- `SparseMap<T>`, the value-bearing sparse/dense map primitive for ECS-style typed component storage and other integer-key workloads
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
- deterministic PageRank with bounded convergence, explicit damping/tolerance policy, and dangling-node redistribution
- Kruskal minimum spanning forests, reusing `UnionFind` for connectivity
- `dinic_max_flow`, exact directed flow with per-edge flows and a minimum-cut certificate
- `hopcroft_karp`, bipartite matching with a minimum vertex cover
- `hungarian_assignment`, rectangular signed-integer assignment with primal/dual certificates

Flow and matching augmentation are iterative, including for long paths. Network optimization uses caller-owned dense indices and edge lists; products still own mapping, eligibility, costs and allocation rules. See [the search and network guide](docs/search-and-network-kernels.md) for API contracts and runnable examples.

### Search, selection, and sorting

`search-kernels` contains:

- `quickselect`, an in-place deterministic three-way selection algorithm
- `top_k_smallest`, which combines selection with sorting only the requested result set
- `top_k_by`, stable ranked stream selection without materializing the entire input
- `BloomFilter`, deterministic probabilistic membership over byte-oriented keys using the shared `BitSet` primitive
- stable LSD radix-sort kernels for `u32` and `u64`, checked against Rust's stable sort

Selection and sorting results are checked against simple standard-library oracles. Bloom-filter tests enforce the no-false-negative contract while keeping false-positive behavior explicit.

### Similarity and bounded search

`similarity-kernels` provides exact Levenshtein, sorted-set operations/Jaccard,
k-way merging, shingles/rolling hashes, MinHash/SimHash, Hamming distance,
BK-tree search and single-word byte-oriented Myers distance.
`levenshtein_bounded` returns the exact distance up to a caller-selected limit;
`LevenshteinWorkspace` reuses narrow scratch rows for repeated bounded queries.
Tokenization, normalization, approximate-index policy and ranking remain outside
these kernels.

### Noise

`noise-kernels` provides a deterministic seeded `Permutation` plus allocation-free
2D/3D Perlin and 2D/3D/4D Simplex sampling. The fixed permutation state is shared
infrastructure for later Worley kernels; terrain composition and world-generation
policy remain consumer-owned.

### Statistics

`statistics-kernels` contains `RunningStats`, a Welford-style streaming accumulator for count, mean, population/sample variance, and standard deviation. Accumulators can be merged without replaying observations, with tests against batch calculations and large-offset fixtures.

## Consuming

Consumers depend on the workspace crates through Cargo git dependencies pinned to an exact `rev`.

## Development

```bash
python3 scripts/validate_verification_matrix.py
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```
