# Bounded search and network optimization

This expansion closes source-registry issue #76 and implements the three
mechanisms proposed in issue #36. It adds no production dependencies. It does
not add min-cost flow without a concrete consumer, nor tokenization, ranking,
scheduling rules, implicit capacities, or application-level graph authority.

## Bounded generic edit distance

```rust
use similarity_kernels::{levenshtein_bounded, LevenshteinWorkspace};
assert_eq!(levenshtein_bounded(b"kitten", b"sitting", 2), None);
let mut scratch = LevenshteinWorkspace::new();
assert_eq!(scratch.distance(b"kitten", b"sitting", 3), Some(3));
assert_eq!(scratch.distance(&["a", "b"], &["a", "c"], 1), Some(1));
```

`Some(d)` is an exact unit-cost distance with `d <= limit`; `None` means the
limit was exceeded. It is not an approximate distance, and must not be supplied
as a metric to the BK-tree (which needs real distances for sound pruning).
Existing unrestricted `levenshtein` and byte-oriented Myers APIs are unchanged.
Inputs are borrowed and never cloned or retained. Unicode segmentation remains
caller policy; callers can pass characters, tokens, or arbitrary `Eq` elements.

The algorithm rejects impossible length gaps before comparisons/allocations,
trims common prefixes and suffixes, evaluates only the relevant diagonal band,
and stops once the row cannot meet the limit. Scratch uses two compact rows,
not two full input-width arrays; `usize::MAX` safely requests unrestricted
computation. A workspace retains capacity from its largest previous query.
Time is O(n+m + max(n,m)*min(min(n,m), limit+1)) and newly required scratch is
O(min(min(n,m), limit+1)). Warm repeated queries reuse storage; fresh workspaces
and unrestricted limits are measured separately rather than hidden in setup.

Exhaustive ternary sequences through length five match unrestricted DP at five
limits; additional tests cover dirty-row reuse, both length orientations, generic
tokens, narrow-band boundaries and long common affixes. A counted-`Eq` test
limits width-five work to 5n+4 comparisons. The reference-comparison fixture at
1,024 elements requires 1,048,576 comparisons with full DP versus 5,116 with the
band. Allocator evidence requires zero allocations and reallocations after
warming, including early rejection.

## Maximum flow and minimum cut

```rust
use graph_kernels::{CapacityEdge, dinic_max_flow};
let edges = [
    CapacityEdge { from: 0, to: 1, capacity: 5 },
    CapacityEdge { from: 1, to: 2, capacity: 3 },
];
let flow = dinic_max_flow(3, &edges, 0, 2).unwrap();
assert_eq!(flow.value, 3);
assert_eq!(flow.edge_flows, [3, 3]);
assert_eq!(flow.source_side, [true, true, false]);
```

Vertices are dense caller-owned indices. Invalid endpoints and equal source/sink
are explicit errors. Zero capacities, self edges, parallel and antiparallel
edges are supported. Every invocation starts at zero flow without mutating or
annotating input edges. Original edge order is preserved in `edge_flows`; self
edges carry zero. Individual capacities/flows are `u64`, while the total is
`u128`, so parallel maximum-capacity edges do not wrap the result.

Dinic uses residual BFS levels and an iterative blocking-flow search. It reuses
queue/path storage during a query and does not recurse with graph depth.
General worst-case time is O(V²E), storage O(V+E). The source-side reachability
set certifies a minimum cut; tests independently check capacity bounds,
conservation and equality of flow and cut. Work counters report phases,
augmentations and residual-edge scans, not wall time.

All 4,096 four-vertex directed unit networks match exhaustive cut enumeration.
Further fixtures cover reverse-edge rerouting, weighted duplicate edges,
wide totals and errors. Disjoint-path families require one blocking phase with
linear edge scans, and a 20,000-vertex path runs on a 64 KiB thread stack.

## Bipartite matching and minimum cover

```rust
use graph_kernels::hopcroft_karp;
let matching = hopcroft_karp(2, 2, &[(0,0), (0,1), (1,0)]).unwrap();
assert_eq!(matching.size, 2);
assert_eq!(matching.left_to_right, [Some(1), Some(0)]);
assert_eq!(matching.cover_left.len() + matching.cover_right.len(), 2);
```

Left and right partitions have independent zero-based indices and may have
different sizes. Empty partitions, isolated vertices and duplicate edges are
valid; invalid endpoints are errors. Ascending left roots and caller edge order
make selection deterministic; no lexicographically-smallest promise is made.
Both mate arrays and a sorted minimum vertex cover are returned. Tests verify
that each edge is covered and the cover size equals the matching cardinality.

Hopcroft--Karp augments shortest alternating paths without recursion. Isolated
left vertices are excluded from per-phase resets/scans, preserving O(V+E sqrt(V))
time with O(V+E) memory. Tests enumerate all 65,536 four-by-four bipartite graphs
against independent matching enumeration. An adversarial long-path family
requires two phases with bounded edge work; its 20,000-vertex variant also runs
on a 64 KiB stack.

## Rectangular minimum-cost assignment

```rust
use graph_kernels::hungarian_assignment;
let assignment = hungarian_assignment(&[[1, 2], [2, 100]]).unwrap();
assert_eq!(assignment.row_to_column, [1, 0]);
assert_eq!(assignment.total_cost, 4); // Row-by-row greedy would cost 101.
```

Every row is assigned to one distinct column. Rows must have equal width and
there must be at least as many columns as rows; the kernel does not silently
transpose the problem or drop requested rows. Empty input is valid. Costs are
signed `i64`; all values, including extrema, are ordinary costs, not forbidden
edge sentinels. Potentials and total cost widen to `i128`.

The rectangular Hungarian implementation uses shortest augmenting paths in
O(rows² * columns) time and O(rows + columns) scratch/output space. Input rows
are borrowed via `AsRef<[i64]>`. Row/column traversal defines deterministic ties.
The dual certificate satisfies `u[i]+v[j] <= cost[i][j]` with equality at assigned
edges, non-positive column potentials, and zero potential for unused columns.
Its objective equals the returned primal cost. Exhaustive small signed matrices,
random rectangular matrices, extreme costs, and the greedy counterexample are
verified independently. Column-scan budgets protect dense scaling.

## Source installation and update boundaries

Both `search-kernels` and standalone `selection` now advertise `top_k_by`.
`similarity-kernels` advertises every currently exported mechanism, including
bounded distance, and is distributed as a dependency-free crate source set.
The graph source set includes flow, matching and assignment. Crate entries
include all `src` modules and every explicitly declared Cargo target: leaving
out a benchmark file made even `cargo check --lib` fail for previous copied
crates. Installed upstream manifests remain byte-for-byte identical; tests
supply only consumer-owned workspace metadata and application wiring.

```sh
python3 scripts/source_registry.py install similarity-kernels graph-kernels --root ../consumer
python3 scripts/source_registry.py status --root ../consumer --require-clean
python3 scripts/test_catalog_install.py
```

Real-catalog tests import every advertised symbol, compile all installed targets,
exercise mechanisms from a fresh executable, and verify file hashes afterward.
They also cover idempotent installs, unchanged-layout upstream updates,
local-divergence refusal, three-way conflict export and explicit acceptance.

Existing installations whose file inventories change still report
`layout-changed` and require explicit reconciliation. The update safety gate has
not been bypassed or made to overwrite locally owned files. Newly installed
similarity sources have complete layouts; source-registry APIs do not silently
rewrite consumer Cargo workspaces or module wiring.

## Reproducible evidence

```sh
cargo test -p similarity-kernels -p graph-kernels
cargo test -p similarity-kernels -p graph-kernels --release
python3 scripts/test_catalog_install.py
cargo test -p similarity-kernels -p graph-kernels
cargo bench -p similarity-kernels --bench expansion
cargo bench -p graph-kernels --bench optimization
```

The benchmark suites are invoked directly through `cargo bench` in CI. Blocking
correctness, work and warm-allocation contracts live in ordinary Rust tests;
Divan output is retained as raw benchmark evidence rather than parsed into a
second test framework. No separate full-workspace validation workflow is added.

The new APIs did not exist historically. Comparisons therefore use exact full
Levenshtein DP, a simple Edmonds--Karp implementation, single-path matching and
small exhaustive assignment, not invented older kernel implementations. The
flow/matching/assignment kernels additionally return certificates; the simple
timing references return scalar optima (and flow scan counts). These are not
benchmarks against tuned provider libraries. Time is informational, with all slower controls retained; algorithmic work,
allocation and correctness gates are asserted by Cargo-native Rust tests. The
existing numerical/boundary Cargo suites remain intact.
