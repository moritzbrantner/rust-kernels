# Input-boundary and lifecycle audit

Baseline: `8e719d3568cf1a10c98f20bef81e629cd58d25db` (merged PR #103).
The same native harness has **seven failing regressions and four passing
controls** on that revision; all eleven pass after these repairs. These are
remaining kernel defects, not new application features. Previous numerical
audit cases and budgets remain in place.

## Repairs

### Conservative octree enclosure

Reconstructing the root from a `f32` center and half extent could round away an
input bound. With one collider at x = `-3668.735` and two overlapping point
colliders at x = `1.05e-10`, the old root ended at zero and both positive
colliders disappeared during subdivision. Large finite bounds could also
cause a center/span overflow and panic.

Root sums and spans now use `f64`, padded bounds round outward, and the final
bounds explicitly include the source union even when the coordinate magnitudes
are too different for `f64` arithmetic to retain both. At range limits where a
finite cube cannot fit, only padding is clamped; the root becomes a conservative
finite box, never an exclusion of valid input. Subdivision midpoints use `f64`
as well. Touching still counts as overlap and pair/trace ordering is unchanged.

The subdivision also borrows rather than clones its parent, moves child member
buffers, and traverses contiguous child indices without duplicate child lists.
Tests cover arbitrary finite floating-point intervals, signs, adjacent values,
subnormals, parent/child enclosure, leaf membership, reversed input order, and
agreement with the naive pair oracle. The existing sparse-scene work budget is
retained. Allocation-profiled regular scenes verify identical pairs, exact-test
counts, node counts and result checksums before comparing cost.

### Heap-backed strongly connected components

Tarjan traversal previously consumed one native stack frame per DFS vertex.
The regression launches a subprocess, creates a **64 KiB stack thread**, and
checks a **20,000-node chain and cycle**. The old child aborts from stack overflow;
the fixed traversal returns all expected components. Isolating the child keeps
the regression itself diagnosable instead of aborting the entire test run.

Explicit heap frames retain the current vertex and next outgoing edge; returning
from a child propagates its lowlink just as recursive Tarjan does. Frame storage
is reused across disconnected roots. Test-only counters require each stored
edge to be processed once and at most one pending frame per discovered vertex.
They do not instrument production builds. The output still uses materialization
order (seeds first, then newly discovered neighbors), not a new DFS ordering.

An independent transitive-reachability oracle checks all 4,096 directed
four-vertex graphs without self-edges. Separate cases cover self-loops, parallel
edges, finished-component boundaries, multiple seeds and empty graphs.
Normalization still sorts components; the linear edge-visit bound describes
Tarjan traversal, not a new O(V+E) promise for the entire normalized public API.
The safety change uses one extra heap buffer; it is not presented as a universal
latency improvement over native recursion.

### Top-k limits are not input lengths

`top_k_by` formerly reserved `k` slots before inspecting its iterator. Even an
empty iterator with `k = usize::MAX` panicked. Modest oversizing also retained
unnecessary storage: requesting 65,536 results from four `u64` values retained
131,072 `u64` slots (1 MiB) on the comparison toolchain.

The initial reservation now uses `min(k, iterator.size_hint().0)`. Exact-size
iterators avoid growth while unknown-length streams grow only as candidates
arrive. The zero-limit path still does not touch the iterator or comparator.
All values, ranking and encounter-order tie semantics remain unchanged.

The comparison's four-value result retains eight slots (64 bytes) after the
repair. That exact capacity is evidence for the pinned standard library, not an
API promise about its future in-place `map`/`collect` implementation. The native
evidence runner rejects a retained-capacity regression on this fixed fixture;
empty-input allocation calls must remain zero. An ordinary 4,096-candidate
benchmark and an unknown-length stream expose any costs outside the oversized
limit case. A deterministic comparison counter bounds the eight-result scan at
8n comparisons for defined adversarial descending fixtures.

### Finite observations must not poison `RunningStats::push`

The previous repair covered merging, but `push(-1e308); push(1e308)` still
overflowed the Welford delta and mean. Adding zero afterward produced a NaN mean
and an incorrectly zero variance because variance's non-negative clamp masked
the invalid second moment.

The ordinary finite-delta update keeps its historical evaluation order. Only
an overflowing difference takes an overflow-safe convex mean update. Such an
extreme difference necessarily makes the stored second moment unrepresentable,
so it becomes positive infinity. Subsequent observations retain a finite mean
and honestly infinite variance instead of poisoning the accumulator.

Tests cover both input signs, maximum finite observations, continued use,
`push`/`merge` agreement and unchanged state after non-finite input rejection.
Ordinary fixtures are compared bit-for-bit with the historical update sequence.
There is still no promise of unlimited-precision second moments; genuinely
unrepresentable results may be infinite.

## Reproduce

```sh
cargo test --workspace --all-features
cargo test -p graph-kernels -p octree-kernels -p search-kernels -p statistics-kernels --release
python3 scripts/test_audit_evidence.py
python3 scripts/audit-evidence.py --suite boundary
cargo bench -p graph-kernels -p octree-kernels -p search-kernels -p statistics-kernels --bench boundary_cases
```

The shared runner's `--suite numerical` (default) retains the original 12-test /
27-benchmark contract. `--suite boundary` selects the new 11-test / 26-benchmark
contract. Both build identical current harness sources against separate base
and candidate libraries using isolated target directories and one copied
dependency lock. A supplied `--baseline-root` can replace the detached worktree.
Missing tests, compilation failures, dropped workloads and inconsistent
allocation counts fail the runner. NaN/Infinity observations are serialized as
explicit strings rather than invalid JSON or invented finite values.

Boundary results go to `target/boundary-audit-evidence/summary.md` and
`results.json`, with raw per-trial logs and fingerprints. The existing **Kernel
audit evidence** CI job runs both suites and publishes both summaries; no second
full Rust/registry validation workflow is added.

Boundary timings use three alternating-order trials, each with 50 Divan samples
of four invocations and allocation profiling on both revisions. New blocking
budgets cover zero eager allocations for empty top-k, at most 239 allocation
calls for the 256-body octree control, and SCC edge-visit counts. Shared-runner
latency remains informational; the old wrong-answer and aborting cases are
correctness evidence, not equivalent-result speed baselines. Raw results retain
all 26 cases, including slower controls, rather than only favorable ratios.
