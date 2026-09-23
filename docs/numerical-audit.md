# Numerical correctness and trace-cost audit

The reference revision is `ddec70d25398341c7130ee68a95e4e0db34e34fc`.
The native regression harness has **nine cases that fail on that revision and
three ordinary-input controls that pass**. All twelve pass with the fixes.
The full workspace additionally runs scale sweeps, independent oracles and
execution-level work gates; passing the twelve-case comparison alone is not the
complete validation contract.

## Reproduce the evidence

```sh
# Expanded regressions, independent oracles and deterministic work budgets.
cargo test -p geometry-kernels -p statistics-kernels
cargo test -p geometry-kernels -p statistics-kernels --release

# The same current native test/benchmark source against base and candidate.
python3 scripts/test_audit_evidence.py
python3 scripts/audit-evidence.py

# Developer-facing allocation-profiled Divan matrices, current revision only.
cargo bench -p geometry-kernels -p statistics-kernels --bench audit_numerics
```

The comparison creates a detached worktree for the base and separate build
outputs for the two revisions. It does not check out over, overwrite, or clean
up the caller's working tree. An existing source directory can instead be
supplied with `--baseline-root /path/to/baseline`. The default output is
`target/audit-evidence/`.

Both builds use the identical current test and benchmark harness, the same
Rust executable and a copied dependency lock. The report records source and
harness SHA-256 fingerprints, Rust/Cargo versions, host details, optimization
profile, `RUSTFLAGS`, allocation-profiler version, dependency lock, and raw
per-trial output. Workload v1 contains **27 timed cases**; missing workloads,
compiler failures, missing tests and duplicated observations are errors, not
successful evidence. The nine expected baseline failures are explicitly named.

`summary.md` shows the before/after comparison. `results.json` contains all
27 workloads and their individual trials, allocation/reallocation counts and
correctness probes. The path-scoped **Numerical audit evidence** workflow
publishes the table in its job summary and retains the report and logs as an
artifact. It does not upload compiler binaries or build caches.

## 1. Segment and capsule distance

The original segment-segment routine used
`|u|² |v|² - (u·v)²`, comparing this length-to-the-fourth quantity with a fixed
squared-length epsilon. Short perpendicular segments were therefore treated as
parallel. Long nearly parallel directions also lost the denominator to
cancellation. Both defects propagate into capsule collision decisions.

The corrected routine obtains the denominator from `|u × v|²`, uses cross-product
forms for the unconstrained interior parameters, and retains the endpoint
clamping/reprojection cases. Nonzero `f32` segments are no longer collapsed to
points merely because they are short. Calculations widen their endpoints to
`f64`; this does not introduce arbitrary tolerances into the segment contract.

| Fixture | Before | After |
|---|---|---|
| Perpendicular segments crossing at zero, endpoints at ±`1e-7` | Distance ≈ `1e-7` | Distance `0`, both parameters `0.5` |
| The same segments as capsules, each radius `1e-8` | Separated | Overlapping |
| `(0,0)→(1e8,1)` against `(0,1)→(1e8,0)` | Distance ≈ `1` | Distance `0` |

Regression coverage includes an independent convex one-dimensional distance
minimization oracle, skew segments, endpoint minima, swap/reversal invariance,
point segments, and scale sweeps including subnormal `f32` coordinates. The
Divan matrix separately measures ordinary and scaled crossings, near-parallel
segments and the small capsule case. These are correctness repairs; a timing
from the old wrong-result path is not an equivalent-result speed comparison.

## 2. GJK scale handling

A line's triple-cross search direction has length-cubed units, a triangle's
normal has length-squared units, and a tetrahedron's volume has length-cubed
units. Their raw magnitudes cannot all be compared with the same distance
threshold.

The corrected tests compare geometric distances/altitudes without taking extra
square roots: squared triple-cross magnitude against `epsilon² × edge_length⁴`,
squared triangle area against `epsilon² × longest_edge_length²`, and squared
tetrahedron volume against `epsilon² × largest_face_area²`. Face normals already
needed for visibility are reused for the volume test. A small search vector by
itself is no longer accepted as proof of intersection; the simplex reducers
own that proof. The public tolerance remains a caller-selected world-space
length, not a new consumer-specific scaling policy.

Overlapping spheres of radius `1e-7`, with centers `(0,0,0)` and
`(1.4e-7,1.4e-7,0)`, previously exhausted the 32-iteration budget. The corrected
kernel returns `Intersecting`. The reproducible power-of-two small-sphere
benchmark (`2^-24`) goes from **32 iterations / indeterminate** to
**6 iterations / intersecting**.

Tests compare scaled spheres with the analytical oracle, rotated boxes with
SAT, and planar shapes through the existing planar adapter. They also verify
GJK trace-prefix parity. Easy sphere fixtures have a deterministic **at most
eight iterations** budget across scales. This is a bounded scenario contract,
not a claim that every possible convex query converges in eight iterations.

## 3. Support directions

The old sphere/capsule helper used a generic geometric normalization threshold
and substituted positive X when normalization failed. Thus both `(0,1e-13,0)`
and `(0,1e308,0)` returned the wrong support point for a unit sphere.

The support-specific helper now uses the ordinary normalization fast path when
its squared length is normal, and component scaling otherwise. Only an actual
zero direction receives the deterministic X fallback. Capsule endpoint
selection uses the normalized direction too, preventing endpoint dot products
from both overflowing to infinity. The separate public `math3::normalized`
policy is unchanged.

Tests exercise positive direction rescaling from the smallest subnormal through
large finite vectors, analytic extrema, both capsule endpoints, and zero-input
behavior. Benchmark setup is excluded; ordinary and extreme-direction queries
are timed separately.

## 4. Running-statistics merge

The original parallel-variance correction multiplied `delta²` by both counts
before dividing. Merging `[0,0]` with `[1e154,1e154]` overflowed an intermediate
and returned infinite population variance, although `2.5e307` is representable.

The merge now forms the normalized count weight before multiplying the
correction. It also scales a finite mean update before multiplication and uses
a convex weighted sum when opposite finite endpoint means would overflow the
subtraction. It preserves representable subnormal accumulated `m2` as well as
large finite results. This is an intentional numerical bug fix: floating-point
merge terms are regrouped, so bit-for-bit identity with every old merge is not
promised. Existing ordinary-input tolerances and the no-replay API are retained.

Tests cover singleton and partitioned merges, reversed merge order, highly
unbalanced counts, representable subnormal `m2`, finite means at opposite
extremes, and legitimately infinite variance. They do not claim unlimited
precision or that an actually unrepresentable variance becomes finite.

## 5. EPA trace construction and the expansion defect exposed by benchmarking

The previous no-trace path constructed an `EpaTraceStep` and cloned its polytope
before discovering that no trace sink existed. Snapshot creation is now lazy:
its closure is evaluated only when tracing was requested. Whole-execution,
thread-local test counters surround actual snapshot construction and require
**zero snapshots and zero copied trace support points** in ordinary EPA,
including iteration-capped/terminal paths. These counters are compiled out of
production. A separate test rejects any evaluation of a disabled snapshot
factory, and traced/untraced results remain equal.

The broader benchmarks exposed another error: inserting a new support point
between only the chosen edge's endpoints could retain adjacent vertices hidden
by that point. Overlapping 16- and 64-vertex polygon fixtures then terminated as
`DegenerateSimplex`. Expansion now removes the contiguous visible edge chain,
preserving convexity, origin enclosure, surviving vertex order and the existing
vector's capacity. The tests compare penetration with an independent polygon
SAT oracle and check every retained trace polytope for convexity and enclosure.

The four-vertex no-trace fixture already returned the correct answer before the
repair. The comparison requires **identical GJK seed, EPA result and witnesses**
for that fixture, so it supports an isolated performance comparison. It uses
**six allocation calls before and four afterward**, with no per-iteration trace
clones. The four-allocation limit is enforced by the native evidence runner.
For 16/64 vertices the old answer was invalid, so their timings are reported but
are explicitly not labeled equivalent-result speedups.

## Interpreting timings

The runner alternates baseline/candidate order across three trials; each trial
uses 200 Divan samples of 64 calls with one thread. It reports the median of the
trial medians. Divan's allocation profiler is active on both revisions, so
these timings include profiler overhead. Raw wall-clock results on a shared
runner are informational; **no nanosecond or percentage timing gate is added**.

Use the correctness outcomes, GJK iteration counts, zero-snapshot tests and
allocator counts as the deterministic evidence. Consult the generated table
for latency: more robust numerical predicates can cost more on ordinary inputs,
and a previously incorrect early exit is not a valid faster baseline.
The existing search Iai-Callgrind thresholds remain unchanged.
