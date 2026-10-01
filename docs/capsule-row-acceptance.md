# Capsule row contact and work acceptance

The fixed-orientation sphere/capsule, box/capsule and capsule/capsule production
paths retain analytic segment-distance or bounded slab/SAT formulations. Geometric
queries have no render vertices, dynamic scratch or generic convex fallback.
Consumers still own physical response, temporal admission and motion policy.

## Independent reference

`capsule_row_reference.rs` minimizes squared distance from a skeleton point to the
other shape's skeleton or closed box with a generic one-dimensional convex search.
Its 80 golden-section iterations use no production closest-segment cross products,
box feature breakpoints, SAT or clipping helpers. The remaining point/segment or
point/box distance is computed directly. The reference compares contact presence
and resolved distance with tolerance `2e-8 * max(shape radii)` at dimensional scales
1e-6, 1 and 1e6. It does not supply signed containment depth inside a box; independent
support-plane clearance certificates test the selected production penetration
normal/depth instead, in both pair orders. Nonnegative resolved contacts also check
opposite normals and equal separation on reversal. Coincident symmetric contact
normals may be nonunique; their actual clearance is the required invariant.

The corpus includes 2304 seeded pair cases and 540 explicit edge cases: parallel
and nearly parallel/antiparallel skeletons, zero separation, skeleton endpoints,
containment, touching and near misses, rotated boxes/capsules, collapsed and tiny
nonzero skeletons. Independent support formulas also drive generic GJK diagnostics:

| Corpus | Converged | Intersecting | No progress | Iteration limit |
| --- | ---: | ---: | ---: | ---: |
| Seeded | 1852 | 359 | 2 | 91 |
| Edge cases | 189 | 282 | 45 | 24 |

Only converged/intersecting generic results assert generic agreement. The 162
unresolved probes remain unresolved and are independently covered by convex
minimization and clearance certificates. They never count as reference successes
or misses. The generic configuration uses 128 iterations per phase, relative
improvement tolerance 1e-12 and world epsilon `1e-9 * max(shape radii)`.

108 fast translation sweeps cross targets while both endpoints are separated,
including 0.01-half-width boxes, moving targets, oblique skeletons and swapped pair
orders. Independent static-distance bisection supplies the first-contact time.
The comparison budget derives from the production world-space convergence
`1e-9 * (1 + sum(shape radii))` divided by relative speed. This covers translation
with fixed orientations; it does not claim rotational CCD or chronological
multi-impact response.

## Actual work and resources

`PrimitiveWork3::segment_distance_evaluations` counts entered closest segment/point,
segment/segment and segment/box problems, including discarded queries.
`segment_feature_tests` counts each projected endpoint-region clamp, endpoint-region
branch, box slab or breakpoint visit, and point/interval distance candidate visited.
These are geometric work items, not machine instructions. Numeric validity checks,
sorting, normal construction and triangle-only feature work are excluded. Shared
segment/segment problems used by capsule/wedge also contribute these counters;
this does not provide complete wedge-row feature accounting.

| Contact pair | Segment-distance problems | Segment feature visits | SAT axes | Support/vertex scans |
| --- | ---: | ---: | ---: | ---: |
| Sphere/capsule | 1 | <=1 | 0 | 0 |
| Capsule/capsule | 1 | <=4 | 0 | 0 |
| Box/capsule | <=6 | <=46 | <=6 | 0 |

For a separated box problem the bound includes at most three slab visits, six
plane breakpoints, the initial point, eight breakpoint candidates and seven smooth
intervals with three coordinate classifications plus one distance candidate each:
`3 + 6 + 1 + 8 + 7*(3+1) = 46`. A separated skeleton crossing every box plane reaches
that exact ceiling. Intersecting skeletons instead test at most six SAT axes and
six point/segment tie candidates. Sweep work includes each entered contact query;
the existing 128-iteration limit remains unchanged.

Long-thin fixtures use half lengths through 1e9 with radius 1e-6 and retain these
bounds, independent of render-mesh complexity. A Cargo-native allocator ratchet
executes 6144 warmed mixed queries, including containment and separation, with zero
allocations or reallocations. The inline box breakpoint array holds eight f64
values (64 bytes); no heap query scratch capacity grows.

Measurement environment on 2026-09-30: AMD Ryzen 7 5700X (8 cores/16 threads),
Linux 7.0.0-34-generic, target `x86_64-unknown-linux-gnu`, rustc 1.98.1
(`48a229cea`, LLVM 22.1.8), Cargo 1.98.1. Divan uses the optimized Cargo bench
profile; paired probes use the optimized release profile, with no custom
`RUSTFLAGS` or `CARGO_ENCODED_RUSTFLAGS`. Runs share this developer host without CPU
affinity or exclusive scheduling. Counted code producer:
`e3bdc928d2833fa832912a1df437321e09db32fc`; `primitive3.rs` SHA-256:
`5943c77a04f0e63bfc857bd45e461f5512e89f2aeba36875c6590471c9d2ddb7`.
Subsequent documentation-only corrections preserve that measured code.

Divan's `capsule_row_contacts` measures three complete calls per sample. A local
50-sample/eight-iteration run reported a 316.6 ns median; this is developer evidence,
not a timing guarantee or an optimization claim. Additional before/after probes
compare 6147 exact rotated contact/witness, sweep and prior-work products against
the unchanged 1c5f291 baseline in five alternating-order trials; all products match.
Each timing trial completes 600,000 queries. Raw elapsed nanoseconds are recorded
below; this mixed, noisy developer comparison makes no speedup claim. Counters add
16 bytes to the inline `PrimitiveWork3` product.

| Trial | Baseline ns | Counted ns |
| --- | ---: | ---: |
| 0 | 61687188 | 61106755 |
| 1 | 60466520 | 61084593 |
| 2 | 60250624 | 61138886 |
| 3 | 60081274 | 61308264 |
| 4 | 60225055 | 62794291 |


```sh
cargo test -p geometry-kernels --test capsule_row_reference -- --nocapture
cargo test -p geometry-kernels --test allocation_regressions
cargo bench -p geometry-kernels --bench primitive3 -- capsule_row_contacts
```

The broader primitive verification matrix remains partial: this evidence applies
to the capsule sphere/box/capsule row, not all ten contact pairs. Uniform mass/COM
products and capsule ray helpers have their own independent fixtures. Convention
sourceRevision: `e6acb5310afaf15c0cba24f87108f5f4ad1bedc3`.
