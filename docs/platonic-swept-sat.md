# Experimental tetrahedron/octahedron swept SAT (issue #188)

`geometry_kernels::primitive3::platonic` is an **experimental** measurement
fixture. It is not part of the `PrimitiveShape3` pair matrix, is not dispatched
by `primitive3::try_swept_time`, has no discrete contact query, and carries no
stability promise. No Physics Engine shape, consumer semantics or dependency is
added.

## Fixture convention

Both regular solids are centred on their centroid and sized by circumradius `R`.

| Solid | Local vertices | Canonical face axes | Unique edge directions |
| --- | --- | --- | --- |
| Tetrahedron | `R s / sqrt(3)`, `s` in `(1,1,1)`, `(1,-1,-1)`, `(-1,1,-1)`, `(-1,-1,1)` | 4 outward normals `-s_i / sqrt(3)` (face `i` opposite vertex `i`) | 6: `(e_i ± e_j) / sqrt(2)` |
| Octahedron | `±R e_x`, `±R e_y`, `±R e_z` | 4 of the 8 normals `(±1,±1,±1) / sqrt(3)`, one per antipodal pair | the same 6 |
| Box (baseline operand) | `±h` per axis | 3 local axes | 3 local axes |

`PlatonicBody3::axes` maps local X/Y/Z to world space and must be orthonormal.
Support ties resolve to the lowest documented vertex index; for the octahedron
and box a zero component selects the positive side.

## Algorithm and exactness

`try_platonic_swept_interval(a, b, dt, margin, max_iterations, work)` returns
the normalized contact interval `[enter, exit]` (exit clipped to the window end)
or `Ok(None)` when one axis separates the whole window. Errors follow
`try_swept_time`: `InvalidInput`, `NonFiniteComputation` and `IterationLimit`
(the budget bounds inspected axes) are never misses.

It reuses the existing swept-SAT slab loop (`sweep::swept_sat_interval`, now
shared with the box/wedge `poly_sweep_time`) and adds only shape-specific axes
and projections:

- Candidate axes: faces of `A`, faces of `B`, then all `A`-edge x `B`-edge
  crosses, generated lazily so an early separating axis skips the rest.
  Generic orientations test 44 axes for tetrahedron/octahedron pairs, 25 with a
  box and 15 for box/box.
- Work happens in `A`'s frame, so `A`'s projection needs no rotation. Crosses are
  not normalized (interval times are scale-invariant per axis); the margin is
  scaled by the axis length instead.
- Parallel-edge crosses (`|e_a x e_b|^2 <= 1e-12`) are skipped and duplicates are
  retested rather than removed. Omitting an axis can only admit contact, never
  hide it, so there is no false-negative culling.

The result is exact, up to floating-point rounding, **only for fixed-orientation
translation**: the face normals and edge crosses span every face normal of the
Minkowski difference. Solids that rotate independently need a different algorithm.

## Evidence

`cargo test -p geometry-kernels --locked --test platonic_swept_sat` covers:

- Fixture identities: circumradius, centroid, inradius face planes, edge
  lengths, and a one-to-one edge-direction mapping.
- Support points against vertex enumeration (512 rotations per shape) and
  deterministic ties.
- Closed-form entry/exit for octahedron vertex–vertex and tetrahedron
  edge–edge contact, window clipping, and margins.
- Initial overlap, separating from an overlap, receding/parallel/short and
  stationary misses.
- Exact edge–edge grazing interval, and an offset graze (`+1e-6`) that only an
  edge-cross axis separates, with sampled positive separation.
- Nearly parallel faces (relative rotation `1e-3` to `1e-12` rad, and 0).
- Uniform scale `1e-3` to `1e6` invariance.
- Repeated-call bit identity and swapped-order agreement.
- Box parity with the existing `primitive3` box/box sweep, including identical
  axis and support counts.
- Budget and invalid-input errors, plus work ratchets. A hit inspects every
  candidate axis and four support queries per axis.

The independent oracle in `tests/support/platonic.rs` is a generic
conservative-advancement cast over `gjk_distance` on `ConvexHull3` world
vertices. It shares no axis or projection code with the kernel. Over 2,304 random
sweeps across all six pairs, entry agrees within `REFERENCE_TIME_TOLERANCE =
1e-7`, and exit agrees with the reversed cast. Every miss has 65 sampled
separations above `REFERENCE_DISTANCE_TOLERANCE = 1e-9 * (R_a + R_b)`.
`allocation_regressions` records zero allocations and reallocations over 4,608
warmed sweeps.

## Measurement

Command: `cargo bench -p geometry-kernels --bench primitive3 -- platonic
--sample-count 100 --threads 1 --color never`. Each batch has 64 randomly
oriented pairs. Medians are per batch:

- `hits`: every pair hits, which is the SAT worst case because no axis can accept early.
- `receding`: no pair hits.
- `near_miss`: 67–94 % of pairs hit, depending on the pair.

| Pair | SAT hits | GJK cast hits | SAT receding | GJK receding | SAT near miss | GJK near miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tetra/tetra | 165.9 µs | 107.2 µs | 9.35 µs | 22.05 µs | 117.5 µs | 100.2 µs |
| octa/octa | 149.2 µs | 128.4 µs | 7.91 µs | 25.64 µs | 120.2 µs | 151.9 µs |
| tetra/octa | 141.2 µs | 98.8 µs | 8.34 µs | 23.34 µs | 126.3 µs | 114.2 µs |
| tetra/box | 82.0 µs | 110.7 µs | 8.39 µs | 29.69 µs | 71.1 µs | 141.7 µs |
| octa/box | 79.2 µs | 218.7 µs | 7.86 µs | 29.83 µs | 69.1 µs | 163.9 µs |
| box/box | 48.9 µs | 225.2 µs | 7.77 µs | 31.28 µs | 44.4 µs | 218.4 µs |
| existing `primitive3` box/box | 76.6 µs | — | 27.7 µs | — | 72.7 µs | — |

Edge–edge graze miss (single sweep): SAT 617 ns (9 axes) and GJK cast 1.36 µs (2
advancement steps).

Work per hit batch: 2,816 axes and 11,264 support queries for tetrahedron and
octahedron pairs, against about 185 GJK distance solves for the reference cast.
Receding batches stop after 89–104 axes in total, about 1.5 per pair.

Environment: rustc/cargo 1.99.1, x86_64 Linux 6.18 on a shared 4-vCPU Intel
Xeon @ 2.10 GHz VM, with no RUSTFLAGS. This is a single shared-host developer
probe, and its wall-clock numbers are advisory only. On the same host the
existing `fast_sphere_sphere_sweep` measured about 135–170 ns, so compare
ratios, not absolute values. An Iai-Callgrind sentinel is not warranted while
nothing is promoted.

A first eager variant built the axis list with the existing O(n²) `push_axis`
deduplication. Its tetra/tetra hits took 290 µs and receding batches took
100 µs. For 44 candidates, building the list cost more than testing it, which is
why the kernel now generates axes lazily.

## Conclusion

- **Tetrahedron: keep experimental, do not promote.** Exact swept SAT is about
  1.5× slower than the generic GJK cast on hit-heavy batches, because 36 of its
  44 axes are edge crosses. It is 2.4× faster when pairs recede and 2.2× faster
  on the grazing worst case for GJK. Drop it unless a follow-up prunes edge pairs
  (Gauss-map arc overlap) to the few that form real Minkowski faces.
- **Octahedron: keep experimental.** It is closer to break-even: 1.16× slower on
  hits, 3.2× faster on receding pairs and 1.26× faster on near misses. Promotion
  also depends on edge-pair pruning and on a consumer that needs octahedra.
- **Mixed pairs with a box and the box baseline:** the A-frame, lazy and
  dedup-free projection path beats both the GJK cast and the existing
  `poly_sweep_time` box/box (48.9 µs against 76.6 µs on hits, 7.8 µs against
  27.7 µs when receding). That speedup applies to the existing box path and is
  worth a separate, consumer-checked `primitive3` slice. It does not justify
  adding these solids to `PrimitiveShape3`.
