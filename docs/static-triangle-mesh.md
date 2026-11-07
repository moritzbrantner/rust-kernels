# Prepared static triangle surfaces

`geometry_kernels::triangle_mesh::PreparedTriangleMesh3` prepares immutable f64
vertices and indexed triangles, two-sided unit normals, and a deterministic local
BVH. Construction rejects empty data, non-finite vertices, invalid indices,
degenerate triangles and unrepresentable geometry. Input triangle indices are
stable IDs; BVH sorting does not change them. The mesh is a set of surfaces:
holes remain open, winding does not select a solid interior, and separate meshes
can represent independently traversable floors and ceilings at the same X/Z.

Capsule contact and fixed-orientation translational sweep queries also cover
spheres when the segment endpoints coincide. Segment casts report the first
triangle and orient its normal against travel. Equal-time sweeps prefer the most
head-on contact, then input triangle ID; equal-depth overlaps and equal-time rays
prefer input triangle ID. Capsule normals point from the triangle toward the
segment. An intersecting segment chooses its midpoint's side, using winding only
when the midpoint lies exactly on the plane. The reported penetration then
includes the segment endpoint on the other side. Starting-touch sweeps ignore
surfaces being left or followed tangentially. Polytope overlaps consume the
existing `heightfield::ConvexPolytope3` contract and shared triangle SAT kernel.
Rotating sweeps and generic polytope sweeps are not provided.

`box_contacts_counted(center, axes, half_extents, margin, work, emit)` accepts a
box's f64 orthonormal local axes in mesh space, positive half extents and a
nonnegative speculative margin. It emits `TriangleMeshBoxManifold3` per triangle,
with a normal toward the box and one to four `TriangleMeshBoxPoint3` witness pairs
in `points[..len]`. Signed separation is measured along that normal. Face patches
use reference/incident clipping; edge pairs use segment witnesses. Speculative
corner contacts use Euclidean distance, so projected SAT gaps alone cannot admit
an out-of-reach corner. Near-equal SAT axes prefer faces deterministically.
Callbacks follow retained BVH order, rather than sorted triangle IDs. Callers own
cross-triangle patch reduction, contact persistence and response. No contacts
means no callback; a later arithmetic error may follow earlier callbacks.

Other queries return `Result<Option<_>, TriangleMeshError3>`: a valid miss is `Ok(None)`;
invalid input, non-finite arithmetic, and exhausted conservative advancement are
errors. Finite data alone does not guarantee every difference/product fits f64.
No numeric failure is converted into a miss. Sweep fractions are f64; consumers
with narrower physical state own conservative narrowing and response policy.

`capsule_sweep` solves each triangle in the swept bounds by Newton steps from below
and searches them best-first, sharing the search of the heightfield sweep
(`sweep_search.rs`): every pending triangle carries a lower bound on its touch (its
current Newton time, raised to a conservative bound from the gap between the capsule
and the triangle's plane), the triangle with the lowest bound (then lowest ID) takes
the next step, and the search stops once that bound passes the best settled hit or the
end of the motion. BVH candidate gathering is unchanged. `max_iterations` caps the
steps per triangle. The sweep returns `Err` only when a triangle fails (runs out of
steps, makes no progress or computes a non-finite value) at a bound at or before the
earliest settled hit (or the end of the motion when nothing is hit); later failures
cannot change the result and are ignored, so the result does not depend on visit
order. Compared with the earlier BVH-order kernel, every `Ok` result is bit-identical
and every `Err` was the same `Err` there; some earlier `Err` results, where a triangle
visited before the winner ran out of steps after it, or a triangle the plane bound
shows unreachable ran out of steps, now return the hit (or no hit) the earlier kernel
settles to with unbounded steps. A seeded differential fuzz
(`triangle_mesh/sweep_differential_tests.rs`) enforces exactly this against a copy of
the earlier kernel. `triangles_tested` counts triangles that take at least one step;
`sweep_iterations` counts steps. For 256 character capsules on the floor of a closed
1968-triangle cave, steps drop from 2376/4160/2098/8247 to 1755/1707/1691/3976
(walk/fall/skim/fast diagonal) and the `cave_capsule_sweep` Divan medians drop by about
10/47/10/28%.

Preparation retains normalized triangle edges and inward clipping planes for box
queries, so repeated queries rotate validated directions instead of reconstructing
them. Exact cross axes already covered by box face axes are skipped. Clipping
reuses each vertex's plane distance. Contact-patch reduction updates nearest selected-witness distances
incrementally in normalized coordinates, avoiding repeated square roots. Preparation owns all retained allocations. `preparation()` reports triangle/node
counts and vector-capacity bytes, excluding object/allocator overhead. Queries
allocate nothing and report query count, BVH node visits, exact triangle tests,
normal reuses and sweep iterations through `TriangleMeshWork3`. BVH construction
uses deterministic median splits; local queries visit intersecting node bounds
rather than scanning the full mesh. A broad query covering the complete mesh may
still require linear work.

Validation is Cargo-native: unit tests compare BVH contacts, sweeps and rays with
unpruned per-triangle traversal, exercise both sides, winding, shared edges,
vertices, holes, initial overlap, failure statuses and translated coordinates.
A local-query doubling family checks logarithmic visits and bounded exact tests;
the allocation regression checks warm query allocation counts. Existing
heightfield integration tests protect its unchanged one-sided crossing policy.
Run `cargo bench -p geometry-kernels --bench triangle_mesh` for preparation and
query timings; timings are advisory, not a portable speed gate.
