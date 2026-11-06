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

Preparation owns all retained allocations. `preparation()` reports triangle/node
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
