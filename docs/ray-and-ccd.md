# Ray queries and continuous sphere sweeps

`geometry-kernels` now owns a small query layer for rays and first-contact sphere motion.

## Ray contract

`Ray3::new(origin, direction)` accepts finite f32 inputs and rejects the zero
vector. Direction magnitude is normalized robustly and is not semantically
significant. Query parameters are therefore **world-space distances**, not
arbitrary multiples of the input vector.

```rust
use geometry_kernels::{Ray3, Sphere, ray_sphere};

let ray = Ray3::new([-5.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
let hit = ray_sphere(ray, Sphere::new([0.0; 3], 1.0)).unwrap();
assert_eq!(hit.enter_distance, 4.0);
assert_eq!(hit.exit_distance, 6.0);
```

Both ray/sphere and ray/AABB return an interval. A ray that starts inside a
shape reports `enter_distance = 0` and `starts_inside = true`. Touching counts
as a hit. Intersections entirely behind the origin are not hits.

The AABB implementation uses the slab method. Tests compare it with independent
face-plane enumeration over generated fixtures. Sphere tests compare the
cross-product line-distance formulation with an independent quadratic oracle.
Direction rescaling from subnormal through maximum finite f32 values is covered.

## Ray and segment versus triangle

`ray_triangle(ray, [a, b, c], max_distance, culling)` returns the hit distance
(world space, like the other ray queries), point, barycentric weights of
`a`, `b`, `c` and the facing. `segment_triangle(start, end, triangle, culling)`
reports the normalized fraction between the endpoints instead.

```rust
use geometry_kernels::{Ray3, TriangleCulling, TriangleFacing, ray_triangle};

let triangle = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
let ray = Ray3::new([0.5, 0.5, 3.0], [0.0, 0.0, -1.0]);
let hit = ray_triangle(ray, triangle, f64::INFINITY, TriangleCulling::BackFaces).unwrap();
assert_eq!(hit.distance, 3.0);
assert_eq!(hit.barycentric, [0.5, 0.25, 0.25]);
assert_eq!(hit.facing, TriangleFacing::Front);
```

- The front face is the side the counter-clockwise normal `(b - a) × (c - a)`
  points to. `TriangleCulling::BackFaces` rejects back-face hits;
  `TriangleCulling::None` reports both with their facing.
- Edges, vertices, distance zero and `max_distance` (segment endpoints) are
  inclusive. The tests run in f64 after widening the f32 inputs and admit
  each quantity within a bound on its own rounding error before clamping,
  so exact boundary hits are not lost to rounding.
- Rays parallel to the plane, including coplanar rays crossing the triangle,
  and degenerate (zero-area) triangles never hit. A zero-length segment never
  hits.
- The queries do not allocate. Triangle iteration, BVH traversal and collision
  response stay with the consumer.

Tests compare 8,192 generated queries with an independent plane-intersection
and signed-area oracle (skipping inputs within rounding of an edge), cover
front/back, edge, vertex, parallel, coplanar, degenerate and bounded cases, and
check direction-scale and translation invariance.

## Continuous sphere/sphere contact

`sphere_sphere_time_of_impact` returns the earliest contact for one sphere
moving linearly against a stationary sphere during a caller-selected time
window. Starts-overlapping returns time zero even with zero velocity.

```rust
use geometry_kernels::{Sphere, sphere_sphere_time_of_impact};

let moving = Sphere::new([-5.0, 0.0, 0.0], 1.0);
let target = Sphere::new([0.0; 3], 1.0);
let hit = sphere_sphere_time_of_impact(moving, [2.0, 0.0, 0.0], target, 10.0).unwrap();
assert_eq!(hit.time, 1.5);
```

The kernel deliberately stops at time of impact. Collision response, impulses,
timestep advancement, filtering, broad-phase traversal and scene authority
remain consumer responsibilities. General convex CCD and mesh-level ray
traversal are separate follow-up decisions.

Generated tests compare 4,096 sweeps with a direct time-domain quadratic oracle
and cover velocity scaling, bounded windows, stationary/receding motion and
initial overlap.

## Cargo-native benchmarks

```sh
cargo bench -p geometry-kernels --bench ray_queries
```

The Divan suite measures deterministic batches of 64, 1,024 and 4,096 AABBs,
spheres, sphere sweeps and triangles (hit-heavy, mixed and miss-heavy). Input construction is outside the timed region and
every workload returns a checksum so the query loop cannot disappear. Raw
wall-clock measurements remain informational.

## Checked fixed-orientation primitive sweeps

`primitive3::try_swept_time(a, b, dt, margin, max_iterations, work)` supports
sphere, box, capsule and wedge pairs with fixed orientations and linear motion.
It returns normalized first-contact time in `[0, 1]`, or `Ok(None)` after a
separating-plane test proves a miss over the requested interval. Contact at the
initial pose returns zero. The non-negative margin expands contact admission.

Search failure provides no collision-absence conclusion: callers must handle
`InvalidInput`, `NonFiniteComputation` and `IterationLimit` separately. The limit
bounds conservative-advance iterations or swept-SAT axes, including a contact
query that admits a hit. Stationary curved pairs use a current-contact query
without search iterations; stationary polyhedral pairs still inspect SAT axes.
Work includes iterations performed before an error. A caller may retry with a
deliberate policy or return an error; treating it as a miss can tunnel through a
collider. No numerical fallback or automatic budget increase is provided.

The existing `swept_time` Option API remains a compatibility wrapper using 128
iterations. It cannot distinguish failure from a miss. Collision-authoritative
consumers should use the checked API. Finite inputs do not guarantee that every
intermediate remains finite: for example, uniformly scaling an analytic capsule/
sphere hit by `1e160` overflows the current squared-distance calculation. The
checked query reports this failure rather than claiming collision absence.
Angular trajectories and general convex continuous collision are outside this
fixed-orientation contract.
