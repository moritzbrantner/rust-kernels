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
remain consumer responsibilities. General convex CCD and triangle/mesh ray
queries are separate follow-up decisions.

Generated tests compare 4,096 sweeps with a direct time-domain quadratic oracle
and cover velocity scaling, bounded windows, stationary/receding motion and
initial overlap.

## Cargo-native benchmarks

```sh
cargo bench -p geometry-kernels --bench ray_queries
```

The Divan suite measures deterministic batches of 64, 1,024 and 4,096 AABBs,
spheres and sphere sweeps. Input construction is outside the timed region and
every workload returns a checksum so the query loop cannot disappear. Raw
wall-clock measurements remain informational.
