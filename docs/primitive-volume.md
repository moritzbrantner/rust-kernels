# Primitive volume products

`primitive3::try_volume_properties` computes uniform-solid volume, local centroid,
and the normalized second central moment matrix for the existing sphere, box,
local-Y capsule and wedge shapes. All products use f64 and the constructors' units.
The moment is `integral((x-c)(x-c)^T dV)/V`, with squared length units; rows and
columns follow local X/Y/Z. Consumers own density, mass, physical inertia, world
poses and response. A wedge's bounding-box origin is distinct from its geometric
centroid `[-half_x/3, -half_y/3, 0]`.

| Shape | Volume | Centroid |
| --- | --- | --- |
| Sphere | `4*pi*r^3/3` | `[0,0,0]` |
| Box | `8*half_x*half_y*half_z` | `[0,0,0]` |
| Capsule | `pi*r^2*(2*half_segment+4*r/3)` | `[0,0,0]` |
| Wedge | `4*half_x*half_y*half_z` | `[-half_x/3,-half_y/3,0]` |

The matrix includes the wedge's negative XY central product. Capsule products
integrate both hemispheres about the complete solid's centroid; zero skeleton
length gives exactly the sphere limit. Direct products outside the f64 arithmetic
budget return `PrimitiveVolumeError3::Unrepresentable`, including overflow and
underflow of positive volume or diagonal moments. This API promises no universal
magnitude range for the existing constructors' finite positive dimensions.

Each query uses a fixed set of scalar operations with no iteration or scratch.
The inline result occupies 104 bytes. Cargo-native allocation instrumentation
checks 4096 warmed mixed queries with zero allocation/reallocation calls.
Independent tests integrate capsule circular slices with three-point
Gauss-Legendre quadrature and wedges through three tetrahedra. Fixtures include
1e-6 through 1e12 dimensional scales, sphere limits and scale covariance; explicit
relative tolerances compare integral products. Contact-pair differential coverage
is a separate contract and remains partial in the verification matrix.

Run:

```sh
cargo test -p geometry-kernels --test primitive_volume
cargo test -p geometry-kernels --test allocation_regressions
cargo bench -p geometry-kernels --bench primitive3 primitive_volume_properties
```

The Divan workload measures four complete calls per sample. Timing is developer
evidence; the Cargo-native allocation assertion is the resource contract.
