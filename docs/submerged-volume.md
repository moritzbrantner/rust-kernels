# Submerged volume and centroid

`primitive3::try_submerged_properties` and `primitive3::try_hull_submerged_properties`
return the volume and world-space centroid of a posed solid on the submerged
side `{x : dot(x - plane_point, plane_normal) <= 0}` of a plane. The normal points
out of the fluid and need not be unit length. All arithmetic is f64. Density,
buoyancy, drag and the fluid model belong to consumers.

| Shape | Method | Error |
| --- | --- | --- |
| Sphere | Closed-form spherical cap | Rounding |
| Box, wedge | Exact plane clip; boundary triangles summed as tetrahedra with an apex on the plane | Rounding |
| Convex hull (`Hull3`, `Polytope`) | Same clip over the hull's outward triangles | Rounding |
| Capsule hemispheres | Divergence theorem: `V = (r*S + h*W)/3`, first moment from the closed surface's vector area; `S` is the two-cap spherical area (Tovchigrechko & Vakser, 2001) | Rounding |
| Capsule cylinder | Circular-segment slices integrated with a fixed 16-point Gauss–Legendre rule in `psi = acos(abs(u))` between the breakpoints `u = +-1` | Rounding (mean slice area within `1e-13 r^2` of the closed form) |

The cylinder's slice integrands (area, axial weight, `(1-u^2)^(3/2)`) and the
Jacobian `sin(psi)` are trigonometric polynomials in `psi`, so the 16-point rule
is exact to rounding. Breakpoint cuts carry `u = +-1` exactly and negative offsets
are mirrored, so a planar normal one ulp from a breakpoint does not move the result.

A fully submerged primitive returns exactly the volume and posed centroid of
`try_volume_properties`. An empty result returns volume `0` and the pose position
as centroid. Axes must be orthonormal within `1e-9`. Non-finite pose or plane,
a zero normal, a hull that is not a closed polytope, non-finite hull vertices or
out-of-range triangle indices return `SubmergedError3::InvalidInput`; f64
overflow returns `SubmergedError3::NonFiniteComputation`.

Primitive queries are a fixed sequence of scalar operations. Hull queries are
linear in vertices plus triangles. Neither allocates; the allocation ratchet runs
5120 warmed mixed queries.

## Evidence

- Empty and full results for every primitive and four poses; full results equal
  `try_volume_properties` bit for bit.
- Volume is monotone and continuous across 2000 plane levels per shape, pose and
  normal (including a near-axis-parallel normal); each step is bounded by
  `pi * reach^2 * dz`.
- Primitives and an irregular 40-point hull match f64 midpoint voxel integration
  (72^3 and 80^3 cells) within 1% of the full volume and 1% of the radius; the
  residual is the voxel discretization.
- Capsules match an independent adaptive Simpson slice integral within `1e-9`
  of the full volume and `1e-8` of the length scale; zero-length capsules match
  the sphere cap within `1e-13`; a box hull matches the box primitive within `1e-12`.
- Symmetric bodies give centroids on the symmetry axis; mirrored tilts give
  mirrored centroids; repeated calls are bit-identical.

Run:

```sh
cargo test -p geometry-kernels --test submerged_volume
cargo test -p geometry-kernels --lib submerged
cargo test -p geometry-kernels --test allocation_regressions submerged
cargo bench -p geometry-kernels --bench primitive3 submerged_properties
```
