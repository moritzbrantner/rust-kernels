# f64 primitive segment rays

`primitive3::try_ray_cast` tests the complete finite segment
`origin + displacement * [0, 1]` against the current sphere, cuboid, capsule or
wedge. It returns the first occupied fraction, world point, outward surface
normal and stable local feature. Bodies remain caller-owned. Target velocity
does not move a snapshot target; no angular or time-dependent sweep is implied.

Initial touching is included even when moving away. A strictly interior origin
returns fraction zero, its original point, `Interior` identity and a normal
opposing displacement. That point is an interior witness rather than a surface
point. At surface edge ties, declared plane order is stable; capsule cap ties
retain negative then positive cap before the cylinder side. A zero-length
capsule uses the sphere identity. Feature indices are documented on the enum.

Zero displacement, non-finite inputs and invalid applicable axes return
`InvalidInput`. Non-finite intermediate/output or detected normalized dimension
underflow returns `NonFiniteComputation`, distinct from a geometric miss.
Shape constructors validate dimensions. Applicable axes are orthonormal;
sphere axes are irrelevant and capsule needs only local Y. No allocations or
iterative searches occur. Work counts every started query and visited quadratic
or clipping plane, including misses/errors.

Calculations normalize the local offset, displacement and dimensions together
before products. Curved tangency and initial boundary classification use a
32-machine-epsilon relative squared-distance band; polyhedral initial boundary
classification uses the corresponding projected-distance band. This rounding
band permits a nearby boundary witness and is not a collision skin. There is
no promise for arbitrary finite magnitudes or ill-conditioned dimension ratios.
Uniform scales from `1e-150` to `1e150` have pose/feature/normal controls.

Independent scalar occupancy and bisection validate 4096 seeded crossings.
Known cases cover capsule caps/side/axis-parallel rays, zero-length capsules,
inside/touching/behind, tangent/near miss, wedge slope and finite endpoint
coverage. Fixed work ceilings are one sphere or three capsule quadratics and
six box or five wedge planes. `cargo bench -p geometry-kernels --bench
ray_queries -- f64_primitive_segment_ray` measures complete kernel calls with
Divan; timings are advisory and do not describe a world query or game workload.
