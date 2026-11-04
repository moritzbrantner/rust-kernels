# Experimental ellipsoid primitive

`primitive3::ellipsoid` is an experimental f64 kernel for a triaxial ellipsoid
`E = center + L * unit_ball`, `L = R * diag(a, b, c)`. It is not a
`PrimitiveShape3` kind, does not extend the primitive pair matrix and is not
consumed by Physics Engine. Consumers own mass/density policy, response and CCD
admission.

## Shape, pose and errors (#189)

`Ellipsoid3::try_new([a, b, c])` rejects non-finite (`NonFiniteSemiaxis`) and
zero or negative (`NonPositiveSemiaxis`) semiaxes. `EllipsoidPose3::try_new`
holds the center and world-space local axes (the columns of `R`) separately and
rejects non-finite input and axes whose Gram matrix differs from the identity by
more than `ELLIPSOID_AXES_TOLERANCE = 1e-10` (`NonOrthonormalAxes`). A zero
support direction or halfspace normal is `ZeroDirection`; an overflowing or
underflowing intermediate is `Unrepresentable`. Nothing is clamped.

| Query | Closed form |
| --- | --- |
| `try_support_point(d)` | `center + L Lᵀ d / ‖Lᵀ d‖` (positive-scale invariant) |
| `try_support_radius(d)` | `‖Lᵀ d̂‖` |
| `try_aabb_half_extents()` | component `i` is `‖row_i(L)‖`, exact for the pose |
| `Ellipsoid3::outer_radius` / `inner_radius` | `max(a,b,c)` / `min(a,b,c)` |
| `EllipsoidBody3::outer_radius` / `inner_radius` | the same, widened/narrowed by the accepted axis slack (`1 + 2τ`, `1 - 4τ`) |
| `try_halfspace_sweep(Δ, p, n)` | fixed-orientation translational first touch with `{x : (x-p)·n <= 0}` |
| `try_pose_displacement_bound` | `‖Δc‖ + ‖(R₁-R₀) diag(a,b,c)‖_F`, rounded up |
| `try_rotational_displacement_bound` | `‖Δc‖ + min(θ, 2) * outer`, rounded up |
| `try_inertia_diagonal(m)` | `[m(b²+c²)/5, m(a²+c²)/5, m(a²+b²)/5]` |

The halfspace sweep is exact up to f64 rounding because the gap is affine in the
fraction under fixed orientation; the normal need not be unit length. Rotating
motion is not swept: the two displacement certificates bound how far any
material point can move, so a future conservative advancement may step by at
most `gap / bound` of the interval. `EllipsoidBody3` also implements
`SupportMap3` for the generic GJK/EPA kernels (a zero direction maps to +X as for
the other support maps). There is no arbitrary ellipsoid-pair quadratic time of
impact.

Independent evidence (`tests/ellipsoid_primitive.rs`) maps a 96 × 192
latitude/longitude unit-sphere grid through `L`. Its hull is inscribed, so each
analytic support/extent/separation must be at least as far as the sampled value
and within `SAMPLE_TOLERANCE = 1.1e-3` of the support radius (twice
`1 - cos` of the grid spacing). Fixtures cover 1:1:1, 1.5:1:1, 4:1:1, 0.3:2:0.7
and 50:0.05:1 shapes under seeded rotations, radius containment including axes at
the orthonormality tolerance, orientation invariance, 1e-150..1e150 scale
covariance, tangent/receding/overlapping halfspace starts, both displacement
certificates against every sampled material point, and inertia against
64³-cell midpoint integration.

## Measurements (#189)

Divan, `cargo bench -p geometry-kernels --bench primitive3 -- ellipsoid`,
rustc 1.99.1, default features, bench profile, 4-vCPU Intel Xeon @ 2.10 GHz
(shared cloud runner). Each sample is a batch of 64 calls over varied
directions/rotations/displacements built outside the timed region; the table
gives the median per call. Timing is advisory developer evidence.

| Kernel | 1:1:1 | 1.5:1:1 | 4:1:1 | adverse 1e4:1:1e-3 | Existing kernels |
| --- | --- | --- | --- | --- | --- |
| Support point | 38.2 ns | 38.6 ns | 38.3 ns | 38.3 ns | sphere 13.4, capsule 14.1, box 17.4 ns |
| AABB extents | 10.1 ns | 10.2 ns | 10.2 ns | 10.2 ns | sphere 1.2, capsule 1.8, box 2.6 ns |
| Halfspace cast | 20.9 ns | 21.8 ns | 21.6 ns | 21.7 ns | sphere/capsule halfspace 11.7 ns, box vs slab swept SAT 1171 ns |

Cost does not depend on aspect ratio. The support map is about 3× a sphere or
capsule support (one square root and one reciprocal plus a 3×3 transform); the
halfspace cast is under 2× the capsule halfspace closed form and far below the
nearest box equivalent.

**Conclusion (#189): promote experimental kernel.** Keep `primitive3::ellipsoid`
as a public experimental module: it is exact against the sampled reference,
explicit about every failure and constant-cost across aspect ratios. Do not add it
to `PrimitiveShape3` or Physics Engine yet. Smallest follow-up: a measured
ellipsoid pair query through the existing GJK/EPA kernels on `SupportMap3`, only
if a consumer capability needs it.

Focused commands:

```sh
cargo test -p geometry-kernels --test ellipsoid_primitive
cargo bench -p geometry-kernels --bench primitive3 -- ellipsoid
```
