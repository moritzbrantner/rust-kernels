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
| Support point | 39.0 ns | 38.9 ns | 38.9 ns | 38.8 ns | sphere 12.8, capsule 14.2, box 17.9 ns |
| AABB extents | 9.7 ns | 9.7 ns | 9.7 ns | 9.7 ns | sphere 0.9, capsule 1.5, box 2.6 ns |
| Halfspace cast | 20.8 ns | 21.8 ns | 21.9 ns | 21.7 ns | sphere/capsule halfspace 11.4 ns, box vs slab swept SAT 1264 ns |

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

## Shared-transform homothetic pair cast (#190)

Two ellipsoids `cA + sA L ball` and `cB + sB L ball` sharing one transform
`L = R diag(a,b,c)` map under `L⁻¹` to spheres of radii `sA` and `sB`; with fixed
orientations and constant velocities the relative center moves linearly there,
so first contact is the sphere root of `‖p' + v' t‖ = sA + sB`. The cast reuses
the crate's checked sphere–sphere fraction (`continuous`, from #192) rather than
another quadratic solver.

- `EllipsoidSharedFrame3::try_new(base, axes)` precomputes the rows of
  `L⁻¹ = diag(1/a,1/b,1/c) R⁻¹` with the exact adjugate inverse of the supplied
  axes (not `Rᵀ`). `try_cast(a, b, dt)` takes `ScaledEllipsoidMotion3 { center,
  velocity, scale }` for each body.
- `try_homothetic_pair_cast(a, va, b, vb, dt)` accepts two posed
  `EllipsoidBody3` only when the axes are bitwise identical and the semiaxes are
  exactly proportional, decided by comparing exact double-length products
  (`mul_add` error terms) after exact power-of-two normalization. Everything else
  is `EllipsoidPairCast3::Unsupported(DifferentOrientation |
  NonHomotheticShapes | UnverifiableProportion)`; there is no tolerance-based
  acceptance, and f64 semiaxes that are only nominally proportional (`0.1` vs
  `0.1 * 3.0`) are rejected. Callers with a known shared base use the frame API.
- Results are `InitiallyOverlapping` (strict interior overlap at the start),
  `Hit { fraction, contact }` (including a touching start at `0`) or `Miss`.
  The witness is `cA(t) + L (sA u')` and the normal is the normalized
  `L⁻ᵀ u'`, the outward normal of A towards B; spherical impulse or inertia in
  transformed coordinates is not provided.
- The sphere-space contact radius is widened by a bound on the transform's
  rounding (`16ε` times the absolute row/vector products), so grazing paths
  within rounding report contact slightly early instead of being missed. The
  widening scales with the frame's axis-ratio condition number: a 1e8 needle
  reports contact up to about `1e-6` early in the fixtures. A later f32 consumer
  must still narrow conservatively at its own boundary.

Out of scope: differently shaped or oriented pairs, angular motion and margins
(an offset ellipsoid is not an ellipsoid). **The generic ellipsoid pair
fallback is absent**: unsupported pairs carry no collision conclusion.

Evidence (`tests/ellipsoid_pair_ccd.rs`): analytic head-on, diagonal, overlapping,
stationary, separating, short and crossing paths; 32 seeded exact tangencies
(hit near `t = 1/2`) with `1 ± 1e-9` offsets (hit/miss); near-zero displacements;
1e-140..1e140 scale covariance through the checked entry; a 1e4:1:1e-4 needle;
inverse-transpose normals; and 600 seeded pairs over 1:1:1, 1.5:1:1, 4:1:1,
0.3:2:0.7 and 10:1:0.2 frames (a quarter at 400 combined sizes per 1/60 s step)
against an independent reference: conservative advancement on GJK distance over
the analytic `SupportMap3`, stepping by the separating-plane lower bound
(`tests/support/ellipsoid_reference.rs`). The last run had 445 hits (109
high-speed) and 155 misses with identical classification, no unresolved
reference search, and hit times within `200 * tolerance / closing` (worst
observed 89). Every hit witness satisfies both independent implicit equations
`‖diag(1/s) Rᵀ (x - c)‖ = 1` and has opposite surface normals.

### Measurements (#190)

Same fingerprint as above; `cargo bench -p geometry-kernels --bench primitive3 --
ellipsoid_pair`. Each sample casts 64 hitting fixtures (B at 20 combined sizes,
closing within one 1/60 s step, rotated frames); median per call:

| Cast | 1:1:1 | 1.5:1:1 | 4:1:1 | adverse 1e4:1:1e-3 |
| --- | --- | --- | --- | --- |
| `EllipsoidSharedFrame3::try_cast` (prepared) | 166 ns | 171 ns | 166 ns | 166 ns |
| `try_homothetic_pair_cast` (checks + frame) | 190 ns | 191 ns | 191 ns | 191 ns |
| Generic CA/GJK reference | 36.3 µs | 50.8 µs | 54.7 µs | 50.4 µs |

For comparison the existing analytic `fast_sphere_sphere_sweep` call through
`try_swept_time` measured 92 ns in the same run. The exact precondition checks and
frame construction add about 25 ns. The
specialized cast is about 220–330× cheaper than the generic support-based search
on the same fixtures and does not depend on aspect ratio.

**Conclusion (#190): retain experimental.** The cast is exact in its declared
domain, explicit outside it and far cheaper than the generic path, but no
consumer needs ellipsoid pairs and the generic fallback that unsupported pairs
would require does not exist. Promotion requires a distinct consumer/parity
issue; this change does not expand the Physics Engine API.

```sh
cargo test -p geometry-kernels --test ellipsoid_pair_ccd
cargo bench -p geometry-kernels --bench primitive3 -- ellipsoid_pair
```
