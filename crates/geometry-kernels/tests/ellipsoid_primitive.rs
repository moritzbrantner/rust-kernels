//! Experimental ellipsoid primitive (#189) against an independent sampled-surface reference.
//!
//! The reference tessellates the unit sphere on a latitude/longitude grid and maps it through
//! `L = R diag(a,b,c)`. Its hull is inscribed in the ellipsoid, so every analytic support
//! value must be at least the sampled one, and at most `SAMPLE_TOLERANCE` (relative to the
//! support radius) larger, where `SAMPLE_TOLERANCE >= 1 - cos(grid spacing)`.

use geometry_kernels::math3::{Vec3, add, dot, length, scale, sub};
use geometry_kernels::primitive3::ellipsoid::{
    ELLIPSOID_AXES_TOLERANCE, Ellipsoid3, EllipsoidBody3, EllipsoidError3, EllipsoidPose3,
    try_pose_displacement_bound, try_rotational_displacement_bound,
};

const RINGS: usize = 96;
const SEGMENTS: usize = 192;
/// `1 - cos(delta)` for the grid's largest angular gap (`delta < 2*pi/SEGMENTS`), doubled.
const SAMPLE_TOLERANCE: f64 = 1.1e-3;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next()
    }

    fn unit(&mut self) -> Vec3 {
        loop {
            let v = [
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            ];
            let l = length(v);
            if l > 0.1 && l <= 1.0 {
                return scale(v, 1.0 / l);
            }
        }
    }

    fn axes(&mut self) -> [Vec3; 3] {
        let w = self.range(-1.0, 1.0);
        quaternion_axes(w, scale(self.unit(), self.next()))
    }
}

fn quaternion_axes(w: f64, v: Vec3) -> [Vec3; 3] {
    let n = (w * w + dot(v, v)).sqrt();
    let (w, x, y, z) = (w / n, v[0] / n, v[1] / n, v[2] / n);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        [
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ],
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn ellipsoid(semiaxes: Vec3, center: Vec3, axes: [Vec3; 3]) -> EllipsoidBody3 {
    EllipsoidBody3::new(
        Ellipsoid3::try_new(semiaxes).unwrap(),
        EllipsoidPose3::try_new(center, axes).unwrap(),
    )
}

fn map(body: EllipsoidBody3, u: Vec3) -> Vec3 {
    let s = body.shape.semiaxes();
    let axes = body.pose.axes();
    let mut point = body.pose.center();
    for index in 0..3 {
        point = add(point, scale(axes[index], s[index] * u[index]));
    }
    point
}

fn unit_samples() -> Vec<Vec3> {
    let mut samples = vec![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
    for ring in 1..RINGS {
        let polar = std::f64::consts::PI * ring as f64 / RINGS as f64;
        for segment in 0..SEGMENTS {
            let azimuth = std::f64::consts::TAU * segment as f64 / SEGMENTS as f64;
            samples.push([
                polar.sin() * azimuth.cos(),
                polar.sin() * azimuth.sin(),
                polar.cos(),
            ]);
        }
    }
    samples
}

fn surface(body: EllipsoidBody3, samples: &[Vec3]) -> Vec<Vec3> {
    samples.iter().map(|&u| map(body, u)).collect()
}

/// `‖L⁻¹(x - c)‖` with `L⁻¹ = diag(1/a,1/b,1/c) Rᵀ` for orthonormal axes.
fn implicit(body: EllipsoidBody3, point: Vec3) -> f64 {
    let s = body.shape.semiaxes();
    let r = sub(point, body.pose.center());
    let axes = body.pose.axes();
    length(std::array::from_fn(|i| dot(axes[i], r) / s[i]))
}

const SHAPES: [Vec3; 5] = [
    [1.0, 1.0, 1.0],
    [1.5, 1.0, 1.0],
    [4.0, 1.0, 1.0],
    [0.3, 2.0, 0.7],
    [50.0, 0.05, 1.0],
];

#[test]
fn invalid_shapes_poses_and_directions_are_explicit_errors() {
    use EllipsoidError3::*;
    assert_eq!(
        Ellipsoid3::try_new([1.0, f64::NAN, 1.0]),
        Err(NonFiniteSemiaxis)
    );
    assert_eq!(
        Ellipsoid3::try_new([1.0, f64::INFINITY, 1.0]),
        Err(NonFiniteSemiaxis)
    );
    assert_eq!(
        Ellipsoid3::try_new([1.0, 0.0, 1.0]),
        Err(NonPositiveSemiaxis)
    );
    assert_eq!(
        Ellipsoid3::try_new([1.0, 1.0, -2.0]),
        Err(NonPositiveSemiaxis)
    );
    assert_eq!(
        EllipsoidPose3::try_axis_aligned([f64::NAN, 0.0, 0.0]),
        Err(NonFiniteInput)
    );
    let skew = [[1.0, 0.0, 0.0], [1.0e-6, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert_eq!(
        EllipsoidPose3::try_new([0.0; 3], skew),
        Err(NonOrthonormalAxes)
    );
    let stretched = [[1.0 + 1.0e-6, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert_eq!(
        EllipsoidPose3::try_new([0.0; 3], stretched),
        Err(NonOrthonormalAxes)
    );
    let body = ellipsoid([1.0, 2.0, 3.0], [0.0; 3], Rng(7).axes());
    assert_eq!(body.try_support_point([0.0; 3]), Err(ZeroDirection));
    assert_eq!(body.try_support_radius([0.0; 3]), Err(ZeroDirection));
    assert_eq!(
        body.try_support_point([f64::NAN, 0.0, 0.0]),
        Err(NonFiniteInput)
    );
    assert_eq!(
        body.try_halfspace_contact([0.0; 3], [0.0; 3]),
        Err(ZeroDirection)
    );
    assert_eq!(
        body.try_halfspace_sweep([f64::INFINITY, 0.0, 0.0], [0.0; 3], [0.0, 1.0, 0.0]),
        Err(NonFiniteInput)
    );
    assert_eq!(body.shape.try_inertia_diagonal(0.0), Err(InvalidParameter));
    assert_eq!(
        body.shape.try_inertia_diagonal(f64::NAN),
        Err(NonFiniteInput)
    );
    assert_eq!(
        try_rotational_displacement_bound(body.shape, [0.0; 3], -1.0),
        Err(InvalidParameter)
    );
    let huge = Ellipsoid3::try_new([1e300, 1e300, 1e300]).unwrap();
    assert_eq!(huge.try_volume(), Err(Unrepresentable));
    assert_eq!(huge.try_inertia_diagonal(1e300), Err(Unrepresentable));
    let tiny = Ellipsoid3::try_new([1e-200, 1e-200, 1e-200]).unwrap();
    assert_eq!(tiny.try_volume(), Err(Unrepresentable));
}

#[test]
fn support_map_matches_sampled_surface_and_lies_on_the_ellipsoid() {
    let samples = unit_samples();
    let mut rng = Rng(0x189);
    for semiaxes in SHAPES {
        for _ in 0..16 {
            let center = [
                rng.range(-5.0, 5.0),
                rng.range(-5.0, 5.0),
                rng.range(-5.0, 5.0),
            ];
            let body = ellipsoid(semiaxes, center, rng.axes());
            let points = surface(body, &samples);
            for _ in 0..32 {
                let direction = scale(rng.unit(), rng.range(1e-3, 1e3));
                let unit = scale(direction, 1.0 / length(direction));
                let support = body.try_support_point(direction).unwrap();
                let radius = body.try_support_radius(direction).unwrap();
                let sampled = points
                    .iter()
                    .map(|&p| dot(sub(p, center), unit))
                    .fold(f64::NEG_INFINITY, f64::max);
                assert!((implicit(body, support) - 1.0).abs() <= 1e-12);
                assert!((dot(sub(support, center), unit) - radius).abs() <= 1e-12 * radius);
                assert!(radius >= sampled - 1e-12 * radius, "{radius} < {sampled}");
                assert!(radius - sampled <= SAMPLE_TOLERANCE * radius);
                // Positive scaling of the direction does not move the support point.
                let rescaled = body.try_support_point(scale(direction, 1e5)).unwrap();
                assert!(length(sub(rescaled, support)) <= 1e-12 * body.shape.outer_radius());
            }
        }
    }
}

#[test]
fn aabb_extents_are_exact_row_norms_and_contain_the_surface() {
    let samples = unit_samples();
    let mut rng = Rng(0xaabb);
    for semiaxes in SHAPES {
        for _ in 0..16 {
            let body = ellipsoid(semiaxes, [1.0, -2.0, 3.0], rng.axes());
            let half = body.try_aabb_half_extents().unwrap();
            let (min, max) = body.try_aabb().unwrap();
            let points = surface(body, &samples);
            for axis in 0..3 {
                let mut e = [0.0; 3];
                e[axis] = 1.0;
                let support = body.try_support_radius(e).unwrap();
                assert!((half[axis] - support).abs() <= 1e-14 * support);
                let sampled = points
                    .iter()
                    .map(|p| (p[axis] - body.pose.center()[axis]).abs())
                    .fold(0.0, f64::max);
                assert!(half[axis] >= sampled - 1e-12 * half[axis]);
                assert!(half[axis] - sampled <= SAMPLE_TOLERANCE * half[axis]);
                assert!(
                    points
                        .iter()
                        .all(|p| p[axis] >= min[axis] - 1e-12 * half[axis]
                            && p[axis] <= max[axis] + 1e-12 * half[axis])
                );
            }
        }
    }
}

#[test]
fn radius_bounds_contain_the_surface_and_are_orientation_invariant() {
    let samples = unit_samples();
    let mut rng = Rng(0x5ad);
    for semiaxes in SHAPES {
        let shape = Ellipsoid3::try_new(semiaxes).unwrap();
        let outer = semiaxes.into_iter().fold(0.0, f64::max);
        let inner = semiaxes.into_iter().fold(f64::INFINITY, f64::min);
        assert_eq!((shape.outer_radius(), shape.inner_radius()), (outer, inner));
        let reference = ellipsoid(semiaxes, [0.0; 3], quaternion_axes(1.0, [0.0; 3]));
        for _ in 0..16 {
            let axes = rng.axes();
            let body = ellipsoid(semiaxes, [3.0, 1.0, -4.0], axes);
            assert!(body.outer_radius() >= outer && body.inner_radius() <= inner);
            for &u in &samples {
                let distance = length(sub(map(body, u), body.pose.center()));
                assert!(distance <= body.outer_radius() && distance >= body.inner_radius());
            }
            // Rotating the body and the direction together preserves the support radius.
            let local = rng.unit();
            let world = add(
                add(scale(axes[0], local[0]), scale(axes[1], local[1])),
                scale(axes[2], local[2]),
            );
            let rotated = body.try_support_radius(world).unwrap();
            let unrotated = reference.try_support_radius(local).unwrap();
            assert!((rotated - unrotated).abs() <= 1e-13 * outer);
        }
        // The extremal support radii are the outer and inner semiaxes.
        let body = ellipsoid(semiaxes, [0.0; 3], rng.axes());
        let axes = body.pose.axes();
        let radii = axes.map(|axis| body.try_support_radius(axis).unwrap());
        for (radius, semiaxis) in radii.into_iter().zip(semiaxes) {
            assert!((radius - semiaxis).abs() <= 1e-13 * outer);
        }
    }
}

#[test]
fn axes_near_the_tolerance_keep_radius_bounds_conservative() {
    // Columns scaled by (1 ± τ/2.5) stay accepted; sampled surface points must stay inside the
    // pose-level bounds although `max`/`min` alone would be violated.
    let scale_up = 1.0 + ELLIPSOID_AXES_TOLERANCE / 2.5;
    let axes = [
        [scale_up, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0 - scale_up],
    ];
    let body = ellipsoid([3.0, 2.0, 1.0], [0.0; 3], axes);
    for u in unit_samples() {
        let distance = length(sub(map(body, u), [0.0; 3]));
        assert!(distance <= body.outer_radius() && distance >= body.inner_radius());
    }
}

#[test]
fn halfspace_sweep_matches_sampled_reference() {
    let samples = unit_samples();
    let mut rng = Rng(0x9a1e);
    let mut hits = 0;
    for semiaxes in SHAPES {
        for _ in 0..64 {
            let body = ellipsoid(
                semiaxes,
                [rng.range(-5.0, 5.0), rng.range(2.0, 60.0), 0.0],
                rng.axes(),
            );
            let normal = scale(add([0.0, 1.0, 0.0], scale(rng.unit(), 0.3)), 3.0);
            let unit = scale(normal, 1.0 / length(normal));
            let plane = [rng.range(-1.0, 1.0), 0.0, rng.range(-1.0, 1.0)];
            let displacement = scale(rng.unit(), rng.range(1.0, 200.0));
            let points = surface(body, &samples);
            let sampled = points
                .iter()
                .map(|&p| dot(sub(p, plane), unit))
                .fold(f64::INFINITY, f64::min);
            let contact = body.try_halfspace_contact(plane, normal).unwrap();
            let radius = body.try_support_radius(unit).unwrap();
            assert!(contact.separation <= sampled + 1e-12 * radius);
            assert!(sampled - contact.separation <= SAMPLE_TOLERANCE * radius);
            let rate = dot(displacement, unit);
            let hit = body
                .try_halfspace_sweep(displacement, plane, normal)
                .unwrap();
            if contact.separation <= 0.0 {
                assert_eq!(hit.unwrap().fraction, 0.0);
                continue;
            }
            // The inscribed reference touches no earlier than the ellipsoid.
            let reference = (rate < 0.0).then(|| sampled / -rate);
            match hit {
                Some(hit) => {
                    hits += 1;
                    let reference = reference.unwrap();
                    let fraction_tolerance = SAMPLE_TOLERANCE * radius / -rate;
                    assert!(hit.fraction <= reference + 1e-12);
                    assert!(reference - hit.fraction <= fraction_tolerance + 1e-12);
                    assert_eq!(hit.contact.separation, 0.0);
                    assert!(dot(sub(hit.contact.point, plane), unit).abs() <= 1e-10 * radius);
                    let moved = EllipsoidBody3::new(
                        body.shape,
                        body.pose
                            .try_with_center(add(
                                body.pose.center(),
                                scale(displacement, hit.fraction),
                            ))
                            .unwrap(),
                    );
                    assert!((implicit(moved, hit.contact.point) - 1.0).abs() <= 1e-10);
                }
                None => assert!(
                    reference.is_none_or(|t| t > 1.0),
                    "miss but sampled hull hits at {reference:?}"
                ),
            }
        }
    }
    assert!(hits > 40, "only {hits} moving hits exercised");
}

#[test]
fn halfspace_sweep_handles_receding_tangent_and_overlapping_starts() {
    let body = ellipsoid(
        [4.0, 1.0, 1.0],
        [0.0, 5.0, 0.0],
        quaternion_axes(1.0, [0.0; 3]),
    );
    let up = [0.0, 1.0, 0.0];
    assert_eq!(
        body.try_halfspace_sweep([0.0, 3.0, 0.0], [0.0; 3], up),
        Ok(None)
    );
    assert_eq!(
        body.try_halfspace_sweep([9.0, 0.0, 0.0], [0.0; 3], up),
        Ok(None)
    );
    assert_eq!(
        body.try_halfspace_sweep([0.0, -3.9, 0.0], [0.0; 3], up),
        Ok(None)
    );
    let exact = body
        .try_halfspace_sweep([0.0, -4.0, 0.0], [0.0; 3], up)
        .unwrap()
        .unwrap();
    assert_eq!(exact.fraction, 1.0);
    assert_eq!(exact.contact.point, [0.0, 0.0, 0.0]);
    let touching = body
        .try_halfspace_sweep([1.0, 0.0, 0.0], [0.0, 4.0, 0.0], up)
        .unwrap()
        .unwrap();
    assert_eq!((touching.fraction, touching.contact.separation), (0.0, 0.0));
    let inside = body
        .try_halfspace_sweep([0.0, 1.0, 0.0], [0.0, 4.5, 0.0], up)
        .unwrap()
        .unwrap();
    assert_eq!(inside.fraction, 0.0);
    assert_eq!(inside.contact.separation, -0.5);
    // A 4:1:1 body tipped onto its long axis reaches the floor four units earlier.
    let tipped = ellipsoid(
        [4.0, 1.0, 1.0],
        [0.0, 5.0, 0.0],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    );
    let hit = tipped
        .try_halfspace_sweep([0.0, -2.0, 0.0], [0.0; 3], up)
        .unwrap()
        .unwrap();
    assert_eq!(hit.fraction, 0.5);
}

#[test]
fn extreme_scales_are_covariant() {
    let mut rng = Rng(0xe7);
    let axes = rng.axes();
    let base = ellipsoid([4.0, 1.0, 0.25], [0.5, 9.0, -1.0], axes);
    let direction = rng.unit();
    let up = [0.0, 1.0, 0.0];
    let displacement = [3.0, -20.0, 1.0];
    let base_support = base.try_support_point(direction).unwrap();
    let base_half = base.try_aabb_half_extents().unwrap();
    let base_hit = base
        .try_halfspace_sweep(displacement, [0.0; 3], up)
        .unwrap()
        .unwrap();
    for k in [1e-150, 1e-30, 1e30, 1e150] {
        let body = ellipsoid(
            scale(base.shape.semiaxes(), k),
            scale(base.pose.center(), k),
            axes,
        );
        let support = body.try_support_point(direction).unwrap();
        assert!(length(sub(scale(support, 1.0 / k), base_support)) <= 1e-13);
        let half = body.try_aabb_half_extents().unwrap();
        assert!(length(sub(scale(half, 1.0 / k), base_half)) <= 1e-13);
        let hit = body
            .try_halfspace_sweep(scale(displacement, k), [0.0; 3], up)
            .unwrap()
            .unwrap();
        assert!((hit.fraction - base_hit.fraction).abs() <= 1e-13);
    }
}

#[test]
fn displacement_certificates_bound_every_sampled_material_point() {
    let samples = unit_samples();
    let mut rng = Rng(0xca);
    for semiaxes in SHAPES {
        let shape = Ellipsoid3::try_new(semiaxes).unwrap();
        for _ in 0..24 {
            let axes0 = rng.axes();
            let axes1 = if rng.next() < 0.2 { axes0 } else { rng.axes() };
            let c0 = [rng.range(-3.0, 3.0), 0.0, rng.range(-3.0, 3.0)];
            let c1 = add(c0, scale(rng.unit(), rng.range(0.0, 5.0)));
            let from = ellipsoid(semiaxes, c0, axes0);
            let to = ellipsoid(semiaxes, c1, axes1);
            let pose_bound = try_pose_displacement_bound(shape, from.pose, to.pose).unwrap();
            // Rotation angle of R₁R₀ᵀ: trace = Σ_i axes1_i · axes0_i.
            let trace: f64 = (0..3).map(|i| dot(axes1[i], axes0[i])).sum();
            let angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos() + 1e-12;
            let angle_bound = try_rotational_displacement_bound(shape, sub(c1, c0), angle).unwrap();
            let mut worst: f64 = 0.0;
            for &u in &samples {
                worst = worst.max(length(sub(map(to, u), map(from, u))));
            }
            // The sampled displacement carries its own position rounding (~ε·|x|).
            worst -= 1e-14 * (length(c0) + length(c1) + shape.outer_radius());
            assert!(worst <= pose_bound, "{worst} > {pose_bound}");
            assert!(worst <= angle_bound, "{worst} > {angle_bound}");
            // Pure translation is certified exactly.
            if axes0 == axes1 {
                assert!(pose_bound - length(sub(c1, c0)) <= 1e-14 * pose_bound.max(1.0));
            }
        }
    }
}

#[test]
fn inertia_is_anisotropic_and_matches_grid_integration() {
    for semiaxes in [
        [1.0, 1.0, 1.0],
        [1.5, 1.0, 1.0],
        [4.0, 1.0, 1.0],
        [0.3, 2.0, 0.7],
    ] {
        let shape = Ellipsoid3::try_new(semiaxes).unwrap();
        let mass = 3.5;
        let inertia = shape.try_inertia_diagonal(mass).unwrap();
        let volume = shape.try_volume().unwrap();
        let [a, b, c] = semiaxes;
        assert!((volume - 4.0 / 3.0 * std::f64::consts::PI * a * b * c).abs() <= 1e-14 * volume);
        // Midpoint grid integration of the solid (independent of the closed form).
        let n = 64;
        let mut count = 0.0;
        let mut second = [0.0; 3];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let u = [i, j, k].map(|v| -1.0 + (2.0 * f64::from(v) + 1.0) / f64::from(n));
                    if dot(u, u) <= 1.0 {
                        count += 1.0;
                        for axis in 0..3 {
                            second[axis] += (u[axis] * semiaxes[axis]).powi(2);
                        }
                    }
                }
            }
        }
        let moments = second.map(|value| mass * value / count);
        let expected = [
            moments[1] + moments[2],
            moments[0] + moments[2],
            moments[0] + moments[1],
        ];
        for axis in 0..3 {
            assert!(
                (inertia[axis] - expected[axis]).abs() <= 5e-3 * expected[axis],
                "{inertia:?} vs {expected:?}"
            );
        }
        assert!((inertia[0] - mass * (b * b + c * c) / 5.0).abs() <= 1e-14 * inertia[0]);
        if a != b {
            assert!(inertia[0] != inertia[2]);
        }
    }
    let sphere = Ellipsoid3::try_new([2.0; 3]).unwrap();
    assert_eq!(sphere.try_inertia_diagonal(5.0).unwrap(), [8.0; 3]);
}

#[test]
fn generic_support_map_agrees_with_the_checked_support() {
    use geometry_kernels::support::SupportMap3;
    let body = ellipsoid([2.0, 0.5, 1.0], [1.0, 2.0, 3.0], Rng(3).axes());
    let direction = [0.3, -0.2, 0.9];
    assert_eq!(
        SupportMap3::support_point(&body, direction),
        body.try_support_point(direction).unwrap()
    );
    assert_eq!(
        SupportMap3::support_point(&body, [0.0; 3]),
        body.try_support_point([1.0, 0.0, 0.0]).unwrap()
    );
}
