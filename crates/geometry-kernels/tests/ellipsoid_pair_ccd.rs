//! Shared-transform homothetic ellipsoid pair TOI (#190) against analytic fixtures and an
//! independent generic reference (conservative advancement over GJK on support maps).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration tests are not production code; a panic is the intended failure report"
)]

mod support;

use geometry_kernels::math3::{Vec3, add, cross, dot, length, scale, sub};
use geometry_kernels::primitive3::ellipsoid::{
    Ellipsoid3, EllipsoidBody3, EllipsoidError3, EllipsoidPairCast3, EllipsoidPairContact3,
    EllipsoidPairToi3, EllipsoidPairUnsupported3, EllipsoidPose3, EllipsoidSharedFrame3,
    ScaledEllipsoidMotion3, try_homothetic_pair_cast,
};
use support::ellipsoid_reference::{ReferenceToi, conservative_advancement};

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

const IDENTITY: [Vec3; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn body(semiaxes: Vec3, center: Vec3, axes: [Vec3; 3]) -> EllipsoidBody3 {
    EllipsoidBody3::new(
        Ellipsoid3::try_new(semiaxes).unwrap(),
        EllipsoidPose3::try_new(center, axes).unwrap(),
    )
}

/// `L x'` for the shared transform (world offset of a sphere-space vector).
fn world(semiaxes: Vec3, axes: [Vec3; 3], local: Vec3) -> Vec3 {
    let mut point = [0.0; 3];
    for i in 0..3 {
        point = add(point, scale(axes[i], semiaxes[i] * local[i]));
    }
    point
}

/// Independent implicit value `‖diag(1/s) Rᵀ (x - c)‖` and its unit gradient.
fn implicit(semiaxes: Vec3, axes: [Vec3; 3], center: Vec3, point: Vec3) -> (f64, Vec3) {
    let r = sub(point, center);
    let local: Vec3 = std::array::from_fn(|i| dot(axes[i], r) / semiaxes[i]);
    let mut gradient = [0.0; 3];
    for i in 0..3 {
        gradient = add(gradient, scale(axes[i], local[i] / semiaxes[i]));
    }
    (length(local), scale(gradient, 1.0 / length(gradient)))
}

struct Pair {
    base: Vec3,
    axes: [Vec3; 3],
    a: ScaledEllipsoidMotion3,
    b: ScaledEllipsoidMotion3,
    dt: f64,
}

impl Pair {
    fn cast(&self) -> EllipsoidPairToi3 {
        EllipsoidSharedFrame3::try_new(Ellipsoid3::try_new(self.base).unwrap(), self.axes)
            .unwrap()
            .try_cast(self.a, self.b, self.dt)
            .unwrap()
    }

    fn bodies(&self) -> (EllipsoidBody3, EllipsoidBody3) {
        (
            body(scale(self.base, self.a.scale), self.a.center, self.axes),
            body(scale(self.base, self.b.scale), self.b.center, self.axes),
        )
    }

    fn size(&self) -> f64 {
        self.base.into_iter().fold(0.0, f64::max) * (self.a.scale + self.b.scale)
    }

    /// The true contact condition: the witness lies on both boundaries and the normals of
    /// the two ellipsoids at it are opposite.
    fn check_contact(&self, fraction: f64, contact: EllipsoidPairContact3, tolerance: f64) {
        let ca = add(self.a.center, scale(self.a.velocity, self.dt * fraction));
        let cb = add(self.b.center, scale(self.b.velocity, self.dt * fraction));
        let (fa, na) = implicit(scale(self.base, self.a.scale), self.axes, ca, contact.point);
        let (fb, nb) = implicit(scale(self.base, self.b.scale), self.axes, cb, contact.point);
        assert!((fa - 1.0).abs() <= tolerance, "A implicit {fa}");
        assert!((fb - 1.0).abs() <= tolerance, "B implicit {fb}");
        assert!((length(contact.normal) - 1.0).abs() <= 1e-14);
        assert!(
            length(sub(contact.normal, na)) <= tolerance,
            "{:?} vs {na:?}",
            contact.normal
        );
        assert!(length(add(na, nb)) <= tolerance, "normals not opposite");
    }
}

/// A pair described in sphere space: `start`/`travel` of B relative to A in `L⁻¹` units.
fn sphere_space_pair(
    base: Vec3,
    axes: [Vec3; 3],
    scales: (f64, f64),
    start: Vec3,
    travel: Vec3,
) -> Pair {
    let center_a = [3.0, -2.0, 1.0];
    Pair {
        base,
        axes,
        a: ScaledEllipsoidMotion3 {
            center: center_a,
            velocity: [0.0, 0.5, 0.0],
            scale: scales.0,
        },
        b: ScaledEllipsoidMotion3 {
            center: add(center_a, world(base, axes, start)),
            velocity: add([0.0, 0.5, 0.0], world(base, axes, travel)),
            scale: scales.1,
        },
        dt: 1.0,
    }
}

#[test]
fn preconditions_are_exact_and_unsupported_pairs_are_explicit() {
    let axes = Rng(1).axes();
    let a = body([1.0, 2.0, 3.0], [0.0; 3], axes);
    let supported = body([3.0, 6.0, 9.0], [10.0, 0.0, 0.0], axes);
    let v = [-20.0, 0.0, 0.0];
    assert!(matches!(
        try_homothetic_pair_cast(a, [0.0; 3], supported, v, 1.0),
        Ok(EllipsoidPairCast3::Supported(EllipsoidPairToi3::Hit { .. }))
    ));
    // One ulp of orientation or shape difference is unsupported, never accepted fuzzily.
    let mut nudged = axes;
    nudged[0][0] = f64::from_bits(nudged[0][0].to_bits() + 1);
    let rotated = body([3.0, 6.0, 9.0], [10.0, 0.0, 0.0], nudged);
    assert_eq!(
        try_homothetic_pair_cast(a, [0.0; 3], rotated, v, 1.0),
        Ok(EllipsoidPairCast3::Unsupported(
            EllipsoidPairUnsupported3::DifferentOrientation
        ))
    );
    let skewed = body(
        [3.0, 6.0, f64::from_bits(9.0_f64.to_bits() + 1)],
        [10.0, 0.0, 0.0],
        axes,
    );
    assert_eq!(
        try_homothetic_pair_cast(a, [0.0; 3], skewed, v, 1.0),
        Ok(EllipsoidPairCast3::Unsupported(
            EllipsoidPairUnsupported3::NonHomotheticShapes
        ))
    );
    let sphere = body([2.0; 3], [10.0, 0.0, 0.0], axes);
    assert_eq!(
        try_homothetic_pair_cast(a, [0.0; 3], sphere, v, 1.0),
        Ok(EllipsoidPairCast3::Unsupported(
            EllipsoidPairUnsupported3::NonHomotheticShapes
        ))
    );
    // f64 values that are not exactly 3x apart are not homothetic, whatever was intended.
    let tenths = body([0.1, 0.2, 0.7], [0.0; 3], axes);
    let tripled = body([0.1 * 3.0, 0.2 * 3.0, 0.7 * 3.0], [5.0, 0.0, 0.0], axes);
    let exact = (0.1_f64 * 3.0) * 0.2 == (0.2_f64 * 3.0) * 0.1
        && (0.1_f64 * 3.0) * 0.7 == (0.7_f64 * 3.0) * 0.1;
    let result = try_homothetic_pair_cast(tenths, [0.0; 3], tripled, v, 1.0).unwrap();
    assert!(exact || matches!(result, EllipsoidPairCast3::Unsupported(_)));
    // Extreme in-shape ratios cannot be decided exactly.
    let sliver = body([1.0, 1e-300, 1.0], [0.0; 3], IDENTITY);
    let sliver2 = body([2.0, 2e-300, 2.0], [9.0, 0.0, 0.0], IDENTITY);
    assert_eq!(
        try_homothetic_pair_cast(sliver, [0.0; 3], sliver2, v, 1.0),
        Ok(EllipsoidPairCast3::Unsupported(
            EllipsoidPairUnsupported3::UnverifiableProportion
        ))
    );
    // Very large and very small common scales stay decidable (power-of-two normalization);
    // the scale is a power of two so the scaled f64 semiaxes remain exactly proportional.
    for k in [2.0_f64.powi(-990), 2.0_f64.powi(990)] {
        let small = body(scale([1.0, 2.0, 3.0], k), [0.0; 3], axes);
        let large = body(scale([3.0, 6.0, 9.0], k), [10.0 * k, 0.0, 0.0], axes);
        let result = try_homothetic_pair_cast(small, [0.0; 3], large, scale(v, k), 1.0);
        assert!(
            matches!(result, Ok(EllipsoidPairCast3::Supported(_))),
            "{k:e}: {result:?}"
        );
    }
}

#[test]
fn invalid_motion_is_an_error() {
    let frame =
        EllipsoidSharedFrame3::try_new(Ellipsoid3::try_new([1.0; 3]).unwrap(), IDENTITY).unwrap();
    let motion = |scale: f64, center: Vec3| ScaledEllipsoidMotion3 {
        center,
        velocity: [0.0; 3],
        scale,
    };
    let ok = motion(1.0, [0.0; 3]);
    assert_eq!(
        frame.try_cast(ok, motion(0.0, [5.0; 3]), 1.0),
        Err(EllipsoidError3::InvalidParameter)
    );
    assert_eq!(
        frame.try_cast(ok, motion(1.0, [5.0; 3]), -1.0),
        Err(EllipsoidError3::InvalidParameter)
    );
    assert_eq!(
        frame.try_cast(ok, motion(1.0, [f64::NAN; 3]), 1.0),
        Err(EllipsoidError3::NonFiniteInput)
    );
    assert_eq!(
        frame.try_cast(ok, motion(f64::INFINITY, [5.0; 3]), 1.0),
        Err(EllipsoidError3::NonFiniteInput)
    );
    let overflow = ScaledEllipsoidMotion3 {
        velocity: [f64::MAX, 0.0, 0.0],
        ..motion(1.0, [5.0; 3])
    };
    assert_eq!(
        frame.try_cast(ok, overflow, 4.0),
        Err(EllipsoidError3::Unrepresentable)
    );
    assert_eq!(
        EllipsoidSharedFrame3::try_new(
            Ellipsoid3::try_new([1.0; 3]).unwrap(),
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        Err(EllipsoidError3::NonOrthonormalAxes)
    );
}

#[test]
fn analytic_head_on_overlap_stationary_and_separating_cases() {
    let axes = Rng(2).axes();
    let base = [4.0, 1.0, 0.5];
    // Head on: radius 3, start 10 apart, closing 20 per interval: t = 7/20.
    let pair = sphere_space_pair(base, axes, (1.0, 2.0), [-10.0, 0.0, 0.0], [20.0, 0.0, 0.0]);
    let EllipsoidPairToi3::Hit { fraction, contact } = pair.cast() else {
        panic!("head-on pair must hit");
    };
    // The rounding slack only ever moves contact earlier.
    assert!(
        (0.35 - 1e-13..=0.35 + 1e-15).contains(&fraction),
        "{fraction}"
    );
    pair.check_contact(fraction, contact, 1e-11);
    // Through a non-axis direction in sphere space.
    let pair = sphere_space_pair(base, axes, (0.5, 0.25), [3.0, -4.0, 2.0], [-6.0, 8.0, -4.0]);
    let EllipsoidPairToi3::Hit { fraction, contact } = pair.cast() else {
        panic!("diagonal pair must hit");
    };
    let distance = length([3.0, -4.0, 2.0]);
    assert!((fraction - (distance - 0.75) / (2.0 * distance)).abs() <= 1e-13);
    pair.check_contact(fraction, contact, 1e-11);

    let overlap = sphere_space_pair(base, axes, (1.0, 2.0), [2.9, 0.0, 0.0], [5.0, 0.0, 0.0]);
    assert_eq!(overlap.cast(), EllipsoidPairToi3::InitiallyOverlapping);
    let stationary = sphere_space_pair(base, axes, (1.0, 2.0), [3.5, 0.0, 0.0], [0.0; 3]);
    assert_eq!(stationary.cast(), EllipsoidPairToi3::Miss);
    let separating = sphere_space_pair(base, axes, (1.0, 2.0), [3.5, 0.0, 0.0], [1.0, 0.1, 0.0]);
    assert_eq!(separating.cast(), EllipsoidPairToi3::Miss);
    let short = sphere_space_pair(base, axes, (1.0, 2.0), [-10.0, 0.0, 0.0], [6.9, 0.0, 0.0]);
    assert_eq!(short.cast(), EllipsoidPairToi3::Miss);
    let crossing = sphere_space_pair(base, axes, (1.0, 2.0), [5.0, 4.0, 0.0], [0.0, 0.0, 9.0]);
    assert_eq!(crossing.cast(), EllipsoidPairToi3::Miss);
    // Exactly touching at the start on an axis-aligned frame is a hit at zero, not overlap.
    let touching = sphere_space_pair(
        [2.0, 1.0, 0.5],
        IDENTITY,
        (1.0, 2.0),
        [3.0, 0.0, 0.0],
        [0.0; 3],
    );
    let EllipsoidPairToi3::Hit { fraction, contact } = touching.cast() else {
        panic!("touching start must hit");
    };
    assert_eq!(fraction, 0.0);
    touching.check_contact(0.0, contact, 1e-14);
    assert_eq!(contact.normal, [1.0, 0.0, 0.0]);
}

#[test]
fn tangent_and_near_tangent_paths() {
    let mut rng = Rng(0x7a);
    for _ in 0..32 {
        let axes = rng.axes();
        let base = [
            rng.range(0.2, 5.0),
            rng.range(0.2, 5.0),
            rng.range(0.2, 5.0),
        ];
        let side = rng.unit();
        let along = rng.unit();
        let perpendicular = sub(side, scale(along, dot(side, along)));
        let perpendicular = scale(perpendicular, 1.0 / length(perpendicular));
        let radius = 3.0;
        let path = |offset: f64| {
            let start = add(scale(along, -10.0), scale(perpendicular, radius * offset));
            sphere_space_pair(base, axes, (1.0, 2.0), start, scale(along, 20.0))
        };
        // Exact tangency is reported as contact (no missed grazing impact) near t = 1/2.
        let pair = path(1.0);
        let EllipsoidPairToi3::Hit { fraction, contact } = pair.cast() else {
            panic!("tangent path must report contact");
        };
        assert!((fraction - 0.5).abs() <= 1e-6, "{fraction}");
        pair.check_contact(fraction, contact, 1e-7);
        assert!(matches!(
            path(1.0 - 1e-9).cast(),
            EllipsoidPairToi3::Hit { .. }
        ));
        assert_eq!(path(1.0 + 1e-9).cast(), EllipsoidPairToi3::Miss);
    }
}

#[test]
fn near_zero_coefficients_and_extreme_scales() {
    let axes = Rng(3).axes();
    let base = [4.0, 1.0, 0.25];
    // A displacement many orders below the gap: no contact, no spurious root.
    let creeping = sphere_space_pair(base, axes, (1.0, 1.0), [-3.0, 0.0, 0.0], [1e-200, 0.0, 0.0]);
    assert_eq!(creeping.cast(), EllipsoidPairToi3::Miss);
    // A tiny gap closed by a tiny displacement.
    let near = sphere_space_pair(
        base,
        axes,
        (1.0, 1.0),
        [-2.0 - 1e-9, 0.0, 0.0],
        [2e-9, 0.0, 0.0],
    );
    let EllipsoidPairToi3::Hit { fraction, .. } = near.cast() else {
        panic!("tiny closing must hit");
    };
    // The ~1e-14 rounding slack is large against a 2e-9 displacement: contact is reported
    // up to slack/displacement early, never late.
    assert!((0.4999..=0.5 + 1e-9).contains(&fraction), "{fraction}");
    // Scale covariance through the checked body entry from 1e-140 to 1e140.
    let reference = sphere_space_pair(base, axes, (1.0, 3.0), [7.0, -2.0, 1.0], [-12.0, 3.0, -1.0]);
    let EllipsoidPairToi3::Hit {
        fraction: expected,
        contact: expected_contact,
    } = reference.cast()
    else {
        panic!("reference pair must hit");
    };
    for k in [1e-140, 1e-20, 1.0, 1e20, 1e140] {
        let (a, b) = reference.bodies();
        let a = body(
            scale(a.shape.semiaxes(), k),
            scale(a.pose.center(), k),
            axes,
        );
        let b = body(
            scale(b.shape.semiaxes(), k),
            scale(b.pose.center(), k),
            axes,
        );
        let result = try_homothetic_pair_cast(
            a,
            scale(reference.a.velocity, k),
            b,
            scale(reference.b.velocity, k),
            1.0,
        )
        .unwrap();
        let EllipsoidPairCast3::Supported(EllipsoidPairToi3::Hit { fraction, contact }) = result
        else {
            panic!("scaled pair must hit: {result:?}");
        };
        assert!((fraction - expected).abs() <= 1e-13);
        assert!(length(sub(scale(contact.point, 1.0 / k), expected_contact.point)) <= 1e-12);
        assert!(length(sub(contact.normal, expected_contact.normal)) <= 1e-12);
    }
    // Near-degenerate but valid semiaxes.
    let needle = sphere_space_pair(
        [1e4, 1.0, 1e-4],
        axes,
        (1.0, 0.5),
        [-5.0, 0.2, 0.1],
        [10.0, 0.0, 0.0],
    );
    let EllipsoidPairToi3::Hit { fraction, contact } = needle.cast() else {
        panic!("needle pair must hit");
    };
    let perpendicular_squared: f64 = 0.2 * 0.2 + 0.1 * 0.1;
    let expected = (5.0 - (2.25 - perpendicular_squared).sqrt()) / 10.0;
    // The 1e8 axis-ratio condition number amplifies transform rounding accordingly.
    // Its rounding slack is ~cond·ε, so contact is up to ~1e-6 early and never late.
    assert!(
        (expected - 1e-6..=expected + 1e-12).contains(&fraction),
        "{fraction} vs {expected}"
    );
    needle.check_contact(fraction, contact, 1e-5);
}

#[test]
fn seeded_pairs_match_the_generic_conservative_advancement_reference() {
    let mut rng = Rng(0x190);
    let (mut hits, mut misses, mut fast, mut unresolved) = (0, 0, 0, 0);
    let mut worst: f64 = 0.0;
    let shapes = [
        [1.0, 1.0, 1.0],
        [1.5, 1.0, 1.0],
        [4.0, 1.0, 1.0],
        [0.3, 2.0, 0.7],
        [10.0, 1.0, 0.2],
    ];
    for case in 0..600 {
        let base = shapes[case % shapes.len()];
        let axes = rng.axes();
        let scales = (rng.range(0.2, 3.0), rng.range(0.2, 3.0));
        let size = base.into_iter().fold(0.0, f64::max) * (scales.0 + scales.1);
        let high_speed = case % 4 == 0;
        let distance = if high_speed { 400.0 } else { 4.0 } * size;
        let center_a = [
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
        ];
        let direction = rng.unit();
        let center_b = add(center_a, scale(direction, distance));
        // Aim back at A with a lateral miss offset of up to about one combined size.
        let aim = add(
            scale(direction, -2.0 * distance),
            scale(rng.unit(), rng.range(0.0, 1.2) * size),
        );
        let velocity_a = scale(rng.unit(), rng.range(0.0, 0.2) * size);
        let pair = Pair {
            base,
            axes,
            a: ScaledEllipsoidMotion3 {
                center: center_a,
                velocity: velocity_a,
                scale: scales.0,
            },
            b: ScaledEllipsoidMotion3 {
                center: center_b,
                velocity: add(velocity_a, scale(aim, 60.0)),
                scale: scales.1,
            },
            dt: 1.0 / 60.0,
        };
        let (a, b) = pair.bodies();
        let tolerance = 1e-9 * pair.size();
        let reference = conservative_advancement(
            a,
            pair.a.velocity,
            b,
            pair.b.velocity,
            pair.dt,
            tolerance,
            100_000,
        );
        let closing = length(sub(pair.b.velocity, pair.a.velocity)) * pair.dt;
        match (pair.cast(), reference) {
            (EllipsoidPairToi3::Hit { fraction, contact }, ReferenceToi::Hit(expected)) => {
                hits += 1;
                fast += usize::from(high_speed);
                // CA stops within `tolerance` of contact, so it precedes the true time by about
                // `tolerance / closing` (more for grazing approach).
                assert!(
                    (fraction - expected).abs() <= 1e-12 + 200.0 * tolerance / closing,
                    "case {case}: {fraction} vs reference {expected}"
                );
                worst = worst.max((fraction - expected).abs() / (tolerance / closing));
                pair.check_contact(fraction, contact, 1e-8);
            }
            (EllipsoidPairToi3::Miss, ReferenceToi::Miss) => misses += 1,
            (_, ReferenceToi::Unresolved) => unresolved += 1,
            (actual, expected) => panic!("case {case}: {actual:?} vs reference {expected:?}"),
        }
    }
    eprintln!("hits {hits} misses {misses} fast {fast} unresolved {unresolved} worst {worst}");
    assert!(hits >= 150 && misses >= 150, "hits {hits}, misses {misses}");
    assert!(fast >= 30, "only {fast} high-speed hits");
    assert!(
        unresolved <= 6,
        "{unresolved} unresolved reference searches"
    );
}

#[test]
fn normals_follow_the_inverse_transpose_not_the_sphere_direction() {
    // For a 4:1 frame the sphere-space direction (1,1)/√2 maps to a world normal tilted
    // toward the short axis; reusing the sphere direction as the world normal is wrong.
    let pair = sphere_space_pair(
        [4.0, 1.0, 1.0],
        IDENTITY,
        (1.0, 1.0),
        scale([1.0, 1.0, 0.0], 4.0 / 2.0_f64.sqrt()),
        scale([-1.0, -1.0, 0.0], 4.0),
    );
    let EllipsoidPairToi3::Hit { contact, fraction } = pair.cast() else {
        panic!("pair must hit");
    };
    pair.check_contact(fraction, contact, 1e-12);
    let expected = [1.0 / 4.0, 1.0, 0.0];
    let expected = scale(expected, 1.0 / length(expected));
    assert!(length(sub(contact.normal, expected)) <= 1e-14);
    assert!(length(cross(contact.normal, [1.0, 1.0, 0.0])) > 0.5);
}
