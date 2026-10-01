use super::*;
use crate::primitive3::PrimitiveShape3 as Shape;

fn body(shape: Shape) -> PrimitiveBody3 {
    PrimitiveBody3::axis_aligned(shape, [0.0; 3], [0.0; 3])
}
fn cast(body: PrimitiveBody3, o: Vec3, d: Vec3) -> Option<PrimitiveRayHit3> {
    try_ray_cast(body, o, d, &mut PrimitiveRayWork3::default()).unwrap()
}

#[test]
fn all_shapes_admit_boundary_endpoints_and_reject_a_shorter_segment() {
    for shape in [
        Shape::sphere(1.0),
        Shape::cuboid([1.0; 3]),
        Shape::capsule(2.0, 1.0),
        Shape::wedge([1.0; 3]),
    ] {
        let b = body(shape);
        let hit = cast(b, [-3.0, -0.5, 0.0], [6.0, 0.0, 0.0]).unwrap();
        assert!(!hit.starts_inside);
        assert!(dot(hit.normal, [-1.0, 0.0, 0.0]) > 0.8);
        assert!(cast(b, [-3.0, -0.5, 0.0], [1.0, 0.0, 0.0]).is_none());
        // Exact known endpoint for each geometry, independent of a previous ray.
        let x = if shape.kind() == PrimitiveKind3::Sphere {
            -(0.75_f64).sqrt()
        } else {
            -1.0
        };
        let end = cast(b, [x - 1.0, -0.5, 0.0], [1.0, 0.0, 0.0]).unwrap();
        assert!((end.fraction - 1.0).abs() < 1e-14);
    }
}

#[test]
fn capsule_side_caps_axis_parallel_and_sphere_degeneracy_are_analytic() {
    use PrimitiveRayFeature3::*;
    let capsule = body(Shape::capsule(2.0, 1.0));
    let side = cast(capsule, [-3.0, 0.0, 0.0], [6.0, 0.0, 0.0]).unwrap();
    assert!((side.fraction - 1.0 / 3.0).abs() < 1e-14);
    assert_eq!(side.feature, CapsuleSide);
    for (o, d, normal, feature) in [
        (
            [0.0, -5.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, -1.0, 0.0],
            CapsuleNegativeCap,
        ),
        (
            [0.0, 5.0, 0.0],
            [0.0, -10.0, 0.0],
            [0.0, 1.0, 0.0],
            CapsulePositiveCap,
        ),
    ] {
        let hit = cast(capsule, o, d).unwrap();
        assert!((hit.fraction - 0.2).abs() < 1e-14);
        assert_eq!(hit.normal, normal);
        assert_eq!(hit.feature, feature);
    }
    let sphere = cast(
        body(Shape::capsule(0.0, 1.0)),
        [-3.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
    )
    .unwrap();
    assert_eq!(
        sphere,
        cast(body(Shape::sphere(1.0)), [-3.0, 0.0, 0.0], [6.0, 0.0, 0.0]).unwrap()
    );
}

#[test]
fn touching_inside_tangent_and_behind_contracts_have_independent_answers() {
    for shape in [
        Shape::sphere(1.0),
        Shape::cuboid([1.0; 3]),
        Shape::capsule(2.0, 1.0),
        Shape::wedge([1.0; 3]),
    ] {
        let b = body(shape);
        let inside = cast(b, [-0.5, -0.5, 0.0], [2.0, 0.0, 0.0]).unwrap();
        assert_eq!(inside.fraction, 0.0);
        assert!(inside.starts_inside);
        assert_eq!(inside.feature, PrimitiveRayFeature3::Interior);
        assert_eq!(inside.normal, [-1.0, 0.0, 0.0]);
        let touch = cast(b, [-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]).unwrap();
        assert_eq!(touch.fraction, 0.0);
        assert!(!touch.starts_inside);
        assert_eq!(touch.normal, [-1.0, 0.0, 0.0]);
        assert!(cast(b, [-3.0, 0.0, 0.0], [-2.0, 0.0, 0.0]).is_none());
    }
    let tangent = cast(body(Shape::sphere(1.0)), [-2.0, 1.0, 0.0], [4.0, 0.0, 0.0]).unwrap();
    assert_eq!(tangent.fraction, 0.5);
    assert_eq!(tangent.normal, [0.0, 1.0, 0.0]);
    assert!(
        cast(
            body(Shape::sphere(1.0)),
            [-2.0, 1.0 + 1e-8, 0.0],
            [4.0, 0.0, 0.0]
        )
        .is_none()
    );
    let slope = cast(
        body(Shape::wedge([2.0, 1.0, 3.0])),
        [0.0, 3.0, 0.0],
        [0.0, -6.0, 0.0],
    )
    .unwrap();
    assert_eq!(slope.fraction, 0.5);
    assert_eq!(slope.feature, PrimitiveRayFeature3::WedgeFace(2));
    let expected = unit_or_zero([1.0, 2.0, 0.0]);
    assert!((dot(slope.normal, expected) - 1.0).abs() < 1e-14);
}

#[test]
fn pose_and_scale_covariance_preserve_features_normals_and_work_ceilings() {
    let axes = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    for s in [1e-150, 1e-12, 1.0, 1e12, 1e150] {
        for shape in [
            Shape::sphere(s),
            Shape::cuboid([s; 3]),
            Shape::capsule(2.0 * s, s),
            Shape::wedge([s; 3]),
        ] {
            let mut b = body(shape);
            b.axes = axes;
            b.position = [s * 9.0, s * -3.0, s * 2.0];
            let o = add(b.position, rotate_local(axes, [-3.0 * s, -0.5 * s, 0.0]));
            let d = rotate_local(axes, [6.0 * s, 0.0, 0.0]);
            let mut work = PrimitiveRayWork3::default();
            let actual = try_ray_cast(b, o, d, &mut work).unwrap().unwrap();
            let expected =
                cast(body(shape), [-3.0 * s, -0.5 * s, 0.0], [6.0 * s, 0.0, 0.0]).unwrap();
            assert!((actual.fraction - expected.fraction).abs() < 1e-14);
            assert_eq!(actual.feature, expected.feature);
            let n = rotate_local(axes, expected.normal);
            assert!(sub(actual.normal, n).into_iter().all(|v| v.abs() < 1e-14));
            assert_eq!(work.queries, 1);
            assert!(work.planes_tested <= 6);
            assert!(work.quadratic_tests <= 3);
        }
    }
}

// Independent occupancy: scalar inequalities, not the production clipping/quadratics.
fn occupied(shape: Shape, p: Vec3) -> bool {
    let h = shape.half_extents();
    match shape.kind() {
        PrimitiveKind3::Sphere => p.iter().map(|v| v * v).sum::<f64>() <= h[0] * h[0],
        PrimitiveKind3::Box => (0..3).all(|i| p[i].abs() <= h[i]),
        PrimitiveKind3::Wedge => {
            p[0] >= -h[0] && p[1] >= -h[1] && p[0] / h[0] + p[1] / h[1] <= 0.0 && p[2].abs() <= h[2]
        }
        PrimitiveKind3::Capsule => {
            let (half, r) = shape.capsule_parts();
            let y = (p[1].abs() - half).max(0.0);
            p[0] * p[0] + p[2] * p[2] + y * y <= r * r
        }
    }
}

#[test]
fn seeded_crossings_match_independent_occupancy_bisection() {
    let mut seed = 0x194_u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 32) as f64 / u32::MAX as f64 * 2.0 - 1.0
    };
    for shape in [
        Shape::sphere(1.0),
        Shape::cuboid([1.0, 2.0, 0.5]),
        Shape::capsule(2.0, 0.5),
        Shape::wedge([1.0, 2.0, 0.5]),
    ] {
        for _ in 0..1024 {
            // Start outside, end in a known strict interior neighbourhood.
            let o = [4.0 + next().abs(), 4.0 * next(), 4.0 * next()];
            let end = [-0.2 + 0.1 * next(), -0.3 + 0.1 * next(), 0.1 * next()];
            assert!(!occupied(shape, o));
            assert!(occupied(shape, end));
            let d = sub(end, o);
            let mut lo = 0.0;
            let mut hi = 1.0;
            for _ in 0..60 {
                let t = (lo + hi) * 0.5;
                if occupied(shape, add(o, scale(d, t))) {
                    hi = t;
                } else {
                    lo = t;
                }
            }
            let hit = cast(body(shape), o, d).unwrap();
            assert!(
                (hit.fraction - hi).abs() < 1e-12,
                "{shape:?} {o:?} {end:?} {hit:?} {hi}"
            );
            assert!(dot(hit.normal, d) <= 1e-12);
        }
    }
}

#[test]
fn invalid_inputs_and_unrepresentable_computations_are_errors_not_misses() {
    let b = body(Shape::sphere(1.0));
    for (o, d) in [
        ([f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0; 3], [0.0; 3]),
    ] {
        assert_eq!(
            try_ray_cast(b, o, d, &mut PrimitiveRayWork3::default()),
            Err(PrimitiveRayError3::InvalidInput)
        );
    }
    assert_eq!(
        try_ray_cast(
            b,
            [f64::MAX, 0.0, 0.0],
            [f64::MAX, 0.0, 0.0],
            &mut PrimitiveRayWork3::default()
        ),
        Err(PrimitiveRayError3::NonFiniteComputation)
    );
    let mut invalid = body(Shape::cuboid([1.0; 3]));
    invalid.axes[0] = [2.0, 0.0, 0.0];
    assert_eq!(
        try_ray_cast(
            invalid,
            [-3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            &mut PrimitiveRayWork3::default()
        ),
        Err(PrimitiveRayError3::InvalidInput)
    );
}

#[test]
fn nonfinite_plane_parameter_is_an_error_and_initial_cap_seams_keep_feature_order() {
    assert_eq!(
        try_ray_cast(
            body(Shape::cuboid([1.0; 3])),
            [2.0, 0.0, 0.0],
            [-1e-320, 1.0, 0.0],
            &mut PrimitiveRayWork3::default()
        ),
        Err(PrimitiveRayError3::NonFiniteComputation)
    );
    let b = body(Shape::capsule(2.0, 1.0));
    for (y, feature) in [
        (-2.0, PrimitiveRayFeature3::CapsuleNegativeCap),
        (2.0, PrimitiveRayFeature3::CapsulePositiveCap),
    ] {
        let hit = cast(b, [1.0, y, 0.0], [1.0, 0.0, 0.0]).unwrap();
        assert_eq!(hit.feature, feature);
        assert_eq!(hit.fraction, 0.0);
        assert_eq!(hit.normal, [1.0, 0.0, 0.0]);
    }
}
