use super::closest::closest_segment_segment;
use super::wedge::{closest_segment_wedge, segment_wedge_interval, wedge_triangles};
use super::*;
use crate::math3::{Vec3, add, dot, length, length_squared, scale, sub};

#[test]
fn closest_segments_preserve_crossings_across_scales() {
    for scale in [1.0e-150, 1.0e-12, 1.0, 1.0e9, 1.0e150] {
        let points = closest_segment_points(
            [-2.0 * scale, 0.0, 0.0],
            [2.0 * scale, 0.0, 0.0],
            [scale, -3.0 * scale, 0.0],
            [scale, scale, 0.0],
        )
        .unwrap();
        for point in [points.0, points.1] {
            assert!(
                length(sub(point, [scale, 0.0, 0.0])) <= 1.0e-14 * scale,
                "scale={scale}, point={point:?}"
            );
        }
    }
}

#[test]
fn closest_segments_handle_skew_endpoints_parallel_and_degenerate_cases() {
    let cases = [
        (
            [0.0; 3],
            [2.0, 0.0, 0.0],
            [1.0, -1.0, 3.0],
            [1.0, 1.0, 3.0],
            ([1.0, 0.0, 0.0], [1.0, 0.0, 3.0]),
        ),
        (
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            ([1.0, 0.0, 0.0], [2.0, 1.0, 0.0]),
        ),
        (
            [0.0; 3],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            ([1.0, 0.0, 0.0], [1.0, 1.0, 0.0]),
        ),
        (
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0; 3],
            [2.0, 0.0, 0.0],
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ),
        (
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            ([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
        ),
    ];
    for (p1, q1, p2, q2, expected) in cases {
        let actual = closest_segment_points(p1, q1, p2, q2).unwrap();
        assert_eq!(actual, expected);
        let reversed = closest_segment_points(p2, q2, p1, q1).unwrap();
        assert!(
            (length(sub(actual.0, actual.1)) - length(sub(reversed.0, reversed.1))).abs() < 1.0e-14
        );
    }
}

#[test]
fn closest_segments_resolve_nearly_parallel_crossings() {
    for slope in [1.0e-4, 1.0e-8, 1.0e-12] {
        let points = closest_segment_points(
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, -slope, 0.0],
            [1.0, slope, 0.0],
        )
        .unwrap();
        assert!(
            length(points.0) < 1.0e-14,
            "slope={slope}, points={points:?}"
        );
        assert!(
            length(points.1) < 1.0e-14,
            "slope={slope}, points={points:?}"
        );
    }
}

#[test]
fn closest_segments_satisfy_membership_and_independent_distance_bounds() {
    let mut state = 12_345_u64;
    let mut coordinate = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((state >> 32) % 2001) as f64 / 100.0 - 10.0
    };
    for _ in 0..128 {
        let endpoints: [Vec3; 4] = std::array::from_fn(|_| std::array::from_fn(|_| coordinate()));
        let [p1, q1, p2, q2] = endpoints;
        let (left, right) = closest_segment_points(p1, q1, p2, q2).unwrap();
        for (point, start, end) in [(left, p1, q1), (right, p2, q2)] {
            let delta = sub(end, start);
            let offset = sub(point, start);
            let t = dot(offset, delta) / length_squared(delta);
            assert!((-1.0e-14..=1.0 + 1.0e-14).contains(&t));
            assert!(length(sub(offset, scale(delta, t))) < 1.0e-12);
        }
        let distance = length(sub(left, right));
        // Exhaustive independently interpolated candidates provide an upper bound.
        for i in 0..=16 {
            for j in 0..=16 {
                let a = add(
                    scale(p1, 1.0 - f64::from(i) / 16.0),
                    scale(q1, f64::from(i) / 16.0),
                );
                let b = add(
                    scale(p2, 1.0 - f64::from(j) / 16.0),
                    scale(q2, f64::from(j) / 16.0),
                );
                assert!(distance <= length(sub(a, b)) + 1.0e-12);
            }
        }
        for (a, b, c, d) in [(q1, p1, p2, q2), (p2, q2, p1, q1)] {
            let points = closest_segment_points(a, b, c, d).unwrap();
            assert!((length(sub(points.0, points.1)) - distance).abs() < 1.0e-12);
        }
    }
}

#[test]
fn closest_segments_reject_non_finite_and_unrepresentable_arithmetic() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            closest_segment_points([invalid, 0.0, 0.0], [1.0; 3], [0.0; 3], [2.0; 3]),
            None
        );
    }
    assert_eq!(
        closest_segment_points(
            [-f64::MAX, 0.0, 0.0],
            [f64::MAX, 0.0, 0.0],
            [0.0; 3],
            [1.0; 3]
        ),
        None
    );
}

fn body(shape: PrimitiveShape3, position: Vec3) -> PrimitiveBody3 {
    PrimitiveBody3::axis_aligned(shape, position, [0.0; 3])
}

#[test]
fn segment_wedge_closest_pair_matches_every_triangle() {
    let half = [1.5, 0.75, 0.8];
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1_u64 << 53) as f64 * 8.0 - 4.0
    };
    let mut work = PrimitiveWork3::default();
    let mut checked = 0;
    for _ in 0..2_000 {
        let a = [next(), next(), next()];
        let b = [next(), next(), next()];
        if segment_wedge_interval(a, b, half).is_some() {
            continue;
        }
        let mut brute = f64::INFINITY;
        for triangle in wedge_triangles(half) {
            for endpoint in [a, b] {
                let q = closest_point_triangle(endpoint, triangle[0], triangle[1], triangle[2]);
                brute = brute.min(length(sub(q, endpoint)));
            }
            for (p, q) in [(0, 1), (1, 2), (2, 0)] {
                let (x, y) = closest_segment_segment(a, b, triangle[p], triangle[q], &mut work);
                brute = brute.min(length(sub(y, x)));
            }
        }
        let (x, y) = closest_segment_wedge(a, b, half, &mut work);
        assert!((length(sub(y, x)) - brute).abs() < 1.0e-12, "{a:?} {b:?}");
        checked += 1;
    }
    assert!(checked > 500);
}

#[test]
fn primitive_pair_matrix_is_complete_canonical_and_dense() {
    let mut seen = [false; 10];
    for left in PrimitiveKind3::ALL {
        for right in PrimitiveKind3::ALL {
            let (pair, reversed) = PrimitivePair3::canonical(left, right);
            let (first, second) = pair.kinds();
            assert!(first <= second);
            assert_eq!(pair, PrimitivePair3::ALL[pair.index()]);
            assert_eq!(reversed, left > right);
            seen[pair.index()] = true;
        }
    }
    assert!(seen.into_iter().all(|value| value));
}

fn shape_for_kind(kind: PrimitiveKind3) -> PrimitiveShape3 {
    match kind {
        PrimitiveKind3::Sphere => PrimitiveShape3::sphere(1.0),
        PrimitiveKind3::Box => PrimitiveShape3::cuboid([1.0, 0.8, 1.2]),
        PrimitiveKind3::Capsule => PrimitiveShape3::capsule(0.7, 0.6),
        PrimitiveKind3::Wedge => PrimitiveShape3::wedge([1.2, 0.9, 1.1]),
    }
}

#[test]
fn every_primitive_pair_executes_its_registered_kernel() {
    for pair in PrimitivePair3::ALL {
        let (left, right) = pair.kinds();
        let a = body(shape_for_kind(left), [0.0; 3]);
        let b = body(shape_for_kind(right), [0.25, 0.1, -0.05]);
        let mut work = PrimitiveWork3::default();
        let contact = query(a, b, &mut work);
        assert!(contact.separation.is_finite(), "{pair:?}");
        assert!(contact.normal.into_iter().all(f64::is_finite), "{pair:?}");
        assert_eq!(work.pair_dispatches[pair.index()], 1, "{pair:?}");
        assert_eq!(work.pair_dispatches.into_iter().sum::<u64>(), 1, "{pair:?}");
    }
}

#[test]
fn capsule_sphere_and_capsule_capsule_use_analytic_segment_distance() {
    let capsule = body(PrimitiveShape3::capsule(2.0, 0.5), [0.0; 3]);
    let sphere = body(PrimitiveShape3::sphere(0.5), [0.0, 3.0, 0.0]);
    let mut work = PrimitiveWork3::default();
    let contact = query(capsule, sphere, &mut work);
    assert!(contact.separation.abs() < 1.0e-12);
    assert_eq!(work.vertex_tests, 0);

    let other = body(PrimitiveShape3::capsule(2.0, 0.5), [0.9, 0.0, 0.0]);
    let contact = query(capsule, other, &mut work);
    assert!(contact.separation < 0.0);
    assert_eq!(work.vertex_tests, 0);
}

#[test]
fn sphere_wedge_respects_sloped_face_instead_of_bounding_box() {
    let wedge = body(PrimitiveShape3::wedge([2.0; 3]), [0.0; 3]);
    let near = body(PrimitiveShape3::sphere(0.8), [1.0, 0.0, 0.0]);
    let far = body(PrimitiveShape3::sphere(0.2), [1.5, 1.5, 0.0]);
    let mut work = PrimitiveWork3::default();
    assert!(query(near, wedge, &mut work).separation < 0.0);
    assert!(query(far, wedge, &mut work).separation > 0.0);
}

#[test]
fn capsule_wedge_is_symmetric_for_rotated_capsule() {
    let wedge = body(PrimitiveShape3::wedge([2.0; 3]), [0.0; 3]);
    let capsule = PrimitiveBody3::new(
        PrimitiveShape3::capsule(1.5, 0.4),
        [0.0, 0.5, 0.0],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    );
    let mut work = PrimitiveWork3::default();
    let forward = query(capsule, wedge, &mut work);
    let reverse = query(wedge, capsule, &mut work);
    assert!((forward.separation - reverse.separation).abs() < 1.0e-12);
    assert!(length(add(forward.normal, reverse.normal)) < 1.0e-12);
}

#[test]
fn wedge_sat_has_a_fixed_small_work_ceiling() {
    let a = body(PrimitiveShape3::wedge([2.0, 1.0, 3.0]), [0.0; 3]);
    let angle = 30.0_f64.to_radians();
    let (sin, cos) = angle.sin_cos();
    let b = PrimitiveBody3::new(
        PrimitiveShape3::wedge([1.5, 2.0, 1.0]),
        [3.0, 0.0, 0.0],
        [[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    );
    let mut work = PrimitiveWork3::default();
    let contact = query(a, b, &mut work);
    assert!(contact.separation.is_finite());
    assert!(work.axes_tested <= 24, "{work:?}");
    assert!(work.vertex_tests <= 600, "{work:?}");
}

#[test]
fn partial_capsule_box_penetration_uses_the_whole_segment() {
    let capsule = PrimitiveBody3::new(
        PrimitiveShape3::capsule(1.0, 0.1),
        [1.5, 0.0, 0.0],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    );
    let cuboid = body(PrimitiveShape3::cuboid([1.0; 3]), [0.0; 3]);
    let mut work = PrimitiveWork3::default();
    let contact = query(capsule, cuboid, &mut work);
    assert!((contact.separation + 0.6).abs() < 1.0e-12, "{contact:?}");
}

#[test]
fn partial_capsule_wedge_penetration_uses_the_whole_segment() {
    let capsule = PrimitiveBody3::new(
        PrimitiveShape3::capsule(1.0, 0.1),
        [0.5, 0.0, 0.0],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0; 3],
    );
    let wedge = body(PrimitiveShape3::wedge([1.0; 3]), [0.0; 3]);
    let mut work = PrimitiveWork3::default();
    let contact = query(capsule, wedge, &mut work);
    let expected = -(0.5_f64 / 2.0_f64.sqrt() + 0.1);
    assert!(
        (contact.separation - expected).abs() < 1.0e-12,
        "{contact:?}"
    );
}

#[test]
fn stationary_overlap_reports_time_zero() {
    let left = body(PrimitiveShape3::sphere(1.0), [0.0; 3]);
    let right = body(PrimitiveShape3::sphere(1.0), [1.5, 0.0, 0.0]);
    let mut work = PrimitiveWork3::default();
    assert_eq!(swept_time(left, right, 1.0, 0.0, &mut work), Some(0.0));
}

#[test]
fn zero_margin_does_not_expand_a_near_miss() {
    let moving = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::sphere(1.0),
        [-3.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
    );
    let target = body(PrimitiveShape3::sphere(1.0), [0.0, 2.000_000_5, 0.0]);
    let mut work = PrimitiveWork3::default();
    assert_eq!(swept_time(moving, target, 1.0, 0.0, &mut work), None);
}

#[test]
fn glancing_capsule_sweep_uses_normal_closing_speed() {
    let capsule = body(PrimitiveShape3::capsule(0.0, 0.1), [-400.0, 1.0, 0.0]);
    let capsule = PrimitiveBody3 {
        velocity: [800.0, -2.0, 0.0],
        ..capsule
    };
    let target = body(PrimitiveShape3::cuboid([500.0, 0.01, 10.0]), [0.0; 3]);
    let mut work = PrimitiveWork3::default();
    let time =
        swept_time(capsule, target, 1.0, 0.02, &mut work).expect("glancing capsule sweep must hit");
    assert!((time - 0.435).abs() < 1.0e-9, "time={time}");
    assert!(work.sweep_iterations <= 3, "{work:?}");
}

#[test]
fn fast_capsule_conservative_advancement_reaches_thin_wedge() {
    let capsule = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::capsule(0.5, 0.1),
        [-1.0, -1.0, 10.0],
        [0.0, 0.0, -10_000.0],
    );
    let wedge = body(PrimitiveShape3::wedge([4.0, 4.0, 0.02]), [0.0; 3]);
    let mut work = PrimitiveWork3::default();
    let time = swept_time(capsule, wedge, 1.0 / 60.0, 0.02, &mut work)
        .expect("fast capsule must reach thin wedge");
    assert!((0.0..=1.0).contains(&time));
    assert!(work.sweep_iterations > 0);
    assert!(work.sweep_iterations < 128);
}

#[test]
fn support_and_bounds_are_allocation_free_value_operations() {
    let wedge = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::wedge([2.0, 1.0, 3.0]),
        [1.0, 2.0, 3.0],
        [0.0; 3],
    );
    let mut work = PrimitiveWork3::default();
    assert_eq!(bounds_extents(wedge), [2.0, 1.0, 3.0]);
    assert_eq!(support_point(wedge, [1.0, 0.0, 0.0], &mut work)[0], 3.0);
    assert_eq!(work.vertex_tests, 6);

    let cuboid =
        PrimitiveBody3::axis_aligned(PrimitiveShape3::cuboid([1.0; 3]), [0.0; 3], [0.0; 3]);
    let tiny = support_point(cuboid, [1.0e-13, 0.0, 0.0], &mut work);
    let scaled = support_point(cuboid, [1.0, 0.0, 0.0], &mut work);
    assert_eq!(tiny[0], 1.0);
    assert_eq!(tiny[0], scaled[0]);

    for shape in [
        PrimitiveShape3::sphere(1.0),
        PrimitiveShape3::capsule(0.7, 0.4),
    ] {
        let body = PrimitiveBody3::axis_aligned(shape, [0.0; 3], [0.0; 3]);
        let tiny = support_point(body, [1.0e-13, 0.0, 0.0], &mut work);
        let scaled = support_point(body, [1.0, 0.0, 0.0], &mut work);
        assert_eq!(tiny, scaled);
    }
}

#[test]
fn capsule_box_penetration_includes_capsule_axis_edge_axes() {
    let inv_sqrt_two = 1.0 / 2.0_f64.sqrt();
    let capsule = PrimitiveBody3::new(
        PrimitiveShape3::capsule(0.9 * 2.0_f64.sqrt(), 0.1),
        [0.0; 3],
        [
            [inv_sqrt_two, inv_sqrt_two, 0.0],
            [inv_sqrt_two, -inv_sqrt_two, 0.0],
            [0.0, 0.0, 1.0],
        ],
        [0.0; 3],
    );
    let cuboid = body(PrimitiveShape3::cuboid([1.0, 1.0, 100.0]), [0.0; 3]);
    let mut work = PrimitiveWork3::default();
    let contact = query(capsule, cuboid, &mut work);
    let expected = -(2.0_f64.sqrt() + 0.1);
    assert!(
        (contact.separation - expected).abs() < 1.0e-12,
        "{contact:?}"
    );
    assert!(work.axes_tested >= 4, "{work:?}");
}

#[test]
fn wedge_inside_tolerance_scales_with_geometry() {
    let wedge = body(PrimitiveShape3::wedge([1.0e-9; 3]), [0.0; 3]);
    let sphere = body(PrimitiveShape3::sphere(1.0e-12), [-1.05e-9, -5.0e-10, 0.0]);
    let mut work = PrimitiveWork3::default();
    assert!(query(sphere, wedge, &mut work).separation > 0.0);
}

#[test]
fn tiny_nonzero_sweep_displacement_is_not_stationary() {
    let left = body(PrimitiveShape3::sphere(1.0e-16), [0.0; 3]);
    let right = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::sphere(1.0e-16),
        [5.0e-16, 0.0, 0.0],
        [-4.0e-16, 0.0, 0.0],
    );
    let mut work = PrimitiveWork3::default();
    let time = swept_time(left, right, 1.0, 0.0, &mut work)
        .expect("tiny but nonzero relative motion must still reach contact");
    assert!((time - 0.75).abs() < 1.0e-12, "time={time}");
}
