//! Independent convex minimization and support certificates for the capsule row.
use geometry_kernels::{
    gjk_distance::{GjkDistanceConfig, GjkDistanceStatus, gjk_distance_with_config},
    math3::{Vec3, add, cross, dot, length, neg, scale, sub},
    primitive3::{PrimitiveBody3, PrimitiveKind3, PrimitiveShape3, PrimitiveWork3, query},
    support::SupportMap3,
};

fn radius_and_half(body: PrimitiveBody3) -> (f64, f64) {
    let extents = body.shape.half_extents();
    match body.shape.kind() {
        PrimitiveKind3::Sphere => (extents[0], 0.0),
        PrimitiveKind3::Capsule => (extents[0], extents[1] - extents[0]),
        _ => (0.0, 0.0),
    }
}

// Independent support formulas, never primitive3::support_point/query helpers.
fn support(body: PrimitiveBody3, direction: Vec3) -> Vec3 {
    match body.shape.kind() {
        PrimitiveKind3::Box => {
            let half = body.shape.half_extents();
            let mut point = body.position;
            for (axis, extent) in body.axes.into_iter().zip(half) {
                let sign = if dot(axis, direction) < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                point = add(point, scale(axis, sign * extent));
            }
            point
        }
        PrimitiveKind3::Sphere | PrimitiveKind3::Capsule => {
            let (radius, half) = radius_and_half(body);
            let sign = if dot(body.axes[1], direction) < 0.0 {
                -1.0
            } else {
                1.0
            };
            let core = add(body.position, scale(body.axes[1], sign * half));
            let norm = length(direction);
            if norm == 0.0 {
                core
            } else {
                add(core, scale(direction, radius / norm))
            }
        }
        PrimitiveKind3::Wedge => panic!("capsule row fixture only"),
    }
}
struct IndependentSupport(PrimitiveBody3);
impl SupportMap3 for IndependentSupport {
    fn support_point(&self, direction: Vec3) -> Vec3 {
        support(self.0, direction)
    }
}

fn axis_point(body: PrimitiveBody3, t: f64) -> Vec3 {
    let (_, half) = radius_and_half(body);
    add(body.position, scale(body.axes[1], (2.0 * t - 1.0) * half))
}
fn squared(vector: Vec3) -> f64 {
    dot(vector, vector)
}
fn distance_to_core(point: Vec3, target: PrimitiveBody3) -> f64 {
    match target.shape.kind() {
        PrimitiveKind3::Sphere => squared(sub(point, target.position)),
        PrimitiveKind3::Capsule => {
            let (_, half) = radius_and_half(target);
            let start = axis_point(target, 0.0);
            let delta = scale(target.axes[1], 2.0 * half);
            let denominator = squared(delta);
            let t = if denominator == 0.0 {
                0.0
            } else {
                (dot(sub(point, start), delta) / denominator).clamp(0.0, 1.0)
            };
            squared(sub(point, add(start, scale(delta, t))))
        }
        PrimitiveKind3::Box => {
            let relative = sub(point, target.position);
            target
                .axes
                .into_iter()
                .zip(target.shape.half_extents())
                .map(|(axis, half)| {
                    let outside = (dot(relative, axis).abs() - half).max(0.0);
                    outside * outside
                })
                .sum()
        }
        PrimitiveKind3::Wedge => panic!("capsule row fixture only"),
    }
}

// A generic one-dimensional convex minimizer, independent of production
// feature breakpoints, cross-product segment solve, SAT axes and clipping.
fn minimize(mut evaluate: impl FnMut(f64) -> f64) -> f64 {
    let ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let (mut low, mut high) = (0.0, 1.0);
    let (mut left, mut right) = (1.0 - ratio, ratio);
    let (mut left_value, mut right_value) = (evaluate(left), evaluate(right));
    let mut best = evaluate(0.0).min(evaluate(1.0));
    for _ in 0..80 {
        best = best.min(left_value).min(right_value);
        if left_value <= right_value {
            high = right;
            right = left;
            right_value = left_value;
            left = high - ratio * (high - low);
            left_value = evaluate(left);
        } else {
            low = left;
            left = right;
            left_value = right_value;
            right = low + ratio * (high - low);
            right_value = evaluate(right);
        }
    }
    best.min(left_value).min(right_value)
}
fn reference_separation(capsule: PrimitiveBody3, other: PrimitiveBody3) -> f64 {
    let core_distance = minimize(|t| distance_to_core(axis_point(capsule, t), other)).sqrt();
    core_distance - radius_and_half(capsule).0 - radius_and_half(other).0
}
fn axes(direction: Vec3) -> [Vec3; 3] {
    let y = scale(direction, 1.0 / length(direction));
    let helper = if y[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let x = cross(y, helper);
    let x = scale(x, 1.0 / length(x));
    [x, y, cross(x, y)]
}
fn pose(shape: PrimitiveShape3, position: Vec3, direction: Vec3) -> PrimitiveBody3 {
    PrimitiveBody3::new(shape, position, axes(direction), [0.0; 3])
}
fn check(capsule: PrimitiveBody3, other: PrimitiveBody3, statuses: &mut [usize; 4]) {
    let geometry_scale = capsule.shape.radius().max(other.shape.radius());
    let tolerance = 2e-8 * geometry_scale;
    let expected = reference_separation(capsule, other);
    let mut work = PrimitiveWork3::default();
    let contact = query(capsule, other, &mut work);
    assert_eq!(work.pair_dispatches.iter().sum::<u64>(), 1);
    assert_eq!(work.support_evaluations, 0);
    assert_eq!(work.vertex_tests, 0);
    match other.shape.kind() {
        PrimitiveKind3::Sphere => {
            assert_eq!(work.segment_distance_evaluations, 1);
            assert!(work.segment_feature_tests <= 1);
        }
        PrimitiveKind3::Capsule => {
            assert_eq!(work.segment_distance_evaluations, 1);
            assert!(work.segment_feature_tests <= 4);
        }
        PrimitiveKind3::Box => {
            assert!(work.segment_distance_evaluations <= 6);
            assert!(work.segment_feature_tests <= 46);
            assert!(work.axes_tested <= 6);
            if contact.separation > tolerance {
                assert_eq!(work.segment_distance_evaluations, 1);
            }
        }
        PrimitiveKind3::Wedge => unreachable!("capsule row fixture only"),
    }
    assert!(contact.normal.into_iter().all(f64::is_finite) && contact.separation.is_finite());
    assert!((length(contact.normal) - 1.0).abs() < 1e-12);
    assert_eq!(
        contact.separation > tolerance,
        expected > tolerance,
        "independent miss mismatch: {capsule:?}, {other:?}, {contact:?}, reference {expected}"
    );
    if other.shape.kind() != PrimitiveKind3::Box || expected > 0.0 {
        assert!(
            (contact.separation - expected).abs() <= tolerance,
            "distance mismatch: {contact:?}, reference {expected}"
        );
    }
    let reverse = query(other, capsule, &mut PrimitiveWork3::default());
    assert!((reverse.separation - contact.separation).abs() <= tolerance);
    if contact.separation > tolerance {
        assert!(length(add(contact.normal, reverse.normal)) < 1e-8);
    }
    for (left, right, result) in [(capsule, other, contact), (other, capsule, reverse)] {
        if result.separation < -tolerance {
            let moved = PrimitiveBody3 {
                position: add(
                    right.position,
                    scale(result.normal, -result.separation + 1e-5 * geometry_scale),
                ),
                ..right
            };
            let gap = dot(
                sub(
                    support(moved, neg(result.normal)),
                    support(left, result.normal),
                ),
                result.normal,
            );
            assert!(
                gap >= 0.99e-5 * geometry_scale,
                "clearance mismatch: {left:?}, {right:?}, {result:?}, gap {gap}"
            );
        }
    }
    let generic = gjk_distance_with_config(
        &IndependentSupport(capsule),
        &IndependentSupport(other),
        GjkDistanceConfig {
            max_iterations: 128,
            relative_tolerance: 1e-12,
            epsilon: 1e-9 * geometry_scale,
        },
    );
    let index = match generic.status {
        GjkDistanceStatus::Converged => {
            let distance = generic
                .closest
                .expect("converged GJK supplies distance")
                .distance;
            assert!(
                (distance - expected.max(0.0)).abs() <= tolerance,
                "generic converged mismatch: {generic:?}, reference {expected}"
            );
            0
        }
        GjkDistanceStatus::Intersecting => {
            assert!(
                expected <= tolerance,
                "generic false intersection: {generic:?}, reference {expected}"
            );
            1
        }
        GjkDistanceStatus::NoProgress => 2,
        GjkDistanceStatus::IterationLimit => 3,
    };
    statuses[index] += 1;
}

#[test]
fn deterministic_capsule_row_matches_independent_distance_and_generic_diagnostics() {
    let mut state = 0x173b_0a55_cafe_1024_u64;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    let mut statuses = [0; 4];
    for dimension_scale in [1e-6, 1.0, 1e6] {
        for _ in 0..256 {
            let a = pose(
                PrimitiveShape3::capsule(
                    (0.25 + random()) * dimension_scale,
                    (0.1 + random() * 0.5) * dimension_scale,
                ),
                std::array::from_fn(|_| (random() - 0.5) * 10.0 * dimension_scale),
                std::array::from_fn(|_| random() - 0.5),
            );
            let center = add(
                a.position,
                std::array::from_fn(|_| (random() - 0.5) * 5.0 * dimension_scale),
            );
            let half = std::array::from_fn(|_| (0.25 + random()) * dimension_scale);
            let direction = std::array::from_fn(|_| random() - 0.5);
            for shape in [
                PrimitiveShape3::sphere(half[0]),
                PrimitiveShape3::cuboid(half),
                PrimitiveShape3::capsule(half[1], half[0]),
            ] {
                check(a, pose(shape, center, direction), &mut statuses);
            }
        }
    }
    println!(
        "capsule row generic diagnostics [converged, intersecting, no-progress, limit]: {statuses:?}"
    );
    assert_eq!(statuses.iter().sum::<usize>(), 2304);
    assert!(statuses[0] > 0 && statuses[1] > 0);
}

#[test]
fn capsule_row_edge_cases_have_independent_clearance_and_bounded_work() {
    let mut statuses = [0; 4];
    for unit in [1e-6, 1.0, 1e6] {
        for half in [0.0, 1e-12 * unit, 2.0 * unit] {
            let a = pose(
                PrimitiveShape3::capsule(half, 0.25 * unit),
                [0.0; 3],
                [0.0, 1.0, 0.0],
            );
            for offset in [
                [0.0; 3],
                [0.0, 2.0 * unit, 0.0],
                [0.75 * unit, 0.0, 0.0],
                [0.75001 * unit, 0.0, 0.0],
                [1.0 * unit, 2.0 * unit, 0.0],
            ] {
                for direction in [
                    [0.0, 1.0, 0.0],
                    [1e-12, 1.0, 0.0],
                    [1e-9, -1.0, 0.0],
                    [1.0, 2.0, 3.0],
                ] {
                    for shape in [
                        PrimitiveShape3::sphere(0.5 * unit),
                        PrimitiveShape3::capsule(1.5 * unit, 0.5 * unit),
                        PrimitiveShape3::cuboid([0.5 * unit, 0.75 * unit, 0.25 * unit]),
                    ] {
                        check(a, pose(shape, offset, direction), &mut statuses);
                    }
                }
            }
        }
    }
    println!(
        "capsule row edge diagnostics [converged, intersecting, no-progress, limit]: {statuses:?}"
    );
    assert_eq!(statuses.iter().sum::<usize>(), 540);
}

#[test]
fn fast_capsule_row_sweeps_match_independent_static_distance_bisection() {
    use geometry_kernels::primitive3::try_swept_time;
    let mut cases = 0;
    for unit in [1e-6, 1.0, 1e6] {
        for direction in [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]] {
            for target_velocity in [-2.0, 0.0, 2.0] {
                for shape in [
                    PrimitiveShape3::sphere(0.25 * unit),
                    PrimitiveShape3::cuboid([0.01 * unit, 2.0 * unit, 2.0 * unit]),
                    PrimitiveShape3::capsule(2.0 * unit, 0.01 * unit),
                ] {
                    let mut a = pose(
                        PrimitiveShape3::capsule(0.5 * unit, 0.2 * unit),
                        [-10.0 * unit, 0.0, 0.0],
                        direction,
                    );
                    let mut b = pose(shape, [0.0; 3], [0.0, 1.0, 0.0]);
                    a.velocity = [20.0 * unit, 0.0, 0.0];
                    b.velocity = [target_velocity * unit, 0.0, 0.0];
                    let separation = |t| {
                        let moved_a = PrimitiveBody3 {
                            position: add(a.position, scale(a.velocity, t)),
                            ..a
                        };
                        let moved_b = PrimitiveBody3 {
                            position: add(b.position, scale(b.velocity, t)),
                            ..b
                        };
                        reference_separation(moved_a, moved_b)
                    };
                    assert!(separation(0.0) > 0.0 && separation(1.0) > 0.0);
                    let (mut low, mut high) = (0.0, 10.0 / (20.0 - target_velocity));
                    assert!(separation(high) < 0.0);
                    for _ in 0..64 {
                        let middle = (low + high) * 0.5;
                        if separation(middle) > 0.0 {
                            low = middle;
                        } else {
                            high = middle;
                        }
                    }
                    let relative_speed = (20.0 - target_velocity) * unit;
                    let tolerance =
                        4e-9 * (1.0 + a.shape.radius() + b.shape.radius()) / relative_speed;
                    for (left, right) in [(a, b), (b, a)] {
                        let mut work = PrimitiveWork3::default();
                        let actual = try_swept_time(left, right, 1.0, 0.0, 128, &mut work)
                            .expect("finite supported sweep succeeds")
                            .expect("crossed thin target must be hit");
                        assert!(
                            (actual - high).abs() <= tolerance,
                            "{shape:?}: actual {actual}, reference {high}"
                        );
                        assert!(work.sweep_iterations > 0 && work.sweep_iterations <= 128);
                        assert!(work.segment_feature_tests <= 46 * work.sweep_iterations);
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 108);
}

#[test]
fn long_thin_capsules_have_fixed_work_without_render_geometry() {
    for half in [1.0, 1e3, 1e6, 1e9] {
        let a = pose(
            PrimitiveShape3::capsule(half, 1e-6),
            [0.0; 3],
            [0.0, 1.0, 0.0],
        );
        for b in [
            pose(PrimitiveShape3::sphere(1e-6), [0.0; 3], [0.0, 1.0, 0.0]),
            pose(
                PrimitiveShape3::capsule(half, 1e-6),
                [0.0; 3],
                [1.0, 0.0, 0.0],
            ),
            pose(
                PrimitiveShape3::cuboid([0.5, 0.5, 0.5]),
                [0.0; 3],
                [0.0, 1.0, 0.0],
            ),
        ] {
            let mut work = PrimitiveWork3::default();
            let contact = query(a, b, &mut work);
            assert!(contact.separation.is_finite() && contact.separation < 0.0);
            assert!(work.segment_distance_evaluations <= 6);
            assert!(work.segment_feature_tests <= 46);
            assert!(work.axes_tested <= 6);
            assert_eq!(work.support_evaluations, 0);
            assert_eq!(work.vertex_tests, 0);
            assert_eq!(work.sweep_iterations, 0);
            assert_eq!(work.pair_dispatches.iter().sum::<u64>(), 1);
        }
    }
}

#[test]
fn separated_segment_crossing_every_box_plane_hits_the_fixed_feature_ceiling() {
    // All six slab-plane times lie inside the skeleton, but their three
    // coordinate intervals do not intersect. Eight breakpoint candidates and
    // seven smooth intervals therefore exercise the entire distance partition.
    let capsule = pose(
        PrimitiveShape3::capsule(56.0_f64.sqrt(), 0.001),
        [0.9, 3.0, 5.0],
        [4.0, 8.0, 12.0],
    );
    let box_body = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::cuboid([0.5, 0.75, 0.25]),
        [0.0; 3],
        [0.0; 3],
    );
    let mut work = PrimitiveWork3::default();
    let contact = query(capsule, box_body, &mut work);
    assert!(contact.separation > 0.0);
    assert_eq!(work.segment_distance_evaluations, 1);
    assert_eq!(work.segment_feature_tests, 46);
    assert_eq!(work.axes_tested, 0);
    assert_eq!(work.support_evaluations, 0);
    assert_eq!(work.vertex_tests, 0);
    assert!((contact.separation - reference_separation(capsule, box_body)).abs() < 1e-10);
}
