//! Native regressions for the numerical audit at ddec70d.
use geometry_kernels::{
    Sphere,
    gjk::gjk_intersection,
    planar::gjk_intersection_planar_xy,
    primitives::{Capsule, Segment3, capsule_capsule, segment_segment},
    sphere_sphere,
    support::{ConvexHull3, SupportMap3},
};

#[test]
fn audit_crossing_short_segments_have_zero_distance() {
    let left = Segment3::new([-1e-7, 0.0, 0.0], [1e-7, 0.0, 0.0]);
    let right = Segment3::new([0.0, -1e-7, 0.0], [0.0, 1e-7, 0.0]);
    let actual = segment_segment(left, right);
    assert!(actual.distance < 1e-15, "{actual:?}");
    assert_eq!(actual.left_parameter, 0.5);
    assert_eq!(actual.right_parameter, 0.5);
}

#[test]
fn audit_crossing_short_capsules_overlap() {
    let left = Capsule::new([-1e-7, 0.0, 0.0], [1e-7, 0.0, 0.0], 1e-8);
    let right = Capsule::new([0.0, -1e-7, 0.0], [0.0, 1e-7, 0.0], 1e-8);
    let actual = capsule_capsule(left, right);
    assert!(
        actual.overlaps,
        "Crossing axes must overlap for positive radii: {actual:?}"
    );
}

#[test]
fn audit_intersecting_near_parallel_segments_do_not_collapse_to_endpoint_case() {
    let left = Segment3::new([0.0, 0.0, 0.0], [1e8, 1.0, 0.0]);
    let right = Segment3::new([0.0, 1.0, 0.0], [1e8, 0.0, 0.0]);
    let actual = segment_segment(left, right);
    assert!(actual.distance < 1e-9, "{actual:?}");
}

#[test]
fn audit_sphere_support_preserves_small_nonzero_direction() {
    let sphere = Sphere::new([0.0; 3], 1.0);
    assert_eq!(sphere.support_point([0.0, 1e-13, 0.0]), [0.0, 1.0, 0.0]);
}

#[test]
fn audit_sphere_support_preserves_large_finite_direction() {
    let sphere = Sphere::new([0.0; 3], 1.0);
    assert_eq!(sphere.support_point([0.0, 1e308, 0.0]), [0.0, 1.0, 0.0]);
}

#[test]
fn audit_capsule_support_preserves_small_nonzero_direction() {
    let capsule = Capsule::new([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0);
    assert_eq!(
        capsule.support_point([0.0, 1e-13, 0.0]),
        capsule.support_point([0.0, 1.0, 0.0])
    );
}

#[test]
fn audit_gjk_agrees_with_sphere_oracle_for_well_resolved_small_shapes() {
    let left = Sphere::new([0.0; 3], 1e-7);
    let right = Sphere::new([1.4e-7, 1.4e-7, 0.0], 1e-7);
    assert!(sphere_sphere(left, right).overlaps);
    let actual = gjk_intersection(&left, &right);
    assert_eq!(actual.intersection(), Some(true), "{actual:?}");
}

#[test]
fn audit_planar_adapter_resolves_small_overlapping_hulls() {
    let s = 1e-7;
    let left_points = [[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]];
    let right_points = left_points.map(|p| [p[0] + 0.5 * s, p[1] + 0.25 * s, 0.0]);
    let left = ConvexHull3::new(&left_points);
    let right = ConvexHull3::new(&right_points);
    let actual = gjk_intersection_planar_xy(&left, &right);
    assert_eq!(actual.intersection(), Some(true), "{actual:?}");
}

#[test]
fn audit_control_unit_scale_spheres_overlap() {
    let left = Sphere::new([0.0; 3], 1.0);
    let right = Sphere::new([1.4, 1.4, 0.0], 1.0);
    assert_eq!(gjk_intersection(&left, &right).intersection(), Some(true));
}

#[test]
fn audit_control_unit_scale_crossing_segments_have_zero_distance() {
    let left = Segment3::new([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let right = Segment3::new([0.0, -1.0, 0.0], [0.0, 1.0, 0.0]);
    assert_eq!(segment_segment(left, right).distance, 0.0);
}

#[test]
fn segment_crossings_are_scale_invariant_across_f32_range() {
    for exponent in [-140, -120, -60, -24, 0, 24, 60, 120] {
        let s = 2.0_f64.powi(exponent) as f32;
        assert!(
            s > 0.0 && s.is_finite(),
            "invalid scale fixture at {exponent}"
        );
        let left = Segment3::new([-s, 0.0, 0.0], [s, 0.0, 0.0]);
        let right = Segment3::new([0.0, -s, 0.0], [0.0, s, 0.0]);
        let relation = segment_segment(left, right);
        assert_eq!(relation.distance, 0.0, "exponent={exponent}: {relation:?}");
        assert_eq!(
            (relation.left_parameter, relation.right_parameter),
            (0.5, 0.5)
        );
        assert!(
            capsule_capsule(
                Capsule::from_segment(left, s / 4.0),
                Capsule::from_segment(right, s / 4.0)
            )
            .overlaps
        );
    }
}

#[test]
fn short_nonzero_segments_are_not_points() {
    use geometry_kernels::primitives::point_segment;
    let end = [1e-15_f32, 0.0, 0.0];
    let segment = Segment3::new([0.0; 3], end);
    let relation = point_segment(end.map(f64::from), segment);
    assert_eq!(relation.parameter, 1.0);
    assert_eq!(relation.distance, 0.0);
    let relation = segment_segment(segment, Segment3::new(end, end));
    assert_eq!(relation.left_parameter, 1.0);
    assert_eq!(relation.distance, 0.0);
}

#[test]
fn near_parallel_skew_and_endpoint_minima_remain_correct() {
    let left = Segment3::new([0.0, 0.0, 0.0], [1e8, 1.0, 0.0]);
    let right = Segment3::new([0.0, 1.0, 2.0], [1e8, 0.0, 2.0]);
    let relation = segment_segment(left, right);
    assert_eq!(relation.distance, 2.0);
    assert_eq!(
        (relation.left_parameter, relation.right_parameter),
        (0.5, 0.5)
    );
    let left = Segment3::new([0.0; 3], [1.0, 0.0, 0.0]);
    let right = Segment3::new([2.0, 1.0, 0.0], [2.0, -1.0, 0.0]);
    let relation = segment_segment(left, right);
    assert_eq!(relation.distance, 1.0);
    assert_eq!(
        (relation.left_parameter, relation.right_parameter),
        (1.0, 0.5)
    );
}

// Independent convex one-dimensional minimization: fix the right parameter,
// project onto the left segment, then minimize over the right parameter. It
// never uses the kernel's cross-product denominator or parallel predicate.
fn distance_oracle(left: Segment3, right: Segment3) -> f64 {
    let a = left.start64();
    let b = left.end64();
    let c = right.start64();
    let d = right.end64();
    let u: [f64; 3] = std::array::from_fn(|i| b[i] - a[i]);
    let u2: f64 = u.iter().map(|x| x * x).sum();
    let at = |t: f64| {
        let q: [f64; 3] = std::array::from_fn(|i| c[i] + t * (d[i] - c[i]));
        let s = if u2 == 0.0 {
            0.0
        } else {
            (u.iter()
                .enumerate()
                .map(|(i, x)| x * (q[i] - a[i]))
                .sum::<f64>()
                / u2)
                .clamp(0.0, 1.0)
        };
        (0..3)
            .map(|i| (a[i] + s * u[i] - q[i]).powi(2))
            .sum::<f64>()
    };
    let (mut low, mut high) = (0.0, 1.0);
    for _ in 0..120 {
        let t1 = (2.0 * low + high) / 3.0;
        let t2 = (low + 2.0 * high) / 3.0;
        if at(t1) < at(t2) {
            high = t2;
        } else {
            low = t1;
        }
    }
    at(0.5 * (low + high)).min(at(0.0)).min(at(1.0)).sqrt()
}

#[test]
fn segments_match_independent_oracle_and_endpoint_permutations() {
    let mut state = 0x8bad_f00d_u32;
    for exponent in [-24, 0, 24] {
        let scale = 2.0_f32.powi(exponent);
        for _ in 0..128 {
            let mut point = || {
                std::array::from_fn(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((state >> 24) as f32 - 128.0) * scale / 16.0
                })
            };
            let left = Segment3::new(point(), point());
            let right = Segment3::new(point(), point());
            let expected = distance_oracle(left, right);
            let tolerance = f64::from(scale) * 1e-10;
            for (left, right) in [
                (left, right),
                (right, left),
                (Segment3::new(left.end, left.start), right),
                (left, Segment3::new(right.end, right.start)),
            ] {
                let actual = segment_segment(left, right);
                assert!(
                    (actual.distance - expected).abs() <= tolerance,
                    "left={left:?} right={right:?} actual={actual:?} oracle={expected}"
                );
                assert!((0.0..=1.0).contains(&actual.left_parameter));
                assert!((0.0..=1.0).contains(&actual.right_parameter));
            }
        }
    }
}

#[test]
fn radial_support_is_invariant_under_positive_direction_rescaling() {
    let sphere = Sphere::new([0.0; 3], 1.0);
    let capsule = Capsule::new([-2.0, -1.0, 0.0], [1.0, 2.0, 0.0], 0.5);
    for direction in [[0.0, 1.0, 0.0], [1.0, -2.0, 3.0], [-5.0, 0.0, 2.0]] {
        let expected_sphere = sphere.support_point(direction);
        let expected_capsule = capsule.support_point(direction);
        for magnitude in [f64::from_bits(1), 1e-308, 1e-150, 1e-13, 1.0, 1e150, 1e307] {
            let scaled = direction.map(|component| component * magnitude);
            for (actual, expected) in [
                (sphere.support_point(scaled), expected_sphere),
                (capsule.support_point(scaled), expected_capsule),
            ] {
                for axis in 0..3 {
                    assert!(
                        (actual[axis] - expected[axis]).abs() < 1e-14,
                        "direction={direction:?}, magnitude={magnitude}, actual={actual:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn capsule_endpoint_projection_cannot_overflow_for_finite_direction() {
    let capsule = Capsule::new([0.0, 2.0, 0.0], [0.0, 3.0, 0.0], 1.0);
    assert_eq!(capsule.support_point([0.0, 1e308, 0.0]), [0.0, 4.0, 0.0]);
}

#[test]
fn zero_direction_keeps_the_documented_deterministic_fallback() {
    let sphere = Sphere::new([1.0, 2.0, 3.0], 2.0);
    assert_eq!(sphere.support_point([-0.0, 0.0, 0.0]), [3.0, 2.0, 3.0]);
}

#[test]
fn gjk_sphere_oracle_and_work_budget_hold_across_scales() {
    for exponent in [-30, -24, -10, 0, 10, 24] {
        let s = 2.0_f64.powi(exponent) as f32;
        let left = Sphere::new([0.0; 3], s);
        for center in [
            [0.0, 0.0, 0.0],
            [1.4, 1.4, 0.0],
            [2.0, 0.0, 0.0],
            [2.05, 0.0, 0.0],
            [1.5, 1.5, 0.0],
            [3.0, 1.0, 0.0],
        ] {
            let right = Sphere::new(center.map(|v| v * s), s);
            let expected = sphere_sphere(left, right).overlaps;
            for (a, b) in [(left, right), (right, left)] {
                let result = gjk_intersection(&a, &b);
                assert_eq!(
                    result.intersection(),
                    Some(expected),
                    "exponent={exponent} center={center:?}: {result:?}"
                );
                assert!(
                    result.iterations <= 8,
                    "easy sphere queries must not exhaust the iteration budget: {result:?}"
                );
            }
        }
    }
}

#[test]
fn scaled_obb_queries_match_sat_and_gjk_trace() {
    use geometry_kernels::{
        gjk_trace::gjk_intersection_trace,
        obb3::{Obb3, obb3_sat},
    };
    for exponent in [-20, -10, 0, 10, 20] {
        let s = 2.0_f64.powi(exponent) as f32;
        let left = Obb3::new([0.0; 3], [1.4 * s, 0.8 * s, 0.9 * s], [0.2, 0.4, -0.15]);
        for center in [[1.2, 0.35, 0.25], [4.0, 1.5, 0.8]] {
            let right = Obb3::new(
                center.map(|v| v * s),
                [s, 0.7 * s, 0.8 * s],
                [-0.3, 0.15, 0.55],
            );
            let result = gjk_intersection(&left, &right);
            assert_eq!(result.intersection(), Some(obb3_sat(left, right).overlaps));
            let trace = gjk_intersection_trace(&left, &right);
            assert_eq!(trace.result, result);
            assert_eq!(trace.steps.len(), result.iterations);
            assert_eq!(
                trace.steps.last().unwrap().terminal_status,
                Some(result.status)
            );
        }
    }
}

#[test]
fn planar_adapter_handles_scaled_overlaps_and_separation() {
    for exponent in [-30, -24, -10, 0, 10, 24] {
        let s = 2.0_f64.powi(exponent);
        let points = [[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]];
        let left = ConvexHull3::new(&points);
        for (offset, expected) in [(0.5, true), (3.0, false)] {
            let other = points.map(|p| [p[0] + offset * s, p[1] + 0.25 * s, 0.0]);
            let right = ConvexHull3::new(&other);
            assert_eq!(
                gjk_intersection_planar_xy(&left, &right).intersection(),
                Some(expected),
                "exponent={exponent} offset={offset}"
            );
        }
    }
}

#[test]
fn configured_gjk_tolerance_scales_across_f32_range() {
    use geometry_kernels::gjk::{GjkConfig, gjk_intersection_with_config};
    for exponent in [-140, -120, -60, -24, 0, 24, 60, 120] {
        let s = 2.0_f64.powi(exponent) as f32;
        let config = GjkConfig {
            epsilon: f64::from(s) * 1e-12,
            ..GjkConfig::default()
        };
        let left = Sphere::new([0.0; 3], s);
        for factor in [1.4, 1.5] {
            let right = Sphere::new([factor * s, factor * s, 0.0], s);
            let result = gjk_intersection_with_config(&left, &right, config);
            assert_eq!(
                result.intersection(),
                Some(sphere_sphere(left, right).overlaps),
                "exponent={exponent} factor={factor}: {result:?}"
            );
            assert!(result.iterations <= 8);
        }
    }
}

fn regular_polygon(count: usize, offset: [f64; 2]) -> Vec<[f64; 3]> {
    (0..count)
        .map(|i| {
            let angle = std::f64::consts::TAU * i as f64 / count as f64;
            [angle.cos() + offset[0], angle.sin() + offset[1], 0.0]
        })
        .collect()
}

// Independent polygon SAT penetration oracle: project every vertex onto each
// edge normal of both shapes. It does not use GJK simplexes or EPA expansion.
fn polygon_sat_depth(left: &[[f64; 3]], right: &[[f64; 3]]) -> f64 {
    let mut depth = f64::INFINITY;
    for points in [left, right] {
        for i in 0..points.len() {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            let (x, y) = (b[1] - a[1], a[0] - b[0]);
            let length = x.hypot(y);
            let project = |vertices: &[[f64; 3]]| {
                vertices
                    .iter()
                    .map(|p| (p[0] * x + p[1] * y) / length)
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| {
                        (min.min(v), max.max(v))
                    })
            };
            let (left_min, left_max) = project(left);
            let (right_min, right_max) = project(right);
            depth = depth.min((left_max - right_min).min(right_max - left_min));
        }
    }
    depth
}

#[test]
fn epa_expansion_remains_convex_and_matches_independent_sat() {
    use geometry_kernels::epa::{
        EpaStatus, epa_penetration_planar_xy, epa_penetration_trace_planar_xy,
    };
    for count in [4, 8, 16, 32, 64] {
        let left_points = regular_polygon(count, [0.0, 0.0]);
        for offset in [[1.25, 0.2], [0.3, -0.5], [-0.75, 0.1]] {
            let right_points = regular_polygon(count, offset);
            let left = ConvexHull3::new(&left_points);
            let right = ConvexHull3::new(&right_points);
            let gjk = gjk_intersection_planar_xy(&left, &right);
            assert_eq!(gjk.intersection(), Some(true));
            let trace = epa_penetration_trace_planar_xy(&left, &right, &gjk);
            assert_eq!(trace.result, epa_penetration_planar_xy(&left, &right, &gjk));
            assert_eq!(
                trace.result.status,
                EpaStatus::Converged,
                "count={count} offset={offset:?}: {trace:?}"
            );
            let expected = polygon_sat_depth(&left_points, &right_points);
            assert!((trace.result.penetration.unwrap().depth - expected).abs() <= 1e-9);
            for step in &trace.steps {
                for i in 0..step.polytope.len() {
                    let a = step.polytope[i].point;
                    let b = step.polytope[(i + 1) % step.polytope.len()].point;
                    let c = step.polytope[(i + 2) % step.polytope.len()].point;
                    let turn = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
                    let origin_side = (b[0] - a[0]) * -a[1] - (b[1] - a[1]) * -a[0];
                    assert!(turn >= -1e-12, "nonconvex step {step:?}");
                    assert!(origin_side >= -1e-12, "origin escaped {step:?}");
                }
            }
        }
    }
}
