//! Experimental exact tetrahedron/octahedron translational swept SAT (issue #188).
//!
//! Quick rerun: `cargo test -p geometry-kernels --locked --test platonic_swept_sat`.

mod support;

use geometry_kernels::math3::{Vec3, cross, dot, length, scale, sub};
use geometry_kernels::primitive3::platonic::{
    PlatonicBody3, PlatonicKind3, PlatonicShape3, PlatonicSweepInterval3, PlatonicWork3,
    try_platonic_swept_interval,
};
use geometry_kernels::primitive3::{
    PrimitiveBody3, PrimitiveShape3, PrimitiveSweepError3, PrimitiveWork3, try_swept_time,
};
use support::platonic::{
    DT, Lcg, REFERENCE_DISTANCE_TOLERANCE, ReferenceCast, axis_angle, conservative_advancement,
    random_pairs, separation,
};

/// Fixture identities (vertex radii, face planes, edge lengths) hold to this relative error.
const FIXTURE_TOLERANCE: f64 = 1.0e-14;
/// Closed-form contact times are reproduced to this normalized-time error.
const ANALYTIC_TIME_TOLERANCE: f64 = 1.0e-12;
/// Agreement with the conservative-advancement reference in normalized time. The reference
/// stops within `REFERENCE_DISTANCE_TOLERANCE * R` of contact while closing at `>= 0.1 R`
/// per window in these fixtures, so its time error is far below this bound.
const REFERENCE_TIME_TOLERANCE: f64 = 1.0e-7;
/// Sampled-miss certificate: every sample keeps more than this relative separation unless
/// the miss is a deliberate graze.
const SAMPLE_COUNT: usize = 65;

const BUDGET: u32 = 64;

fn tetra(r: f64) -> PlatonicShape3 {
    PlatonicShape3::tetrahedron(r)
}

fn octa(r: f64) -> PlatonicShape3 {
    PlatonicShape3::octahedron(r)
}

fn cube(h: f64) -> PlatonicShape3 {
    PlatonicShape3::cuboid([h, 0.75 * h, 1.25 * h])
}

fn sweep(
    a: PlatonicBody3,
    b: PlatonicBody3,
    dt: f64,
    margin: f64,
) -> (Option<PlatonicSweepInterval3>, PlatonicWork3) {
    let mut work = PlatonicWork3::default();
    let result = try_platonic_swept_interval(a, b, dt, margin, BUDGET, &mut work)
        .expect("finite fixtures resolve within budget");
    (result, work)
}

fn close(actual: f64, expected: f64, tolerance: f64) -> bool {
    (actual - expected).abs() <= tolerance
}

#[test]
fn fixture_convention_vertices_faces_and_edges() {
    let sqrt3 = 3.0_f64.sqrt();
    for (shape, inradius, edge_length, faces, edges) in [
        (tetra(2.0), 2.0 / 3.0, 2.0 * (8.0_f64 / 3.0).sqrt(), 4, 6),
        (octa(2.0), 2.0 / sqrt3, 2.0 * 2.0_f64.sqrt(), 4, 6),
    ] {
        let (vertices, len) = shape.local_vertices();
        let vertices = &vertices[..len];
        let centroid = vertices
            .iter()
            .fold([0.0; 3], |sum, v| geometry_kernels::math3::add(sum, *v));
        assert!(length(centroid) <= FIXTURE_TOLERANCE);
        for v in vertices {
            assert!(close(length(*v), 2.0, 2.0 * FIXTURE_TOLERANCE));
        }
        assert_eq!(shape.local_face_axes().len(), faces);
        for &normal in shape.local_face_axes() {
            assert!(close(length(normal), 1.0, FIXTURE_TOLERANCE));
            // Each canonical axis and (octahedron) its negation bound a real face: exactly
            // three vertices lie on the support plane at the inradius.
            let signs: &[f64] = if shape.kind() == PlatonicKind3::Octahedron {
                &[1.0, -1.0]
            } else {
                &[1.0]
            };
            for &sign in signs {
                let n = scale(normal, sign);
                let on_plane = vertices
                    .iter()
                    .filter(|v| close(dot(**v, n), inradius, 4.0 * FIXTURE_TOLERANCE))
                    .count();
                assert_eq!(on_plane, 3, "{shape:?} face {n:?}");
                assert!(
                    vertices
                        .iter()
                        .all(|v| dot(*v, n) <= inradius + 4.0 * FIXTURE_TOLERANCE)
                );
            }
        }
        // Every polyhedron edge (vertex pair at edge length) is parallel to exactly one
        // listed direction, and every listed direction is used.
        let directions = shape.local_edge_directions();
        assert_eq!(directions.len(), edges);
        let mut used = vec![0; edges];
        let mut edge_count = 0;
        for i in 0..len {
            for j in i + 1..len {
                let delta = sub(vertices[j], vertices[i]);
                if !close(length(delta), edge_length, 4.0 * FIXTURE_TOLERANCE) {
                    continue;
                }
                edge_count += 1;
                let matches: Vec<_> = (0..edges)
                    .filter(|&k| length(cross(delta, directions[k])) <= 1.0e-13)
                    .collect();
                assert_eq!(matches.len(), 1);
                used[matches[0]] += 1;
            }
        }
        let expected_edges = if shape.kind() == PlatonicKind3::Tetrahedron {
            6
        } else {
            12
        };
        assert_eq!(edge_count, expected_edges);
        assert!(used.iter().all(|&count| count > 0));
    }
}

#[test]
fn support_points_match_vertex_enumeration_and_break_ties_deterministically() {
    let mut rng = Lcg(0x188);
    for shape in [tetra(1.3), octa(0.7), cube(0.9)] {
        for _ in 0..512 {
            let body = PlatonicBody3::new(shape, [0.5, -2.0, 3.0], rng.rotation(), [0.0; 3]);
            let direction = scale(rng.direction(), 1.0 + 10.0 * rng.unit());
            let mut work = PlatonicWork3::default();
            let support = body.support_point(direction, &mut work);
            let (vertices, len) = body.world_vertices(0.0);
            let best = vertices[..len]
                .iter()
                .map(|v| dot(*v, direction))
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(close(
                dot(support, direction),
                best,
                1.0e-12 * best.abs().max(1.0)
            ));
            assert_eq!(work.support_queries, 1);
        }
    }
    let mut work = PlatonicWork3::default();
    // Tetrahedron vertices 0 and 1 tie along +x: the lower index wins.
    let body = PlatonicBody3::axis_aligned(tetra(1.0), [0.0; 3], [0.0; 3]);
    let first = body.support_point([1.0, 0.0, 0.0], &mut work);
    assert_eq!(first, tetra(1.0).local_vertices().0[0]);
    assert_eq!(body.support_point([1.0, 0.0, 0.0], &mut work), first);
    // Octahedron +x/+y tie selects +x; zero components select the positive side.
    let body = PlatonicBody3::axis_aligned(octa(1.0), [0.0; 3], [0.0; 3]);
    assert_eq!(
        body.support_point([1.0, 1.0, 0.0], &mut work),
        [1.0, 0.0, 0.0]
    );
    assert_eq!(body.support_point([0.0; 3], &mut work), [1.0, 0.0, 0.0]);
    let body = PlatonicBody3::axis_aligned(cube(1.0), [0.0; 3], [0.0; 3]);
    assert_eq!(body.support_point([0.0; 3], &mut work), [1.0, 0.75, 1.25]);
}

#[test]
fn entering_and_exiting_intervals_match_closed_form() {
    // Octahedra meet vertex-to-vertex when centres are 2R apart along x.
    let a = PlatonicBody3::axis_aligned(octa(1.0), [0.0; 3], [0.0; 3]);
    let b = PlatonicBody3::axis_aligned(octa(1.0), [5.0, 0.0, 0.0], [-10.0, 0.0, 0.0]);
    let (hit, work) = sweep(a, b, 1.0, 0.0);
    let hit = hit.expect("head-on octahedra collide");
    assert!(close(hit.enter, 0.3, ANALYTIC_TIME_TOLERANCE));
    assert!(close(hit.exit, 0.7, ANALYTIC_TIME_TOLERANCE));
    // Aligned octahedra: 8 face axes plus 30 non-parallel edge crosses.
    assert_eq!(work.axes_tested, 38);

    // Identical tetrahedra meet edge-to-edge (A's +x edge, B's -x edge) at 2R/sqrt(3).
    let gap = 2.0 / 3.0_f64.sqrt();
    let a = PlatonicBody3::axis_aligned(tetra(1.0), [0.0; 3], [0.0; 3]);
    let b = PlatonicBody3::axis_aligned(tetra(1.0), [3.0, 0.0, 0.0], [-6.0, 0.0, 0.0]);
    let hit = sweep(a, b, 1.0, 0.0).0.expect("tetrahedra collide");
    assert!(close(hit.enter, (3.0 - gap) / 6.0, ANALYTIC_TIME_TOLERANCE));
    assert!(close(hit.exit, (3.0 + gap) / 6.0, ANALYTIC_TIME_TOLERANCE));

    // A window that ends during contact clips exit to 1.
    let hit = sweep(a, b, 0.5, 0.0).0.expect("contact before window end");
    assert!(close(hit.enter, (3.0 - gap) / 3.0, ANALYTIC_TIME_TOLERANCE));
    assert_eq!(hit.exit, 1.0);

    // The margin widens the interval symmetrically.
    let hit = sweep(a, b, 1.0, 0.6).0.expect("margin contact");
    assert!(close(
        hit.enter,
        (3.0 - gap - 0.6) / 6.0,
        ANALYTIC_TIME_TOLERANCE
    ));
    assert!(close(
        hit.exit,
        (3.0 + gap + 0.6) / 6.0,
        ANALYTIC_TIME_TOLERANCE
    ));
}

#[test]
fn initial_overlap_reports_time_zero() {
    let a = PlatonicBody3::axis_aligned(tetra(1.0), [0.0; 3], [0.0; 3]);
    let b = PlatonicBody3::new(
        octa(1.0),
        [0.2, 0.1, 0.0],
        axis_angle([0.0, 0.6, 0.8], 0.4),
        [0.0; 3],
    );
    assert_eq!(
        sweep(a, b, DT, 0.0).0,
        Some(PlatonicSweepInterval3 {
            enter: 0.0,
            exit: 1.0
        })
    );
    // Separating from overlap: enter stays zero, exit is when octahedron vertices part.
    let a = PlatonicBody3::axis_aligned(octa(1.0), [0.0; 3], [0.0; 3]);
    let b = PlatonicBody3::axis_aligned(octa(1.0), [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]);
    let hit = sweep(a, b, 1.0, 0.0).0.expect("initial overlap");
    assert_eq!(hit.enter, 0.0);
    assert!(close(hit.exit, 0.25, ANALYTIC_TIME_TOLERANCE));
    // Within margin but not touching still counts at time zero.
    let b = PlatonicBody3::axis_aligned(octa(1.0), [2.05, 0.0, 0.0], [0.0; 3]);
    assert_eq!(sweep(a, b, DT, 0.1).0.map(|hit| hit.enter), Some(0.0));
    assert_eq!(sweep(a, b, DT, 0.0).0, None);
}

#[test]
fn separated_and_non_closing_trajectories_miss() {
    let a = PlatonicBody3::axis_aligned(tetra(1.0), [0.0; 3], [0.0; 3]);
    for (position, velocity) in [
        ([3.0, 0.0, 0.0], [6.0, 0.0, 0.0]),  // receding
        ([3.0, 0.0, 0.0], [0.0, 50.0, 0.0]), // parallel pass with a gap
        ([3.0, 0.0, 0.0], [0.0; 3]),         // stationary and apart
        ([3.0, 0.0, 0.0], [-0.5, 0.0, 0.0]), // closing but not reaching
    ] {
        let b = PlatonicBody3::axis_aligned(octa(1.0), position, velocity);
        assert_eq!(sweep(a, b, 1.0, 0.0).0, None, "{position:?} {velocity:?}");
        assert!(matches!(
            conservative_advancement(a, b, 1.0, 0.0),
            ReferenceCast::Miss { .. }
        ));
    }
}

#[test]
fn edge_edge_graze_is_a_contact_interval_and_offset_graze_misses() {
    // A's +x edge (direction (0,1,1)) and B's -x edge (direction (0,1,-1)) lie in the plane
    // x = 1/sqrt(3) and slide across each other for |shift| <= 2/sqrt(3).
    let gap = 2.0 / 3.0_f64.sqrt();
    let a = PlatonicBody3::axis_aligned(tetra(1.0), [0.0; 3], [0.0; 3]);
    let b = PlatonicBody3::axis_aligned(tetra(1.0), [gap, -3.0, 0.0], [0.0, 6.0, 0.0]);
    let margin = 1.0e-12;
    let hit = sweep(a, b, 1.0, margin).0.expect("grazing edges touch");
    assert!(close(hit.enter, (3.0 - gap) / 6.0, 1.0e-11));
    assert!(close(hit.exit, (3.0 + gap) / 6.0, 1.0e-11));
    for step in 0..=8 {
        let t = hit.enter + (hit.exit - hit.enter) * f64::from(step) / 8.0;
        let (distance, _) = separation(a, b, t).expect("reference resolves");
        assert!(distance <= margin + REFERENCE_DISTANCE_TOLERANCE);
    }

    let shifted = PlatonicBody3 {
        position: [gap + 1.0e-6, -3.0, 0.0],
        ..b
    };
    let (miss, work) = sweep(a, shifted, 1.0, 0.0);
    assert_eq!(miss, None);
    // Only the edge-cross axis (the x axis) separates these solids.
    assert!(work.axes_tested > 4);
    for step in 0..SAMPLE_COUNT {
        let t = step as f64 / (SAMPLE_COUNT - 1) as f64;
        let (distance, _) = separation(a, shifted, t).expect("reference resolves");
        assert!(distance > 0.5e-6);
    }
}

#[test]
fn nearly_parallel_faces_agree_with_reference_and_never_miss() {
    let axis = [0.48, 0.6, 0.64];
    for shape in [tetra(1.0), octa(1.0), cube(1.0)] {
        let normal = scale(shape.local_face_axes()[0], -1.0);
        for angle in [1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12, 0.0] {
            let a = PlatonicBody3::axis_aligned(shape, [0.0; 3], [0.0; 3]);
            // B approaches A's face 0 (negated for octahedron/box so it is an outward face).
            let approach = if shape.kind() == PlatonicKind3::Tetrahedron {
                scale(normal, -1.0)
            } else {
                normal
            };
            let b = PlatonicBody3::new(
                shape,
                scale(approach, 4.0),
                axis_angle(axis, angle),
                scale(approach, -6.0),
            );
            let hit = sweep(a, b, 1.0, 0.0)
                .0
                .expect("approach along a face normal hits");
            match conservative_advancement(a, b, 1.0, 0.0) {
                ReferenceCast::Hit { time, .. } => {
                    assert!(
                        close(hit.enter, time, REFERENCE_TIME_TOLERANCE),
                        "{shape:?} angle {angle}: {} vs {time}",
                        hit.enter
                    );
                }
                other => panic!("reference {other:?} for {shape:?} angle {angle}"),
            }
        }
    }
}

#[test]
fn uniform_scale_preserves_normalized_interval() {
    let base: [[PlatonicBody3; 2]; 32] =
        random_pairs(0x5ca1e, tetra(1.0), octa(0.8), (0.0, 0.3), false);
    for [a, b] in base {
        let (reference, _) = sweep(a, b, DT, 0.0);
        let reference = reference.expect("offsets below the summed inradii hit");
        for factor in [1.0e-3, 1.0e3, 1.0e6] {
            let scaled = |body: PlatonicBody3, shape| PlatonicBody3 {
                shape,
                position: scale(body.position, factor),
                velocity: scale(body.velocity, factor),
                ..body
            };
            let a = scaled(a, tetra(factor));
            let b = scaled(b, octa(0.8 * factor));
            let hit = sweep(a, b, DT, 0.0).0.expect("scaled pair hits");
            assert!(close(hit.enter, reference.enter, 1.0e-12));
            assert!(close(hit.exit, reference.exit, 1.0e-12));
        }
    }
}

#[test]
fn repeated_and_swapped_queries_are_deterministic() {
    let pairs: [[PlatonicBody3; 2]; 64] =
        random_pairs(0xd37, octa(1.0), tetra(1.2), (0.0, 1.2), false);
    for [a, b] in pairs {
        let (first, first_work) = sweep(a, b, DT, 0.01);
        let (second, second_work) = sweep(a, b, DT, 0.01);
        assert_eq!(first, second);
        assert_eq!(first_work, second_work);
        let (swapped, _) = sweep(b, a, DT, 0.01);
        match (first, swapped) {
            (Some(x), Some(y)) => {
                assert!(close(x.enter, y.enter, ANALYTIC_TIME_TOLERANCE));
                assert!(close(x.exit, y.exit, ANALYTIC_TIME_TOLERANCE));
            }
            (None, None) => {}
            other => panic!("order changed the outcome: {other:?}"),
        }
    }
}

/// Differential test against the generic GJK conservative-advancement cast. Entry agrees
/// with the reference cast, exit agrees with the reversed reference cast, and every miss is
/// certified by sampled positive separation.
#[test]
fn random_sweeps_match_generic_convex_reference() {
    let shapes = [
        (tetra(1.0), tetra(0.7)),
        (octa(1.0), octa(1.3)),
        (tetra(1.0), octa(1.0)),
        (tetra(0.9), cube(1.0)),
        (octa(1.1), cube(0.8)),
        (cube(1.0), cube(0.6)),
    ];
    let mut hits = 0;
    let mut misses = 0;
    for (seed, (left, right)) in shapes.into_iter().enumerate() {
        for receding in [false, true] {
            let pairs: [[PlatonicBody3; 2]; 192] =
                random_pairs(0x188_0000 + seed as u64, left, right, (0.0, 1.0), receding);
            for [a, b] in pairs {
                let (result, _) = sweep(a, b, DT, 0.0);
                let reference = conservative_advancement(a, b, DT, 0.0);
                match (result, reference) {
                    (Some(hit), ReferenceCast::Hit { time, .. }) => {
                        hits += 1;
                        assert!(close(hit.enter, time, REFERENCE_TIME_TOLERANCE));
                        // Exit: the reversed cast from the window end.
                        let end = |body: PlatonicBody3| PlatonicBody3 {
                            position: geometry_kernels::math3::add(
                                body.position,
                                scale(body.velocity, DT),
                            ),
                            velocity: scale(body.velocity, -1.0),
                            ..body
                        };
                        match conservative_advancement(end(a), end(b), DT, 0.0) {
                            ReferenceCast::Hit { time, .. } => {
                                assert!(close(hit.exit, 1.0 - time, REFERENCE_TIME_TOLERANCE));
                            }
                            other => panic!("reverse reference {other:?}"),
                        }
                    }
                    (None, ReferenceCast::Miss { .. }) => {
                        misses += 1;
                        let size = a.shape.circumradius() + b.shape.circumradius();
                        for step in 0..SAMPLE_COUNT {
                            let t = DT * step as f64 / (SAMPLE_COUNT - 1) as f64;
                            let (distance, _) = separation(a, b, t).expect("reference resolves");
                            assert!(distance > REFERENCE_DISTANCE_TOLERANCE * size);
                        }
                    }
                    other => panic!("{left:?}/{right:?} disagreement: {other:?}"),
                }
            }
        }
    }
    assert!(hits > 400 && misses > 900, "hits {hits}, misses {misses}");
}

/// The platonic box operand reproduces the existing `primitive3` box/box swept SAT.
#[test]
fn box_pairs_reproduce_existing_primitive_box_sweep() {
    let pairs: [[PlatonicBody3; 2]; 256] =
        random_pairs(0xb0c5, cube(1.0), cube(0.6), (0.0, 1.0), false);
    let as_primitive = |body: PlatonicBody3, half: Vec3| {
        PrimitiveBody3::new(
            PrimitiveShape3::cuboid(half),
            body.position,
            body.axes,
            body.velocity,
        )
    };
    for [a, b] in pairs {
        let (result, work) = sweep(a, b, DT, 0.01);
        let mut baseline_work = PrimitiveWork3::default();
        let baseline = try_swept_time(
            as_primitive(a, [1.0, 0.75, 1.25]),
            as_primitive(b, [0.6, 0.45, 0.75]),
            DT,
            0.01,
            BUDGET,
            &mut baseline_work,
        )
        .expect("baseline resolves");
        match (result, baseline) {
            (Some(hit), Some(time)) => assert!(close(hit.enter, time, ANALYTIC_TIME_TOLERANCE)),
            (None, None) => {}
            other => panic!("box parity: {other:?}"),
        }
        assert_eq!(work.axes_tested, baseline_work.axes_tested);
        assert_eq!(work.support_queries, baseline_work.support_evaluations);
    }
}

#[test]
fn work_counters_for_generic_orientations() {
    // Generic orientations keep every face axis and edge cross.
    for (left, right, axes, vertex_tests_per_axis) in [
        (tetra(1.0), tetra(1.0), 44, 8),
        (octa(1.0), octa(1.0), 44, 6),
        (tetra(1.0), octa(1.0), 44, 7),
        (tetra(1.0), cube(1.0), 25, 7),
        (octa(1.0), cube(1.0), 25, 6),
        (cube(1.0), cube(1.0), 15, 6),
    ] {
        let [[a, b]]: [[PlatonicBody3; 2]; 1] = random_pairs(7, left, right, (0.0, 0.0), false);
        let (hit, work) = sweep(a, b, DT, 0.0);
        assert!(hit.is_some());
        // A hit inspects every candidate axis: swept SAT has no early accept.
        assert_eq!(work.axes_tested, axes, "{left:?}/{right:?}");
        assert_eq!(work.support_queries, 4 * axes);
        assert_eq!(work.vertex_tests, vertex_tests_per_axis * axes);
    }
    // Aligned identical solids skip the six parallel-edge crosses.
    let a = PlatonicBody3::axis_aligned(tetra(1.0), [0.0; 3], [0.0; 3]);
    let b = PlatonicBody3::axis_aligned(tetra(1.0), [3.0, 0.0, 0.0], [-6.0, 0.0, 0.0]);
    assert_eq!(sweep(a, b, 1.0, 0.0).1.axes_tested, 38);
}

#[test]
fn budget_and_invalid_input_are_errors_not_misses() {
    let a = PlatonicBody3::axis_aligned(tetra(1.0), [0.0; 3], [0.0; 3]);
    let b = PlatonicBody3::new(
        octa(1.0),
        [3.0, 0.2, 0.1],
        axis_angle([0.0, 0.6, 0.8], 0.3),
        [-6.0, 0.0, 0.0],
    );
    let mut work = PlatonicWork3::default();
    assert_eq!(
        try_platonic_swept_interval(a, b, 1.0, 0.0, 5, &mut work),
        Err(PrimitiveSweepError3::IterationLimit)
    );
    assert_eq!(work.axes_tested, 5);
    for (dt, margin, position) in [
        (-1.0, 0.0, [0.0; 3]),
        (f64::NAN, 0.0, [0.0; 3]),
        (1.0, -1.0, [0.0; 3]),
        (1.0, 0.0, [f64::INFINITY, 0.0, 0.0]),
    ] {
        let a = PlatonicBody3 { position, ..a };
        assert_eq!(
            try_platonic_swept_interval(a, b, dt, margin, BUDGET, &mut work),
            Err(PrimitiveSweepError3::InvalidInput)
        );
    }
    let fast = PlatonicBody3 {
        velocity: [f64::MAX, 0.0, 0.0],
        ..b
    };
    let slow = PlatonicBody3 {
        velocity: [-f64::MAX, 0.0, 0.0],
        ..a
    };
    assert_eq!(
        try_platonic_swept_interval(slow, fast, 1.0, 0.0, BUDGET, &mut work),
        Err(PrimitiveSweepError3::NonFiniteComputation)
    );
}

/// Work ratchet for the Divan `platonic` batches (64 pairs each): a hit pays every
/// candidate axis, a receding pair usually stops on the first one or two.
#[test]
fn bench_batch_work_ratchet() {
    for (left, right, axes_per_hit) in [
        (tetra(1.0), tetra(0.8), 44),
        (octa(1.0), octa(0.8), 44),
        (tetra(1.0), PlatonicShape3::cuboid([0.8, 0.6, 1.0]), 25),
    ] {
        for receding in [false, true] {
            let offsets = if receding { (0.0, 1.0) } else { (0.0, 0.2) };
            let pairs: [[PlatonicBody3; 2]; 64] =
                random_pairs(0x188, left, right, offsets, receding);
            let mut work = PlatonicWork3::default();
            let mut hits = 0;
            for [a, b] in pairs {
                hits += usize::from(
                    try_platonic_swept_interval(a, b, DT, 0.0, BUDGET, &mut work)
                        .expect("resolves")
                        .is_some(),
                );
            }
            if receding {
                assert_eq!(hits, 0);
                assert!(work.axes_tested <= 2 * 64, "{}", work.axes_tested);
            } else {
                assert_eq!(hits, 64);
                assert_eq!(work.axes_tested, axes_per_hit * 64);
            }
            assert_eq!(work.support_queries, 4 * work.axes_tested);
        }
    }
}
