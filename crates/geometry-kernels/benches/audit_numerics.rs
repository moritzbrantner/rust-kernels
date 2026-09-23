//! Workload v1. Correctness probes are separate from allocation-profiled timing.
//! `scripts/audit-evidence.py` compiles this exact harness against both revisions.
use divan::{Bencher, black_box};
use geometry_kernels::{
    Sphere,
    epa::{epa_penetration_planar_xy, epa_penetration_trace_planar_xy},
    gjk::{GjkResult, GjkStatus, gjk_intersection},
    planar::gjk_intersection_planar_xy,
    primitives::{Capsule, Segment3, capsule_capsule, segment_segment},
    support::{ConvexHull3, SupportMap3},
};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    if std::env::args().any(|argument| argument == "--audit-probe") {
        probe();
    } else {
        divan::main();
    }
}

fn crossing(scale: f32) -> (Segment3, Segment3) {
    (
        Segment3::new([-scale, 0.0, 0.0], [scale, 0.0, 0.0]),
        Segment3::new([0.0, -scale, 0.0], [0.0, scale, 0.0]),
    )
}

fn spheres(exponent: i32) -> (Sphere, Sphere) {
    let scale = 2.0_f64.powi(exponent) as f32;
    (
        Sphere::new([0.0; 3], scale),
        Sphere::new([1.4 * scale, 1.4 * scale, 0.0], scale),
    )
}

fn polygon(count: usize, offset: [f64; 2]) -> Vec<[f64; 3]> {
    (0..count)
        .map(|i| {
            let angle = std::f64::consts::TAU * i as f64 / count as f64;
            [angle.cos() + offset[0], angle.sin() + offset[1], 0.0]
        })
        .collect()
}

fn epa_seed(left: &ConvexHull3<'_>, right: &ConvexHull3<'_>) -> GjkResult {
    let gjk = gjk_intersection_planar_xy(left, right);
    assert_eq!(gjk.status, GjkStatus::Intersecting, "invalid EPA workload");
    gjk
}

#[divan::bench(args = [-24, 0, 24])]
fn segment_crossing(bencher: Bencher, exponent: i32) {
    let (left, right) = crossing(2.0_f64.powi(exponent) as f32);
    bencher.bench_local(|| black_box(segment_segment(black_box(left), black_box(right))));
}

#[divan::bench]
fn segment_near_parallel(bencher: Bencher) {
    let left = Segment3::new([0.0; 3], [1e8, 1.0, 0.0]);
    let right = Segment3::new([0.0, 1.0, 0.0], [1e8, 0.0, 0.0]);
    bencher.bench_local(|| black_box(segment_segment(black_box(left), black_box(right))));
}

#[divan::bench]
fn capsule_small_crossing(bencher: Bencher) {
    let (left, right) = crossing(1e-7);
    let left = Capsule::from_segment(left, 1e-8);
    let right = Capsule::from_segment(right, 1e-8);
    bencher.bench_local(|| black_box(capsule_capsule(black_box(left), black_box(right))));
}

#[divan::bench(args = [-1023, -44, 0, 1023])]
fn sphere_support(bencher: Bencher, exponent: i32) {
    let sphere = Sphere::new([0.0; 3], 1.0);
    let direction = [0.0, 2.0_f64.powi(exponent), 0.0];
    bencher.bench_local(|| black_box(sphere.support_point(black_box(direction))));
}

#[divan::bench(args = [-1023, -44, 0, 1023])]
fn capsule_support(bencher: Bencher, exponent: i32) {
    let capsule = Capsule::new([0.0, 2.0, 0.0], [0.0, 3.0, 0.0], 1.0);
    let direction = [0.0, 2.0_f64.powi(exponent), 0.0];
    bencher.bench_local(|| black_box(capsule.support_point(black_box(direction))));
}

#[divan::bench(args = [-24, 0, 24])]
fn gjk_overlap(bencher: Bencher, exponent: i32) {
    let (left, right) = spheres(exponent);
    bencher.bench_local(|| black_box(gjk_intersection(black_box(&left), black_box(&right))));
}

#[divan::bench(args = [4, 16, 64])]
fn epa_no_trace(bencher: Bencher, vertices: usize) {
    let left_points = polygon(vertices, [0.0, 0.0]);
    let right_points = polygon(vertices, [1.25, 0.2]);
    let left = ConvexHull3::new(&left_points);
    let right = ConvexHull3::new(&right_points);
    let gjk = epa_seed(&left, &right);
    bencher.bench_local(|| {
        black_box(epa_penetration_planar_xy(
            black_box(&left),
            black_box(&right),
            black_box(&gjk),
        ))
    });
}

#[divan::bench(args = [4, 16, 64])]
fn epa_trace(bencher: Bencher, vertices: usize) {
    let left_points = polygon(vertices, [0.0, 0.0]);
    let right_points = polygon(vertices, [1.25, 0.2]);
    let left = ConvexHull3::new(&left_points);
    let right = ConvexHull3::new(&right_points);
    let gjk = epa_seed(&left, &right);
    bencher.bench_local(|| {
        black_box(epa_penetration_trace_planar_xy(
            black_box(&left),
            black_box(&right),
            black_box(&gjk),
        ))
    });
}

fn emit(case: &str, metric: &str, value: impl std::fmt::Debug) {
    println!("AUDIT\t{case}\t{metric}\t{value:?}");
}

fn probe() {
    let (left, right) = crossing(1e-7);
    let relation = segment_segment(left, right);
    emit("segment_small", "distance", relation.distance);
    emit("segment_small", "left_parameter", relation.left_parameter);
    emit("segment_small", "right_parameter", relation.right_parameter);
    emit(
        "capsule_small",
        "overlaps",
        capsule_capsule(
            Capsule::from_segment(left, 1e-8),
            Capsule::from_segment(right, 1e-8),
        )
        .overlaps,
    );
    emit(
        "segment_near_parallel",
        "distance",
        segment_segment(
            Segment3::new([0.0; 3], [1e8, 1.0, 0.0]),
            Segment3::new([0.0, 1.0, 0.0], [1e8, 0.0, 0.0]),
        )
        .distance,
    );
    let sphere = Sphere::new([0.0; 3], 1.0);
    for (case, magnitude) in [("support_small", 1e-13), ("support_large", 1e308)] {
        emit(case, "point", sphere.support_point([0.0, magnitude, 0.0]));
    }
    for exponent in [-24, 0, 24] {
        let (left, right) = spheres(exponent);
        let result = gjk_intersection(&left, &right);
        let case = format!("gjk_{exponent}");
        emit(&case, "status", result.status);
        emit(&case, "iterations", result.iterations);
    }
    for vertices in [4, 16, 64] {
        let left_points = polygon(vertices, [0.0, 0.0]);
        let right_points = polygon(vertices, [1.25, 0.2]);
        let left = ConvexHull3::new(&left_points);
        let right = ConvexHull3::new(&right_points);
        let gjk = epa_seed(&left, &right);
        let result = epa_penetration_planar_xy(&left, &right, &gjk);
        let trace = epa_penetration_trace_planar_xy(&left, &right, &gjk);
        assert_eq!(result, trace.result);
        let case = format!("epa_{vertices}");
        // Exact seeds and witnesses make the isolated EPA timing comparable.
        emit(&case, "seed", gjk);
        emit(&case, "result", result);
        emit(&case, "status", result.status);
        emit(&case, "iterations", result.iterations);
        emit(
            &case,
            "trace_points",
            trace
                .steps
                .iter()
                .map(|step| step.polytope.len())
                .sum::<usize>(),
        );
    }
}
