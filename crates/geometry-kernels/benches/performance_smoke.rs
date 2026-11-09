#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "benchmarks are not production code; a panic is the intended failure report"
)]
//! Deterministic instruction-count sentinels for optimized contact paths. Each
//! guards a measured optimization against silent regression (5% soft limit
//! against the pull request base, see `.performance/contract.json`):
//!
//! - box/triangle-mesh contacts: branch-free SAT and iterative BVH walk (#225)
//! - fresh box/hull frames: branch-free edge-cone marks, one plane allocation (#226, #227)
//! - reused box/hull frames: inline clip polygons and points (#228)

use geometry_kernels::segment_hull::{BoxHullWork, PreparedBoxHull, PreparedSegmentHull};
use geometry_kernels::triangle_mesh::{PreparedTriangleMesh3, TriangleMeshWork3};
use iai_callgrind::{
    Callgrind, EventKind, LibraryBenchmarkConfig, library_benchmark, library_benchmark_group, main,
};
use std::hint::black_box;
use std::sync::Arc;

#[path = "../tests/support/contact_fixtures.rs"]
mod fixtures;
use fixtures::{Axes, Vec3};

fn cave_case(tilted: bool) -> (PreparedTriangleMesh3, Vec<Vec3>, Axes) {
    let mesh = PreparedTriangleMesh3::try_new(fixtures::cave()).unwrap();
    let boxes = fixtures::cave_boxes(&mesh);
    let axes = if tilted {
        fixtures::tilted()
    } else {
        fixtures::IDENTITY
    };
    (mesh, boxes, axes)
}

#[library_benchmark]
#[bench::resting_64(cave_case(false))]
#[bench::tilted_64(cave_case(true))]
fn box_mesh_contacts((mesh, boxes, axes): (PreparedTriangleMesh3, Vec<Vec3>, Axes)) -> usize {
    let mut manifolds = 0;
    let mut work = TriangleMeshWork3::default();
    for &centre in &boxes {
        mesh.box_contacts_counted(
            black_box(centre),
            axes,
            fixtures::CAVE_BOX_HALF,
            0.02,
            &mut work,
            |manifold| manifolds += manifold.len,
        )
        .unwrap();
    }
    black_box(manifolds)
}

type RockCase = (Arc<PreparedSegmentHull>, Vec<(Vec3, Axes)>);

fn rock_case() -> RockCase {
    (fixtures::rock(), fixtures::rock_queries())
}

// A new box orientation every query: frame, deferred planes and contact.
#[library_benchmark]
#[bench::tumbling_64(rock_case())]
fn box_hull_fresh_frames((hull, queries): RockCase) -> usize {
    let mut contacts = 0;
    for &(centre, axes) in &queries {
        let mut work = BoxHullWork::default();
        let frame = PreparedBoxHull::new(hull.clone(), [0.35; 3], axes, &mut work).unwrap();
        let contact = frame.contact(black_box(centre), 0.02, &mut work).unwrap();
        contacts += contact.map_or(0, |c| c.len);
    }
    black_box(contacts)
}

fn prepared_rock_case() -> (PreparedBoxHull, Vec<(Vec3, Axes)>) {
    let (hull, queries) = rock_case();
    let mut work = BoxHullWork::default();
    let frame = PreparedBoxHull::new(hull, [0.35; 3], queries[0].1, &mut work).unwrap();
    // Prepare and retain the deferred planes outside the measurement.
    frame.contact([0.0, 1.1, 0.0], 0.02, &mut work).unwrap();
    (frame, queries)
}

// One prepared orientation resting at 64 places: the steady state.
#[library_benchmark]
#[bench::resting_64(prepared_rock_case())]
fn box_hull_reused_frame((frame, queries): (PreparedBoxHull, Vec<(Vec3, Axes)>)) -> usize {
    let mut contacts = 0;
    let mut work = BoxHullWork::default();
    for &(centre, _) in &queries {
        let contact = frame.contact(black_box(centre), 0.02, &mut work).unwrap();
        contacts += contact.map_or(0, |c| c.len);
    }
    black_box(contacts)
}

library_benchmark_group!(
    name = contact_smoke;
    benchmarks = box_mesh_contacts, box_hull_fresh_frames, box_hull_reused_frame
);

fn benchmark_config() -> LibraryBenchmarkConfig {
    let mut callgrind = Callgrind::default();
    callgrind
        .soft_limits([(EventKind::Ir, 5.0)])
        .fail_fast(true);
    let mut config = LibraryBenchmarkConfig::default();
    config.tool(callgrind);
    config
}

main!(
    config = benchmark_config();
    library_benchmark_groups = contact_smoke
);
