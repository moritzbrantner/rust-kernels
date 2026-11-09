#![allow(
    clippy::unwrap_used,
    reason = "benchmark setup failures must fail the benchmark"
)]
//! Box/hull contacts against a 32-vertex authored rock proxy (centimetres).
use divan::{Bencher, black_box};
use geometry_kernels::segment_hull::{BoxHullWork, PreparedBoxHull};
#[path = "../tests/support/contact_fixtures.rs"]
mod fixtures;
fn main() {
    divan::main();
}
use fixtures::{rock, rock_queries as queries};

/// A new orientation every query: the frame and its deferred edge axes are prepared each time.
#[divan::bench]
fn rock_box_contact_fresh(bencher: Bencher) {
    let (hull, queries) = (rock(), queries());
    bencher.bench_local(|| {
        for &(centre, axes) in &queries {
            let mut work = BoxHullWork::default();
            let frame = PreparedBoxHull::new(hull.clone(), [0.35; 3], axes, &mut work).unwrap();
            black_box(frame.contact(black_box(centre), 0.02, &mut work).unwrap());
        }
    });
}
/// One prepared orientation, many translations.
#[divan::bench]
fn rock_box_contact_reused(bencher: Bencher) {
    let (hull, queries) = (rock(), queries());
    let mut work = BoxHullWork::default();
    let frame = PreparedBoxHull::new(hull, [0.35; 3], queries[0].1, &mut work).unwrap();
    bencher.bench_local(|| {
        for &(centre, _) in &queries {
            black_box(frame.contact(black_box(centre), 0.02, &mut work).unwrap());
        }
    });
}
