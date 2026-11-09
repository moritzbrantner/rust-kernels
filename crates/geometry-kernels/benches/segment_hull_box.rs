#![allow(
    clippy::unwrap_used,
    reason = "benchmark setup failures must fail the benchmark"
)]
//! Box/hull contacts against a 32-vertex authored rock proxy (centimetres).
use divan::{Bencher, black_box};
use geometry_kernels::hull::{Hull3, QuickHullWork, quickhull3};
use geometry_kernels::segment_hull::{BoxHullWork, PreparedBoxHull, PreparedSegmentHull};
use std::sync::Arc;
fn main() {
    divan::main();
}
const ROCK: [[i32; 3]; 32] = [
    [-160, 5, 45],
    [-157, 1, 47],
    [-160, 72, 16],
    [-125, 109, 58],
    [-149, 79, 46],
    [-150, 1, -37],
    [-77, 214, -22],
    [-78, 216, -8],
    [-79, 193, 28],
    [-43, 220, 11],
    [-151, 5, -37],
    [-134, 84, -88],
    [-84, 1, -107],
    [-28, 80, -133],
    [-97, 197, -33],
    [-68, 161, -98],
    [-20, 199, -85],
    [21, 95, 122],
    [64, 2, 150],
    [66, 5, 147],
    [16, 126, 106],
    [18, 215, -10],
    [72, 183, -6],
    [54, 1, -150],
    [158, 90, -69],
    [152, 1, -89],
    [155, 0, -110],
    [160, 1, -107],
    [7, 200, -77],
    [45, 195, -66],
    [105, 174, -58],
    [154, 122, -53],
];
fn rock() -> Arc<PreparedSegmentHull> {
    let points: Vec<_> = ROCK
        .iter()
        .map(|p| [f64::from(p[0]), f64::from(p[1] - 110), f64::from(p[2])].map(|v| v * 0.01))
        .collect();
    let mut hull = Hull3::default();
    quickhull3(&points, &mut QuickHullWork::default(), &mut hull);
    Arc::new(PreparedSegmentHull::new(hull).unwrap())
}
/// 64 deterministic tumbling orientations and resting-contact centres around the rock.
fn queries() -> Vec<([f64; 3], [[f64; 3]; 3])> {
    (0..64)
        .map(|i| {
            let (a, b) = (f64::from(i) * 0.37, f64::from(i) * 0.91);
            let ((s, c), (t, d)) = (a.sin_cos(), b.sin_cos());
            let axes = [[c, 0.0, -s], [s * t, d, c * t], [s * d, -t, c * d]];
            let (u, w) = (f64::from(i) * 2.39996, 0.3 + 0.4 * f64::from(i % 5) / 4.0);
            let centre = [
                1.9 * w.cos() * u.cos(),
                1.1 + 0.9 * w.sin(),
                1.9 * w.cos() * u.sin(),
            ];
            (centre, axes)
        })
        .collect()
}
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
