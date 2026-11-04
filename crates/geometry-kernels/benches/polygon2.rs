#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "benchmarks are not production code; a panic is the intended failure report"
)]

use divan::{Bencher, black_box};
use geometry_kernels::polygon2::{
    ConvexDecomposition2, Polygon2Work, SimplicityCheck, Vec2, convex_decompose_polygon2,
    triangulate_polygon2,
};
use std::f64::consts::TAU;

fn main() {
    divan::main();
}

const SIZES: [usize; 3] = [16, 256, 2048];

#[derive(Clone, Copy, Debug)]
enum Shape {
    /// Regular n-gon: every vertex convex.
    Convex,
    /// Seeded star with random radii: about half the vertices reflex.
    Star,
    /// Comb with n / 4 teeth: long reflex chains.
    Comb,
}

const SHAPES: [Shape; 3] = [Shape::Convex, Shape::Star, Shape::Comb];

fn polygon(shape: Shape, n: usize) -> Vec<Vec2> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ n as u64;
    let mut unit = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    match shape {
        Shape::Convex => (0..n)
            .map(|i| {
                let t = i as f64 / n as f64 * TAU;
                [t.cos(), t.sin()]
            })
            .collect(),
        Shape::Star => (0..n)
            .map(|i| {
                let t = (i as f64 + 0.1 + 0.8 * unit()) / n as f64 * TAU;
                let r = 1.0 - 0.9 * unit();
                [r * t.cos(), r * t.sin()]
            })
            .collect(),
        Shape::Comb => {
            let teeth = (n / 4).max(1);
            let mut p = vec![[0.0, 0.0], [teeth as f64 * 2.0 - 1.0, 0.0]];
            for k in (0..teeth).rev() {
                let x = k as f64 * 2.0;
                p.extend([[x + 1.0, 3.0], [x, 3.0]]);
                if k > 0 {
                    p.extend([[x, 1.0], [x - 1.0, 1.0]]);
                }
            }
            p
        }
    }
}

#[divan::bench(args = SIZES, consts = [0, 1, 2])]
fn triangulate_validated<const S: usize>(bencher: Bencher, n: usize) {
    let points = polygon(SHAPES[S], n);
    let mut work = Polygon2Work::default();
    let mut out = Vec::new();
    bencher.bench_local(|| {
        triangulate_polygon2(
            black_box(&points),
            SimplicityCheck::Validate,
            &mut work,
            &mut out,
        )
        .unwrap();
        black_box(out.len())
    });
}

#[divan::bench(args = SIZES, consts = [0, 1, 2])]
fn triangulate_assume_simple<const S: usize>(bencher: Bencher, n: usize) {
    let points = polygon(SHAPES[S], n);
    let mut work = Polygon2Work::default();
    let mut out = Vec::new();
    bencher.bench_local(|| {
        triangulate_polygon2(
            black_box(&points),
            SimplicityCheck::AssumeSimple,
            &mut work,
            &mut out,
        )
        .unwrap();
        black_box(out.len())
    });
}

#[divan::bench(args = SIZES, consts = [0, 1, 2])]
fn convex_decompose_assume_simple<const S: usize>(bencher: Bencher, n: usize) {
    let points = polygon(SHAPES[S], n);
    let mut work = Polygon2Work::default();
    let mut out = ConvexDecomposition2::default();
    bencher.bench_local(|| {
        convex_decompose_polygon2(
            black_box(&points),
            SimplicityCheck::AssumeSimple,
            &mut work,
            &mut out,
        )
        .unwrap();
        black_box(out.len())
    });
}
