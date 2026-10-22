use divan::{Bencher, black_box};
use geometry_kernels::hull::{Hull3, QuickHullWork, quickhull2, quickhull3};

fn main() {
    divan::main();
}

const SIZES: [usize; 3] = [100, 10_000, 100_000];

fn unit_stream(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

#[derive(Clone, Copy, Debug)]
enum Distribution {
    /// Uniform in a cube.
    Uniform,
    /// Every point on the unit sphere: all points are hull vertices.
    Sphere,
    /// Tight Gaussian-ish clusters.
    Clustered,
    /// A ball inside a cube's eight corners: nearly everything is interior.
    Interior,
}

fn cloud(distribution: Distribution, n: usize) -> Vec<[f64; 3]> {
    let mut unit = unit_stream(0x9e37_79b9_7f4a_7c15 ^ n as u64);
    let mut points = Vec::with_capacity(n);
    match distribution {
        Distribution::Uniform => points.extend((0..n).map(|_| [unit(), unit(), unit()])),
        Distribution::Sphere => points.extend((0..n).map(|_| {
            loop {
                let p = [unit(), unit(), unit()];
                let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                if len > 0.1 && len <= 1.0 {
                    break p.map(|c| c / len);
                }
            }
        })),
        Distribution::Clustered => {
            let centers: Vec<[f64; 3]> = (0..8).map(|_| [unit(), unit(), unit()]).collect();
            points.extend((0..n).map(|i| {
                let c = centers[i % centers.len()];
                let jitter = (unit() + unit() + unit()) * 0.03;
                [c[0] + jitter, c[1] + unit() * 0.03, c[2] + unit() * 0.03]
            }));
        }
        Distribution::Interior => {
            points.extend(
                (0..n.saturating_sub(8)).map(|_| [unit() * 0.5, unit() * 0.5, unit() * 0.5]),
            );
            points.extend(
                (0..8).map(|i| [i & 1, i >> 1 & 1, i >> 2 & 1].map(|b| f64::from(b) * 2.0 - 1.0)),
            );
        }
    }
    points
}

const DISTRIBUTIONS: [Distribution; 4] = [
    Distribution::Uniform,
    Distribution::Sphere,
    Distribution::Clustered,
    Distribution::Interior,
];

#[divan::bench(args = SIZES, consts = [0, 1, 2, 3])]
fn quickhull3_points<const D: usize>(bencher: Bencher, n: usize) {
    let points = cloud(DISTRIBUTIONS[D], n);
    let mut work = QuickHullWork::default();
    let mut out = Hull3::default();
    bencher.bench_local(|| {
        quickhull3(black_box(&points), &mut work, &mut out);
        black_box(out.triangles.len())
    });
}

#[divan::bench(args = SIZES, consts = [0, 1, 2, 3])]
fn quickhull2_points<const D: usize>(bencher: Bencher, n: usize) {
    let points: Vec<[f64; 2]> = cloud(DISTRIBUTIONS[D], n)
        .into_iter()
        .map(|p| [p[0], p[1]])
        .collect();
    let mut work = QuickHullWork::default();
    let mut out = Vec::new();
    bencher.bench_local(|| {
        quickhull2(black_box(&points), &mut work, &mut out);
        black_box(out.len())
    });
}
