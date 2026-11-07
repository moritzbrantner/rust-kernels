//! Capsule contact and sweep against rolling terrain, for character-sized capsules resting on,
//! skimming and falling onto it (the MMORPG terrain workload).
#![allow(
    clippy::unwrap_used,
    reason = "benchmark setup failures must fail the benchmark"
)]
use divan::{Bencher, black_box};
use geometry_kernels::heightfield::{HeightfieldData3, HeightfieldDiagonal3, PreparedHeightfield3};

fn main() {
    divan::main();
}

fn terrain() -> PreparedHeightfield3 {
    let n = 65;
    let columns: Vec<f64> = (0..n).map(|i| i as f64 - 32.0).collect();
    let rows = columns.clone();
    let heights = (0..n * n)
        .map(|k| {
            let (x, z) = (columns[k % n], rows[k / n]);
            1.2 * (x * 0.21).sin() * (z * 0.17).cos() + 0.3 * (x * 0.9 + z * 0.4).sin()
        })
        .collect();
    PreparedHeightfield3::try_new(HeightfieldData3 {
        columns,
        rows,
        heights,
        active_cells: None,
        diagonal: HeightfieldDiagonal3::Fixed,
    })
    .unwrap()
}

const RADIUS: f64 = 0.35;
const HALF: f64 = 0.55;

/// Capsule segments whose bottom sits `gap` above the terrain directly below.
fn poses(field: &PreparedHeightfield3, gap: f64) -> Vec<([f64; 3], [f64; 3])> {
    (0..256)
        .map(|i| {
            let (x, z) = (
                f64::from(i % 16) * 3.1 - 24.0,
                f64::from(i / 16) * 2.9 - 23.0,
            );
            let y = field.sample(x, z).unwrap().surface.unwrap().point[1];
            let bottom = y + RADIUS + gap;
            ([x, bottom, z], [x, bottom + 2.0 * HALF, z])
        })
        .collect()
}

#[divan::bench(args = [-0.01, 0.0, 0.02, 0.5])]
fn capsule_contact(bencher: Bencher, gap: f64) {
    let field = terrain();
    let poses = poses(&field, gap);
    bencher.bench_local(|| {
        for &(p, q) in &poses {
            black_box(field.capsule_contact(black_box(p), q, RADIUS, 0.0));
        }
    });
}

/// Walking (a step across the ground with a little gravity), and falling onto it.
#[divan::bench(args = ["walk", "fall"])]
fn capsule_sweep(bencher: Bencher, motion: &str) {
    let field = terrain();
    let (gap, displacement) = match motion {
        "walk" => (0.02, [0.12, -0.03, 0.07]),
        _ => (0.5, [0.05, -1.2, 0.0]),
    };
    let poses = poses(&field, gap);
    bencher.bench_local(|| {
        for &(p, q) in &poses {
            black_box(
                field
                    .capsule_sweep(black_box(p), q, RADIUS, displacement, 64)
                    .unwrap(),
            );
        }
    });
}
