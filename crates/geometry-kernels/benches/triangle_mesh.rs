#![allow(
    clippy::unwrap_used,
    reason = "benchmark setup failures must fail the benchmark"
)]
use divan::{Bencher, black_box};
use geometry_kernels::triangle_mesh::{PreparedTriangleMesh3, TriangleMeshData3};
fn main() {
    divan::main();
}
fn data(count: usize) -> TriangleMeshData3 {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    for i in 0..count {
        let x = i as f64 * 5.;
        let at = vertices.len() as u32;
        vertices.extend([[x - 1., 0., -1.], [x + 1., 0., -1.], [x, 0., 1.]]);
        triangles.push([at, at + 2, at + 1]);
    }
    TriangleMeshData3 {
        vertices,
        triangles,
    }
}
#[divan::bench(args=[64,256,1024])]
fn prepare(bencher: Bencher, count: usize) {
    bencher
        .with_inputs(|| data(count))
        .bench_values(|data| black_box(PreparedTriangleMesh3::try_new(data).unwrap()));
}
#[divan::bench(args=[64,256,1024])]
fn capsule_sweep(bencher: Bencher, count: usize) {
    let mesh = PreparedTriangleMesh3::try_new(data(count)).unwrap();
    bencher.bench_local(|| {
        black_box(
            mesh.capsule_sweep(
                black_box([0., 2., 0.]),
                [0., 2.5, 0.],
                0.3,
                [0., -4., 0.],
                64,
            )
            .unwrap(),
        )
    });
}
#[divan::bench(args=[64,256,1024])]
fn segment_hit(bencher: Bencher, count: usize) {
    let mesh = PreparedTriangleMesh3::try_new(data(count)).unwrap();
    bencher.bench_local(|| {
        black_box(
            mesh.segment_hit(black_box([0., 2., 0.]), [0., -2., 0.])
                .unwrap(),
        )
    });
}

#[divan::bench(args=[64,256,1024])]
fn box_manifold(bencher: Bencher, count: usize) {
    let mesh = PreparedTriangleMesh3::try_new(data(count)).unwrap();
    bencher.bench_local(|| {
        mesh.box_contacts_counted(
            black_box([0., 0.49, 0.]),
            black_box([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
            [0.5; 3],
            0.05,
            &mut geometry_kernels::triangle_mesh::TriangleMeshWork3::default(),
            |m| {
                black_box(m);
            },
        )
        .unwrap();
    });
}

#[divan::bench(args=[64,256,1024])]
fn rotated_box_manifold(bencher: Bencher, count: usize) {
    let mesh = PreparedTriangleMesh3::try_new(data(count)).unwrap();
    let (s, c) = 0.31_f64.sin_cos();
    bencher.bench_local(|| {
        mesh.box_contacts_counted(
            black_box([0., 0.49, 0.]),
            black_box([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]]),
            [0.5; 3],
            0.05,
            &mut geometry_kernels::triangle_mesh::TriangleMeshWork3::default(),
            |m| {
                black_box(m);
            },
        )
        .unwrap();
    });
}

/// The closed cave of the `triangle_mesh` sweep tests: a wobbling tube of 41 rings of 24
/// vertices along X with a flattened floor and capped ends (1968 triangles).
fn cave() -> TriangleMeshData3 {
    let (rings, around, size, wobble) = (41_usize, 24_usize, 3.0, 0.1);
    let mut vertices = Vec::new();
    let half = (rings - 1) as f64 * 0.5;
    for i in 0..rings {
        let x = i as f64 - half;
        for k in 0..around {
            let angle = std::f64::consts::TAU * k as f64 / around as f64;
            let r = size
                * (1.0
                    + wobble
                        * ((0.7 * x + 3.0 * angle).sin() + 0.5 * (1.3 * x - 2.0 * angle).cos()));
            let y = (r * angle.sin()).max(-0.6 * size + 0.05 * (0.9 * x).sin());
            vertices.push([x, y, r * angle.cos()]);
        }
    }
    let ring = |i: usize, k: usize| (i * around + k % around) as u32;
    let mut triangles = Vec::new();
    for i in 0..rings - 1 {
        for k in 0..around {
            triangles.push([ring(i, k), ring(i + 1, k), ring(i + 1, k + 1)]);
            triangles.push([ring(i, k), ring(i + 1, k + 1), ring(i, k + 1)]);
        }
    }
    for (i, x) in [(0, -half - 0.5), (rings - 1, half + 0.5)] {
        let centre = vertices.len() as u32;
        vertices.push([x, 0.0, 0.0]);
        for k in 0..around {
            triangles.push([centre, ring(i, k + 1), ring(i, k)]);
        }
    }
    TriangleMeshData3 {
        vertices,
        triangles,
    }
}

/// 256 character capsules (radius 0.35, half height 0.55) on the cave floor, each swept once.
#[divan::bench(args = ["walk", "fall", "skim", "fast diagonal"])]
fn cave_capsule_sweep(bencher: Bencher, motion: &str) {
    let mesh = PreparedTriangleMesh3::try_new(cave()).unwrap();
    let (radius, half) = (0.35, 0.55);
    let (gap, d) = match motion {
        "walk" => (0.02, [0.12, -0.03, 0.07]),
        "fall" => (0.5, [0.05, -1.2, 0.0]),
        "skim" => (0.0, [0.3, 0.0, 0.2]),
        _ => (0.1, [1.5, -0.8, 1.1]),
    };
    let capsules: Vec<_> = (0..256)
        .map(|i| {
            let (x, z) = (
                f64::from(i % 32) * 1.1 - 17.0,
                f64::from(i / 32) * 0.4 - 1.4,
            );
            let floor = mesh
                .segment_hit([x, 0.0, z], [x, -4.0, z])
                .unwrap()
                .unwrap()
                .point[1];
            let bottom = floor + radius + gap;
            ([x, bottom, z], [x, bottom + 2.0 * half, z])
        })
        .collect();
    bencher.bench_local(|| {
        for &(p, q) in &capsules {
            black_box(mesh.capsule_sweep(black_box(p), q, radius, d, 64).unwrap());
        }
    });
}

/// 64 boxes (half extents 0.4 x 0.3 x 0.5) resting on the cave floor, axis aligned or tilted,
/// each queried once for contacts with a 0.02 margin.
#[divan::bench(args = ["resting", "tilted"])]
fn cave_box_contacts(bencher: Bencher, pose: &str) {
    let mesh = PreparedTriangleMesh3::try_new(cave()).unwrap();
    let half = [0.4, 0.3, 0.5];
    let axes = if pose == "resting" {
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    } else {
        let (s, c) = 0.21_f64.sin_cos();
        let (s2, c2) = 0.37_f64.sin_cos();
        [[c, s, 0.], [-s * c2, c * c2, s2], [s * s2, -c * s2, c2]]
    };
    let boxes: Vec<_> = (0..64)
        .map(|i| {
            let (x, z) = (
                f64::from(i % 16) * 2.1 - 16.0,
                f64::from(i / 16) * 0.5 - 0.8,
            );
            let floor = mesh
                .segment_hit([x, 0.0, z], [x, -4.0, z])
                .unwrap()
                .unwrap()
                .point[1];
            [x, floor + 0.29, z]
        })
        .collect();
    let mut work = geometry_kernels::triangle_mesh::TriangleMeshWork3::default();
    bencher.bench_local(|| {
        for &center in &boxes {
            mesh.box_contacts_counted(black_box(center), axes, half, 0.02, &mut work, |m| {
                black_box(m);
            })
            .unwrap();
        }
    });
}
