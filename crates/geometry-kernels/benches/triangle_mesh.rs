#![allow(
    clippy::unwrap_used,
    reason = "benchmark setup failures must fail the benchmark"
)]
use divan::{Bencher, black_box};
use geometry_kernels::triangle_mesh::{PreparedTriangleMesh3, TriangleMeshData3};
#[path = "../tests/support/contact_fixtures.rs"]
mod fixtures;
use fixtures::cave;
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
    let axes = if pose == "resting" {
        fixtures::IDENTITY
    } else {
        fixtures::tilted()
    };
    let boxes = fixtures::cave_boxes(&mesh);
    let mut work = geometry_kernels::triangle_mesh::TriangleMeshWork3::default();
    bencher.bench_local(|| {
        for &center in &boxes {
            mesh.box_contacts_counted(
                black_box(center),
                axes,
                fixtures::CAVE_BOX_HALF,
                0.02,
                &mut work,
                |m| {
                    black_box(m);
                },
            )
            .unwrap();
        }
    });
}
