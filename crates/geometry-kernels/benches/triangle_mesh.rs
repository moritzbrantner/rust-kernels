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
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
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
