use divan::{Bencher, black_box};
use geometry_kernels::primitive3::{
    PrimitiveBody3, PrimitiveShape3, PrimitiveWork3, query, swept_time,
};

fn main() {
    divan::main();
}

fn capsule() -> PrimitiveBody3 {
    PrimitiveBody3::axis_aligned(PrimitiveShape3::capsule(2.0, 0.5), [0.0; 3], [0.0; 3])
}

fn wedge() -> PrimitiveBody3 {
    PrimitiveBody3::axis_aligned(
        PrimitiveShape3::wedge([2.0, 2.0, 2.0]),
        [1.0, 0.5, 0.0],
        [0.0; 3],
    )
}

#[divan::bench]
fn capsule_wedge_contact(bencher: Bencher) {
    let capsule = capsule();
    let wedge = wedge();
    bencher.bench_local(|| {
        let mut work = PrimitiveWork3::default();
        black_box(query(
            black_box(capsule),
            black_box(wedge),
            black_box(&mut work),
        ))
    });
}

#[divan::bench]
fn fast_capsule_wedge_sweep(bencher: Bencher) {
    let moving = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::capsule(0.5, 0.1),
        [-1.0, -1.0, 10.0],
        [0.0, 0.0, -10_000.0],
    );
    let target =
        PrimitiveBody3::axis_aligned(PrimitiveShape3::wedge([4.0, 4.0, 0.02]), [0.0; 3], [0.0; 3]);
    bencher.bench_local(|| {
        let mut work = PrimitiveWork3::default();
        black_box(swept_time(
            black_box(moving),
            black_box(target),
            black_box(1.0 / 60.0),
            black_box(0.02),
            black_box(&mut work),
        ))
    });
}
