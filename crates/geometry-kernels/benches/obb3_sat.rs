use criterion::{Criterion, black_box, criterion_group, criterion_main};
use geometry_kernels::obb3::{Obb3, obb3_sat};

fn obb3_sat_cases(c: &mut Criterion) {
    let axis_aligned = Obb3::new([0.0; 3], [1.0, 2.0, 0.75], [0.0; 3]);
    let axis_aligned_overlap = Obb3::new([1.5, 0.0, 0.0], [1.0; 3], [0.0; 3]);
    let axis_aligned_separation = Obb3::new([2.01, 0.0, 0.0], [1.0; 3], [0.0; 3]);

    let rotated = Obb3::new([0.0, 0.0, 0.0], [1.4, 0.8, 0.9], [0.2, 0.4, -0.15]);
    let rotated_overlap = Obb3::new([1.2, 0.35, 0.25], [1.0, 0.7, 0.8], [-0.3, 0.15, 0.55]);
    let rotated_separation = Obb3::new([4.0, 1.5, 0.8], [1.0, 0.7, 0.8], [-0.3, 0.15, 0.55]);

    let mut group = c.benchmark_group("obb3_sat");

    group.bench_function("axis_aligned_overlap", |b| {
        b.iter(|| {
            black_box(obb3_sat(
                black_box(axis_aligned),
                black_box(axis_aligned_overlap),
            ))
        });
    });
    group.bench_function("axis_aligned_separation", |b| {
        b.iter(|| {
            black_box(obb3_sat(
                black_box(axis_aligned),
                black_box(axis_aligned_separation),
            ))
        });
    });
    group.bench_function("rotated_overlap", |b| {
        b.iter(|| black_box(obb3_sat(black_box(rotated), black_box(rotated_overlap))));
    });
    group.bench_function("rotated_separation", |b| {
        b.iter(|| black_box(obb3_sat(black_box(rotated), black_box(rotated_separation))));
    });

    group.finish();
}

criterion_group!(benches, obb3_sat_cases);
criterion_main!(benches);
