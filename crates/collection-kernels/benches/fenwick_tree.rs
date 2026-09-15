use collection_kernels::FenwickTree;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

const ITEMS: usize = 65_536;

fn generated_values() -> Vec<i64> {
    (0..ITEMS)
        .map(|index| ((index * 17 + 11) % 1_009) as i64 - 504)
        .collect()
}

fn bulk_build(c: &mut Criterion) {
    let values = generated_values();
    c.bench_function("fenwick/from_slice_65536", |b| {
        b.iter(|| black_box(FenwickTree::from_slice(black_box(&values))))
    });
}

fn incremental_build(c: &mut Criterion) {
    let values = generated_values();
    c.bench_function("fenwick/incremental_add_65536", |b| {
        b.iter(|| {
            let mut tree = FenwickTree::new(values.len());
            for (index, &value) in values.iter().enumerate() {
                tree.add(index, value);
            }
            black_box(tree)
        });
    });
}

criterion_group!(benches, bulk_build, incremental_build);
criterion_main!(benches);
