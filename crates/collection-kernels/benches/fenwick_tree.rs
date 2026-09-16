use collection_kernels::FenwickTree;
use divan::{Bencher, counter::ItemsCount};

const SIZES: &[usize] = &[256, 4_096, 65_536];

fn main() {
    divan::main();
}

fn generated_values(len: usize) -> Vec<i64> {
    (0..len)
        .map(|index| ((index * 17 + 11) % 1_009) as i64 - 504)
        .collect()
}

#[divan::bench(args = SIZES, skip_ext_time)]
fn ordered_build(bencher: Bencher, len: usize) {
    bencher
        .with_inputs(|| generated_values(len))
        .counter(ItemsCount::new(len))
        .bench_local_values(|values| divan::black_box(FenwickTree::from_slice(&values)));
}

#[divan::bench(args = SIZES, skip_ext_time)]
fn linear_build(bencher: Bencher, len: usize) {
    bencher
        .with_inputs(|| generated_values(len))
        .counter(ItemsCount::new(len))
        .bench_local_values(|values| divan::black_box(FenwickTree::from_slice_linear(&values)));
}
