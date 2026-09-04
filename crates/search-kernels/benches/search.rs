use divan::{counter::ItemsCount, Bencher};
use search_kernels::{quickselect, radix_sort_u32, top_k_smallest};

const SIZES: &[usize] = &[256, 4_096, 65_536];

fn main() {
    divan::main();
}

fn generated_values(len: usize) -> Vec<u32> {
    let mut state = 0x1234_5678_u32;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        })
        .collect()
}

#[divan::bench(args = SIZES, skip_ext_time)]
fn radix_sort(bencher: Bencher, len: usize) {
    bencher
        .with_inputs(|| generated_values(len))
        .counter(ItemsCount::new(len))
        .bench_local_refs(|values| radix_sort_u32(values));
}

#[divan::bench(args = SIZES, skip_ext_time)]
fn middle_quickselect(bencher: Bencher, len: usize) {
    bencher
        .with_inputs(|| generated_values(len))
        .counter(ItemsCount::new(len))
        .bench_local_refs(|values| {
            let nth = values.len() / 2;
            let _ = divan::black_box(quickselect(values, nth));
        });
}

#[divan::bench(args = SIZES, skip_ext_time)]
fn smallest_one_percent(bencher: Bencher, len: usize) {
    let k = (len / 100).max(1);
    bencher
        .with_inputs(|| generated_values(len))
        .counter(ItemsCount::new(len))
        .bench_local_values(|values| divan::black_box(top_k_smallest(&values, k)));
}
