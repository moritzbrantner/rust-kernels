use divan::{Bencher, counter::ItemsCount};
use similarity_kernels::jaccard_similarity_sorted_unique;

const ITEMS_PER_SET: usize = 65_536;

fn main() {
    divan::main();
}

fn interleaved_sets() -> (Vec<u32>, Vec<u32>) {
    let left = (0..ITEMS_PER_SET).map(|index| (index * 2) as u32).collect();
    let right = (0..ITEMS_PER_SET)
        .map(|index| (index * 2 + 1) as u32)
        .collect();
    (left, right)
}

#[divan::bench(skip_ext_time)]
fn jaccard_interleaved_disjoint(bencher: Bencher) {
    let (left, right) = interleaved_sets();
    bencher
        .counter(ItemsCount::new(left.len() + right.len()))
        .bench_local(|| divan::black_box(jaccard_similarity_sorted_unique(&left, &right)));
}
