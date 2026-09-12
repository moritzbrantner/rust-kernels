use divan::{Bencher, counter::ItemsCount};
use similarity_kernels::{BkTree, levenshtein, myers_levenshtein_bytes};

const INDEX_SIZE: usize = 4_096;
const TEXT_SIZE: usize = 4_096;

fn main() {
    divan::main();
}

fn integer_tree() -> BkTree<i32> {
    let mut tree = BkTree::new();
    tree.insert(2_048, absolute_distance);
    for value in 0..INDEX_SIZE as i32 {
        if value != 2_048 {
            tree.insert(value, absolute_distance);
        }
    }
    tree
}

fn absolute_distance(left: &i32, right: &i32) -> usize {
    left.abs_diff(*right) as usize
}

fn pattern() -> Vec<u8> {
    (0..64).map(|index| b'a' + (index % 23) as u8).collect()
}

fn text() -> Vec<u8> {
    (0..TEXT_SIZE)
        .map(|index| b'a' + ((index * 7 + 3) % 23) as u8)
        .collect()
}

#[divan::bench(skip_ext_time)]
fn bk_tree_radius_two(bencher: Bencher) {
    let tree = integer_tree();
    bencher
        .counter(ItemsCount::new(INDEX_SIZE))
        .bench_local(|| divan::black_box(tree.search(&2_048, 2, absolute_distance)));
}

#[divan::bench(skip_ext_time)]
fn levenshtein_64_by_4096(bencher: Bencher) {
    let pattern = pattern();
    let text = text();
    bencher
        .counter(ItemsCount::new(TEXT_SIZE))
        .bench_local(|| divan::black_box(levenshtein(&pattern, &text)));
}

#[divan::bench(skip_ext_time)]
fn myers_64_by_4096(bencher: Bencher) {
    let pattern = pattern();
    let text = text();
    bencher
        .counter(ItemsCount::new(TEXT_SIZE))
        .bench_local(|| divan::black_box(myers_levenshtein_bytes(&pattern, &text).unwrap()));
}
