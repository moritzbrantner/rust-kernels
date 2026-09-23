//! Exact full-DP reference versus bounded queries on identical deterministic inputs.
use divan::{Bencher, black_box};
use similarity_kernels::{LevenshteinWorkspace, levenshtein, levenshtein_bounded};
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();
fn pair(n: usize) -> (Vec<u8>, Vec<u8>) {
    (
        (0..n).map(|i| (i % 23) as u8).collect(),
        (0..n).map(|i| ((i + 1) % 23) as u8).collect(),
    )
}
#[divan::bench(args=[64,256,1024])]
fn full_reference(bencher: Bencher, n: usize) {
    let (a, b) = pair(n);
    bencher.bench_local(|| black_box(levenshtein(black_box(&a), black_box(&b))));
}
#[divan::bench(args=[64,256,1024])]
fn bounded_fresh(bencher: Bencher, n: usize) {
    let (a, b) = pair(n);
    bencher.bench_local(|| {
        black_box(levenshtein_bounded(
            black_box(&a),
            black_box(&b),
            black_box(2),
        ))
    });
}
#[divan::bench(args=[64,256,1024])]
fn bounded_reused(bencher: Bencher, n: usize) {
    let (a, b) = pair(n);
    let mut scratch = LevenshteinWorkspace::new();
    assert_eq!(scratch.distance(&a, &b, 2), Some(2));
    bencher.bench_local(|| black_box(scratch.distance(black_box(&a), black_box(&b), black_box(2))));
}
#[divan::bench(args=[64,256,1024])]
fn length_rejection(bencher: Bencher, n: usize) {
    let a = vec![0_u8; n];
    let b = vec![1_u8; n + 8];
    let mut scratch = LevenshteinWorkspace::new();
    bencher.bench_local(|| black_box(scratch.distance(black_box(&a), black_box(&b), black_box(2))));
}
#[divan::bench(args=[64,256,1024])]
fn rejected_bounded(bencher: Bencher, n: usize) {
    let a = vec![0_u8; n];
    let b = vec![1_u8; n];
    let mut scratch = LevenshteinWorkspace::new();
    assert_eq!(scratch.distance(&a, &b, 2), None);
    bencher.bench_local(|| black_box(scratch.distance(black_box(&a), black_box(&b), black_box(2))));
}
#[divan::bench(args=[64,256,1024])]
fn rejected_full_reference(bencher: Bencher, n: usize) {
    let a = vec![0_u8; n];
    let b = vec![1_u8; n];
    bencher.bench_local(|| black_box(levenshtein(black_box(&a), black_box(&b))));
}
fn main() {
    if !std::env::args().any(|a| a == "--expansion-probe") {
        divan::main();
        return;
    }
    use std::cell::Cell;
    struct Counted<'a>(u8, &'a Cell<usize>);
    impl PartialEq for Counted<'_> {
        fn eq(&self, b: &Self) -> bool {
            self.1.set(self.1.get() + 1);
            self.0 == b.0
        }
    }
    impl Eq for Counted<'_> {}
    for n in [64, 256, 1024] {
        let (a, b) = pair(n);
        let calls = Cell::new(0);
        let a: Vec<_> = a.into_iter().map(|v| Counted(v, &calls)).collect();
        let b: Vec<_> = b.into_iter().map(|v| Counted(v, &calls)).collect();
        let exact = levenshtein(&a, &b);
        let full_calls = calls.get();
        calls.set(0);
        let result = levenshtein_bounded(&a, &b, 2);
        assert_eq!(result, Some(exact));
        assert_eq!(exact, 2);
        println!("AUDIT\tedit_{n}\tdistance\t{exact}");
        println!("AUDIT\tedit_{n}\tfull_equality_calls\t{full_calls}");
        println!("AUDIT\tedit_{n}\tbounded_equality_calls\t{}", calls.get());
        assert_eq!(levenshtein_bounded(&vec![0_u8; n], &vec![1_u8; n], 2), None);
    }
}
