//! Boundary workload v1; input setup and oracle checks are outside timing.
use divan::{Bencher, black_box};
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();
fn emit(case: &str, metric: &str, value: impl std::fmt::Debug) {
    println!("AUDIT\t{case}\t{metric}\t{value:?}");
}
fn main() {
    if std::env::args().any(|arg| arg == "--audit-probe") {
        probe();
    } else {
        divan::main();
    }
}

use search_kernels::top_k_by;

#[divan::bench(args = [8, 65536])]
fn empty_limit(bencher: Bencher, limit: usize) {
    bencher.bench_local(|| {
        black_box(top_k_by(
            std::iter::empty::<u64>(),
            black_box(limit),
            Ord::cmp,
        ))
    });
}

#[divan::bench(args = [8, 65536])]
fn short_limit(bencher: Bencher, limit: usize) {
    bencher.bench_local(|| {
        black_box(top_k_by(
            black_box([3_u64, 1, 2, 0]),
            black_box(limit),
            Ord::cmp,
        ))
    });
}

#[divan::bench(args = [8, 65536])]
fn stream_unknown(bencher: Bencher, limit: usize) {
    bencher.bench_local(|| {
        let mut source = black_box([3_u64, 1, 2, 0]).into_iter();
        black_box(top_k_by(
            std::iter::from_fn(|| source.next()),
            black_box(limit),
            Ord::cmp,
        ))
    });
}

#[divan::bench]
fn ranked_4096(bencher: Bencher) {
    bencher.bench_local(|| black_box(top_k_by((0..black_box(4096_u64)).rev(), 40, Ord::cmp)));
}

fn probe() {
    for limit in [8, 65536] {
        let result = top_k_by([3_u64, 1, 2, 0], limit, Ord::cmp);
        assert_eq!(result, [0, 1, 2, 3]);
        emit(&format!("top_k_{limit}"), "values", &result);
        // Result capacity is observable retained-memory evidence, not a promise
        // about the standard library's future map/collect allocation strategy.
        emit(
            &format!("top_k_{limit}"),
            "retained_capacity",
            result.capacity(),
        );
    }
}
