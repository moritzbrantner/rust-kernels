//! Workload v1. Setup is excluded; correctness is reported outside timing.
use divan::{Bencher, black_box, counter::ItemsCount};
use statistics_kernels::RunningStats;

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn stats(values: impl IntoIterator<Item = f64>) -> RunningStats {
    let mut stats = RunningStats::new();
    stats.extend(values);
    stats
}

fn main() {
    if std::env::args().any(|argument| argument == "--audit-probe") {
        for (case, magnitude) in [("merge_ordinary", 2.0), ("merge_large", 1e154)] {
            let mut left = stats([0.0, 0.0]);
            left.merge(&stats([magnitude, magnitude]));
            println!("AUDIT\t{case}\tmean\t{:?}", left.mean().unwrap());
            println!(
                "AUDIT\t{case}\tvariance\t{:?}",
                left.population_variance().unwrap()
            );
        }
    } else {
        divan::main();
    }
}

#[divan::bench]
fn merge_ordinary(bencher: Bencher) {
    bencher
        .with_inputs(|| (stats([0.0, 0.0]), stats([2.0, 2.0])))
        .bench_local_values(|(mut left, right)| {
            left.merge(black_box(&right));
            black_box(left)
        });
}

#[divan::bench]
fn merge_large_finite(bencher: Bencher) {
    bencher
        .with_inputs(|| (stats([0.0, 0.0]), stats([1e154, 1e154])))
        .bench_local_values(|(mut left, right)| {
            left.merge(black_box(&right));
            black_box(left)
        });
}

#[divan::bench(args = [4, 64, 1024])]
fn merge_chunks(bencher: Bencher, chunks: usize) {
    let inputs = (0..chunks)
        .map(|i| stats([i as f64, i as f64 + 0.5]))
        .collect::<Vec<_>>();
    bencher.counter(ItemsCount::new(chunks)).bench_local(|| {
        let mut result = RunningStats::new();
        for chunk in black_box(&inputs) {
            result.merge(chunk);
        }
        black_box(result)
    });
}
