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

use statistics_kernels::RunningStats;

#[divan::bench(args = [16, 256, 4096])]
fn push_ordinary(bencher: Bencher, n: usize) {
    let values = (0..n).map(|v| (v % 31) as f64 - 15.0).collect::<Vec<_>>();
    bencher.bench_local(|| {
        let mut stats = RunningStats::new();
        stats.extend(black_box(&values).iter().copied());
        black_box(stats)
    });
}

#[divan::bench]
fn push_extremes(bencher: Bencher) {
    bencher.bench_local(|| {
        let mut stats = RunningStats::new();
        stats.extend(black_box([-1e308, 1e308, 0.0]));
        black_box(stats)
    });
}

fn probe() {
    let mut extreme = RunningStats::new();
    extreme.extend([-1e308, 1e308, 0.0]);
    emit("push_extremes", "mean", extreme.mean().unwrap());
    emit(
        "push_extremes",
        "variance",
        extreme.population_variance().unwrap(),
    );
    for n in [16, 256, 4096] {
        let mut stats = RunningStats::new();
        stats.extend((0..n).map(|v| (v % 31) as f64 - 15.0));
        emit(
            &format!("push_{n}"),
            "mean_bits",
            stats.mean().unwrap().to_bits(),
        );
        emit(
            &format!("push_{n}"),
            "variance_bits",
            stats.population_variance().unwrap().to_bits(),
        );
    }
}
