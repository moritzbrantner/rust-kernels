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

use graph_kernels::strongly_connected_components;

#[divan::bench(args = [64, 512, 2048])]
fn scc_chain(bencher: Bencher, n: usize) {
    bencher.bench_local(|| {
        black_box(strongly_connected_components([0_usize], |&v| {
            (v + 1 < black_box(n)).then_some(v + 1)
        }))
    });
}

#[divan::bench(args = [64, 512, 2048])]
fn scc_cycle(bencher: Bencher, n: usize) {
    bencher.bench_local(|| {
        black_box(strongly_connected_components([0_usize], |&v| {
            [(v + 1) % black_box(n)]
        }))
    });
}

#[divan::bench(args = [64, 512, 2048])]
fn scc_disconnected(bencher: Bencher, n: usize) {
    bencher.bench_local(|| {
        black_box(strongly_connected_components(0..black_box(n), |_| {
            std::iter::empty::<usize>()
        }))
    });
}

fn probe() {
    for n in [64, 512, 2048] {
        let chain = strongly_connected_components([0_usize], |&v| (v + 1 < n).then_some(v + 1));
        let cycle = strongly_connected_components([0_usize], |&v| [(v + 1) % n]);
        assert_eq!(chain, (0..n).map(|v| vec![v]).collect::<Vec<_>>());
        assert_eq!(cycle, vec![(0..n).collect::<Vec<_>>()]);
        emit(&format!("scc_{n}"), "chain_components", chain.len());
        emit(&format!("scc_{n}"), "cycle_components", cycle.len());
        emit(
            &format!("scc_{n}"),
            "checksum",
            chain.iter().flatten().sum::<usize>(),
        );
    }
}
