use criterion::{Criterion, black_box, criterion_group, criterion_main};
use graph_kernels::{PageRankConfig, page_rank};

const NODES: usize = 2_048;

fn neighbors(node: &usize) -> Vec<usize> {
    if node % 17 == 0 {
        Vec::new()
    } else {
        vec![
            (node + 1) % NODES,
            (node * 7 + 3) % NODES,
            (node * 13 + 5) % NODES,
        ]
    }
}

fn iterative_page_rank(c: &mut Criterion) {
    c.bench_function("page_rank/2048_nodes_bounded_40", |b| {
        b.iter(|| {
            let report = page_rank(
                0..NODES,
                neighbors,
                PageRankConfig {
                    damping: 0.85,
                    tolerance: 0.0,
                    max_iterations: 40,
                },
            )
            .expect("benchmark configuration is valid");
            black_box((report.iterations, report.residual, report.scores));
        });
    });
}

criterion_group!(benches, iterative_page_rank);
criterion_main!(benches);
