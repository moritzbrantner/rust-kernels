use criterion::{Criterion, black_box, criterion_group, criterion_main};
use graph_kernels::topological_sort;

const NODES: u32 = 16_384;
const OUTGOING: u32 = 8;

fn range_neighbors(node: &u32) -> std::ops::Range<u32> {
    let start = node + 1;
    start..(start + OUTGOING).min(NODES)
}

fn materialize_range_neighbors(c: &mut Criterion) {
    c.bench_function("topology/materialize_range_neighbors_16384", |b| {
        b.iter(|| {
            let order = topological_sort(0..NODES, range_neighbors)
                .expect("forward-only benchmark graph is acyclic");
            black_box(order.len());
        });
    });
}

criterion_group!(benches, materialize_range_neighbors);
criterion_main!(benches);
