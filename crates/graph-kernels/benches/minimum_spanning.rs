use criterion::{Criterion, black_box, criterion_group, criterion_main};
use graph_kernels::{WeightedEdge, kruskal_minimum_spanning_forest};

const VERTICES: usize = 4_096;
const EDGES: usize = 32_768;

fn generated_edges() -> Vec<WeightedEdge<u32>> {
    (0..EDGES)
        .map(|index| {
            let a = (index % VERTICES) as u32;
            let b = ((index * 37 + 17) % VERTICES) as u32;
            let weight = ((index * 104_729 + 31) % 1_000_003) as u64;
            WeightedEdge::new(a, b, weight)
        })
        .collect()
}

fn kruskal_dense_input(c: &mut Criterion) {
    let edges = generated_edges();
    c.bench_function("kruskal/index_and_sort_32768_edges", |b| {
        b.iter_batched(
            || edges.clone(),
            |edges| {
                let forest = kruskal_minimum_spanning_forest(0..VERTICES as u32, edges);
                black_box((forest.total_weight, forest.component_count));
            },
            criterion::BatchSize::LargeInput,
        );
    });
}

criterion_group!(benches, kruskal_dense_input);
criterion_main!(benches);
