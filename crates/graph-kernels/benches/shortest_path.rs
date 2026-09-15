use std::hash::{Hash, Hasher};

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use graph_kernels::dijkstra;

const NODES: u32 = 4_096;

#[derive(Clone, Eq, PartialEq)]
struct WideNode {
    id: u32,
    payload: [u8; 128],
}

impl WideNode {
    fn new(id: u32) -> Self {
        Self {
            id,
            payload: [id as u8; 128],
        }
    }
}

impl Hash for WideNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

fn neighbors(node: &WideNode) -> Option<(WideNode, u64)> {
    (node.id + 1 < NODES).then(|| (WideNode::new(node.id + 1), 1))
}

fn dijkstra_wide_chain(c: &mut Criterion) {
    c.bench_function("shortest_path/dijkstra_wide_chain_4096", |b| {
        b.iter(|| {
            let path = dijkstra(
                WideNode::new(0),
                |node| node.id == NODES - 1,
                neighbors,
            )
            .expect("chain endpoint is reachable");
            black_box((path.cost, path.nodes.len()));
        });
    });
}

criterion_group!(benches, dijkstra_wide_chain);
criterion_main!(benches);
