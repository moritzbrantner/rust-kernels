//! New capabilities compared with simple independent references, not past versions.
//! Both sides include per-query working storage. Input construction is excluded.
use divan::{Bencher, black_box};
use graph_kernels::{CapacityEdge, dinic_max_flow, hopcroft_karp, hungarian_assignment};
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();
fn paths(n: usize) -> Vec<CapacityEdge> {
    (1..=n)
        .flat_map(|i| {
            [
                CapacityEdge {
                    from: 0,
                    to: i,
                    capacity: 3,
                },
                CapacityEdge {
                    from: i,
                    to: n + 1,
                    capacity: 3,
                },
            ]
        })
        .collect()
}
// Breadth-first, one augmenting path at a time; residual arcs are paired by index.
fn edmonds_karp(n: usize, edges: &[CapacityEdge]) -> (u128, usize) {
    let mut adjacency = vec![Vec::new(); n];
    let mut residual = Vec::new();
    for e in edges {
        let i = residual.len();
        adjacency[e.from].push(i);
        adjacency[e.to].push(i + 1);
        residual.push((e.to, e.capacity));
        residual.push((e.from, 0_u64));
    }
    let mut parent = vec![None::<usize>; n];
    let mut queue = Vec::new();
    let mut value = 0_u128;
    let mut scans = 0;
    loop {
        parent.fill(None);
        parent[0] = Some(usize::MAX);
        queue.clear();
        queue.push(0);
        let mut cursor = 0;
        while cursor < queue.len() && parent[n - 1].is_none() {
            let u = queue[cursor];
            cursor += 1;
            for &edge in &adjacency[u] {
                scans += 1;
                let (v, capacity) = residual[edge];
                if capacity > 0 && parent[v].is_none() {
                    parent[v] = Some(edge);
                    queue.push(v);
                }
            }
        }
        if parent[n - 1].is_none() {
            return (value, scans);
        }
        let mut amount = u64::MAX;
        let mut v = n - 1;
        while v != 0 {
            let e = parent[v].unwrap();
            amount = amount.min(residual[e].1);
            v = residual[e ^ 1].0;
        }
        v = n - 1;
        while v != 0 {
            let e = parent[v].unwrap();
            residual[e].1 -= amount;
            residual[e ^ 1].1 += amount;
            v = residual[e ^ 1].0;
        }
        value += u128::from(amount);
    }
}
fn links(n: usize) -> Vec<(usize, usize)> {
    (0..n)
        .flat_map(|i| (0..8).map(move |k| (i, (i * 17 + k * 13) % n)))
        .collect()
}
// One BFS augment per left root; only cardinality is returned, no cover certificate.
fn single_augment(n: usize, edges: &[(usize, usize)]) -> usize {
    let mut adjacency = vec![Vec::new(); n];
    for &(a, b) in edges {
        adjacency[a].push(b);
    }
    let mut right = vec![None::<usize>; n];
    let mut left = vec![None::<usize>; n];
    let mut parent = vec![None::<usize>; n];
    let mut seen = vec![false; n];
    let mut queue = Vec::new();
    let mut size = 0;
    for root in 0..n {
        parent.fill(None);
        seen.fill(false);
        queue.clear();
        queue.push(root);
        seen[root] = true;
        let mut cursor = 0;
        let mut free = None;
        while cursor < queue.len() && free.is_none() {
            let u = queue[cursor];
            cursor += 1;
            for &v in &adjacency[u] {
                if parent[v].is_some() {
                    continue;
                }
                parent[v] = Some(u);
                if let Some(next) = right[v] {
                    if !seen[next] {
                        seen[next] = true;
                        queue.push(next);
                    }
                } else {
                    free = Some(v);
                    break;
                }
            }
        }
        if let Some(mut v) = free {
            loop {
                let u = parent[v].unwrap();
                let old = left[u];
                left[u] = Some(v);
                right[v] = Some(u);
                if let Some(previous) = old {
                    v = previous;
                } else {
                    break;
                }
            }
            size += 1;
        }
    }
    size
}
fn costs(n: usize) -> Vec<Vec<i64>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| ((i * 137 + j * 79 + i * j * 31) % 997) as i64 - 500)
                .collect()
        })
        .collect()
}
fn exhaustive(costs: &[Vec<i64>], row: usize, used: usize) -> i128 {
    if row == costs.len() {
        return 0;
    }
    (0..costs.len())
        .filter(|&j| used & (1 << j) == 0)
        .map(|j| i128::from(costs[row][j]) + exhaustive(costs, row + 1, used | (1 << j)))
        .min()
        .unwrap()
}
#[divan::bench(args=[32,128,512])]
fn dinic(bencher: Bencher, n: usize) {
    let edges = paths(n);
    bencher.bench_local(|| black_box(dinic_max_flow(n + 2, black_box(&edges), 0, n + 1).unwrap()));
}
#[divan::bench(args=[32,128,512])]
fn edmonds_karp_reference(bencher: Bencher, n: usize) {
    let edges = paths(n);
    bencher.bench_local(|| black_box(edmonds_karp(n + 2, black_box(&edges))));
}
#[divan::bench(args=[32,128,512])]
fn hopcroft(bencher: Bencher, n: usize) {
    let edges = links(n);
    bencher.bench_local(|| black_box(hopcroft_karp(n, n, black_box(&edges)).unwrap()));
}
#[divan::bench(args=[32,128,512])]
fn single_augment_reference(bencher: Bencher, n: usize) {
    let edges = links(n);
    bencher.bench_local(|| black_box(single_augment(n, black_box(&edges))));
}
#[divan::bench(args=[8,32,96])]
fn hungarian(bencher: Bencher, n: usize) {
    let costs = costs(n);
    bencher.bench_local(|| black_box(hungarian_assignment(black_box(&costs)).unwrap()));
}
#[divan::bench]
fn exhaustive_assignment_8(bencher: Bencher) {
    let costs = costs(8);
    bencher.bench_local(|| black_box(exhaustive(black_box(&costs), 0, 0)));
}
fn main() {
    if !std::env::args().any(|a| a == "--expansion-probe") {
        divan::main();
        return;
    }
    for n in [32, 128, 512] {
        let edges = paths(n);
        let flow = dinic_max_flow(n + 2, &edges, 0, n + 1).unwrap();
        let (value, scans) = edmonds_karp(n + 2, &edges);
        assert_eq!(flow.value, value);
        assert_eq!(value, 3 * n as u128);
        println!("AUDIT\tflow_{n}\tvalue\t{value}");
        println!("AUDIT\tflow_{n}\tdinic_scans\t{}", flow.work.edge_scans);
        println!("AUDIT\tflow_{n}\treference_scans\t{scans}");
        println!("AUDIT\tflow_{n}\tphases\t{}", flow.work.phases);
        let edges = links(n);
        let matching = hopcroft_karp(n, n, &edges).unwrap();
        assert_eq!(matching.size, single_augment(n, &edges));
        assert_eq!(matching.size, n);
        assert_eq!(matching.cover_left.len() + matching.cover_right.len(), n);
        println!("AUDIT\tmatching_{n}\tsize\t{}", matching.size);
    }
    let costs = costs(8);
    let result = hungarian_assignment(&costs).unwrap();
    assert_eq!(result.total_cost, exhaustive(&costs, 0, 0));
    println!("AUDIT\tassignment_8\tcost\t{}", result.total_cost);
    let greedy_counterexample = hungarian_assignment(&[[1, 2], [2, 100]]).unwrap();
    assert_eq!(greedy_counterexample.total_cost, 4);
    println!(
        "AUDIT\tassignment_greedy_counterexample\toptimum\t{}",
        greedy_counterexample.total_cost
    );
    println!("AUDIT\tassignment_greedy_counterexample\tgreedy\t101");
}
