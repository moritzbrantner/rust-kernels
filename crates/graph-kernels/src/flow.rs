use std::fmt;

/// One directed edge. Parallel, antiparallel and self edges are allowed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapacityEdge {
    pub from: usize,
    pub to: usize,
    pub capacity: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FlowError {
    InvalidVertex { vertex: usize, vertex_count: usize },
    EqualTerminals,
    TooManyEdges,
}

impl fmt::Display for FlowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid flow network: {self:?}")
    }
}
impl std::error::Error for FlowError {}

/// Deterministic work, not a wall-clock estimate. Scans include residual BFS/DFS.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FlowWork {
    pub phases: usize,
    pub augmentations: usize,
    pub edge_scans: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxFlow {
    /// Widened sum: parallel u64 capacities can yield more than u64::MAX flow.
    pub value: u128,
    /// Net flow for each original edge, in input order. Self edges carry zero.
    pub edge_flows: Vec<u64>,
    /// Residually reachable vertices from source: a minimum-cut certificate.
    pub source_side: Vec<bool>,
    pub work: FlowWork,
}

#[derive(Clone, Copy)]
struct ResidualEdge {
    to: usize,
    capacity: u64,
}

/// Dinic maximum flow and minimum cut, without recursive DFS.
///
/// Vertices are caller-indexed `0..vertex_count`. Every call starts from zero
/// flow; it never mutates the edge list or retains a residual network in it.
/// Input edge order breaks ties deterministically. Integer capacity arithmetic
/// is exact, with u128 total flow. O(V² E) worst-case time, O(V+E) storage.
/// Invalid vertices or equal source/sink return an error before building storage.
///
/// ```
/// use graph_kernels::{CapacityEdge, dinic_max_flow};
/// let edges = [CapacityEdge { from: 0, to: 1, capacity: 7 }];
/// let result = dinic_max_flow(2, &edges, 0, 1).unwrap();
/// assert_eq!(result.value, 7);
/// assert_eq!(result.edge_flows, [7]);
/// assert_eq!(result.source_side, [true, false]);
/// ```
pub fn dinic_max_flow(
    vertex_count: usize,
    edges: &[CapacityEdge],
    source: usize,
    sink: usize,
) -> Result<MaxFlow, FlowError> {
    for vertex in [source, sink]
        .into_iter()
        .chain(edges.iter().flat_map(|e| [e.from, e.to]))
    {
        if vertex >= vertex_count {
            return Err(FlowError::InvalidVertex {
                vertex,
                vertex_count,
            });
        }
    }
    if source == sink {
        return Err(FlowError::EqualTerminals);
    }
    let capacity = edges.len().checked_mul(2).ok_or(FlowError::TooManyEdges)?;
    let mut residual = Vec::with_capacity(capacity);
    let mut adjacency = vec![Vec::new(); vertex_count];
    for edge in edges {
        let index = residual.len();
        adjacency[edge.from].push(index);
        adjacency[edge.to].push(index + 1);
        residual.push(ResidualEdge {
            to: edge.to,
            capacity: edge.capacity,
        });
        residual.push(ResidualEdge {
            to: edge.from,
            capacity: 0,
        });
    }
    let mut levels = vec![usize::MAX; vertex_count];
    let mut next = vec![0; vertex_count];
    let mut queue = Vec::with_capacity(vertex_count);
    let mut path = Vec::<usize>::new();
    let mut bottlenecks = Vec::<u64>::new();
    let mut value = 0_u128;
    let mut work = FlowWork::default();
    loop {
        levels.fill(usize::MAX);
        levels[source] = 0;
        queue.clear();
        queue.push(source);
        let mut cursor = 0;
        while cursor < queue.len() {
            let node = queue[cursor];
            cursor += 1;
            for &index in &adjacency[node] {
                work.edge_scans += 1;
                let edge = residual[index];
                if edge.capacity > 0 && levels[edge.to] == usize::MAX {
                    levels[edge.to] = levels[node] + 1;
                    queue.push(edge.to);
                }
            }
        }
        if levels[sink] == usize::MAX {
            break;
        }
        work.phases += 1;
        next.fill(0);
        loop {
            path.clear();
            bottlenecks.clear();
            bottlenecks.push(u64::MAX);
            let mut node = source;
            loop {
                if node == sink {
                    break;
                }
                let mut found = None;
                while let Some(&index) = adjacency[node].get(next[node]) {
                    work.edge_scans += 1;
                    let edge = residual[index];
                    if edge.capacity > 0 && levels[edge.to] == levels[node] + 1 {
                        found = Some(index);
                        break;
                    }
                    next[node] += 1;
                }
                if let Some(index) = found {
                    path.push(index);
                    bottlenecks.push(
                        bottlenecks
                            .last()
                            .copied()
                            .unwrap()
                            .min(residual[index].capacity),
                    );
                    node = residual[index].to;
                } else if let Some(index) = path.pop() {
                    bottlenecks.pop();
                    node = residual[index ^ 1].to;
                    next[node] += 1;
                } else {
                    break;
                }
            }
            if node != sink {
                break;
            }
            let amount = *bottlenecks.last().unwrap();
            for &index in &path {
                residual[index].capacity -= amount;
                // Each reverse pair sums to its original u64 capacity, even in
                // networks with independent antiparallel original edges.
                residual[index ^ 1].capacity += amount;
            }
            value += u128::from(amount);
            work.augmentations += 1;
        }
    }
    Ok(MaxFlow {
        value,
        edge_flows: edges
            .iter()
            .enumerate()
            .map(|(i, e)| e.capacity - residual[2 * i].capacity)
            .collect(),
        source_side: levels.iter().map(|&level| level != usize::MAX).collect(),
        work,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn edge(from: usize, to: usize, capacity: u64) -> CapacityEdge {
        CapacityEdge { from, to, capacity }
    }

    fn certify(n: usize, edges: &[CapacityEdge], source: usize, sink: usize, flow: &MaxFlow) {
        assert_eq!(flow.edge_flows.len(), edges.len());
        assert_eq!(flow.source_side.len(), n);
        assert!(flow.source_side[source] && !flow.source_side[sink]);
        let mut net = vec![0_i128; n];
        let mut cut = 0_u128;
        for (e, &f) in edges.iter().zip(&flow.edge_flows) {
            assert!(f <= e.capacity);
            net[e.from] += i128::from(f);
            net[e.to] -= i128::from(f);
            if flow.source_side[e.from] && !flow.source_side[e.to] {
                cut += u128::from(e.capacity);
            }
        }
        assert_eq!(cut, flow.value);
        for (i, amount) in net.into_iter().enumerate() {
            assert_eq!(
                amount,
                if i == source {
                    flow.value as i128
                } else if i == sink {
                    -(flow.value as i128)
                } else {
                    0
                }
            );
        }
    }
    fn cut_oracle(n: usize, edges: &[CapacityEdge]) -> u128 {
        (0..1_usize << n)
            .filter(|mask| mask & 1 != 0 && mask & (1 << (n - 1)) == 0)
            .map(|mask| {
                edges
                    .iter()
                    .filter(|e| mask & (1 << e.from) != 0 && mask & (1 << e.to) == 0)
                    .map(|e| u128::from(e.capacity))
                    .sum()
            })
            .min()
            .unwrap()
    }

    #[test]
    fn matches_every_four_vertex_unit_network_and_its_min_cut() {
        let all: Vec<_> = (0..4)
            .flat_map(|a| (0..4).filter(move |&b| b != a).map(move |b| edge(a, b, 1)))
            .collect();
        for mask in 0..1 << all.len() {
            let edges: Vec<_> = all
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, e)| *e)
                .collect();
            let result = dinic_max_flow(4, &edges, 0, 3).unwrap();
            assert_eq!(result.value, cut_oracle(4, &edges));
            certify(4, &edges, 0, 3, &result);
        }
    }
    #[test]
    fn weighted_parallel_antiparallel_self_and_wide_totals() {
        let edges = [
            edge(0, 0, 9),
            edge(0, 1, u64::MAX),
            edge(0, 1, u64::MAX),
            edge(1, 0, u64::MAX),
            edge(1, 2, u64::MAX),
            edge(1, 2, u64::MAX),
        ];
        let result = dinic_max_flow(3, &edges, 0, 2).unwrap();
        assert_eq!(result.value, 2 * u128::from(u64::MAX));
        assert_eq!(result.edge_flows[0], 0);
        certify(3, &edges, 0, 2, &result);
        assert_eq!(result, dinic_max_flow(3, &edges, 0, 2).unwrap());
        let mut state = 123_u32;
        for _ in 0..128 {
            let edges: Vec<_> = (0..16)
                .map(|_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    edge(
                        (state % 5) as usize,
                        ((state >> 8) % 5) as usize,
                        u64::from((state >> 16) % 10),
                    )
                })
                .collect();
            let result = dinic_max_flow(5, &edges, 0, 4).unwrap();
            assert_eq!(result.value, cut_oracle(5, &edges));
            certify(5, &edges, 0, 4, &result);
        }
    }
    #[test]
    fn reverse_residual_rerouting_and_input_errors() {
        // A first path uses 1->3, then flow must be rerouted via its reverse.
        let edges = [
            edge(0, 1, 1),
            edge(0, 2, 1),
            edge(1, 3, 1),
            edge(1, 4, 1),
            edge(2, 3, 1),
            edge(3, 5, 1),
            edge(4, 5, 1),
        ];
        let flow = dinic_max_flow(6, &edges, 0, 5).unwrap();
        assert_eq!(flow.value, 2);
        assert_eq!(flow.edge_flows[2], 0);
        certify(6, &edges, 0, 5, &flow);
        assert!(matches!(
            dinic_max_flow(0, &[], 0, 0),
            Err(FlowError::InvalidVertex { .. })
        ));
        assert_eq!(dinic_max_flow(1, &[], 0, 0), Err(FlowError::EqualTerminals));
        assert!(dinic_max_flow(2, &[edge(0, 2, 1)], 0, 1).is_err());
        assert_eq!(dinic_max_flow(2, &[edge(0, 1, 0)], 0, 1).unwrap().value, 0);
        assert_eq!(dinic_max_flow(2, &[], 0, 1).unwrap().value, 0);
    }
    #[test]
    fn blocking_flow_uses_one_phase_for_disjoint_paths() {
        for n in [32, 256, 2048] {
            let edges: Vec<_> = (1..=n)
                .flat_map(|i| [edge(0, i, 1), edge(i, n + 1, 1)])
                .collect();
            let result = dinic_max_flow(n + 2, &edges, 0, n + 1).unwrap();
            assert_eq!(result.value, n as u128);
            assert_eq!(result.work.phases, 1);
            assert_eq!(result.work.augmentations, n);
            assert!(result.work.edge_scans <= 12 * edges.len());
        }
    }
}
