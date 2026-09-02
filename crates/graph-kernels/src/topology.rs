use std::collections::{HashMap, VecDeque, hash_map::Entry};
use std::hash::Hash;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CycleDetected;

/// Returns a deterministic topological ordering of the supplied nodes and all
/// nodes reachable from them, or `CycleDetected` when the reachable graph is not
/// acyclic.
pub fn topological_sort<N, Nodes, I, Neighbors>(
    nodes: Nodes,
    mut neighbors: Neighbors,
) -> Result<Vec<N>, CycleDetected>
where
    N: Clone + Eq + Hash,
    Nodes: IntoIterator<Item = N>,
    I: IntoIterator<Item = N>,
    Neighbors: FnMut(&N) -> I,
{
    let graph = materialize_graph(nodes, &mut neighbors);
    let mut indegree = vec![0_usize; graph.nodes.len()];
    for outgoing in &graph.adjacency {
        for &next in outgoing {
            indegree[next] += 1;
        }
    }

    let mut ready = VecDeque::new();
    for (index, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            ready.push_back(index);
        }
    }

    let mut order = Vec::with_capacity(graph.nodes.len());
    while let Some(node) = ready.pop_front() {
        order.push(node);
        for &next in &graph.adjacency[node] {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                ready.push_back(next);
            }
        }
    }

    if order.len() != graph.nodes.len() {
        return Err(CycleDetected);
    }

    Ok(order
        .into_iter()
        .map(|index| graph.nodes[index].clone())
        .collect())
}

/// Computes strongly connected components with Tarjan's algorithm.
///
/// Component members and the component list are normalized by first discovery
/// order so deterministic input iteration yields deterministic output.
pub fn strongly_connected_components<N, Nodes, I, Neighbors>(
    nodes: Nodes,
    mut neighbors: Neighbors,
) -> Vec<Vec<N>>
where
    N: Clone + Eq + Hash,
    Nodes: IntoIterator<Item = N>,
    I: IntoIterator<Item = N>,
    Neighbors: FnMut(&N) -> I,
{
    let graph = materialize_graph(nodes, &mut neighbors);
    let mut state = TarjanState::new(graph.nodes.len());

    for node in 0..graph.nodes.len() {
        if state.indices[node].is_none() {
            state.strong_connect(node, &graph.adjacency);
        }
    }

    for component in &mut state.components {
        component.sort_unstable();
    }
    state.components.sort_by_key(|component| component[0]);

    state
        .components
        .into_iter()
        .map(|component| {
            component
                .into_iter()
                .map(|index| graph.nodes[index].clone())
                .collect()
        })
        .collect()
}

struct TarjanState {
    next_index: usize,
    indices: Vec<Option<usize>>,
    lowlink: Vec<usize>,
    stack: Vec<usize>,
    on_stack: Vec<bool>,
    components: Vec<Vec<usize>>,
}

impl TarjanState {
    fn new(len: usize) -> Self {
        Self {
            next_index: 0,
            indices: vec![None; len],
            lowlink: vec![0; len],
            stack: Vec::new(),
            on_stack: vec![false; len],
            components: Vec::new(),
        }
    }

    fn strong_connect(&mut self, node: usize, adjacency: &[Vec<usize>]) {
        let node_index = self.next_index;
        self.next_index += 1;
        self.indices[node] = Some(node_index);
        self.lowlink[node] = node_index;
        self.stack.push(node);
        self.on_stack[node] = true;

        for &next in &adjacency[node] {
            if self.indices[next].is_none() {
                self.strong_connect(next, adjacency);
                self.lowlink[node] = self.lowlink[node].min(self.lowlink[next]);
            } else if self.on_stack[next] {
                if let Some(next_discovery_index) = self.indices[next] {
                    self.lowlink[node] = self.lowlink[node].min(next_discovery_index);
                }
            }
        }

        if self.lowlink[node] != node_index {
            return;
        }

        let mut component = Vec::new();
        while let Some(member) = self.stack.pop() {
            self.on_stack[member] = false;
            component.push(member);
            if member == node {
                break;
            }
        }
        self.components.push(component);
    }
}

struct MaterializedGraph<N> {
    nodes: Vec<N>,
    adjacency: Vec<Vec<usize>>,
}

fn materialize_graph<N, Nodes, I, Neighbors>(
    seeds: Nodes,
    neighbors: &mut Neighbors,
) -> MaterializedGraph<N>
where
    N: Clone + Eq + Hash,
    Nodes: IntoIterator<Item = N>,
    I: IntoIterator<Item = N>,
    Neighbors: FnMut(&N) -> I,
{
    let mut nodes = Vec::new();
    let mut indices = HashMap::new();

    for node in seeds {
        if let Entry::Vacant(entry) = indices.entry(node.clone()) {
            let index = nodes.len();
            entry.insert(index);
            nodes.push(node);
        }
    }

    let mut adjacency = Vec::new();
    let mut cursor = 0;
    while cursor < nodes.len() {
        let node = nodes[cursor].clone();
        let outgoing_nodes: Vec<_> = neighbors(&node).into_iter().collect();
        let mut outgoing = Vec::with_capacity(outgoing_nodes.len());

        for next in outgoing_nodes {
            let next_index = match indices.entry(next.clone()) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let discovered = nodes.len();
                    entry.insert(discovered);
                    nodes.push(next);
                    discovered
                }
            };
            outgoing.push(next_index);
        }

        adjacency.push(outgoing);
        cursor += 1;
    }

    MaterializedGraph { nodes, adjacency }
}

#[cfg(test)]
mod tests {
    use super::{CycleDetected, strongly_connected_components, topological_sort};

    #[test]
    fn topological_sort_orders_dependencies_before_dependents() {
        let neighbors = |node: &char| match node {
            'A' => vec!['B', 'C'],
            'B' => vec!['D'],
            'C' => vec!['D'],
            _ => Vec::new(),
        };

        assert_eq!(
            topological_sort(['A'], neighbors),
            Ok(vec!['A', 'B', 'C', 'D'])
        );
    }

    #[test]
    fn topological_sort_detects_cycles() {
        let neighbors = |node: &u8| match node {
            0 => vec![1],
            1 => vec![2],
            2 => vec![0],
            _ => Vec::new(),
        };

        assert_eq!(topological_sort([0], neighbors), Err(CycleDetected));
    }

    #[test]
    fn tarjan_finds_and_normalizes_strongly_connected_components() {
        let neighbors = |node: &u8| match node {
            0 => vec![1],
            1 => vec![2],
            2 => vec![0, 3],
            3 => vec![4],
            4 => vec![3],
            _ => Vec::new(),
        };

        assert_eq!(
            strongly_connected_components([0, 5], neighbors),
            vec![vec![0, 1, 2], vec![3, 4], vec![5]]
        );
    }
}
