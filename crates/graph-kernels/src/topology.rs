use std::collections::{HashMap, VecDeque, hash_map::Entry};
use std::hash::Hash;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CycleDetected;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageRankConfig {
    pub damping: f64,
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            tolerance: 1.0e-10,
            max_iterations: 100,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageRankError {
    InvalidDamping,
    InvalidTolerance,
    ZeroIterations,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PageRank<N> {
    pub scores: Vec<(N, f64)>,
    pub iterations: usize,
    pub residual: f64,
    pub converged: bool,
}

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
/// Component members and the component list are normalized by materialization
/// order (unique seeds first, then newly discovered neighbors), preserving the
/// existing deterministic output contract. DFS frames live on the heap, so a
/// long path does not consume one native stack frame per vertex.
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

/// Computes PageRank scores for caller-owned directed graph storage.
///
/// Seed and neighbor iteration order define stable discovery order and therefore
/// the order of returned scores. Dangling-node mass is redistributed uniformly
/// on every iteration, so total rank mass remains normalized. Parallel outgoing
/// edges are treated as distinct links.
pub fn page_rank<N, Nodes, I, Neighbors>(
    nodes: Nodes,
    mut neighbors: Neighbors,
    config: PageRankConfig,
) -> Result<PageRank<N>, PageRankError>
where
    N: Clone + Eq + Hash,
    Nodes: IntoIterator<Item = N>,
    I: IntoIterator<Item = N>,
    Neighbors: FnMut(&N) -> I,
{
    validate_page_rank_config(config)?;
    let graph = materialize_graph(nodes, &mut neighbors);
    let node_count = graph.nodes.len();

    if node_count == 0 {
        return Ok(PageRank {
            scores: Vec::new(),
            iterations: 0,
            residual: 0.0,
            converged: true,
        });
    }

    let node_count_f64 = node_count as f64;
    let mut scores = vec![1.0 / node_count_f64; node_count];
    let mut next = vec![0.0; node_count];
    let base = (1.0 - config.damping) / node_count_f64;
    let mut residual = f64::INFINITY;

    for iteration in 1..=config.max_iterations {
        let dangling_mass: f64 = graph
            .adjacency
            .iter()
            .enumerate()
            .filter(|(_, outgoing)| outgoing.is_empty())
            .map(|(index, _)| scores[index])
            .sum();
        let dangling_share = config.damping * dangling_mass / node_count_f64;
        next.fill(base + dangling_share);

        for (source, outgoing) in graph.adjacency.iter().enumerate() {
            if outgoing.is_empty() {
                continue;
            }
            let share = config.damping * scores[source] / outgoing.len() as f64;
            for &target in outgoing {
                next[target] += share;
            }
        }

        residual = scores
            .iter()
            .zip(&next)
            .map(|(left, right)| (left - right).abs())
            .sum();
        std::mem::swap(&mut scores, &mut next);

        if residual <= config.tolerance {
            return Ok(PageRank {
                scores: graph.nodes.into_iter().zip(scores).collect(),
                iterations: iteration,
                residual,
                converged: true,
            });
        }
    }

    Ok(PageRank {
        scores: graph.nodes.into_iter().zip(scores).collect(),
        iterations: config.max_iterations,
        residual,
        converged: false,
    })
}

fn validate_page_rank_config(config: PageRankConfig) -> Result<(), PageRankError> {
    if !config.damping.is_finite() || !(0.0..=1.0).contains(&config.damping) {
        return Err(PageRankError::InvalidDamping);
    }
    if !config.tolerance.is_finite() || config.tolerance < 0.0 {
        return Err(PageRankError::InvalidTolerance);
    }
    if config.max_iterations == 0 {
        return Err(PageRankError::ZeroIterations);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct DfsFrame {
    node: usize,
    next_edge: usize,
}

struct TarjanState {
    // Reuse frame storage across disconnected components.
    frames: Vec<DfsFrame>,
    #[cfg(test)]
    edge_visits: usize,
    #[cfg(test)]
    max_frames: usize,
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
            frames: Vec::new(),
            #[cfg(test)]
            edge_visits: 0,
            #[cfg(test)]
            max_frames: 0,
            next_index: 0,
            indices: vec![None; len],
            lowlink: vec![0; len],
            stack: Vec::new(),
            on_stack: vec![false; len],
            components: Vec::new(),
        }
    }

    fn discover(&mut self, node: usize) {
        let node_index = self.next_index;
        self.next_index += 1;
        self.indices[node] = Some(node_index);
        self.lowlink[node] = node_index;
        self.stack.push(node);
        self.on_stack[node] = true;
        self.frames.push(DfsFrame { node, next_edge: 0 });
        #[cfg(test)]
        {
            self.max_frames = self.max_frames.max(self.frames.len());
        }
    }

    fn strong_connect(&mut self, root: usize, adjacency: &[Vec<usize>]) {
        self.discover(root);
        while let Some(frame) = self.frames.last_mut() {
            let node = frame.node;
            if let Some(&next) = adjacency[node].get(frame.next_edge) {
                frame.next_edge += 1;
                #[cfg(test)]
                {
                    self.edge_visits += 1;
                }
                if let Some(next_index) = self.indices[next] {
                    if self.on_stack[next] {
                        self.lowlink[node] = self.lowlink[node].min(next_index);
                    }
                } else {
                    self.discover(next);
                }
                continue;
            }

            // Resume exactly where recursive Tarjan would return to its caller.
            self.frames.pop();
            if self.indices[node] == Some(self.lowlink[node]) {
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
            if let Some(parent) = self.frames.last() {
                self.lowlink[parent.node] = self.lowlink[parent.node].min(self.lowlink[node]);
            }
        }
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
        if let Entry::Vacant(entry) = indices.entry(node) {
            let index = nodes.len();
            nodes.push(entry.key().clone());
            entry.insert(index);
        }
    }

    let mut adjacency = Vec::new();
    let mut cursor = 0;
    while cursor < nodes.len() {
        let outgoing_nodes = neighbors(&nodes[cursor]).into_iter();
        let (minimum_outgoing, _) = outgoing_nodes.size_hint();
        let mut outgoing = Vec::with_capacity(minimum_outgoing);

        for next in outgoing_nodes {
            let next_index = match indices.entry(next) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let discovered = nodes.len();
                    nodes.push(entry.key().clone());
                    entry.insert(discovered);
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
    use std::hash::{Hash, Hasher};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::{
        CycleDetected, PageRankConfig, PageRankError, materialize_graph, page_rank,
        strongly_connected_components, topological_sort,
    };

    #[derive(Debug)]
    struct CountedNode {
        id: usize,
        clones: Arc<AtomicUsize>,
    }

    impl CountedNode {
        fn new(id: usize, clones: &Arc<AtomicUsize>) -> Self {
            Self {
                id,
                clones: Arc::clone(clones),
            }
        }
    }

    impl Clone for CountedNode {
        fn clone(&self) -> Self {
            self.clones.fetch_add(1, Ordering::Relaxed);
            Self::new(self.id, &self.clones)
        }
    }

    impl PartialEq for CountedNode {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id
        }
    }

    impl Eq for CountedNode {}

    impl Hash for CountedNode {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.id.hash(state);
        }
    }

    #[test]
    fn iterative_tarjan_visits_each_edge_once_and_reuses_frames() {
        for n in [64, 512, 4096] {
            let graph: Vec<Vec<usize>> = (0..n).map(|v| vec![(v + 1) % n, v]).collect();
            let mut state = super::TarjanState::new(n);
            state.strong_connect(0, &graph);
            assert_eq!(state.edge_visits, 2 * n);
            assert_eq!(state.next_index, n);
            assert_eq!(state.max_frames, n);
            assert!(state.frames.is_empty());
            assert!(state.stack.is_empty());
            assert_eq!(state.components.len(), 1);
            assert_eq!(state.components[0].len(), n);
        }
        let graph = vec![Vec::new(); 1024];
        let mut state = super::TarjanState::new(graph.len());
        state.strong_connect(0, &graph);
        let frame_capacity = state.frames.capacity();
        for root in 1..graph.len() {
            state.strong_connect(root, &graph);
        }
        assert_eq!(state.frames.capacity(), frame_capacity);
        assert_eq!(state.max_frames, 1);
        assert_eq!(state.edge_visits, 0);
    }

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
    fn materialization_does_not_clone_already_known_neighbors() {
        const NODE_COUNT: usize = 16;

        let clones = Arc::new(AtomicUsize::new(0));
        let make_seeds = || {
            (0..NODE_COUNT)
                .map(|id| CountedNode::new(id, &clones))
                .collect::<Vec<_>>()
        };

        clones.store(0, Ordering::Relaxed);
        let mut sparse_neighbors = |_node: &CountedNode| Vec::<CountedNode>::new();
        let sparse = materialize_graph(make_seeds(), &mut sparse_neighbors);
        let sparse_clones = clones.load(Ordering::Relaxed);
        assert_eq!(sparse.nodes.len(), NODE_COUNT);
        assert!(sparse.adjacency.iter().all(Vec::is_empty));
        assert_eq!(sparse_clones, NODE_COUNT);
        drop(sparse);

        clones.store(0, Ordering::Relaxed);
        let dense_clones = Arc::clone(&clones);
        let mut dense_neighbors = move |node: &CountedNode| {
            ((node.id + 1)..NODE_COUNT)
                .map(|id| CountedNode::new(id, &dense_clones))
                .collect::<Vec<_>>()
        };
        let dense = materialize_graph(make_seeds(), &mut dense_neighbors);
        let dense_clone_count = clones.load(Ordering::Relaxed);

        assert_eq!(
            dense.adjacency.iter().map(Vec::len).sum::<usize>(),
            NODE_COUNT * (NODE_COUNT - 1) / 2
        );
        assert_eq!(dense_clone_count, sparse_clones);
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
            vec![vec![0, 1, 2], vec![5], vec![3, 4]]
        );
    }

    #[test]
    fn page_rank_keeps_a_cycle_uniform() {
        let report = page_rank(
            [0_u8, 1, 2],
            |node| vec![(*node + 1) % 3],
            PageRankConfig::default(),
        )
        .unwrap();

        assert!(report.converged);
        assert_eq!(report.scores.len(), 3);
        for (_, score) in report.scores {
            assert!((score - 1.0 / 3.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn page_rank_redistributes_dangling_mass() {
        let report = page_rank(
            [0_u8, 1],
            |node| if *node == 0 { vec![1] } else { Vec::new() },
            PageRankConfig::default(),
        )
        .unwrap();

        let total: f64 = report.scores.iter().map(|(_, score)| *score).sum();
        assert!((total - 1.0).abs() < 1.0e-12);
        assert!(report.scores[1].1 > report.scores[0].1);
    }

    #[test]
    fn page_rank_preserves_discovery_order() {
        let report = page_rank(
            ['a'],
            |node| {
                if *node == 'a' {
                    vec!['b', 'c']
                } else {
                    Vec::new()
                }
            },
            PageRankConfig::default(),
        )
        .unwrap();

        assert_eq!(
            report
                .scores
                .iter()
                .map(|(node, _)| *node)
                .collect::<Vec<_>>(),
            vec!['a', 'b', 'c']
        );
    }

    #[test]
    fn page_rank_rejects_invalid_configuration() {
        assert_eq!(
            page_rank(
                [0_u8],
                |_| Vec::<u8>::new(),
                PageRankConfig {
                    damping: 1.1,
                    ..PageRankConfig::default()
                }
            ),
            Err(PageRankError::InvalidDamping)
        );
        assert_eq!(
            page_rank(
                [0_u8],
                |_| Vec::<u8>::new(),
                PageRankConfig {
                    tolerance: f64::NAN,
                    ..PageRankConfig::default()
                }
            ),
            Err(PageRankError::InvalidTolerance)
        );
        assert_eq!(
            page_rank(
                [0_u8],
                |_| Vec::<u8>::new(),
                PageRankConfig {
                    max_iterations: 0,
                    ..PageRankConfig::default()
                }
            ),
            Err(PageRankError::ZeroIterations)
        );
    }
}
