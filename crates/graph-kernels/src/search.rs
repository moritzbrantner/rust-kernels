use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Path<N> {
    pub cost: u64,
    pub nodes: Vec<N>,
}

struct QueueEntry<N> {
    estimated_total: u64,
    cost: u64,
    sequence: u64,
    node: N,
}

impl<N> PartialEq for QueueEntry<N> {
    fn eq(&self, other: &Self) -> bool {
        self.estimated_total == other.estimated_total
            && self.cost == other.cost
            && self.sequence == other.sequence
    }
}

impl<N> Eq for QueueEntry<N> {}

impl<N> Ord for QueueEntry<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .estimated_total
            .cmp(&self.estimated_total)
            .then_with(|| other.cost.cmp(&self.cost))
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

impl<N> PartialOrd for QueueEntry<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Finds a minimum-cost path over non-negative integer edge costs.
///
/// The graph stays external to this kernel: `neighbors` yields outgoing
/// `(node, cost)` edges for the requested node. Paths whose accumulated cost
/// cannot be represented by `u64` are ignored.
pub fn dijkstra<N, I, Goal, Neighbors>(
    start: N,
    is_goal: Goal,
    neighbors: Neighbors,
) -> Option<Path<N>>
where
    N: Clone + Eq + Hash,
    I: IntoIterator<Item = (N, u64)>,
    Goal: FnMut(&N) -> bool,
    Neighbors: FnMut(&N) -> I,
{
    astar(start, is_goal, neighbors, |_: &N| 0)
}

/// Finds a minimum-cost path using an admissible heuristic.
///
/// `heuristic` must not overestimate the remaining cost and should return zero
/// for goal nodes. Neighbor iteration order is used as a deterministic
/// tie-breaker when multiple frontier entries have equal priority.
pub fn astar<N, I, Goal, Neighbors, Heuristic>(
    start: N,
    mut is_goal: Goal,
    mut neighbors: Neighbors,
    mut heuristic: Heuristic,
) -> Option<Path<N>>
where
    N: Clone + Eq + Hash,
    I: IntoIterator<Item = (N, u64)>,
    Goal: FnMut(&N) -> bool,
    Neighbors: FnMut(&N) -> I,
    Heuristic: FnMut(&N) -> u64,
{
    let mut frontier = BinaryHeap::new();
    let mut best_cost = HashMap::new();
    let mut came_from = HashMap::new();
    let mut sequence = 0_u64;

    let start_estimate = heuristic(&start);
    best_cost.insert(start.clone(), 0_u64);
    frontier.push(QueueEntry {
        estimated_total: start_estimate,
        cost: 0,
        sequence,
        node: start,
    });

    while let Some(entry) = frontier.pop() {
        if best_cost.get(&entry.node).copied() != Some(entry.cost) {
            continue;
        }

        if is_goal(&entry.node) {
            return Some(reconstruct_path(&came_from, entry.node, entry.cost));
        }

        for (next, edge_cost) in neighbors(&entry.node) {
            let Some(next_cost) = entry.cost.checked_add(edge_cost) else {
                continue;
            };

            if best_cost
                .get(&next)
                .is_some_and(|known_cost| *known_cost <= next_cost)
            {
                continue;
            }

            came_from.insert(next.clone(), entry.node.clone());
            best_cost.insert(next.clone(), next_cost);
            sequence = sequence.wrapping_add(1);
            frontier.push(QueueEntry {
                estimated_total: next_cost.saturating_add(heuristic(&next)),
                cost: next_cost,
                sequence,
                node: next,
            });
        }
    }

    None
}

fn reconstruct_path<N>(came_from: &HashMap<N, N>, goal: N, cost: u64) -> Path<N>
where
    N: Clone + Eq + Hash,
{
    let mut nodes = vec![goal];
    while let Some(previous) = came_from.get(nodes.last().expect("path always contains its goal")) {
        nodes.push(previous.clone());
    }
    nodes.reverse();
    Path { cost, nodes }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::{Path, astar, dijkstra, reconstruct_path};

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

    fn neighbors(node: &char) -> Vec<(char, u64)> {
        match node {
            'A' => vec![('B', 4), ('C', 2)],
            'B' => vec![('D', 5)],
            'C' => vec![('B', 1), ('D', 8)],
            _ => Vec::new(),
        }
    }

    #[test]
    fn dijkstra_finds_the_minimum_cost_path() {
        assert_eq!(
            dijkstra('A', |node| *node == 'D', neighbors),
            Some(Path {
                cost: 8,
                nodes: vec!['A', 'C', 'B', 'D'],
            })
        );
    }

    #[test]
    fn astar_matches_dijkstra_with_an_admissible_heuristic() {
        let expected = dijkstra('A', |node| *node == 'D', neighbors);
        let actual = astar(
            'A',
            |node| *node == 'D',
            neighbors,
            |node| match node {
                'A' => 6,
                'B' => 5,
                'C' => 6,
                'D' => 0,
                _ => 0,
            },
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn path_reconstruction_clones_each_predecessor_once() {
        const NODE_COUNT: usize = 64;

        let clones = Arc::new(AtomicUsize::new(0));
        let mut came_from = HashMap::new();
        for id in 1..NODE_COUNT {
            came_from.insert(
                CountedNode::new(id, &clones),
                CountedNode::new(id - 1, &clones),
            );
        }

        clones.store(0, Ordering::Relaxed);
        let path = reconstruct_path(&came_from, CountedNode::new(NODE_COUNT - 1, &clones), 17);

        assert_eq!(clones.load(Ordering::Relaxed), NODE_COUNT - 1);
        assert_eq!(path.cost, 17);
        assert_eq!(
            path.nodes.iter().map(|node| node.id).collect::<Vec<_>>(),
            (0..NODE_COUNT).collect::<Vec<_>>()
        );
    }

    #[test]
    fn returns_none_when_the_goal_is_unreachable() {
        assert_eq!(dijkstra('A', |node| *node == 'Z', neighbors), None);
    }

    #[test]
    fn ignores_paths_that_overflow_the_cost_type() {
        let neighbors = |node: &char| match node {
            'A' => vec![('B', u64::MAX), ('C', 5)],
            'B' => vec![('C', 1)],
            _ => Vec::new(),
        };

        assert_eq!(
            dijkstra('A', |node| *node == 'C', neighbors),
            Some(Path {
                cost: 5,
                nodes: vec!['A', 'C'],
            })
        );
    }
}
