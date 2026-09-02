use std::collections::{HashSet, VecDeque};
use std::hash::Hash;

/// Visits every reachable node in breadth-first order.
///
/// Neighbor iteration order determines deterministic tie-breaking among nodes at
/// the same depth. The graph remains caller-owned through the callback.
pub fn breadth_first<N, I, Neighbors>(start: N, mut neighbors: Neighbors) -> Vec<N>
where
    N: Clone + Eq + Hash,
    I: IntoIterator<Item = N>,
    Neighbors: FnMut(&N) -> I,
{
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    let mut frontier = VecDeque::new();

    seen.insert(start.clone());
    frontier.push_back(start);

    while let Some(node) = frontier.pop_front() {
        for next in neighbors(&node) {
            if seen.insert(next.clone()) {
                frontier.push_back(next);
            }
        }
        order.push(node);
    }

    order
}

/// Visits every reachable node in depth-first preorder.
///
/// Neighbor iteration order is preserved by pushing neighbors onto the explicit
/// stack in reverse order.
pub fn depth_first<N, I, Neighbors>(start: N, mut neighbors: Neighbors) -> Vec<N>
where
    N: Clone + Eq + Hash,
    I: IntoIterator<Item = N>,
    Neighbors: FnMut(&N) -> I,
{
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![start];

    while let Some(node) = stack.pop() {
        if !seen.insert(node.clone()) {
            continue;
        }

        let mut outgoing: Vec<_> = neighbors(&node).into_iter().collect();
        outgoing.reverse();
        stack.extend(outgoing);
        order.push(node);
    }

    order
}

#[cfg(test)]
mod tests {
    use super::{breadth_first, depth_first};

    fn neighbors(node: &char) -> Vec<char> {
        match node {
            'A' => vec!['B', 'C'],
            'B' => vec!['D', 'E'],
            'C' => vec!['F'],
            'E' => vec!['F'],
            _ => Vec::new(),
        }
    }

    fn graph_from_mask(mut mask: usize) -> [Vec<u8>; 4] {
        let mut graph = std::array::from_fn(|_| Vec::new());
        for (from, outgoing) in graph.iter_mut().enumerate() {
            for to in 0_usize..4 {
                if from == to {
                    continue;
                }
                if mask & 1 != 0 {
                    outgoing.push(to as u8);
                }
                mask >>= 1;
            }
        }
        graph
    }

    fn breadth_first_oracle(graph: &[Vec<u8>; 4], start: u8) -> Vec<u8> {
        let mut seen = [false; 4];
        seen[usize::from(start)] = true;
        let mut frontier = vec![start];
        let mut order = Vec::new();

        while !frontier.is_empty() {
            let mut next_frontier = Vec::new();
            for node in frontier {
                order.push(node);
                for &next in &graph[usize::from(node)] {
                    if !seen[usize::from(next)] {
                        seen[usize::from(next)] = true;
                        next_frontier.push(next);
                    }
                }
            }
            frontier = next_frontier;
        }

        order
    }

    fn depth_first_oracle(graph: &[Vec<u8>; 4], start: u8) -> Vec<u8> {
        fn visit(graph: &[Vec<u8>; 4], node: u8, seen: &mut [bool; 4], order: &mut Vec<u8>) {
            if seen[usize::from(node)] {
                return;
            }
            seen[usize::from(node)] = true;
            order.push(node);
            for &next in &graph[usize::from(node)] {
                visit(graph, next, seen, order);
            }
        }

        let mut seen = [false; 4];
        let mut order = Vec::new();
        visit(graph, start, &mut seen, &mut order);
        order
    }

    #[test]
    fn breadth_first_preserves_layer_and_neighbor_order() {
        assert_eq!(
            breadth_first('A', neighbors),
            vec!['A', 'B', 'C', 'D', 'E', 'F']
        );
    }

    #[test]
    fn depth_first_preserves_preorder_neighbor_order() {
        assert_eq!(
            depth_first('A', neighbors),
            vec!['A', 'B', 'D', 'E', 'F', 'C']
        );
    }

    #[test]
    fn traversal_terminates_on_cycles_and_visits_each_node_once() {
        let cyclic = |node: &u8| match node {
            0 => vec![1],
            1 => vec![2],
            2 => vec![0],
            _ => Vec::new(),
        };

        assert_eq!(breadth_first(0, cyclic), vec![0, 1, 2]);
        assert_eq!(depth_first(0, cyclic), vec![0, 1, 2]);
    }

    #[test]
    fn exhaustive_four_node_graphs_match_independent_oracles() {
        for mask in 0_usize..(1_usize << 12) {
            let graph = graph_from_mask(mask);

            for start in 0_u8..4 {
                let bfs = breadth_first(start, |node| graph[usize::from(*node)].clone());
                let dfs = depth_first(start, |node| graph[usize::from(*node)].clone());
                assert_eq!(bfs, breadth_first_oracle(&graph, start), "mask={mask}");
                assert_eq!(dfs, depth_first_oracle(&graph, start), "mask={mask}");
            }
        }
    }
}
