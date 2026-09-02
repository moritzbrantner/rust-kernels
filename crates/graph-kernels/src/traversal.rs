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
}
