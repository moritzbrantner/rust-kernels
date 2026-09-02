use std::collections::{HashMap, hash_map::Entry};
use std::hash::Hash;

use collection_kernels::UnionFind;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightedEdge<N> {
    pub a: N,
    pub b: N,
    pub weight: u64,
}

impl<N> WeightedEdge<N> {
    #[must_use]
    pub fn new(a: N, b: N, weight: u64) -> Self {
        Self { a, b, weight }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpanningForest<N> {
    pub edges: Vec<WeightedEdge<N>>,
    pub total_weight: u128,
    pub component_count: usize,
}

/// Computes a deterministic minimum spanning forest with Kruskal's algorithm.
///
/// Vertex iteration establishes stable vertex indices. Endpoints not listed in
/// `vertices` are discovered from edges in edge iteration order. Equal-weight
/// edges are tie-broken by canonical endpoint indices, then original edge order.
pub fn kruskal_minimum_spanning_forest<N, Vertices, Edges>(
    vertices: Vertices,
    edges: Edges,
) -> SpanningForest<N>
where
    N: Clone + Eq + Hash,
    Vertices: IntoIterator<Item = N>,
    Edges: IntoIterator<Item = WeightedEdge<N>>,
{
    let edges: Vec<_> = edges.into_iter().collect();
    let mut nodes = Vec::new();
    let mut indices = HashMap::new();

    for vertex in vertices {
        add_vertex(vertex, &mut nodes, &mut indices);
    }
    for edge in &edges {
        add_vertex(edge.a.clone(), &mut nodes, &mut indices);
        add_vertex(edge.b.clone(), &mut nodes, &mut indices);
    }

    let mut indexed: Vec<_> = edges
        .into_iter()
        .enumerate()
        .map(|(order, edge)| {
            let left = indices[&edge.a];
            let right = indices[&edge.b];
            IndexedEdge {
                left,
                right,
                order,
                edge,
            }
        })
        .collect();

    indexed.sort_by_key(|entry| {
        (
            entry.edge.weight,
            entry.left.min(entry.right),
            entry.left.max(entry.right),
            entry.order,
        )
    });

    let mut union_find = UnionFind::new(nodes.len());
    let mut selected = Vec::new();
    let mut total_weight = 0_u128;

    for entry in indexed {
        if !union_find.union(entry.left, entry.right) {
            continue;
        }

        total_weight += u128::from(entry.edge.weight);
        if entry.left <= entry.right {
            selected.push(entry.edge);
        } else {
            selected.push(WeightedEdge::new(
                entry.edge.b,
                entry.edge.a,
                entry.edge.weight,
            ));
        }
    }

    SpanningForest {
        edges: selected,
        total_weight,
        component_count: union_find.component_count(),
    }
}

struct IndexedEdge<N> {
    left: usize,
    right: usize,
    order: usize,
    edge: WeightedEdge<N>,
}

fn add_vertex<N>(vertex: N, nodes: &mut Vec<N>, indices: &mut HashMap<N, usize>)
where
    N: Clone + Eq + Hash,
{
    if let Entry::Vacant(entry) = indices.entry(vertex.clone()) {
        let index = nodes.len();
        entry.insert(index);
        nodes.push(vertex);
    }
}

#[cfg(test)]
mod tests {
    use super::{SpanningForest, WeightedEdge, kruskal_minimum_spanning_forest};

    fn edge(a: char, b: char, weight: u64) -> WeightedEdge<char> {
        WeightedEdge::new(a, b, weight)
    }

    fn prim_forest_oracle(vertex_count: usize, edges: &[WeightedEdge<u8>]) -> (u128, usize) {
        let mut visited = vec![false; vertex_count];
        let mut total_weight = 0_u128;
        let mut component_count = 0_usize;

        for start in 0..vertex_count {
            if visited[start] {
                continue;
            }

            component_count += 1;
            visited[start] = true;

            loop {
                let mut best: Option<(u64, usize, usize)> = None;

                for (order, candidate) in edges.iter().enumerate() {
                    let a = usize::from(candidate.a);
                    let b = usize::from(candidate.b);
                    if visited[a] == visited[b] {
                        continue;
                    }

                    let next = if visited[a] { b } else { a };
                    let choice = (candidate.weight, order, next);
                    if best.is_none_or(|current| choice < current) {
                        best = Some(choice);
                    }
                }

                let Some((weight, _, next)) = best else {
                    break;
                };
                visited[next] = true;
                total_weight += u128::from(weight);
            }
        }

        (total_weight, component_count)
    }

    #[test]
    fn finds_the_minimum_spanning_tree_for_a_connected_graph() {
        let forest = kruskal_minimum_spanning_forest(
            ['A', 'B', 'C', 'D'],
            [
                edge('A', 'C', 4),
                edge('B', 'D', 5),
                edge('A', 'B', 1),
                edge('C', 'D', 3),
                edge('B', 'C', 2),
            ],
        );

        assert_eq!(
            forest,
            SpanningForest {
                edges: vec![edge('A', 'B', 1), edge('B', 'C', 2), edge('C', 'D', 3)],
                total_weight: 6,
                component_count: 1,
            }
        );
    }

    #[test]
    fn disconnected_inputs_return_a_forest_and_preserve_isolated_vertices() {
        let forest = kruskal_minimum_spanning_forest(
            ['A', 'B', 'C', 'D', 'E'],
            [edge('A', 'B', 1), edge('C', 'D', 2)],
        );

        assert_eq!(forest.edges, vec![edge('A', 'B', 1), edge('C', 'D', 2)]);
        assert_eq!(forest.total_weight, 3);
        assert_eq!(forest.component_count, 3);
    }

    #[test]
    fn equal_weight_ties_are_independent_of_edge_iteration_order() {
        let first = [
            WeightedEdge::new(2_u8, 3, 1),
            WeightedEdge::new(0, 2, 1),
            WeightedEdge::new(1, 3, 1),
            WeightedEdge::new(0, 1, 1),
        ];
        let second = [
            WeightedEdge::new(0_u8, 1, 1),
            WeightedEdge::new(1, 3, 1),
            WeightedEdge::new(0, 2, 1),
            WeightedEdge::new(2, 3, 1),
        ];

        let expected = kruskal_minimum_spanning_forest([0, 1, 2, 3], first);
        let actual = kruskal_minimum_spanning_forest([0, 1, 2, 3], second);
        assert_eq!(actual, expected);
        assert_eq!(actual.edges.len(), 3);
        assert_eq!(actual.component_count, 1);
    }

    #[test]
    fn self_loops_and_cycle_edges_are_rejected_by_union_find() {
        let forest = kruskal_minimum_spanning_forest(
            [0_u8, 1, 2],
            [
                WeightedEdge::new(0, 0, 0),
                WeightedEdge::new(0, 1, 1),
                WeightedEdge::new(1, 2, 1),
                WeightedEdge::new(0, 2, 10),
            ],
        );

        assert_eq!(forest.edges.len(), 2);
        assert_eq!(forest.total_weight, 2);
        assert_eq!(forest.component_count, 1);
    }

    #[test]
    fn kruskal_matches_independent_prim_oracle_for_all_small_graphs() {
        const PAIRS: [(u8, u8); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

        for case in 0..3_usize.pow(PAIRS.len() as u32) {
            let mut encoded = case;
            let mut edges = Vec::new();

            for &(a, b) in &PAIRS {
                let state = encoded % 3;
                encoded /= 3;
                if state != 0 {
                    edges.push(WeightedEdge::new(a, b, (state - 1) as u64));
                }
            }

            let expected = prim_forest_oracle(4, &edges);
            let actual = kruskal_minimum_spanning_forest([0_u8, 1, 2, 3], edges);

            assert_eq!(
                (actual.total_weight, actual.component_count),
                expected,
                "case={case}"
            );
            assert_eq!(
                actual.edges.len(),
                4 - actual.component_count,
                "case={case}"
            );
        }
    }
}
