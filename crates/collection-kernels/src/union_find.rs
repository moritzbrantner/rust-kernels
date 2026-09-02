/// Disjoint-set union with path compression and union by size.
///
/// Equal-size components are joined under the smaller root index so the
/// representative is deterministic for the same sequence of unions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    components: usize,
}

impl UnionFind {
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            size: vec![1; len],
            components: len,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    #[must_use]
    pub fn component_count(&self) -> usize {
        self.components
    }

    #[must_use]
    pub fn find(&mut self, index: usize) -> usize {
        self.assert_index(index);

        let mut root = index;
        while self.parent[root] != root {
            root = self.parent[root];
        }

        let mut current = index;
        while self.parent[current] != current {
            let next = self.parent[current];
            self.parent[current] = root;
            current = next;
        }

        root
    }

    #[must_use]
    pub fn union(&mut self, left: usize, right: usize) -> bool {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);

        if left_root == right_root {
            return false;
        }

        let left_size = self.size[left_root];
        let right_size = self.size[right_root];
        if left_size < right_size || (left_size == right_size && left_root > right_root) {
            std::mem::swap(&mut left_root, &mut right_root);
        }

        self.parent[right_root] = left_root;
        self.size[left_root] += self.size[right_root];
        self.components -= 1;
        true
    }

    #[must_use]
    pub fn connected(&mut self, left: usize, right: usize) -> bool {
        self.find(left) == self.find(right)
    }

    #[must_use]
    pub fn component_size(&mut self, index: usize) -> usize {
        let root = self.find(index);
        self.size[root]
    }

    fn assert_index(&self, index: usize) {
        assert!(
            index < self.len(),
            "union-find index {index} out of bounds for length {}",
            self.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::UnionFind;

    #[test]
    fn starts_with_one_component_per_element() {
        let union_find = UnionFind::new(4);
        assert_eq!(union_find.len(), 4);
        assert_eq!(union_find.component_count(), 4);
        assert!(!union_find.is_empty());
    }

    #[test]
    fn union_tracks_connectivity_sizes_and_component_count() {
        let mut union_find = UnionFind::new(6);

        assert!(union_find.union(0, 1));
        assert!(union_find.union(1, 2));
        assert!(!union_find.union(0, 2));
        assert!(union_find.connected(0, 2));
        assert!(!union_find.connected(0, 3));
        assert_eq!(union_find.component_size(1), 3);
        assert_eq!(union_find.component_count(), 4);
    }

    #[test]
    fn equal_size_unions_choose_the_smaller_root_deterministically() {
        let mut union_find = UnionFind::new(4);

        assert!(union_find.union(3, 2));
        assert!(union_find.union(1, 0));
        assert!(union_find.union(2, 0));

        for index in 0..4 {
            assert_eq!(union_find.find(index), 0);
        }
    }

    #[test]
    fn matches_a_trivial_connectivity_oracle() {
        let edges = [(0, 1), (2, 3), (1, 2), (5, 6), (6, 7)];
        let mut union_find = UnionFind::new(8);
        for &(left, right) in &edges {
            assert!(union_find.union(left, right));
        }

        for left in 0..8 {
            for right in 0..8 {
                assert_eq!(
                    union_find.connected(left, right),
                    oracle_connected(8, &edges, left, right),
                    "connectivity mismatch for ({left}, {right})"
                );
            }
        }
    }

    #[test]
    fn exhaustive_small_graphs_match_connectivity_and_size_oracle() {
        const LEN: usize = 5;
        const EDGES: [(usize, usize); 10] = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
        ];

        for mask in 0_usize..(1_usize << EDGES.len()) {
            let mut union_find = UnionFind::new(LEN);
            let mut selected = Vec::new();

            for (edge_index, &(left, right)) in EDGES.iter().enumerate() {
                if mask & (1_usize << edge_index) == 0 {
                    continue;
                }

                let expected_change = !oracle_connected(LEN, &selected, left, right);
                assert_eq!(
                    union_find.union(left, right),
                    expected_change,
                    "mask={mask}, edge=({left}, {right})"
                );
                selected.push((left, right));
            }

            let expected_components = (0..LEN)
                .filter(|&node| {
                    (0..node).all(|earlier| !oracle_connected(LEN, &selected, node, earlier))
                })
                .count();
            assert_eq!(
                union_find.component_count(),
                expected_components,
                "mask={mask}"
            );

            for left in 0..LEN {
                let expected_size = (0..LEN)
                    .filter(|&right| oracle_connected(LEN, &selected, left, right))
                    .count();
                assert_eq!(
                    union_find.component_size(left),
                    expected_size,
                    "mask={mask}"
                );

                for right in 0..LEN {
                    assert_eq!(
                        union_find.connected(left, right),
                        oracle_connected(LEN, &selected, left, right),
                        "mask={mask}, pair=({left}, {right})"
                    );
                }
            }
        }
    }

    fn oracle_connected(len: usize, edges: &[(usize, usize)], start: usize, goal: usize) -> bool {
        let mut seen = vec![false; len];
        let mut stack = vec![start];
        seen[start] = true;

        while let Some(node) = stack.pop() {
            if node == goal {
                return true;
            }

            for &(left, right) in edges {
                let neighbor = if left == node {
                    Some(right)
                } else if right == node {
                    Some(left)
                } else {
                    None
                };

                match neighbor {
                    Some(neighbor) if !seen[neighbor] => {
                        seen[neighbor] = true;
                        stack.push(neighbor);
                    }
                    _ => {}
                }
            }
        }

        false
    }
}
