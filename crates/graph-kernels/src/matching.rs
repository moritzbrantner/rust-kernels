use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchingError {
    InvalidLeft { vertex: usize, left_count: usize },
    InvalidRight { vertex: usize, right_count: usize },
}
impl fmt::Display for MatchingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid bipartite graph: {self:?}")
    }
}
impl std::error::Error for MatchingError {}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MatchingWork {
    pub phases: usize,
    /// BFS, augmenting-search and minimum-cover adjacency examinations.
    pub edge_scans: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BipartiteMatching {
    pub size: usize,
    pub left_to_right: Vec<Option<usize>>,
    pub right_to_left: Vec<Option<usize>>,
    /// Minimum vertex cover; each side is sorted by caller index.
    pub cover_left: Vec<usize>,
    pub cover_right: Vec<usize>,
    pub work: MatchingWork,
}

struct Search {
    adjacency: Vec<Vec<usize>>,
    left: Vec<Option<usize>>,
    right: Vec<Option<usize>>,
    distance: Vec<usize>,
    next: Vec<usize>,
    work: MatchingWork,
}

impl Search {
    fn augment(
        &mut self,
        root: usize,
        shortest: usize,
        path: &mut Vec<usize>,
        via: &mut Vec<usize>,
    ) -> bool {
        path.clear();
        via.clear();
        path.push(root);
        while let Some(&node) = path.last() {
            let mut descended = false;
            while let Some(&target) = self.adjacency[node].get(self.next[node]) {
                self.next[node] += 1;
                self.work.edge_scans += 1;
                if let Some(child) = self.right[target] {
                    if self.distance[child] == self.distance[node] + 1
                        && self.distance[child] < shortest
                    {
                        via.push(target);
                        path.push(child);
                        descended = true;
                        break;
                    }
                } else if self.distance[node] + 1 == shortest {
                    self.left[node] = Some(target);
                    self.right[target] = Some(node);
                    for (&parent, &right) in path.iter().zip(via.iter()).rev() {
                        self.left[parent] = Some(right);
                        self.right[right] = Some(parent);
                    }
                    return true;
                }
            }
            if !descended {
                self.distance[node] = usize::MAX;
                path.pop();
                via.pop();
            }
        }
        false
    }
}

/// Maximum-cardinality bipartite matching with Hopcroft--Karp, and a minimum
/// vertex cover certificate. Both partitions use independent zero-based indices.
///
/// Duplicate edges are harmless and are not silently sorted or normalized.
/// Ascending left roots and caller edge order break ties deterministically, but
/// the result is not promised to be the lexicographically smallest matching.
/// Iterative augmentation avoids native stack growth on long alternating paths.
/// O(V + E sqrt(V)) time, O(V+E) memory. Inputs are never mutated.
///
/// ```
/// use graph_kernels::hopcroft_karp;
/// let result = hopcroft_karp(2, 2, &[(0,0), (0,1), (1,0)]).unwrap();
/// assert_eq!(result.size, 2);
/// assert_eq!(result.left_to_right, [Some(1), Some(0)]);
/// assert_eq!(result.cover_left.len() + result.cover_right.len(), 2);
/// ```
pub fn hopcroft_karp(
    left_count: usize,
    right_count: usize,
    edges: &[(usize, usize)],
) -> Result<BipartiteMatching, MatchingError> {
    for &(left, right) in edges {
        if left >= left_count {
            return Err(MatchingError::InvalidLeft {
                vertex: left,
                left_count,
            });
        }
        if right >= right_count {
            return Err(MatchingError::InvalidRight {
                vertex: right,
                right_count,
            });
        }
    }
    let mut search = Search {
        adjacency: vec![Vec::new(); left_count],
        left: vec![None; left_count],
        right: vec![None; right_count],
        distance: vec![usize::MAX; left_count],
        next: vec![0; left_count],
        work: MatchingWork::default(),
    };
    for &(left, right) in edges {
        search.adjacency[left].push(right);
    }
    // Exclude isolated left vertices from phase resets/root scans. Otherwise a
    // mostly isolated partition can cost O(V) per phase instead of O(V) total.
    let active: Vec<usize> = search
        .adjacency
        .iter()
        .enumerate()
        .filter(|(_, neighbors)| !neighbors.is_empty())
        .map(|(i, _)| i)
        .collect();
    let mut queue = Vec::with_capacity(left_count);
    let mut path = Vec::new();
    let mut via = Vec::new();
    let mut size = 0;
    loop {
        queue.clear();
        for &left in &active {
            search.distance[left] = usize::MAX;
            if search.left[left].is_none() {
                search.distance[left] = 0;
                queue.push(left);
            }
        }
        let mut shortest = usize::MAX;
        let mut cursor = 0;
        while cursor < queue.len() {
            let left = queue[cursor];
            cursor += 1;
            if search.distance[left] >= shortest {
                continue;
            }
            for &right in &search.adjacency[left] {
                search.work.edge_scans += 1;
                if let Some(next) = search.right[right] {
                    if search.distance[next] == usize::MAX {
                        search.distance[next] = search.distance[left] + 1;
                        queue.push(next);
                    }
                } else {
                    shortest = search.distance[left] + 1;
                }
            }
        }
        if shortest == usize::MAX {
            break;
        }
        search.work.phases += 1;
        for &left in &active {
            search.next[left] = 0;
        }
        for &left in &active {
            if search.left[left].is_none() && search.augment(left, shortest, &mut path, &mut via) {
                size += 1;
            }
        }
    }
    // Alternating reachability from unmatched left vertices gives (L\Z) ∪ (R∩Z).
    let mut seen_left = vec![false; left_count];
    let mut seen_right = vec![false; right_count];
    queue.clear();
    for (left, mate) in search.left.iter().enumerate() {
        if mate.is_none() {
            seen_left[left] = true;
            queue.push(left);
        }
    }
    let mut cursor = 0;
    while cursor < queue.len() {
        let left = queue[cursor];
        cursor += 1;
        for &right in &search.adjacency[left] {
            search.work.edge_scans += 1;
            if search.left[left] == Some(right) || seen_right[right] {
                continue;
            }
            seen_right[right] = true;
            if let Some(next) = search.right[right] {
                if !seen_left[next] {
                    seen_left[next] = true;
                    queue.push(next);
                }
            }
        }
    }
    Ok(BipartiteMatching {
        size,
        left_to_right: search.left,
        right_to_left: search.right,
        cover_left: (0..left_count).filter(|&i| !seen_left[i]).collect(),
        cover_right: (0..right_count).filter(|&i| seen_right[i]).collect(),
        work: search.work,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn brute(left: usize, n: usize, edges: &[(usize, usize)], used: u64) -> usize {
        if left == n {
            return 0;
        }
        let mut best = brute(left + 1, n, edges, used);
        for &(_, right) in edges.iter().filter(|&&(a, _)| a == left) {
            if used & (1 << right) == 0 {
                best = best.max(1 + brute(left + 1, n, edges, used | (1 << right)));
            }
        }
        best
    }
    fn certify(edges: &[(usize, usize)], result: &BipartiteMatching) {
        let mut count = 0;
        for (left, right) in result.left_to_right.iter().enumerate() {
            if let Some(right) = right {
                assert!(edges.contains(&(left, *right)));
                assert_eq!(result.right_to_left[*right], Some(left));
                count += 1;
            }
        }
        for (right, left) in result.right_to_left.iter().enumerate() {
            if let Some(left) = left {
                assert_eq!(result.left_to_right[*left], Some(right));
            }
        }
        assert_eq!(count, result.size);
        assert_eq!(result.cover_left.len() + result.cover_right.len(), count);
        for &(left, right) in edges {
            assert!(result.cover_left.contains(&left) || result.cover_right.contains(&right));
        }
    }
    #[test]
    fn exhaustive_four_by_four_graphs_match_brute_force_and_cover_certificate() {
        for mask in 0_u32..1 << 16 {
            let edges: Vec<_> = (0..16)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| (i / 4, i % 4))
                .collect();
            let result = hopcroft_karp(4, 4, &edges).unwrap();
            assert_eq!(result.size, brute(0, 4, &edges, 0));
            certify(&edges, &result);
        }
    }
    #[test]
    fn rectangular_duplicates_isolated_vertices_and_deterministic_ties() {
        for (a, b, edges) in [
            (3, 5, vec![(0, 2), (0, 2), (1, 2), (1, 3)]),
            (5, 2, vec![(0, 0), (1, 0), (1, 1), (4, 1)]),
            (0, 4, vec![]),
            (4, 0, vec![]),
        ] {
            let result = hopcroft_karp(a, b, &edges).unwrap();
            assert_eq!(result.size, brute(0, a, &edges, 0));
            certify(&edges, &result);
            assert_eq!(result, hopcroft_karp(a, b, &edges).unwrap());
        }
        assert_eq!(
            hopcroft_karp(2, 2, &[(0, 0), (0, 1), (1, 0), (1, 1)])
                .unwrap()
                .left_to_right,
            vec![Some(0), Some(1)]
        );
        assert!(matches!(
            hopcroft_karp(0, 1, &[(0, 0)]),
            Err(MatchingError::InvalidLeft { .. })
        ));
        assert!(matches!(
            hopcroft_karp(1, 0, &[(0, 0)]),
            Err(MatchingError::InvalidRight { .. })
        ));
    }
    #[test]
    fn long_alternating_path_uses_two_phases_and_linear_work() {
        for n in [32, 256, 2048] {
            let mut edges: Vec<_> = (0..n - 1).flat_map(|i| [(i, i + 1), (i, i)]).collect();
            edges.push((n - 1, n - 1));
            let result = hopcroft_karp(n, n, &edges).unwrap();
            assert_eq!(result.size, n);
            assert_eq!(result.work.phases, 2);
            assert!(result.work.edge_scans <= 12 * edges.len());
        }
    }
}
