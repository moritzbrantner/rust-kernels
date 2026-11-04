use std::collections::HashSet;

use spatial_kernels::{Aabb, Body, BroadPhaseResult, BroadPhaseStats, ColliderId, Pair};

#[derive(Clone, Copy, Debug)]
enum NodeKind {
    Leaf(Body),
    Branch { left: usize, right: usize },
}

#[derive(Clone, Copy, Debug)]
struct Node {
    bounds: Aabb,
    kind: NodeKind,
}

/// Immutable bounding-volume hierarchy built by recursively splitting AABBs at
/// the median of their centroids along the longest axis.
#[derive(Clone, Debug, Default)]
pub struct StaticBvh {
    nodes: Vec<Node>,
    root: Option<usize>,
    len: usize,
}

impl StaticBvh {
    #[must_use]
    pub fn build(bodies: &[Body]) -> Self {
        validate_unique_ids(bodies);
        let mut ordered = bodies.to_vec();
        let mut nodes = Vec::with_capacity(bodies.len().saturating_mul(2).saturating_sub(1));
        let root = if ordered.is_empty() {
            None
        } else {
            Some(build_node(&mut nodes, &mut ordered))
        };

        Self {
            nodes,
            root,
            len: bodies.len(),
        }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns collider IDs whose AABBs overlap `query`, sorted by ID.
    #[must_use]
    pub fn query_aabb(&self, query: Aabb) -> Vec<ColliderId> {
        let Some(root) = self.root else {
            return Vec::new();
        };

        let mut hits = Vec::new();
        let mut stack = vec![root];
        while let Some(index) = stack.pop() {
            let node = self.nodes[index];
            if !node.bounds.overlaps(query) {
                continue;
            }

            match node.kind {
                NodeKind::Leaf(body) => hits.push(body.id),
                NodeKind::Branch { left, right } => {
                    stack.push(right);
                    stack.push(left);
                }
            }
        }

        hits.sort_unstable();
        hits
    }

    /// Returns every overlapping collider pair exactly once in canonical order.
    #[must_use]
    pub fn overlapping_pairs(&self) -> Vec<Pair> {
        self.overlapping_pairs_result().pairs
    }

    /// Returns exact overlap pairs plus deterministic broad-phase work counters
    /// for the current retained tree state.
    #[must_use]
    pub fn overlapping_pairs_result(&self) -> BroadPhaseResult {
        let (pairs, aabb_tests) = self.overlapping_pairs_with_tests();
        BroadPhaseResult {
            pairs,
            stats: BroadPhaseStats {
                aabb_tests,
                occupied_cells: None,
            },
        }
    }

    pub(crate) fn overlapping_pairs_with_tests(&self) -> (Vec<Pair>, u64) {
        let Some(root) = self.root else {
            return (Vec::new(), 0);
        };

        let mut pairs = Vec::new();
        let mut aabb_tests = 0_u64;
        self.collect_within(root, &mut pairs, &mut aabb_tests);
        pairs.sort_unstable();
        (pairs, aabb_tests)
    }

    fn collect_within(&self, node_index: usize, pairs: &mut Vec<Pair>, aabb_tests: &mut u64) {
        if let NodeKind::Branch { left, right } = self.nodes[node_index].kind {
            self.collect_within(left, pairs, aabb_tests);
            self.collect_within(right, pairs, aabb_tests);
            self.collect_cross(left, right, pairs, aabb_tests);
        }
    }

    fn collect_cross(
        &self,
        left_index: usize,
        right_index: usize,
        pairs: &mut Vec<Pair>,
        aabb_tests: &mut u64,
    ) {
        let left_node = self.nodes[left_index];
        let right_node = self.nodes[right_index];

        match (left_node.kind, right_node.kind) {
            (NodeKind::Leaf(left), NodeKind::Leaf(right)) => {
                *aabb_tests += 1;
                if left_node.bounds.overlaps(right_node.bounds) {
                    pairs.push(Pair::new(left.id, right.id));
                }
            }
            (left_kind, right_kind) => {
                if !left_node.bounds.overlaps(right_node.bounds) {
                    return;
                }

                match (left_kind, right_kind) {
                    (NodeKind::Branch { left, right }, NodeKind::Leaf(_)) => {
                        self.collect_cross(left, right_index, pairs, aabb_tests);
                        self.collect_cross(right, right_index, pairs, aabb_tests);
                    }
                    (NodeKind::Leaf(_), NodeKind::Branch { left, right }) => {
                        self.collect_cross(left_index, left, pairs, aabb_tests);
                        self.collect_cross(left_index, right, pairs, aabb_tests);
                    }
                    (
                        NodeKind::Branch {
                            left: left_left,
                            right: left_right,
                        },
                        NodeKind::Branch {
                            left: right_left,
                            right: right_right,
                        },
                    ) => {
                        self.collect_cross(left_left, right_left, pairs, aabb_tests);
                        self.collect_cross(left_left, right_right, pairs, aabb_tests);
                        self.collect_cross(left_right, right_left, pairs, aabb_tests);
                        self.collect_cross(left_right, right_right, pairs, aabb_tests);
                    }
                    (NodeKind::Leaf(_), NodeKind::Leaf(_)) => unreachable!(),
                }
            }
        }
    }
}

fn build_node(nodes: &mut Vec<Node>, bodies: &mut [Body]) -> usize {
    let bounds = enclosing_bounds(bodies);
    if bodies.len() == 1 {
        let index = nodes.len();
        nodes.push(Node {
            bounds,
            kind: NodeKind::Leaf(bodies[0]),
        });
        return index;
    }

    let axis = longest_axis(bounds);
    bodies.sort_unstable_by(|left, right| {
        centroid(left.aabb, axis)
            .total_cmp(&centroid(right.aabb, axis))
            .then_with(|| left.id.cmp(&right.id))
    });

    let middle = bodies.len() / 2;
    let (left_bodies, right_bodies) = bodies.split_at_mut(middle);
    let left = build_node(nodes, left_bodies);
    let right = build_node(nodes, right_bodies);
    let index = nodes.len();
    nodes.push(Node {
        bounds,
        kind: NodeKind::Branch { left, right },
    });
    index
}

fn enclosing_bounds(bodies: &[Body]) -> Aabb {
    let mut bounds = bodies[0].aabb;
    for body in &bodies[1..] {
        for axis in 0..3 {
            bounds.min[axis] = bounds.min[axis].min(body.aabb.min[axis]);
            bounds.max[axis] = bounds.max[axis].max(body.aabb.max[axis]);
        }
    }
    bounds
}

fn longest_axis(bounds: Aabb) -> usize {
    let extents = [
        f64::from(bounds.max[0]) - f64::from(bounds.min[0]),
        f64::from(bounds.max[1]) - f64::from(bounds.min[1]),
        f64::from(bounds.max[2]) - f64::from(bounds.min[2]),
    ];
    let mut axis = 0;
    if extents[1] > extents[axis] {
        axis = 1;
    }
    if extents[2] > extents[axis] {
        axis = 2;
    }
    axis
}

fn centroid(bounds: Aabb, axis: usize) -> f64 {
    (f64::from(bounds.min[axis]) + f64::from(bounds.max[axis])) * 0.5
}

pub(crate) fn validate_unique_ids(bodies: &[Body]) {
    let mut ids = HashSet::with_capacity(bodies.len());
    assert!(
        bodies.iter().all(|body| ids.insert(body.id)),
        "collider IDs must be unique"
    );
}
