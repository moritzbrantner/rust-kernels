use std::collections::HashSet;

use spatial_kernels::{Aabb, Body, BroadPhaseResult, BroadPhaseStats, ColliderId, Pair};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StaticBvhVisitKind {
    Descend,
    Pruned,
    LeafTest,
}

impl StaticBvhVisitKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Descend => "descend",
            Self::Pruned => "pruned",
            Self::LeafTest => "leaf-test",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticBvhNodeSnapshot {
    pub index: usize,
    pub bounds: Aabb,
    pub depth: usize,
    pub body: Option<ColliderId>,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub leaf_count: usize,
    pub is_root: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticBvhTraversalStep {
    pub left: usize,
    pub right: usize,
    pub kind: StaticBvhVisitKind,
    /// Number of leaf-object pairs represented by this node pair.
    pub potential_pairs: u64,
    /// Exact leaf pair when `kind == LeafTest`.
    pub pair: Option<Pair>,
    /// Whether the two tested bounds overlap. For `Descend`, this is always
    /// true; for `Pruned`, false; for `LeafTest`, it is the exact result.
    pub overlap: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StaticBvhTraversalTrace {
    pub result: BroadPhaseResult,
    pub root: Option<usize>,
    pub nodes: Vec<StaticBvhNodeSnapshot>,
    pub steps: Vec<StaticBvhTraversalStep>,
    /// Every node-pair visit, including hierarchy-bound checks and leaf tests.
    pub node_pair_visits: u64,
    /// Sum of leaf-object pairs eliminated by `Pruned` steps.
    pub pruned_potential_pairs: u64,
}

impl StaticBvhTraversalTrace {
    #[must_use]
    pub fn represented_pair_count(&self) -> u64 {
        self.pruned_potential_pairs + self.result.stats.aabb_tests
    }
}

#[derive(Clone, Copy, Debug)]
enum NodeKind {
    Leaf(Body),
    Branch { left: usize, right: usize },
}

#[derive(Clone, Copy, Debug)]
struct Node {
    bounds: Aabb,
    kind: NodeKind,
    depth: usize,
    leaf_count: usize,
}

/// Builds and traces the same deterministic median-split static-BVH strategy
/// used by `bvh-kernels`, while keeping all trace allocation in this companion
/// crate. Tests differential-check every result against `StaticBvhBroadPhase`.
#[must_use]
pub fn trace_static_bvh(bodies: &[Body]) -> StaticBvhTraversalTrace {
    validate_unique_ids(bodies);

    if bodies.is_empty() {
        return StaticBvhTraversalTrace {
            result: BroadPhaseResult::default(),
            root: None,
            nodes: Vec::new(),
            steps: Vec::new(),
            node_pair_visits: 0,
            pruned_potential_pairs: 0,
        };
    }

    let mut ordered = bodies.to_vec();
    let mut nodes = Vec::with_capacity(bodies.len().saturating_mul(2).saturating_sub(1));
    let root = build_node(&mut nodes, &mut ordered, 0);

    let mut pairs = Vec::new();
    let mut exact_tests = 0_u64;
    let mut steps = Vec::new();
    let mut node_pair_visits = 0_u64;
    let mut pruned_potential_pairs = 0_u64;
    collect_within(
        &nodes,
        root,
        &mut pairs,
        &mut exact_tests,
        &mut steps,
        &mut node_pair_visits,
        &mut pruned_potential_pairs,
    );
    pairs.sort_unstable();

    let snapshots = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let (body, left, right) = match node.kind {
                NodeKind::Leaf(body) => (Some(body.id), None, None),
                NodeKind::Branch { left, right } => (None, Some(left), Some(right)),
            };
            StaticBvhNodeSnapshot {
                index,
                bounds: node.bounds,
                depth: node.depth,
                body,
                left,
                right,
                leaf_count: node.leaf_count,
                is_root: index == root,
            }
        })
        .collect();

    StaticBvhTraversalTrace {
        result: BroadPhaseResult {
            pairs,
            stats: BroadPhaseStats {
                aabb_tests: exact_tests,
                occupied_cells: None,
            },
        },
        root: Some(root),
        nodes: snapshots,
        steps,
        node_pair_visits,
        pruned_potential_pairs,
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_within(
    nodes: &[Node],
    node_index: usize,
    pairs: &mut Vec<Pair>,
    exact_tests: &mut u64,
    steps: &mut Vec<StaticBvhTraversalStep>,
    node_pair_visits: &mut u64,
    pruned_potential_pairs: &mut u64,
) {
    if let NodeKind::Branch { left, right } = nodes[node_index].kind {
        collect_within(
            nodes,
            left,
            pairs,
            exact_tests,
            steps,
            node_pair_visits,
            pruned_potential_pairs,
        );
        collect_within(
            nodes,
            right,
            pairs,
            exact_tests,
            steps,
            node_pair_visits,
            pruned_potential_pairs,
        );
        collect_cross(
            nodes,
            left,
            right,
            pairs,
            exact_tests,
            steps,
            node_pair_visits,
            pruned_potential_pairs,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_cross(
    nodes: &[Node],
    left_index: usize,
    right_index: usize,
    pairs: &mut Vec<Pair>,
    exact_tests: &mut u64,
    steps: &mut Vec<StaticBvhTraversalStep>,
    node_pair_visits: &mut u64,
    pruned_potential_pairs: &mut u64,
) {
    let left_node = nodes[left_index];
    let right_node = nodes[right_index];
    let potential_pairs = (left_node.leaf_count as u64) * (right_node.leaf_count as u64);
    *node_pair_visits += 1;

    match (left_node.kind, right_node.kind) {
        (NodeKind::Leaf(left), NodeKind::Leaf(right)) => {
            *exact_tests += 1;
            let overlap = left_node.bounds.overlaps(right_node.bounds);
            let pair = Pair::new(left.id, right.id);
            steps.push(StaticBvhTraversalStep {
                left: left_index,
                right: right_index,
                kind: StaticBvhVisitKind::LeafTest,
                potential_pairs,
                pair: Some(pair),
                overlap,
            });
            if overlap {
                pairs.push(pair);
            }
        }
        (left_kind, right_kind) => {
            if !left_node.bounds.overlaps(right_node.bounds) {
                *pruned_potential_pairs += potential_pairs;
                steps.push(StaticBvhTraversalStep {
                    left: left_index,
                    right: right_index,
                    kind: StaticBvhVisitKind::Pruned,
                    potential_pairs,
                    pair: None,
                    overlap: false,
                });
                return;
            }

            steps.push(StaticBvhTraversalStep {
                left: left_index,
                right: right_index,
                kind: StaticBvhVisitKind::Descend,
                potential_pairs,
                pair: None,
                overlap: true,
            });

            match (left_kind, right_kind) {
                (NodeKind::Branch { left, right }, NodeKind::Leaf(_)) => {
                    collect_cross(
                        nodes,
                        left,
                        right_index,
                        pairs,
                        exact_tests,
                        steps,
                        node_pair_visits,
                        pruned_potential_pairs,
                    );
                    collect_cross(
                        nodes,
                        right,
                        right_index,
                        pairs,
                        exact_tests,
                        steps,
                        node_pair_visits,
                        pruned_potential_pairs,
                    );
                }
                (NodeKind::Leaf(_), NodeKind::Branch { left, right }) => {
                    collect_cross(
                        nodes,
                        left_index,
                        left,
                        pairs,
                        exact_tests,
                        steps,
                        node_pair_visits,
                        pruned_potential_pairs,
                    );
                    collect_cross(
                        nodes,
                        left_index,
                        right,
                        pairs,
                        exact_tests,
                        steps,
                        node_pair_visits,
                        pruned_potential_pairs,
                    );
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
                    for (left, right) in [
                        (left_left, right_left),
                        (left_left, right_right),
                        (left_right, right_left),
                        (left_right, right_right),
                    ] {
                        collect_cross(
                            nodes,
                            left,
                            right,
                            pairs,
                            exact_tests,
                            steps,
                            node_pair_visits,
                            pruned_potential_pairs,
                        );
                    }
                }
                (NodeKind::Leaf(_), NodeKind::Leaf(_)) => unreachable!(),
            }
        }
    }
}

fn build_node(nodes: &mut Vec<Node>, bodies: &mut [Body], depth: usize) -> usize {
    let bounds = enclosing_bounds(bodies);
    if bodies.len() == 1 {
        let index = nodes.len();
        nodes.push(Node {
            bounds,
            kind: NodeKind::Leaf(bodies[0]),
            depth,
            leaf_count: 1,
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
    let left = build_node(nodes, left_bodies, depth + 1);
    let right = build_node(nodes, right_bodies, depth + 1);
    let index = nodes.len();
    nodes.push(Node {
        bounds,
        kind: NodeKind::Branch { left, right },
        depth,
        leaf_count: nodes[left].leaf_count + nodes[right].leaf_count,
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

fn validate_unique_ids(bodies: &[Body]) {
    let mut ids = HashSet::with_capacity(bodies.len());
    assert!(
        bodies.iter().all(|body| ids.insert(body.id)),
        "collider IDs must be unique"
    );
}

#[cfg(test)]
mod tests {
    use super::{StaticBvhVisitKind, trace_static_bvh};
    use bvh_kernels::StaticBvhBroadPhase;
    use spatial_kernels::{Aabb, Body, BroadPhase, ColliderId, NaiveBroadPhase};

    fn body(id: ColliderId, center: [f32; 3], half: f32) -> Body {
        Body::new(id, Aabb::from_center_half_extents(center, [half; 3]))
    }

    fn fixture() -> Vec<Body> {
        vec![
            body(10, [0.0, 0.0, 0.0], 0.6),
            body(20, [0.9, 0.0, 0.0], 0.6),
            body(30, [4.0, 4.0, 4.0], 0.5),
            body(40, [-0.8, 0.0, 0.0], 0.6),
            body(50, [1.5, 0.0, 0.0], 1.2),
        ]
    }

    #[test]
    fn traced_result_matches_production_static_bvh() {
        let bodies = fixture();
        let trace = trace_static_bvh(&bodies);
        assert_eq!(trace.result, StaticBvhBroadPhase.detect(&bodies));
    }

    #[test]
    fn trace_accounts_for_every_possible_leaf_pair_once() {
        let bodies = fixture();
        let trace = trace_static_bvh(&bodies);
        let possible = (bodies.len() * (bodies.len() - 1) / 2) as u64;
        assert_eq!(trace.represented_pair_count(), possible);
        assert_eq!(
            trace
                .steps
                .iter()
                .filter(|step| step.kind == StaticBvhVisitKind::LeafTest)
                .count() as u64,
            trace.result.stats.aabb_tests
        );
    }

    #[test]
    fn node_snapshots_have_one_root_and_one_leaf_per_body() {
        let bodies = fixture();
        let trace = trace_static_bvh(&bodies);
        assert_eq!(trace.nodes.iter().filter(|node| node.is_root).count(), 1);
        assert_eq!(
            trace
                .nodes
                .iter()
                .filter(|node| node.body.is_some())
                .count(),
            bodies.len()
        );
        assert_eq!(
            trace.nodes[trace.root.expect("non-empty tree must have root")].leaf_count,
            bodies.len()
        );
    }

    #[test]
    fn trace_is_independent_of_input_order() {
        let bodies = fixture();
        let expected = trace_static_bvh(&bodies);
        let mut reversed = bodies.clone();
        reversed.reverse();
        assert_eq!(trace_static_bvh(&reversed), expected);
    }

    #[test]
    fn sparse_scene_prunes_most_possible_pairs() {
        let bodies: Vec<_> = (0..128)
            .map(|id| body(id, [id as f32 * 10.0, 0.0, 0.0], 0.25))
            .collect();
        let trace = trace_static_bvh(&bodies);
        let naive = NaiveBroadPhase.detect(&bodies);
        assert_eq!(trace.result.pairs, naive.pairs);
        assert!(trace.pruned_potential_pairs > naive.stats.aabb_tests * 9 / 10);
        assert!(trace.result.stats.aabb_tests < naive.stats.aabb_tests / 10);
    }

    #[test]
    fn everything_overlapping_reaches_every_leaf_pair() {
        let bodies: Vec<_> = (0..20).map(|id| body(id, [0.0; 3], 2.0)).collect();
        let trace = trace_static_bvh(&bodies);
        let naive = NaiveBroadPhase.detect(&bodies);
        assert_eq!(trace.result, naive);
        assert_eq!(trace.pruned_potential_pairs, 0);
    }

    #[test]
    fn empty_trace_is_empty() {
        let trace = trace_static_bvh(&[]);
        assert_eq!(trace.root, None);
        assert!(trace.nodes.is_empty());
        assert!(trace.steps.is_empty());
        assert_eq!(trace.result, StaticBvhBroadPhase.detect(&[]));
    }
}
