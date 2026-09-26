use std::collections::{BTreeSet, HashSet};

use spatial_kernels::{
    Aabb, Body, BroadPhase, BroadPhaseResult, BroadPhaseStats, ColliderId, Pair,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OctreeConfig {
    pub max_depth: u8,
    pub leaf_capacity: usize,
}

impl Default for OctreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 6,
            leaf_capacity: 8,
        }
    }
}

impl OctreeConfig {
    #[must_use]
    pub fn new(max_depth: u8, leaf_capacity: usize) -> Self {
        assert!(max_depth > 0, "octree max depth must be positive");
        assert!(leaf_capacity > 0, "octree leaf capacity must be positive");
        Self {
            max_depth,
            leaf_capacity,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OctreeNodeSnapshot {
    pub index: usize,
    pub bounds: Aabb,
    pub depth: u8,
    pub members: Vec<ColliderId>,
    pub children: Vec<usize>,
}

impl OctreeNodeSnapshot {
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OctreeTrace {
    pub result: BroadPhaseResult,
    pub root: Option<usize>,
    pub nodes: Vec<OctreeNodeSnapshot>,
    pub leaf_count: usize,
    pub occupied_leaf_count: usize,
}

/// Deterministic broad phase with a conservatively rounded root cube.
/// At f32 range limits where a finite cube cannot fit, its padded axes are
/// clamped to finite bounds while still enclosing every input AABB.
#[derive(Clone, Copy, Debug, Default)]
pub struct OctreeBroadPhase {
    config: OctreeConfig,
}

impl OctreeBroadPhase {
    #[must_use]
    pub fn new(max_depth: u8, leaf_capacity: usize) -> Self {
        Self {
            config: OctreeConfig::new(max_depth, leaf_capacity),
        }
    }

    #[must_use]
    pub const fn config(self) -> OctreeConfig {
        self.config
    }

    #[must_use]
    pub fn trace(&self, bodies: &[Body]) -> OctreeTrace {
        run_octree(self.config, bodies, true)
    }
}

impl BroadPhase for OctreeBroadPhase {
    fn detect(&self, bodies: &[Body]) -> BroadPhaseResult {
        run_octree(self.config, bodies, false).result
    }
}

#[derive(Debug)]
struct Node {
    bounds: Aabb,
    depth: u8,
    members: Vec<usize>,
    children: Vec<usize>,
}

fn run_octree(config: OctreeConfig, bodies: &[Body], trace: bool) -> OctreeTrace {
    validate_unique_ids(bodies);

    if bodies.is_empty() {
        return OctreeTrace {
            result: BroadPhaseResult::default(),
            root: None,
            nodes: Vec::new(),
            leaf_count: 0,
            occupied_leaf_count: 0,
        };
    }

    let mut ordered_members: Vec<usize> = (0..bodies.len()).collect();
    ordered_members.sort_unstable_by_key(|&index| bodies[index].id);
    let mut nodes = vec![Node {
        bounds: enclosing_cube(bodies),
        depth: 0,
        members: ordered_members,
        children: Vec::new(),
    }];
    subdivide(0, config, bodies, &mut nodes);

    let mut tested = HashSet::new();
    let mut overlaps = BTreeSet::new();
    let mut aabb_tests = 0_u64;

    for node in nodes.iter().filter(|node| node.children.is_empty()) {
        for left in 0..node.members.len() {
            for right in (left + 1)..node.members.len() {
                let a = node.members[left];
                let b = node.members[right];
                let pair = Pair::new(bodies[a].id, bodies[b].id);
                if !tested.insert(pair) {
                    continue;
                }
                aabb_tests += 1;
                if bodies[a].aabb.overlaps(bodies[b].aabb) {
                    overlaps.insert(pair);
                }
            }
        }
    }

    let result = BroadPhaseResult {
        pairs: overlaps.into_iter().collect(),
        stats: BroadPhaseStats {
            aabb_tests,
            occupied_cells: None,
        },
    };

    let leaf_count = nodes.iter().filter(|node| node.children.is_empty()).count();
    let occupied_leaf_count = nodes
        .iter()
        .filter(|node| node.children.is_empty() && !node.members.is_empty())
        .count();

    let snapshots = if trace {
        nodes
            .iter()
            .enumerate()
            .map(|(index, node)| OctreeNodeSnapshot {
                index,
                bounds: node.bounds,
                depth: node.depth,
                members: node
                    .members
                    .iter()
                    .map(|&member| bodies[member].id)
                    .collect(),
                children: node.children.clone(),
            })
            .collect()
    } else {
        Vec::new()
    };

    OctreeTrace {
        result,
        root: Some(0),
        nodes: snapshots,
        leaf_count,
        occupied_leaf_count,
    }
}

fn subdivide(index: usize, config: OctreeConfig, bodies: &[Body], nodes: &mut Vec<Node>) {
    let node = &nodes[index];
    if node.depth >= config.max_depth || node.members.len() <= config.leaf_capacity {
        return;
    }

    let child_depth = node.depth + 1;
    let child_bounds = split_bounds(node.bounds);
    let child_members: [Vec<usize>; 8] = std::array::from_fn(|child| {
        node.members
            .iter()
            .copied()
            .filter(|&member| child_bounds[child].overlaps(bodies[member].aabb))
            .collect()
    });
    // Every recursive branch must make strict progress. If any child would
    // retain the complete parent set, descending into it only duplicates the
    // same candidate population at a deeper level while sibling overlap
    // creates additional copies. This is especially costly for clustered
    // scenes with straddlers near split planes.
    let parent_member_count = node.members.len();
    if child_members
        .iter()
        .any(|members| members.len() == parent_member_count)
    {
        return;
    }

    // Child slots are contiguous: move their member buffers once and walk the
    // index range without cloning parent nodes, member vectors or child lists.
    let first_child = nodes.len();
    nodes[index].children = (first_child..first_child + 8).collect();
    for (bounds, members) in child_bounds.into_iter().zip(child_members) {
        nodes.push(Node {
            bounds,
            depth: child_depth,
            members,
            children: Vec::new(),
        });
    }
    for child_index in first_child..first_child + 8 {
        if !nodes[child_index].members.is_empty() {
            subdivide(child_index, config, bodies, nodes);
        }
    }
}

fn enclosing_cube(bodies: &[Body]) -> Aabb {
    let mut bounds = bodies[0].aabb;
    for body in &bodies[1..] {
        bounds = bounds.union(body.aabb);
    }

    // Compute in f64 so finite f32 endpoints cannot overflow sums or spans.
    // The final min/max against the source union is intentional: even f64 may
    // lose the small endpoint of an extremely asymmetric interval.
    let half_side = (0..3)
        .map(|axis| f64::from(bounds.max[axis]) - f64::from(bounds.min[axis]))
        .fold(1.0_f64, f64::max)
        * 0.5;
    let centers: [f64; 3] = std::array::from_fn(|axis| {
        (f64::from(bounds.min[axis]) + f64::from(bounds.max[axis])) * 0.5
    });
    Aabb::new(
        std::array::from_fn(|axis| {
            round_outward(centers[axis] - half_side, false).min(bounds.min[axis])
        }),
        std::array::from_fn(|axis| {
            round_outward(centers[axis] + half_side, true).max(bounds.max[axis])
        }),
    )
}

// Directed rounding without next_up/next_down, which are newer than our MSRV.
// At f32's extremes a finite cube may not fit: clamp only the padded part and
// retain a conservative finite root box rather than panicking or losing bodies.
fn round_outward(value: f64, up: bool) -> f32 {
    let value = value.clamp(f64::from(f32::MIN), f64::from(f32::MAX));
    let rounded = value as f32;
    let needs_step = if up {
        f64::from(rounded) < value
    } else {
        f64::from(rounded) > value
    };
    if !needs_step {
        return rounded;
    }
    if rounded == 0.0 {
        return if up {
            f32::from_bits(1)
        } else {
            -f32::from_bits(1)
        };
    }
    let bits = if up == rounded.is_sign_positive() {
        rounded.to_bits() + 1
    } else {
        rounded.to_bits() - 1
    };
    f32::from_bits(bits)
}

fn split_bounds(bounds: Aabb) -> [Aabb; 8] {
    let mid: [f32; 3] = std::array::from_fn(|axis| {
        ((f64::from(bounds.min[axis]) + f64::from(bounds.max[axis])) * 0.5) as f32
    });
    std::array::from_fn(|child| {
        let high_x = child & 1 != 0;
        let high_y = child & 2 != 0;
        let high_z = child & 4 != 0;
        Aabb::new(
            [
                if high_x { mid[0] } else { bounds.min[0] },
                if high_y { mid[1] } else { bounds.min[1] },
                if high_z { mid[2] } else { bounds.min[2] },
            ],
            [
                if high_x { bounds.max[0] } else { mid[0] },
                if high_y { bounds.max[1] } else { mid[1] },
                if high_z { bounds.max[2] } else { mid[2] },
            ],
        )
    })
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
    use super::*;
    use spatial_kernels::NaiveBroadPhase;

    fn body(id: ColliderId, center: [f32; 3], half: f32) -> Body {
        Body::new(id, Aabb::from_center_half_extents(center, [half; 3]))
    }

    fn fixture() -> Vec<Body> {
        vec![
            body(10, [-2.0, -2.0, -2.0], 0.8),
            body(20, [-1.2, -2.0, -2.0], 0.8),
            body(30, [2.0, 2.0, 2.0], 0.7),
            body(40, [0.0, 0.0, 0.0], 1.5),
            body(50, [3.5, -3.0, 2.0], 0.5),
            body(60, [-3.5, 3.0, -2.0], 0.5),
        ]
    }

    #[test]
    fn directed_rounding_is_finite_and_outward() {
        for x in [
            -f32::MAX,
            -1e30,
            -1.0,
            -f32::from_bits(1),
            0.0,
            f32::from_bits(1),
            1.0,
            1e30,
            f32::MAX,
        ] {
            for value in [
                f64::from(x),
                f64::from(x) * (1.0 + 1e-10),
                f64::from(x) * (1.0 - 1e-10),
            ] {
                let lo = super::round_outward(value, false);
                let hi = super::round_outward(value, true);
                let clamped = value.clamp(f64::from(f32::MIN), f64::from(f32::MAX));
                assert!(lo.is_finite() && hi.is_finite());
                assert!(f64::from(lo) <= clamped && f64::from(hi) >= clamped);
            }
        }
        assert_eq!(super::round_outward(1e100, true), f32::MAX);
        assert_eq!(super::round_outward(-1e100, false), f32::MIN);
    }

    #[test]
    fn octree_matches_naive_oracle() {
        let bodies = fixture();
        let octree = OctreeBroadPhase::new(5, 2);
        assert_eq!(
            octree.detect(&bodies).pairs,
            NaiveBroadPhase.detect(&bodies).pairs
        );
    }

    #[test]
    fn trace_matches_detect_and_exposes_eight_way_split() {
        let bodies = fixture();
        let octree = OctreeBroadPhase::new(4, 1);
        let trace = octree.trace(&bodies);
        assert_eq!(trace.result, octree.detect(&bodies));
        assert_eq!(trace.root, Some(0));
        assert_eq!(trace.nodes[0].children.len(), 8);
        assert!(
            trace.nodes[0]
                .children
                .iter()
                .all(|&index| trace.nodes[index].depth == 1)
        );
    }

    #[test]
    fn trace_is_independent_of_input_order() {
        let bodies = fixture();
        let expected = OctreeBroadPhase::new(5, 2).trace(&bodies);
        let mut reversed = bodies.clone();
        reversed.reverse();
        assert_eq!(OctreeBroadPhase::new(5, 2).trace(&reversed), expected);
    }

    #[test]
    fn sparse_scene_avoids_most_exact_tests() {
        let bodies: Vec<_> = (0..128)
            .map(|id| {
                let x = (id % 8) as f32 * 20.0;
                let y = ((id / 8) % 4) as f32 * 20.0;
                let z = (id / 32) as f32 * 20.0;
                body(id, [x, y, z], 0.25)
            })
            .collect();
        let naive = NaiveBroadPhase.detect(&bodies);
        let octree = OctreeBroadPhase::new(7, 4).detect(&bodies);
        assert_eq!(octree.pairs, naive.pairs);
        assert!(octree.stats.aabb_tests < naive.stats.aabb_tests / 10);
    }

    #[test]
    fn large_straddling_body_remains_correct() {
        let bodies = vec![
            body(1, [0.0, 0.0, 0.0], 10.0),
            body(2, [-8.0, -8.0, -8.0], 0.5),
            body(3, [8.0, 8.0, 8.0], 0.5),
            body(4, [15.0, 0.0, 0.0], 0.5),
        ];
        let octree = OctreeBroadPhase::new(6, 1);
        assert_eq!(
            octree.detect(&bodies).pairs,
            NaiveBroadPhase.detect(&bodies).pairs
        );
    }

    #[test]
    fn boundary_touching_overlap_is_not_lost() {
        let bodies = vec![
            Body::new(1, Aabb::new([-2.0, -1.0, -1.0], [0.0, 1.0, 1.0])),
            Body::new(2, Aabb::new([0.0, -1.0, -1.0], [2.0, 1.0, 1.0])),
            body(3, [4.0, 0.0, 0.0], 0.5),
        ];
        let result = OctreeBroadPhase::new(5, 1).detect(&bodies);
        assert_eq!(result.pairs, vec![Pair::new(1, 2)]);
    }

    #[test]
    fn subdivision_stops_when_a_child_cannot_reduce_the_candidate_set() {
        let bodies = vec![
            body(1, [0.0, 0.0, 0.0], 10.0),
            body(2, [-8.0, -8.0, -8.0], 0.5),
        ];
        let tree = OctreeBroadPhase::new(6, 1);
        let trace = tree.trace(&bodies);

        assert_eq!(trace.result.pairs, NaiveBroadPhase.detect(&bodies).pairs);
        assert_eq!(trace.nodes.len(), 1);
        assert_eq!(trace.leaf_count, 1);
        assert_eq!(trace.occupied_leaf_count, 1);
    }

    #[test]
    fn everything_overlapping_degrades_without_losing_pairs() {
        let bodies: Vec<_> = (0..20).map(|id| body(id, [0.0; 3], 2.0)).collect();
        let naive = NaiveBroadPhase.detect(&bodies);
        let octree = OctreeBroadPhase::new(6, 2).detect(&bodies);
        assert_eq!(octree.pairs, naive.pairs);
        assert_eq!(octree.stats.aabb_tests, naive.stats.aabb_tests);
    }

    #[test]
    fn empty_tree_has_no_nodes_or_pairs() {
        let trace = OctreeBroadPhase::default().trace(&[]);
        assert_eq!(trace.root, None);
        assert!(trace.nodes.is_empty());
        assert!(trace.result.pairs.is_empty());
    }
}
