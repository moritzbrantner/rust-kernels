use super::{DynamicAabbTree, DynamicAabbTreeBroadPhase, StaticBvh, StaticBvhBroadPhase};
use spatial_kernels::{Aabb, Body, BroadPhase, ColliderId, NaiveBroadPhase};

fn body(id: ColliderId, center: [f32; 3], half: f32) -> Body {
    Body::new(id, Aabb::from_center_half_extents(center, [half; 3]))
}

#[test]
fn predicted_updates_stretch_the_fat_bounds_along_the_motion() {
    let mut tree = DynamicAabbTree::new(0.1);
    tree.insert(body(1, [0.0, 0.0, 0.0], 0.5));
    // A steady 0.3-per-update move to +X: the first update reinserts with room ahead,
    // so the next two moves stay inside the leaf.
    let step = [0.3, 0.0, 0.0];
    assert!(tree.update_predicted(body(1, [0.3, 0.0, 0.0], 0.5), [0.6, 0.0, 0.0]));
    assert!(!tree.update_predicted(body(1, [0.6, 0.0, 0.0], 0.5), step));
    assert!(!tree.update_predicted(body(1, [0.9, 0.0, 0.0], 0.5), step));
    let fat = tree.fat_bounds(1).unwrap();
    let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
    assert!(close(fat.min[0], -0.3) && close(fat.min[1], -0.6) && close(fat.min[2], -0.6));
    assert!(close(fat.max[0], 1.5) && close(fat.max[1], 0.6));
    // Queries still see the exact body.
    assert_eq!(
        tree.query_aabb(Aabb::from_center_half_extents([1.3, 0.0, 0.0], [0.05; 3])),
        [1]
    );
    assert!(
        tree.query_aabb(Aabb::from_center_half_extents([-0.25, 0.0, 0.0], [0.01; 3]))
            .is_empty()
    );
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
fn query_returns_sorted_overlapping_ids() {
    let bvh = StaticBvh::build(&fixture());
    let query = Aabb::from_center_half_extents([0.5, 0.0, 0.0], [0.2; 3]);
    assert_eq!(bvh.query_aabb(query), vec![10, 20, 50]);
}

#[test]
fn broad_phase_matches_the_naive_oracle() {
    let bodies = fixture();
    let expected = NaiveBroadPhase.detect(&bodies);
    let actual = StaticBvhBroadPhase.detect(&bodies);
    assert_eq!(actual.pairs, expected.pairs);
}

#[test]
fn output_is_independent_of_input_order() {
    let bodies = fixture();
    let expected = StaticBvhBroadPhase.detect(&bodies).pairs;
    let mut reversed = bodies.clone();
    reversed.reverse();
    assert_eq!(StaticBvhBroadPhase.detect(&reversed).pairs, expected);
}

#[test]
fn sparse_bvh_avoids_most_leaf_pair_tests() {
    let bodies: Vec<_> = (0..100)
        .map(|id| body(id, [id as f32 * 10.0, 0.0, 0.0], 0.25))
        .collect();

    let naive = NaiveBroadPhase.detect(&bodies);
    let bvh = StaticBvhBroadPhase.detect(&bodies);
    assert_eq!(bvh.pairs, naive.pairs);
    assert!(bvh.stats.aabb_tests < naive.stats.aabb_tests / 10);
    assert_eq!(naive.stats.aabb_tests, 4_950);
}

#[test]
fn retained_result_matches_snapshot_adapter_work_and_pairs() {
    let bodies = fixture();
    let margin = 0.75;
    let mut tree = DynamicAabbTree::new(margin);
    for body in &bodies {
        tree.insert(*body);
    }

    let retained = tree.overlapping_pairs_result();
    let snapshot = DynamicAabbTreeBroadPhase::new(margin).detect(&bodies);
    assert_eq!(retained, snapshot);
    assert_eq!(tree.overlapping_pairs(), retained.pairs);
}

#[test]
fn empty_and_singleton_trees_have_expected_sizes() {
    let empty = StaticBvh::build(&[]);
    assert!(empty.is_empty());
    assert_eq!(empty.node_count(), 0);

    let singleton = StaticBvh::build(&[body(1, [0.0; 3], 1.0)]);
    assert_eq!(singleton.len(), 1);
    assert_eq!(singleton.node_count(), 1);
    assert!(singleton.overlapping_pairs().is_empty());
}

#[test]
fn dynamic_broad_phase_matches_naive_oracle() {
    let bodies = fixture();
    let expected = NaiveBroadPhase.detect(&bodies);
    let actual = DynamicAabbTreeBroadPhase::new(0.25).detect(&bodies);
    assert_eq!(actual.pairs, expected.pairs);
}

#[test]
fn dynamic_tree_supports_insert_query_and_remove() {
    let bodies = fixture();
    let mut tree = DynamicAabbTree::new(0.25);
    for body in &bodies {
        tree.insert(*body);
    }
    let query = Aabb::from_center_half_extents([0.5, 0.0, 0.0], [0.2; 3]);
    assert_eq!(tree.query_aabb(query), vec![10, 20, 50]);
    assert_eq!(tree.remove(20), Some(bodies[1]));
    assert_eq!(tree.query_aabb(query), vec![10, 50]);
    assert_eq!(tree.len(), bodies.len() - 1);
}

#[test]
fn fat_aabb_avoids_small_reinsertions() {
    let mut tree = DynamicAabbTree::new(1.0);
    tree.insert(body(1, [0.0; 3], 0.5));
    let original_fat = tree.fat_bounds(1).expect("inserted body must exist");
    assert!(!tree.update(body(1, [0.25, 0.0, 0.0], 0.5)));
    assert_eq!(tree.fat_bounds(1), Some(original_fat));
    assert!(tree.update(body(1, [2.0, 0.0, 0.0], 0.5)));
    assert_ne!(tree.fat_bounds(1), Some(original_fat));
}

#[test]
fn contained_update_trace_keeps_fat_bounds_and_changes_leaf_snapshot() {
    let mut tree = DynamicAabbTree::new(1.0);
    tree.insert(body(1, [0.0; 3], 0.5));
    let trace = tree.update_with_trace(body(1, [0.25, 0.0, 0.0], 0.5));

    assert!(!trace.reinserted);
    assert_eq!(trace.previous_fat_bounds, trace.current_fat_bounds);
    assert_eq!(trace.height_before, trace.height_after);
    assert_eq!(trace.changed_nodes, vec![0]);
    assert_eq!(trace.before_nodes.len(), 1);
    assert_eq!(trace.after_nodes.len(), 1);
    assert_ne!(
        trace.before_nodes[0].exact_bounds,
        trace.after_nodes[0].exact_bounds
    );
}

#[test]
fn reinsertion_trace_exposes_structural_diff() {
    let mut tree = DynamicAabbTree::new(0.25);
    for body in fixture() {
        tree.insert(body);
    }
    let before_pairs = tree.overlapping_pairs();
    let trace = tree.update_with_trace(body(20, [8.0, 0.0, 0.0], 0.6));

    assert!(trace.reinserted);
    assert_ne!(trace.previous_fat_bounds, trace.current_fat_bounds);
    assert!(!trace.changed_nodes.is_empty());
    assert_eq!(trace.after_nodes.len(), tree.node_count());
    assert_ne!(tree.overlapping_pairs(), before_pairs);
}

#[test]
fn debug_snapshot_has_one_root_and_matches_node_count() {
    let mut tree = DynamicAabbTree::new(0.5);
    for body in fixture() {
        tree.insert(body);
    }
    let nodes = tree.debug_nodes();
    assert_eq!(nodes.len(), tree.node_count());
    assert_eq!(nodes.iter().filter(|node| node.is_root).count(), 1);
    assert_eq!(
        nodes.iter().filter(|node| node.body.is_some()).count(),
        tree.len()
    );
}

#[test]
fn dynamic_pairs_remain_exact_during_incremental_motion() {
    let mut bodies = fixture();
    let mut tree = DynamicAabbTree::new(0.75);
    for body in &bodies {
        tree.insert(*body);
    }

    for step in 0..20 {
        bodies[1] = body(20, [0.9 + step as f32 * 0.15, 0.0, 0.0], 0.6);
        tree.update(bodies[1]);
        assert_eq!(
            tree.overlapping_pairs(),
            NaiveBroadPhase.detect(&bodies).pairs,
            "step {step}"
        );
    }
}

#[test]
fn traced_updates_preserve_exact_pair_parity() {
    let mut bodies = fixture();
    let mut tree = DynamicAabbTree::new(0.5);
    for body in &bodies {
        tree.insert(*body);
    }

    for step in 0..30 {
        bodies[1] = body(20, [0.9 + step as f32 * 0.2, 0.0, 0.0], 0.6);
        let _ = tree.update_with_trace(bodies[1]);
        assert_eq!(
            tree.overlapping_pairs(),
            NaiveBroadPhase.detect(&bodies).pairs,
            "step {step}"
        );
    }
}

#[test]
fn dynamic_pair_traversal_matches_naive_for_clustered_scene() {
    let bodies: Vec<_> = (0..512)
        .map(|id| {
            let cluster = id % 8;
            let local = id / 8;
            let base = [
                if cluster & 1 == 0 { -40.0 } else { 40.0 },
                if cluster & 2 == 0 { -40.0 } else { 40.0 },
                if cluster & 4 == 0 { -40.0 } else { 40.0 },
            ];
            let offset = [
                (local % 4) as f32 * 0.8,
                ((local / 4) % 4) as f32 * 0.8,
                (local / 16) as f32 * 0.8,
            ];
            body(
                id,
                [
                    base[0] + offset[0],
                    base[1] + offset[1],
                    base[2] + offset[2],
                ],
                0.5,
            )
        })
        .collect();

    let naive = NaiveBroadPhase.detect(&bodies);
    let dynamic = DynamicAabbTreeBroadPhase::new(1.25).detect(&bodies);
    assert_eq!(dynamic.pairs, naive.pairs);
    assert!(dynamic.stats.aabb_tests < naive.stats.aabb_tests / 4);
}

#[test]
fn balancing_keeps_sequential_inserts_shallow() {
    let mut tree = DynamicAabbTree::new(0.1);
    for id in 0..128 {
        tree.insert(body(id, [id as f32 * 2.0, 0.0, 0.0], 0.25));
    }
    assert!(tree.height() <= 16, "tree height was {}", tree.height());
}

#[test]
fn sparse_dynamic_tree_avoids_most_exact_pair_tests() {
    let bodies: Vec<_> = (0..100)
        .map(|id| body(id, [id as f32 * 10.0, 0.0, 0.0], 0.25))
        .collect();
    let naive = NaiveBroadPhase.detect(&bodies);
    let dynamic = DynamicAabbTreeBroadPhase::new(0.1).detect(&bodies);
    assert_eq!(dynamic.pairs, naive.pairs);
    assert!(dynamic.stats.aabb_tests < naive.stats.aabb_tests / 10);
}

fn scattered(count: u32) -> Vec<Body> {
    let mut state = 0x9e37_79b9_u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state >> 8) as f32 / (1 << 24) as f32 * 40.0 - 20.0
    };
    (0..count)
        .map(|id| {
            let center = [next(), next(), next()];
            body(id, center, 0.5 + (id % 5) as f32 * 0.5)
        })
        .collect()
}

#[test]
fn visit_aabb_matches_query_aabb() {
    let mut tree = DynamicAabbTree::new(0.1);
    for body in scattered(300) {
        tree.insert(body);
    }
    for id in (0..300).step_by(7) {
        tree.remove(id);
    }
    let query = Aabb::from_center_half_extents([1.0, -2.0, 3.0], [6.0, 5.0, 4.0]);
    let mut visited = Vec::new();
    tree.visit_aabb(query, |body| visited.push(body.id));
    visited.sort_unstable();
    assert_eq!(visited, tree.query_aabb(query));
    assert!(!visited.is_empty());
}

#[test]
fn visit_segment_finds_every_crossed_box_and_prunes_later_ones() {
    let bodies = scattered(3000);
    let mut tree = DynamicAabbTree::new(0.1);
    for body in &bodies {
        tree.insert(*body);
    }
    let (origin, delta) = ([-20.0, -3.0, -1.0], [40.0, 5.0, 3.0]);
    let mut all = Vec::new();
    tree.visit_segment(origin, delta, 1.0, 0.0, |body, t| {
        all.push((body.id, t));
        1.0
    });
    let expected: Vec<_> = bodies
        .iter()
        .filter_map(|b| {
            super::segment::segment_entry(b.aabb, origin, delta, 1.0, 0.0).map(|t| (b.id, t))
        })
        .collect();
    let mut sorted = all.clone();
    sorted.sort_unstable_by_key(|&(id, _)| id);
    assert_eq!(sorted, expected);
    assert!(expected.len() > 3);

    // Pruning at the nearest entry never visits a box entered later.
    let nearest = expected
        .iter()
        .map(|&(_, t)| t)
        .fold(f32::INFINITY, f32::min);
    let mut visited = Vec::new();
    tree.visit_segment(origin, delta, 1.0, 0.0, |body, t| {
        visited.push((body.id, t));
        t
    });
    assert!(visited.iter().any(|&(_, t)| t == nearest));
    let best = visited
        .iter()
        .map(|&(_, t)| t)
        .fold(f32::INFINITY, f32::min);
    assert_eq!(best, nearest);
    assert!(visited.len() < expected.len());
}

#[test]
fn dense_leaf_table_handles_sparse_and_reused_ids() {
    let mut tree = DynamicAabbTree::new(0.0);
    tree.insert(body(1000, [0.0; 3], 1.0));
    tree.insert(body(3, [5.0, 0.0, 0.0], 1.0));
    assert_eq!(tree.len(), 2);
    assert!(tree.remove(1000).is_some());
    assert!(tree.remove(1000).is_none());
    assert!(tree.fat_bounds(4).is_none());
    tree.insert(body(1000, [9.0, 0.0, 0.0], 1.0));
    assert_eq!(tree.len(), 2);
    assert_eq!(
        tree.query_aabb(Aabb::from_center_half_extents([9.0, 0.0, 0.0], [0.5; 3])),
        vec![1000]
    );
}

#[test]
fn rebuild_keeps_query_results_and_supports_further_updates() {
    let bodies = scattered(500);
    let mut tree = DynamicAabbTree::new(0.1);
    for body in &bodies {
        tree.insert(*body);
    }
    let query = Aabb::from_center_half_extents([0.0; 3], [8.0; 3]);
    let before = tree.query_aabb(query);
    let pairs = tree.overlapping_pairs();
    tree.rebuild();
    assert_eq!(tree.len(), 500);
    assert_eq!(tree.node_count(), 999);
    assert_eq!(tree.query_aabb(query), before);
    assert_eq!(tree.overlapping_pairs(), pairs);
    assert!(tree.height() <= 24);
    tree.remove(7);
    tree.insert(body(7, [50.0; 3], 1.0));
    assert!(tree.update(body(8, [-50.0; 3], 1.0)));
    assert_eq!(
        tree.query_aabb(Aabb::from_center_half_extents([50.0; 3], [2.0; 3])),
        vec![7]
    );
    assert_eq!(
        tree.query_aabb(Aabb::from_center_half_extents([-50.0; 3], [2.0; 3])),
        vec![8]
    );
}
