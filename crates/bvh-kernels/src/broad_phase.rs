use spatial_kernels::{Body, BroadPhase, BroadPhaseResult, BroadPhaseStats};

use crate::dynamic_tree::DynamicAabbTree;
use crate::static_bvh::{StaticBvh, validate_unique_ids};

/// Broad-phase adapter that builds a static BVH for each detection call.
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticBvhBroadPhase;

impl BroadPhase for StaticBvhBroadPhase {
    fn detect(&self, bodies: &[Body]) -> BroadPhaseResult {
        let bvh = StaticBvh::build(bodies);
        let (pairs, aabb_tests) = bvh.overlapping_pairs_with_tests();
        BroadPhaseResult {
            pairs,
            stats: BroadPhaseStats {
                aabb_tests,
                occupied_cells: None,
            },
        }
    }
}

/// Broad-phase adapter that builds a dynamic tree from a snapshot. Stateful
/// consumers can keep `DynamicAabbTree` alive and call `update` between frames.
#[derive(Clone, Copy, Debug)]
pub struct DynamicAabbTreeBroadPhase {
    fat_margin: f32,
}

impl Default for DynamicAabbTreeBroadPhase {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl DynamicAabbTreeBroadPhase {
    #[must_use]
    pub fn new(fat_margin: f32) -> Self {
        let _ = DynamicAabbTree::new(fat_margin);
        Self { fat_margin }
    }

    #[must_use]
    pub const fn fat_margin(self) -> f32 {
        self.fat_margin
    }
}

impl BroadPhase for DynamicAabbTreeBroadPhase {
    fn detect(&self, bodies: &[Body]) -> BroadPhaseResult {
        validate_unique_ids(bodies);
        let mut ordered = bodies.to_vec();
        ordered.sort_unstable_by_key(|body| body.id);
        let mut tree = DynamicAabbTree::new(self.fat_margin);
        for body in ordered {
            tree.insert(body);
        }
        tree.overlapping_pairs_result()
    }
}
