//! Reusable spatial kernels with deterministic outputs.
//!
//! The representation stays small: axis-aligned bounding boxes, spatial cell
//! hashing, integer spatial keys, and interchangeable broad-phase algorithms.

mod morton;
mod spatial_hash;

use std::collections::{BTreeSet, HashMap, HashSet};

pub use morton::{
    MORTON_3D_BITS_PER_AXIS, MORTON_3D_MAX_COORD, morton2_decode, morton2_encode, morton3_decode,
    morton3_encode,
};
pub use spatial_hash::{CellCoord3, SpatialHash3D};

pub type ColliderId = u32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    #[must_use]
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        assert!(
            min.iter().chain(max.iter()).all(|bound| bound.is_finite()),
            "AABB bounds must be finite"
        );
        assert!(
            min.iter().zip(max).all(|(min, max)| *min <= max),
            "AABB minimum must not exceed maximum"
        );
        Self { min, max }
    }

    #[must_use]
    pub fn from_center_half_extents(center: [f32; 3], half_extents: [f32; 3]) -> Self {
        assert!(
            center.iter().all(|coordinate| coordinate.is_finite()),
            "AABB center must be finite"
        );
        assert!(
            half_extents
                .iter()
                .all(|extent| extent.is_finite() && *extent >= 0.0),
            "AABB half extents must be non-negative and finite"
        );
        Self::new(
            [
                center[0] - half_extents[0],
                center[1] - half_extents[1],
                center[2] - half_extents[2],
            ],
            [
                center[0] + half_extents[0],
                center[1] + half_extents[1],
                center[2] + half_extents[2],
            ],
        )
    }

    #[must_use]
    pub fn overlaps(self, other: Self) -> bool {
        (0..3).all(|axis| self.min[axis] <= other.max[axis] && self.max[axis] >= other.min[axis])
    }

    #[must_use]
    pub fn contains(self, other: Self) -> bool {
        (0..3).all(|axis| self.min[axis] <= other.min[axis] && self.max[axis] >= other.max[axis])
    }

    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self::new(
            [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        )
    }

    #[must_use]
    pub fn expanded(self, margin: f32) -> Self {
        assert!(
            margin.is_finite() && margin >= 0.0,
            "AABB expansion margin must be non-negative and finite"
        );
        Self::new(
            [
                self.min[0] - margin,
                self.min[1] - margin,
                self.min[2] - margin,
            ],
            [
                self.max[0] + margin,
                self.max[1] + margin,
                self.max[2] + margin,
            ],
        )
    }

    #[must_use]
    pub fn surface_area(self) -> f64 {
        let x = f64::from(self.max[0]) - f64::from(self.min[0]);
        let y = f64::from(self.max[1]) - f64::from(self.min[1]);
        let z = f64::from(self.max[2]) - f64::from(self.min[2]);
        2.0 * (x * y + y * z + z * x)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub id: ColliderId,
    pub aabb: Aabb,
}

impl Body {
    #[must_use]
    pub const fn new(id: ColliderId, aabb: Aabb) -> Self {
        Self { id, aabb }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Pair {
    pub a: ColliderId,
    pub b: ColliderId,
}

impl Pair {
    #[must_use]
    pub fn new(a: ColliderId, b: ColliderId) -> Self {
        assert_ne!(a, b, "a pair must contain two distinct colliders");
        if a < b {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BroadPhaseStats {
    /// Number of unique AABB-vs-AABB tests performed.
    pub aabb_tests: u64,
    /// Number of occupied grid cells, when the algorithm has cells.
    pub occupied_cells: Option<usize>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BroadPhaseResult {
    /// Deterministic, sorted set of overlapping AABB pairs.
    pub pairs: Vec<Pair>,
    pub stats: BroadPhaseStats,
}

pub trait BroadPhase {
    fn detect(&self, bodies: &[Body]) -> BroadPhaseResult;
}

/// O(n²) reference implementation. Keep this intentionally simple: optimized
/// broad phases can be differential-tested against it.
#[derive(Clone, Copy, Debug, Default)]
pub struct NaiveBroadPhase;

impl BroadPhase for NaiveBroadPhase {
    fn detect(&self, bodies: &[Body]) -> BroadPhaseResult {
        validate_unique_ids(bodies);
        let mut pairs = Vec::new();
        let mut aabb_tests = 0_u64;

        for left in 0..bodies.len() {
            for right in (left + 1)..bodies.len() {
                aabb_tests += 1;
                if bodies[left].aabb.overlaps(bodies[right].aabb) {
                    pairs.push(Pair::new(bodies[left].id, bodies[right].id));
                }
            }
        }

        pairs.sort_unstable();
        BroadPhaseResult {
            pairs,
            stats: BroadPhaseStats {
                aabb_tests,
                occupied_cells: None,
            },
        }
    }
}

/// Uniform 3D grid. Bodies are inserted into every spatial-hash cell touched by
/// their AABB; only bodies sharing a cell can become candidate pairs.
#[derive(Clone, Copy, Debug)]
pub struct UniformGridBroadPhase {
    cell_size: f32,
}

impl UniformGridBroadPhase {
    #[must_use]
    pub fn new(cell_size: f32) -> Self {
        let _ = SpatialHash3D::new(cell_size);
        Self { cell_size }
    }

    #[must_use]
    pub const fn cell_size(self) -> f32 {
        self.cell_size
    }
}

impl BroadPhase for UniformGridBroadPhase {
    fn detect(&self, bodies: &[Body]) -> BroadPhaseResult {
        validate_unique_ids(bodies);
        let spatial_hash = SpatialHash3D::new(self.cell_size);
        let mut cells: HashMap<CellCoord3, Vec<usize>> = HashMap::new();

        for (body_index, body) in bodies.iter().enumerate() {
            let (min, max) = spatial_hash.cell_bounds(body.aabb);
            for x in min.x..=max.x {
                for y in min.y..=max.y {
                    for z in min.z..=max.z {
                        cells
                            .entry(CellCoord3::new(x, y, z))
                            .or_default()
                            .push(body_index);
                    }
                }
            }
        }

        let mut tested_indices = HashSet::new();
        let mut overlaps = BTreeSet::new();
        let mut aabb_tests = 0_u64;

        for members in cells.values() {
            for left in 0..members.len() {
                for right in (left + 1)..members.len() {
                    let a = members[left];
                    let b = members[right];
                    let index_pair = if a < b { (a, b) } else { (b, a) };
                    if !tested_indices.insert(index_pair) {
                        continue;
                    }

                    aabb_tests += 1;
                    if bodies[a].aabb.overlaps(bodies[b].aabb) {
                        overlaps.insert(Pair::new(bodies[a].id, bodies[b].id));
                    }
                }
            }
        }

        BroadPhaseResult {
            pairs: overlaps.into_iter().collect(),
            stats: BroadPhaseStats {
                aabb_tests,
                occupied_cells: Some(cells.len()),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Axis3 {
    #[default]
    X,
    Y,
    Z,
}

impl Axis3 {
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

/// Sort-and-sweep broad phase over one axis. Bodies enter the active set when
/// their minimum endpoint is reached and leave after their maximum endpoint.
#[derive(Clone, Copy, Debug, Default)]
pub struct SweepAndPruneBroadPhase {
    axis: Axis3,
}

impl SweepAndPruneBroadPhase {
    #[must_use]
    pub const fn new(axis: Axis3) -> Self {
        Self { axis }
    }

    #[must_use]
    pub const fn axis(self) -> Axis3 {
        self.axis
    }
}

impl BroadPhase for SweepAndPruneBroadPhase {
    fn detect(&self, bodies: &[Body]) -> BroadPhaseResult {
        validate_unique_ids(bodies);
        let axis = self.axis.index();
        let mut order: Vec<usize> = (0..bodies.len()).collect();
        order.sort_unstable_by(|&left, &right| {
            bodies[left].aabb.min[axis]
                .total_cmp(&bodies[right].aabb.min[axis])
                .then_with(|| bodies[left].aabb.max[axis].total_cmp(&bodies[right].aabb.max[axis]))
                .then_with(|| bodies[left].id.cmp(&bodies[right].id))
        });

        let mut active: Vec<usize> = Vec::new();
        let mut pairs = Vec::new();
        let mut aabb_tests = 0_u64;

        for current in order {
            let current_min = bodies[current].aabb.min[axis];
            active.retain(|&other| bodies[other].aabb.max[axis] >= current_min);
            for &other in &active {
                aabb_tests += 1;
                if bodies[current].aabb.overlaps(bodies[other].aabb) {
                    pairs.push(Pair::new(bodies[current].id, bodies[other].id));
                }
            }
            active.push(current);
        }

        pairs.sort_unstable();
        BroadPhaseResult {
            pairs,
            stats: BroadPhaseStats {
                aabb_tests,
                occupied_cells: None,
            },
        }
    }
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
    fn aabb_overlap_is_inclusive_at_boundaries() {
        let left = Aabb::new([0.0; 3], [1.0; 3]);
        let touching = Aabb::new([1.0, 0.25, 0.25], [2.0, 0.75, 0.75]);
        let separated = Aabb::new([1.01, 0.25, 0.25], [2.0, 0.75, 0.75]);

        assert!(left.overlaps(touching));
        assert!(!left.overlaps(separated));
    }

    #[test]
    #[should_panic(expected = "AABB bounds must be finite")]
    fn aabb_rejects_non_finite_bounds() {
        let _ = Aabb::new([f32::NEG_INFINITY, 0.0, 0.0], [f32::INFINITY, 0.0, 0.0]);
    }

    #[test]
    fn aabb_helpers_cover_union_containment_expansion_and_area() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
        let b = Aabb::new([-1.0, 1.0, 2.0], [0.5, 4.0, 5.0]);
        let union = a.union(b);
        assert_eq!(union, Aabb::new([-1.0, 0.0, 0.0], [1.0, 4.0, 5.0]));
        assert!(union.contains(a));
        assert!(union.contains(b));
        assert_eq!(
            a.expanded(1.0),
            Aabb::new([-1.0, -1.0, -1.0], [2.0, 3.0, 4.0])
        );
        assert_eq!(a.surface_area(), 22.0);
    }

    #[test]
    fn pair_order_is_canonical() {
        assert_eq!(Pair::new(9, 2), Pair { a: 2, b: 9 });
    }

    #[test]
    fn grid_matches_naive_across_cell_sizes() {
        let bodies = fixture();
        let expected = NaiveBroadPhase.detect(&bodies).pairs;

        for cell_size in [0.25, 0.5, 1.0, 2.0, 10.0] {
            let actual = UniformGridBroadPhase::new(cell_size).detect(&bodies);
            assert_eq!(actual.pairs, expected, "cell size {cell_size}");
        }
    }

    #[test]
    fn sparse_grid_avoids_most_pair_tests() {
        let bodies: Vec<_> = (0..100)
            .map(|id| body(id, [id as f32 * 10.0, 0.0, 0.0], 0.25))
            .collect();

        let naive = NaiveBroadPhase.detect(&bodies);
        let grid = UniformGridBroadPhase::new(2.0).detect(&bodies);

        assert_eq!(grid.pairs, naive.pairs);
        assert_eq!(grid.stats.aabb_tests, 0);
        assert_eq!(naive.stats.aabb_tests, 4_950);
    }

    #[test]
    fn sweep_and_prune_matches_naive_on_every_axis() {
        let bodies = fixture();
        let expected = NaiveBroadPhase.detect(&bodies).pairs;

        for axis in [Axis3::X, Axis3::Y, Axis3::Z] {
            let actual = SweepAndPruneBroadPhase::new(axis).detect(&bodies);
            assert_eq!(actual.pairs, expected, "axis {axis:?}");
        }
    }

    #[test]
    fn sweep_and_prune_output_is_independent_of_input_order() {
        let bodies = fixture();
        let expected = SweepAndPruneBroadPhase::default().detect(&bodies).pairs;
        let mut reversed = bodies.clone();
        reversed.reverse();
        assert_eq!(
            SweepAndPruneBroadPhase::default().detect(&reversed).pairs,
            expected
        );
    }

    #[test]
    fn sparse_sweep_avoids_most_pair_tests() {
        let bodies: Vec<_> = (0..100)
            .map(|id| body(id, [id as f32 * 10.0, 0.0, 0.0], 0.25))
            .collect();

        let naive = NaiveBroadPhase.detect(&bodies);
        let sweep = SweepAndPruneBroadPhase::new(Axis3::X).detect(&bodies);

        assert_eq!(sweep.pairs, naive.pairs);
        assert_eq!(sweep.stats.aabb_tests, 0);
        assert_eq!(naive.stats.aabb_tests, 4_950);
    }

    #[test]
    fn uniform_grid_and_public_spatial_hash_share_cell_semantics() {
        let hash = SpatialHash3D::new(2.0);
        let bounds = Aabb::new([-0.1, 0.0, 0.0], [2.0, 0.5, 0.5]);
        assert_eq!(
            hash.cell_bounds(bounds),
            (CellCoord3::new(-1, 0, 0), CellCoord3::new(1, 0, 0))
        );
    }
}
