//! Reusable spatial kernels with deterministic outputs.
//!
//! The MVP deliberately keeps the representation small: axis-aligned bounding
//! boxes, a brute-force reference broad phase, and a uniform-grid broad phase.

use std::collections::{BTreeSet, HashMap, HashSet};

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
        assert!(half_extents.iter().all(|extent| *extent >= 0.0));
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

    fn cell_bounds(self, cell_size: f32) -> (Cell, Cell) {
        let cell = |value: f32| (value / cell_size).floor() as i32;
        (
            Cell([cell(self.min[0]), cell(self.min[1]), cell(self.min[2])]),
            Cell([cell(self.max[0]), cell(self.max[1]), cell(self.max[2])]),
        )
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Cell([i32; 3]);

/// Uniform 3D grid. Bodies are inserted into every cell touched by their AABB;
/// only bodies sharing a cell can become candidate pairs.
#[derive(Clone, Copy, Debug)]
pub struct UniformGridBroadPhase {
    cell_size: f32,
}

impl UniformGridBroadPhase {
    #[must_use]
    pub fn new(cell_size: f32) -> Self {
        assert!(
            cell_size.is_finite() && cell_size > 0.0,
            "cell size must be positive and finite"
        );
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
        let mut cells: HashMap<Cell, Vec<usize>> = HashMap::new();

        for (body_index, body) in bodies.iter().enumerate() {
            let (min, max) = body.aabb.cell_bounds(self.cell_size);
            for x in min.0[0]..=max.0[0] {
                for y in min.0[1]..=max.0[1] {
                    for z in min.0[2]..=max.0[2] {
                        cells.entry(Cell([x, y, z])).or_default().push(body_index);
                    }
                }
            }
        }

        // A large body can share multiple cells with the same neighbor. Test
        // each body pair only once, then sort public output deterministically.
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
    fn pair_order_is_canonical() {
        assert_eq!(Pair::new(9, 2), Pair { a: 2, b: 9 });
    }

    #[test]
    fn grid_matches_naive_across_cell_sizes() {
        let bodies = vec![
            body(10, [0.0, 0.0, 0.0], 0.6),
            body(20, [0.9, 0.0, 0.0], 0.6),
            body(30, [4.0, 4.0, 4.0], 0.5),
            body(40, [-0.8, 0.0, 0.0], 0.6),
            body(50, [1.5, 0.0, 0.0], 1.2),
        ];
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
}
