//! **Experimental** fixed-topology regular tetrahedron and octahedron kernels (issue #188).
//!
//! This module is a measurement fixture, not part of the [`super::PrimitiveShape3`] pair
//! matrix: it has no discrete contact query, no dispatch through [`super::try_swept_time`]
//! and no stability promise. It exists to decide whether exact shape-specific swept SAT for
//! these solids is worth promoting; see `docs/platonic-swept-sat.md`.
//!
//! # Convention
//!
//! Both solids are centred on their centroid and scaled by their circumradius `R`.
//!
//! - Tetrahedron: local vertices `R * s / sqrt(3)` for `s` in `(1,1,1)`, `(1,-1,-1)`,
//!   `(-1,1,-1)`, `(-1,-1,1)`. Face `i` is opposite vertex `i` with outward normal
//!   `-s_i / sqrt(3)`; inradius `R / 3`, edge length `R * sqrt(8/3)`.
//! - Octahedron: local vertices `±R e_x`, `±R e_y`, `±R e_z`. Its eight faces have outward
//!   normals `(±1,±1,±1) / sqrt(3)`; inradius `R / sqrt(3)`, edge length `R * sqrt(2)`.
//! - Box: the comparison operand, identical geometry to [`super::PrimitiveShape3::cuboid`].
//!
//! Both regular solids share the six canonical edge directions `(e_i ± e_j) / sqrt(2)`.
//! [`PlatonicBody3::axes`] maps local X/Y/Z to world space and must be orthonormal.
//!
//! # Exactness
//!
//! [`try_platonic_swept_interval`] is exact (up to floating-point rounding) only for
//! fixed-orientation **translation**: every face normal of both solids plus every edge-pair
//! cross product spans the face normals of the Minkowski difference. Independently rotating
//! solids need a different algorithm. Near-parallel edge crosses are dropped (dropping an
//! axis can only admit contact, never hide it); duplicate axes are retested, not removed.

use super::frame::{inverse_rotate, rotate_local};
use super::sweep::{PrimitiveSweepError3, swept_sat_interval};
use crate::math3::{Vec3, add, cross, dot, is_finite, length_squared, scale, sub};

const INV_SQRT_3: f64 = 0.577_350_269_189_625_8;
const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

const TETRAHEDRON_SIGNS: [Vec3; 4] = [
    [1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
];

const UNIT_TETRAHEDRON_VERTICES: [Vec3; 4] = [
    [INV_SQRT_3, INV_SQRT_3, INV_SQRT_3],
    [INV_SQRT_3, -INV_SQRT_3, -INV_SQRT_3],
    [-INV_SQRT_3, INV_SQRT_3, -INV_SQRT_3],
    [-INV_SQRT_3, -INV_SQRT_3, INV_SQRT_3],
];

/// Outward normal of the face opposite each vertex.
const TETRAHEDRON_FACE_NORMALS: [Vec3; 4] = [
    [-INV_SQRT_3, -INV_SQRT_3, -INV_SQRT_3],
    [-INV_SQRT_3, INV_SQRT_3, INV_SQRT_3],
    [INV_SQRT_3, -INV_SQRT_3, INV_SQRT_3],
    [INV_SQRT_3, INV_SQRT_3, -INV_SQRT_3],
];

const UNIT_OCTAHEDRON_VERTICES: [Vec3; 6] = [
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
];

/// One outward normal per antipodal face pair (first component positive).
const OCTAHEDRON_FACE_AXES: [Vec3; 4] = [
    [INV_SQRT_3, INV_SQRT_3, INV_SQRT_3],
    [INV_SQRT_3, INV_SQRT_3, -INV_SQRT_3],
    [INV_SQRT_3, -INV_SQRT_3, INV_SQRT_3],
    [INV_SQRT_3, -INV_SQRT_3, -INV_SQRT_3],
];

/// Unique edge directions shared by both regular solids (first non-zero component positive).
const REGULAR_EDGE_DIRECTIONS: [Vec3; 6] = [
    [0.0, INV_SQRT_2, INV_SQRT_2],
    [0.0, INV_SQRT_2, -INV_SQRT_2],
    [INV_SQRT_2, 0.0, INV_SQRT_2],
    [INV_SQRT_2, 0.0, -INV_SQRT_2],
    [INV_SQRT_2, INV_SQRT_2, 0.0],
    [INV_SQRT_2, -INV_SQRT_2, 0.0],
];

const BOX_AXES: [Vec3; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Edge crosses at or below this squared length (unit edges: sine of the angle) are
/// dropped as parallel edges; matches the existing `primitive3` polytope SAT.
const PARALLEL_EDGE_CROSS_SQUARED: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PlatonicKind3 {
    Tetrahedron,
    Octahedron,
    Box,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlatonicShape3 {
    kind: PlatonicKind3,
    /// Circumradius for the regular solids, half extents for the box.
    size: Vec3,
}

impl PlatonicShape3 {
    /// Regular tetrahedron with the given circumradius.
    #[must_use]
    pub fn tetrahedron(circumradius: f64) -> Self {
        Self::regular(PlatonicKind3::Tetrahedron, circumradius)
    }

    /// Regular octahedron with the given circumradius.
    #[must_use]
    pub fn octahedron(circumradius: f64) -> Self {
        Self::regular(PlatonicKind3::Octahedron, circumradius)
    }

    /// Box comparison operand with the same geometry as `PrimitiveShape3::cuboid`.
    #[must_use]
    pub fn cuboid(half_extents: Vec3) -> Self {
        assert!(
            half_extents
                .into_iter()
                .all(|value| value.is_finite() && value > 0.0),
            "box half extents must be positive and finite"
        );
        Self {
            kind: PlatonicKind3::Box,
            size: half_extents,
        }
    }

    fn regular(kind: PlatonicKind3, circumradius: f64) -> Self {
        assert!(
            circumradius.is_finite() && circumradius > 0.0,
            "circumradius must be positive and finite"
        );
        Self {
            kind,
            size: [circumradius; 3],
        }
    }

    #[must_use]
    pub const fn kind(self) -> PlatonicKind3 {
        self.kind
    }

    /// Circumradius (box: half-diagonal length).
    #[must_use]
    pub fn circumradius(self) -> f64 {
        match self.kind {
            PlatonicKind3::Box => crate::math3::length(self.size),
            PlatonicKind3::Tetrahedron | PlatonicKind3::Octahedron => self.size[0],
        }
    }

    /// Local vertices in the documented order (box: sign-pattern order `x` fastest).
    #[must_use]
    pub fn local_vertices(self) -> ([Vec3; 8], usize) {
        let mut out = [[0.0; 3]; 8];
        match self.kind {
            PlatonicKind3::Tetrahedron => {
                for (slot, vertex) in out.iter_mut().zip(UNIT_TETRAHEDRON_VERTICES) {
                    *slot = scale(vertex, self.size[0]);
                }
                (out, 4)
            }
            PlatonicKind3::Octahedron => {
                for (slot, vertex) in out.iter_mut().zip(UNIT_OCTAHEDRON_VERTICES) {
                    *slot = scale(vertex, self.size[0]);
                }
                (out, 6)
            }
            PlatonicKind3::Box => {
                for (index, slot) in out.iter_mut().enumerate() {
                    *slot = std::array::from_fn(|axis| {
                        if index >> axis & 1 == 0 {
                            -self.size[axis]
                        } else {
                            self.size[axis]
                        }
                    });
                }
                (out, 8)
            }
        }
    }

    /// Canonical SAT face axes, one per face orientation up to sign.
    ///
    /// Tetrahedron: the four outward normals, face `i` opposite vertex `i`. Octahedron and
    /// box: one outward normal per antipodal face pair.
    #[must_use]
    pub fn local_face_axes(self) -> &'static [Vec3] {
        match self.kind {
            PlatonicKind3::Tetrahedron => &TETRAHEDRON_FACE_NORMALS,
            PlatonicKind3::Octahedron => &OCTAHEDRON_FACE_AXES,
            PlatonicKind3::Box => &BOX_AXES,
        }
    }

    /// Unique unit edge directions up to sign.
    #[must_use]
    pub fn local_edge_directions(self) -> &'static [Vec3] {
        match self.kind {
            PlatonicKind3::Tetrahedron | PlatonicKind3::Octahedron => &REGULAR_EDGE_DIRECTIONS,
            PlatonicKind3::Box => &BOX_AXES,
        }
    }

    /// Local `[min, max]` projection onto a local direction, counting candidates scored.
    fn local_extremes(self, direction: Vec3, work: &mut PlatonicWork3) -> (f64, f64) {
        match self.kind {
            PlatonicKind3::Tetrahedron => {
                work.vertex_tests += 4;
                let [x, y, z] = direction;
                let candidates = [x + y + z, x - y - z, -x + y - z, -x - y + z];
                let mut low = candidates[0];
                let mut high = candidates[0];
                for value in &candidates[1..] {
                    low = low.min(*value);
                    high = high.max(*value);
                }
                let unit = self.size[0] * INV_SQRT_3;
                (low * unit, high * unit)
            }
            PlatonicKind3::Octahedron => {
                work.vertex_tests += 3;
                let reach = direction.into_iter().map(f64::abs).fold(0.0, f64::max) * self.size[0];
                (-reach, reach)
            }
            PlatonicKind3::Box => {
                work.vertex_tests += 3;
                let reach = self.size[0] * direction[0].abs()
                    + self.size[1] * direction[1].abs()
                    + self.size[2] * direction[2].abs();
                (-reach, reach)
            }
        }
    }

    /// Local support vertex; ties resolve to the lowest documented vertex index.
    fn local_support(self, direction: Vec3, work: &mut PlatonicWork3) -> Vec3 {
        match self.kind {
            PlatonicKind3::Tetrahedron => {
                work.vertex_tests += 4;
                let mut best = 0;
                let mut best_value = dot(TETRAHEDRON_SIGNS[0], direction);
                for (index, signs) in TETRAHEDRON_SIGNS.iter().enumerate().skip(1) {
                    let value = dot(*signs, direction);
                    if value > best_value {
                        best = index;
                        best_value = value;
                    }
                }
                scale(UNIT_TETRAHEDRON_VERTICES[best], self.size[0])
            }
            PlatonicKind3::Octahedron => {
                work.vertex_tests += 3;
                let mut axis = 0;
                for candidate in 1..3 {
                    if direction[candidate].abs() > direction[axis].abs() {
                        axis = candidate;
                    }
                }
                let mut out = [0.0; 3];
                out[axis] = if direction[axis] < 0.0 {
                    -self.size[0]
                } else {
                    self.size[0]
                };
                out
            }
            PlatonicKind3::Box => {
                work.vertex_tests += 3;
                std::array::from_fn(|axis| {
                    if direction[axis] < 0.0 {
                        -self.size[axis]
                    } else {
                        self.size[axis]
                    }
                })
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlatonicBody3 {
    pub shape: PlatonicShape3,
    pub position: Vec3,
    /// World-space local X/Y/Z axes; must be orthonormal.
    pub axes: [Vec3; 3],
    pub velocity: Vec3,
}

impl PlatonicBody3 {
    #[must_use]
    pub const fn new(
        shape: PlatonicShape3,
        position: Vec3,
        axes: [Vec3; 3],
        velocity: Vec3,
    ) -> Self {
        Self {
            shape,
            position,
            axes,
            velocity,
        }
    }

    #[must_use]
    pub const fn axis_aligned(shape: PlatonicShape3, position: Vec3, velocity: Vec3) -> Self {
        Self::new(shape, position, BOX_AXES, velocity)
    }

    /// World support vertex for `direction`; ties resolve to the lowest vertex index
    /// (box and octahedron: a zero component selects the positive side).
    #[must_use]
    pub fn support_point(self, direction: Vec3, work: &mut PlatonicWork3) -> Vec3 {
        work.support_queries += 1;
        let local = self
            .shape
            .local_support(inverse_rotate(self.axes, direction), work);
        add(self.position, rotate_local(self.axes, local))
    }

    /// World vertices at `time` seconds along the body's velocity.
    #[must_use]
    pub fn world_vertices(self, time: f64) -> ([Vec3; 8], usize) {
        let (mut vertices, len) = self.shape.local_vertices();
        let origin = add(self.position, scale(self.velocity, time));
        for vertex in &mut vertices[..len] {
            *vertex = add(origin, rotate_local(self.axes, *vertex));
        }
        (vertices, len)
    }

    fn projection(self, axis: Vec3, work: &mut PlatonicWork3) -> (f64, f64) {
        // Minimum and maximum each count as one directional support query, matching
        // `PrimitiveWork3::support_evaluations`; the specialization scores both in one pass.
        work.support_queries += 2;
        let center = dot(self.position, axis);
        let (low, high) = self
            .shape
            .local_extremes(inverse_rotate(self.axes, axis), work);
        (center + low, center + high)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlatonicWork3 {
    /// Swept-SAT axes inspected, including the one that proves a miss.
    pub axes_tested: u64,
    /// Directional support queries (a projection interval counts two).
    pub support_queries: u64,
    /// Vertices (tetrahedron) or per-axis extent terms (octahedron, box) scored.
    pub vertex_tests: u64,
}

/// Normalized contact interval over the sweep window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlatonicSweepInterval3 {
    /// First time in `[0, 1]` at which separation is at most the margin.
    pub enter: f64,
    /// Last such time, clipped to the window end `1`.
    pub exit: f64,
}

/// Exact fixed-orientation translational swept SAT for tetrahedron, octahedron and box pairs.
///
/// Returns the normalized contact interval of `a` and `b` over `dt`, `Ok(None)` when a
/// separating axis holds for the whole window, and errors under the same contract as
/// [`super::try_swept_time`]: `max_iterations` bounds inspected axes, and a failed search is
/// never a miss. Initial contact reports `enter == 0`. Angular motion is not swept.
pub fn try_platonic_swept_interval(
    a: PlatonicBody3,
    b: PlatonicBody3,
    dt: f64,
    margin: f64,
    max_iterations: u32,
    work: &mut PlatonicWork3,
) -> Result<Option<PlatonicSweepInterval3>, PrimitiveSweepError3> {
    if !dt.is_finite()
        || dt < 0.0
        || !margin.is_finite()
        || margin < 0.0
        || [a, b].into_iter().any(|body| {
            !is_finite(body.position)
                || !is_finite(body.velocity)
                || !body.axes.into_iter().all(is_finite)
        })
    {
        return Err(PrimitiveSweepError3::InvalidInput);
    }
    let displacement = scale(sub(b.velocity, a.velocity), dt);
    if !is_finite(displacement) {
        return Err(PrimitiveSweepError3::NonFiniteComputation);
    }
    // Work in A's frame: A's projections need no rotation and the relative pose is formed
    // once. Cross axes stay unnormalized (interval times are scale-invariant per axis); the
    // margin is folded into A's interval scaled by the axis length instead.
    let relative_axes = b.axes.map(|axis| inverse_rotate(a.axes, axis));
    let relative_position = inverse_rotate(a.axes, sub(b.position, a.position));
    let local_displacement = inverse_rotate(a.axes, displacement);
    if !is_finite(relative_position) || !relative_axes.into_iter().all(is_finite) {
        return Err(PrimitiveSweepError3::NonFiniteComputation);
    }
    let b_local = PlatonicBody3 {
        position: relative_position,
        axes: relative_axes,
        ..b
    };
    let interval = swept_sat_interval(
        candidate_axes(a.shape, b_local),
        local_displacement,
        0.0,
        max_iterations,
        |axis| {
            work.axes_tested += 1;
            work.support_queries += 2;
            let (a_min, a_max) = a.shape.local_extremes(axis, work);
            let reach = if margin > 0.0 {
                margin * length_squared(axis).sqrt()
            } else {
                0.0
            };
            let (b_min, b_max) = b_local.projection(axis, work);
            [a_min - reach, a_max + reach, b_min, b_max]
        },
    )?;
    Ok(interval.map(|(enter, exit)| PlatonicSweepInterval3 { enter, exit }))
}

/// Lazily yields, in `A`'s local frame, the face axes of `A`, then of `B`, then
/// `A`-edge x `B`-edge crosses in index order. Duplicates are not removed: retesting one is
/// cheaper than an O(n^2) scan and cannot change the interval. Parallel-edge crosses are
/// skipped; crosses are left unnormalized.
fn candidate_axes(a: PlatonicShape3, b: PlatonicBody3) -> impl Iterator<Item = Vec3> {
    let b_world =
        move |local: &'static [Vec3]| local.iter().map(move |&axis| rotate_local(b.axes, axis));
    let faces = a
        .local_face_axes()
        .iter()
        .copied()
        .chain(b_world(b.shape.local_face_axes()));
    let crosses = a.local_edge_directions().iter().flat_map(move |&left| {
        b_world(b.shape.local_edge_directions())
            .map(move |right| cross(left, right))
            .filter(|axis| length_squared(*axis) > PARALLEL_EDGE_CROSS_SQUARED)
    });
    faces.chain(crosses)
}
