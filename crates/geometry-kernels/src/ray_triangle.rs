//! Bounded ray- and segment-vs-triangle intersection.
//!
//! The query uses the [`Ray3`] convention: hit parameters are world-space
//! distances along the unit internal direction. Segments report the
//! normalized fraction between their endpoints instead.
//!
//! Semantics:
//!
//! - **Facing.** A triangle `[a, b, c]` faces the side its counter-clockwise
//!   normal `(b - a) × (c - a)` points to. A ray travelling against that
//!   normal hits the front face. [`TriangleCulling::BackFaces`] rejects back
//!   face hits; [`TriangleCulling::None`] reports both with their facing.
//! - **Boundaries are inclusive.** Hits exactly on an edge or a vertex, at
//!   distance zero (origin on the triangle) and exactly at the maximum
//!   distance (segment endpoints) count, and every other point misses. The
//!   inside, behind-the-origin and beyond-the-bound tests use the exact signs
//!   of their triple-product numerators for the f64 inputs (the widened f32
//!   corners and origin, the unit ray direction, or the exact segment vector
//!   `end - start` even where f64 rounds it):
//!   each numerator is computed in f64 and kept when it clears a
//!   componentwise forward error bound, and is otherwise evaluated exactly
//!   with error-free transformations (two-sum, two-product) into a
//!   fixed-size floating-point expansion. The decision is therefore exact
//!   for ill-conditioned (sliver) triangles too, needs no tolerance, and is
//!   deterministic. Only the reported distance, point and weights are
//!   rounded. Segments are intersected along their raw vector, so the
//!   endpoint bound is compared in fraction space.
//! - **Parallel rays miss.** A ray whose direction lies in the triangle's
//!   plane (within a relative tolerance), including a coplanar ray crossing
//!   the triangle, reports no hit: a coplanar contact has no unique distance
//!   or facing.
//! - **Degenerate triangles miss.** Triangles with zero area (repeated or
//!   collinear corners) have no plane and never hit.
//!
//! The kernel does not allocate. Mesh traversal, BVH ordering and collision
//! response stay with the consumer.

use crate::{
    Ray3,
    math3::{Vec3, add, cross, dot, length_squared, scale, sub},
};

/// Which faces a triangle query may hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriangleCulling {
    /// Report front and back face hits.
    None,
    /// Reject hits on the back face (the ray travels along the normal).
    BackFaces,
}

/// Which side of the triangle the ray hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriangleFacing {
    /// The ray travels against the counter-clockwise normal.
    Front,
    /// The ray travels along the counter-clockwise normal.
    Back,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayTriangleHit {
    /// World-space distance along the ray, in `[0, max_distance]`.
    pub distance: f64,
    pub point: Vec3,
    /// Barycentric weights of corners `a`, `b`, `c`; each is in `[0, 1]` and
    /// they sum to one up to rounding.
    pub barycentric: [f64; 3],
    pub facing: TriangleFacing,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentTriangleHit {
    /// Normalized position between the segment's start (0) and end (1).
    pub fraction: f64,
    pub point: Vec3,
    pub barycentric: [f64; 3],
    pub facing: TriangleFacing,
}

/// Relative tolerance below which a ray counts as parallel to the plane: the
/// sine of the angle between direction and plane, scaled by rounding error.
const PARALLEL_TOLERANCE: f64 = 16.0 * f64::EPSILON;

/// First intersection of `ray` with `triangle` within `max_distance`
/// (inclusive). `max_distance` must be non-negative; it may be infinite.
#[must_use]
pub fn ray_triangle(
    ray: Ray3,
    triangle: [[f32; 3]; 3],
    max_distance: f64,
    culling: TriangleCulling,
) -> Option<RayTriangleHit> {
    assert!(
        max_distance >= 0.0,
        "ray-triangle maximum distance must be non-negative"
    );
    let [a, b, c] = triangle.map(|corner| corner.map(f64::from));
    ray_triangle_f64(
        ray.origin64(),
        ray.direction(),
        [a, b, c],
        max_distance,
        culling,
    )
}

/// Intersection of the closed segment from `start` to `end` with `triangle`.
/// A zero-length segment has no direction and never hits.
#[must_use]
pub fn segment_triangle(
    start: [f32; 3],
    end: [f32; 3],
    triangle: [[f32; 3]; 3],
    culling: TriangleCulling,
) -> Option<SegmentTriangleHit> {
    assert!(
        start
            .iter()
            .chain(end.iter())
            .all(|value| value.is_finite()),
        "segment endpoints must be finite"
    );
    let origin = start.map(f64::from);
    // The exact segment vector: f64 can round `end - start` when the
    // coordinates differ widely in magnitude, so the exact fallbacks keep the
    // tail and the endpoint bound stays exact.
    let exact_delta = exact_difference(end.map(f64::from), origin);
    let delta = exact_delta.map(|[high, _]| high);
    if delta == [0.0; 3] {
        return None;
    }
    let [a, b, c] = triangle.map(|corner| corner.map(f64::from));
    // Intersect along the raw segment vector so the parameter already is the
    // fraction and the endpoint bound is compared without normalization
    // rounding.
    let (fraction, barycentric, facing) =
        intersect_exact_direction(origin, exact_delta, [a, b, c], 1.0, culling)?;
    Some(SegmentTriangleHit {
        fraction,
        point: add(origin, scale(delta, fraction)),
        barycentric,
        facing,
    })
}

fn ray_triangle_f64(
    origin: Vec3,
    direction: Vec3,
    triangle: [Vec3; 3],
    max_distance: f64,
    culling: TriangleCulling,
) -> Option<RayTriangleHit> {
    let (distance, barycentric, facing) =
        intersect(origin, direction, triangle, max_distance, culling)?;
    Some(RayTriangleHit {
        distance,
        point: add(origin, scale(direction, distance)),
        barycentric,
        facing,
    })
}

/// Forward error bound of a computed triple-product numerator, in units of
/// its componentwise magnitude (see [`triple_magnitude`]). Forming the
/// differences, the cross product and the dot product perturbs a computed
/// triple product by at most about `7 ε` of that magnitude (Shewchuk's
/// `orient3d` bound), a segment vector rounded from its exact endpoint
/// difference adds at most half an ulp, and the derived numerators (`w` and
/// the bound gap) add at most two more roundings; twice that leaves margin
/// for rounding in the magnitude itself. Barring underflow, a computed numerator farther from zero
/// than this bound has the sign of the exact one.
const ERROR_BOUND: f64 = 16.0 * f64::EPSILON;

/// `Σ |x_i| (|y_j| |z_k| + |y_k| |z_j|)`: the componentwise magnitude of the
/// triple product `x · (y × z)`. Its rounding error is proportional to this
/// sum, not to `|x| |y × z|`, so terms that cancel exactly (an origin on a
/// vertex) and components that do not take part (near-parallel rays) get the
/// bound they actually need.
fn triple_magnitude(x: Vec3, y: Vec3, z: Vec3) -> f64 {
    let [x, y, z] = [x, y, z].map(|vector| vector.map(f64::abs));
    x[0] * (y[1] * z[2] + y[2] * z[1])
        + x[1] * (y[2] * z[0] + y[0] * z[2])
        + x[2] * (y[0] * z[1] + y[1] * z[0])
}

/// The triple-product numerators [`intersect`] decides on.
#[derive(Clone, Copy)]
enum Numerator {
    /// `(b - a) · (direction × (c - a))`: the determinant, `-(direction ·
    /// normal)`.
    Determinant,
    /// `(origin - a) · (direction × (c - a))`: the weight of `b`.
    U,
    /// `direction · ((origin - a) × (b - a))`: the weight of `c`.
    V,
    /// `direction · ((c - origin) × (b - origin))`: the weight of `a`.
    W,
    /// `(c - a) · ((origin - a) × (b - a))`: the hit parameter.
    T,
    /// `max_parameter · determinant - T`: the room left before the bound.
    BoundGap(f64),
}

/// Returns `value` when it is farther from zero than its forward error bound,
/// so its sign is certain; otherwise the exactly evaluated numerator, rounded
/// once. Either way the result has the sign of the exact numerator.
#[inline]
fn certain_sign(
    value: f64,
    error: f64,
    numerator: Numerator,
    origin: Vec3,
    direction: [Exact; 3],
    corners: [Vec3; 3],
) -> f64 {
    if value.abs() > error || error == 0.0 {
        value
    } else {
        exact_numerator(numerator, origin, direction, corners)
    }
}

/// `numerator` evaluated exactly (see [`exact_triple`]), rounded once. Kept
/// out of line so the fast path neither inlines it nor reserves stack for
/// its expansions.
#[cold]
#[inline(never)]
fn exact_numerator(
    numerator: Numerator,
    origin: Vec3,
    direction: [Exact; 3],
    [a, b, c]: [Vec3; 3],
) -> f64 {
    match numerator {
        Numerator::Determinant => determinant_numerator(direction, [a, b, c]).estimate(),
        Numerator::U => exact_triple(
            exact_difference(origin, a),
            direction,
            exact_difference(c, a),
        )
        .estimate(),
        Numerator::V => exact_triple(
            direction,
            exact_difference(origin, a),
            exact_difference(b, a),
        )
        .estimate(),
        Numerator::W => exact_triple(
            direction,
            exact_difference(c, origin),
            exact_difference(b, origin),
        )
        .estimate(),
        Numerator::T => parameter_numerator(origin, [a, b, c]).estimate(),
        Numerator::BoundGap(max_parameter) => {
            exact_bound_gap(origin, direction, [a, b, c], max_parameter)
        }
    }
}

fn determinant_numerator(direction: [Exact; 3], [a, b, c]: [Vec3; 3]) -> Expansion<TRIPLE_TERMS> {
    exact_triple(exact_difference(b, a), direction, exact_difference(c, a))
}

fn parameter_numerator(origin: Vec3, [a, b, c]: [Vec3; 3]) -> Expansion<TRIPLE_TERMS> {
    exact_triple(
        exact_difference(c, a),
        exact_difference(origin, a),
        exact_difference(b, a),
    )
}

/// Möller–Trumbore along `direction` (any non-zero length): the hit
/// parameter in units of `direction`, within `[0, max_parameter]`, with
/// barycentric weights and facing.
///
/// Inclusion is decided from the exact signs of the triple-product numerators
/// of the f64 inputs: each is computed in f64 and kept when it clears its
/// forward error bound, and evaluated exactly with error-free transformations
/// otherwise (see [`exact_triple`]). Hits exactly on vertices, edges, the
/// origin or the bound therefore count, and every other point outside misses,
/// however ill-conditioned the triangle.
fn intersect(
    origin: Vec3,
    direction: Vec3,
    triangle: [Vec3; 3],
    max_parameter: f64,
    culling: TriangleCulling,
) -> Option<(f64, [f64; 3], TriangleFacing)> {
    intersect_exact_direction(
        origin,
        exact_vector(direction),
        triangle,
        max_parameter,
        culling,
    )
}

/// [`intersect`] along a direction given exactly as `[high, low]` pairs. The
/// f64 fast path uses the high parts; the error bound covers the dropped low
/// parts (at most half an ulp each), and the exact fallbacks use both, so the
/// decision is exact for the unrounded direction.
fn intersect_exact_direction(
    origin: Vec3,
    exact_direction: [Exact; 3],
    [a, b, c]: [Vec3; 3],
    max_parameter: f64,
    culling: TriangleCulling,
) -> Option<(f64, [f64; 3], TriangleFacing)> {
    let direction = exact_direction.map(|[high, _]| high);
    assert!(
        [a, b, c].iter().flatten().all(|value| value.is_finite()),
        "triangle corners must be finite"
    );
    let edge_ab = sub(b, a);
    let edge_ac = sub(c, a);
    let normal = cross(edge_ab, edge_ac);
    let normal_length_squared = length_squared(normal);
    if normal_length_squared == 0.0 {
        return None;
    }
    let direction_length = length_squared(direction).sqrt();
    let p = cross(direction, edge_ac);
    // determinant = -(direction · normal) = u + v + w.
    let determinant = dot(edge_ab, p);
    if determinant.abs() <= PARALLEL_TOLERANCE * normal_length_squared.sqrt() * direction_length {
        return None;
    }
    let corners = [a, b, c];
    // The determinant's exact sign is the facing and orients every other
    // numerator so that inside is >= 0.
    let determinant_error = ERROR_BOUND * triple_magnitude(edge_ab, direction, edge_ac);
    let orientation = match certain_sign(
        determinant,
        determinant_error,
        Numerator::Determinant,
        origin,
        exact_direction,
        corners,
    ) {
        0.0 => return None,
        value => value.signum(),
    };
    let facing = if orientation > 0.0 {
        TriangleFacing::Front
    } else {
        TriangleFacing::Back
    };
    if culling == TriangleCulling::BackFaces && facing == TriangleFacing::Back {
        return None;
    }
    let to_origin = sub(origin, a);
    let q = cross(to_origin, edge_ab);
    let u_error = ERROR_BOUND * triple_magnitude(to_origin, direction, edge_ac);
    let v_error = ERROR_BOUND * triple_magnitude(direction, to_origin, edge_ab);
    let t_error = ERROR_BOUND * triple_magnitude(edge_ac, to_origin, edge_ab);

    // Edge numerators (weights of b, c and a times the determinant), each
    // with the sign of its exact value. They sum exactly to the determinant,
    // so the point is inside the closed triangle exactly when none opposes
    // its sign.
    let u_raw = dot(to_origin, p);
    let u = certain_sign(
        u_raw,
        u_error,
        Numerator::U,
        origin,
        exact_direction,
        corners,
    );
    if u * orientation < 0.0 {
        return None;
    }
    let v_raw = dot(direction, q);
    let v = certain_sign(
        v_raw,
        v_error,
        Numerator::V,
        origin,
        exact_direction,
        corners,
    );
    if v * orientation < 0.0 {
        return None;
    }
    // w = determinant - u - v = direction · ((c - origin) × (b - origin)).
    let w = certain_sign(
        determinant - u_raw - v_raw,
        determinant_error + u_error + v_error,
        Numerator::W,
        origin,
        exact_direction,
        corners,
    );
    if w * orientation < 0.0 {
        return None;
    }

    // Parameter numerator: behind the origin when its sign opposes the
    // determinant's.
    let t_raw = dot(edge_ac, q);
    let t = certain_sign(
        t_raw,
        t_error,
        Numerator::T,
        origin,
        exact_direction,
        corners,
    );
    if t * orientation < 0.0 {
        return None;
    }
    if max_parameter.is_finite() {
        // Beyond the bound when max_parameter · determinant - t opposes the
        // determinant's sign.
        let gap = certain_sign(
            max_parameter * determinant - t_raw,
            max_parameter * determinant_error + t_error,
            Numerator::BoundGap(max_parameter),
            origin,
            exact_direction,
            corners,
        );
        if gap * orientation < 0.0 {
            return None;
        }
    }
    // All four numerators now carry the determinant's sign (or are zero), so
    // the oriented values are non-negative; u + v + w is the determinant.
    let [u, v, w, t] = [u, v, w, t].map(|value| (value * orientation).abs());
    let total = u + v + w;
    let parameter = (t / total).min(max_parameter);
    Some((parameter, [w / total, u / total, v / total], facing))
}

/// `max_parameter · determinant - t` evaluated exactly, rounded once. A
/// `max_parameter · determinant` beyond the f64 range is reported as an
/// infinite gap with the determinant's sign, which is the exact sign because
/// the parameter numerator is finite.
fn exact_bound_gap(
    origin: Vec3,
    direction: [Exact; 3],
    [a, b, c]: [Vec3; 3],
    max_parameter: f64,
) -> f64 {
    let determinant = determinant_numerator(direction, [a, b, c]);
    let parameter = parameter_numerator(origin, [a, b, c]);
    let mut gap = Expansion::<{ 3 * TRIPLE_TERMS }>::new();
    for &term in parameter.terms() {
        gap.add(-term);
    }
    for &term in determinant.terms() {
        let [product, error] = two_product(max_parameter, term);
        if !product.is_finite() {
            return determinant.estimate().signum() * f64::INFINITY;
        }
        gap.add(error);
        gap.add(product);
    }
    gap.estimate()
}

/// An f64 value as an exact `[high, low]` pair: `high + low` is the value.
type Exact = [f64; 2];

fn exact_vector(vector: Vec3) -> [Exact; 3] {
    vector.map(|value| [value, 0.0])
}

/// `left - right` per component, without rounding.
fn exact_difference(left: Vec3, right: Vec3) -> [Exact; 3] {
    std::array::from_fn(|axis| two_difference(left[axis], right[axis]))
}

/// Knuth's two-sum: `[a + b rounded, exact rounding error]`.
fn two_sum(a: f64, b: f64) -> Exact {
    let sum = a + b;
    let b_virtual = sum - a;
    let a_virtual = sum - b_virtual;
    [sum, (a - a_virtual) + (b - b_virtual)]
}

fn two_difference(a: f64, b: f64) -> Exact {
    let difference = a - b;
    let b_virtual = a - difference;
    let a_virtual = difference + b_virtual;
    [difference, (a - a_virtual) + (b_virtual - b)]
}

/// `[a · b rounded, exact rounding error]` via a fused multiply-add, which
/// rounds once on every target.
fn two_product(a: f64, b: f64) -> Exact {
    let product = a * b;
    [product, a.mul_add(b, -product)]
}

/// Upper bound on the components of an exact triple product: six monomials,
/// each a product of three `[high, low]` pairs (eight parts), each part two
/// error-free products deep (four terms).
const TRIPLE_TERMS: usize = 6 * 8 * 4;

/// `x · (y × z)` of exact vectors as a nonoverlapping expansion whose sum is
/// the exact value, barring f64 underflow and overflow: products of three
/// f32-sized differences (or two and a unit direction) stay far inside the
/// f64 exponent range. Only the fallback paths call it.
fn exact_triple(x: [Exact; 3], y: [Exact; 3], z: [Exact; 3]) -> Expansion<TRIPLE_TERMS> {
    let mut sum = Expansion::new();
    for [i, j, k] in [[0, 1, 2], [1, 2, 0], [2, 0, 1]] {
        // x_i (y_j z_k - y_k z_j)
        for (y_axis, z_axis, sign) in [(j, k, 1.0), (k, j, -1.0)] {
            for x_part in x[i] {
                for y_part in y[y_axis] {
                    for z_part in z[z_axis] {
                        if x_part == 0.0 || y_part == 0.0 || z_part == 0.0 {
                            continue;
                        }
                        let [high, low] = two_product(sign * x_part, y_part);
                        for term in two_product(high, z_part)
                            .into_iter()
                            .chain(two_product(low, z_part))
                        {
                            sum.add(term);
                        }
                    }
                }
            }
        }
    }
    sum
}

/// Shewchuk's floating-point expansion with fixed capacity: nonoverlapping
/// components in increasing magnitude whose exact sum is the value, so the
/// largest component carries its sign. Each [`Expansion::add`] grows it by at
/// most one component.
struct Expansion<const N: usize> {
    terms: [f64; N],
    len: usize,
}

impl<const N: usize> Expansion<N> {
    const fn new() -> Self {
        Self {
            terms: [0.0; N],
            len: 0,
        }
    }

    fn terms(&self) -> &[f64] {
        &self.terms[..self.len]
    }

    /// Adds `value` exactly (`GROW-EXPANSION` with zero elimination).
    fn add(&mut self, value: f64) {
        if value == 0.0 {
            return;
        }
        let mut carry = value;
        let mut kept = 0;
        for index in 0..self.len {
            let [sum, error] = two_sum(carry, self.terms[index]);
            if error != 0.0 {
                self.terms[kept] = error;
                kept += 1;
            }
            carry = sum;
        }
        if carry != 0.0 {
            self.terms[kept] = carry;
            kept += 1;
        }
        self.len = kept;
    }

    /// The value rounded once, which has the exact value's sign.
    fn estimate(&self) -> f64 {
        let Some(&largest) = self.terms().last() else {
            return 0.0;
        };
        let sum: f64 = self.terms().iter().sum();
        // The lower components sum to less than the largest; guard against
        // rounding that sum up to cancel it.
        if sum * largest > 0.0 { sum } else { largest }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIANGLE: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];

    #[test]
    fn near_parallel_rays_outside_the_triangle_miss() {
        // Beyond the parallel cutoff, but u = -0.1: a norm-based allowance
        // would admit this as a distance-zero hit.
        let hit = intersect(
            [-0.1, 0.5, 0.0],
            [1.0, 0.0, -4e-15],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            f64::INFINITY,
            TriangleCulling::None,
        );
        assert!(hit.is_none(), "{hit:?}");
    }

    #[test]
    fn origins_on_vertices_of_ill_conditioned_triangles_hit_at_zero() {
        let triangle = [
            [9.8e-6, -9e-10, 9000.0],
            [6.8e-24, 4.9e-25, 2.7e-32],
            [0.0, 0.0, 0.0],
        ];
        for direction in [[4e15, -7.9e-5, 9.4e19], [-4e15, 7.9e-5, -9.4e19]] {
            let (parameter, barycentric, _) = intersect(
                [0.0, 0.0, 0.0],
                direction,
                triangle,
                f64::INFINITY,
                TriangleCulling::None,
            )
            .expect("origin on vertex c is a distance-zero hit");
            // The exact parameter is 0; this triangle is so ill-conditioned
            // that only a rounding-sized parameter can be resolved.
            assert!((0.0..1e-18).contains(&parameter), "{parameter}");
            assert!(
                barycentric
                    .iter()
                    .all(|weight| (0.0..=1.0).contains(weight))
            );
        }
    }

    fn down(x: f32, y: f32) -> Ray3 {
        Ray3::new([x, y, 3.0], [0.0, 0.0, -1.0])
    }

    #[test]
    fn front_and_back_faces_follow_the_counter_clockwise_normal() {
        let front = ray_triangle(
            down(0.5, 0.5),
            TRIANGLE,
            f64::INFINITY,
            TriangleCulling::None,
        )
        .unwrap();
        assert_eq!(front.distance, 3.0);
        assert_eq!(front.point, [0.5, 0.5, 0.0]);
        assert_eq!(front.barycentric, [0.5, 0.25, 0.25]);
        assert_eq!(front.facing, TriangleFacing::Front);

        let up = Ray3::new([0.5, 0.5, -3.0], [0.0, 0.0, 1.0]);
        let back = ray_triangle(up, TRIANGLE, f64::INFINITY, TriangleCulling::None).unwrap();
        assert_eq!(back.facing, TriangleFacing::Back);
        assert_eq!(back.distance, 3.0);
        assert!(ray_triangle(up, TRIANGLE, f64::INFINITY, TriangleCulling::BackFaces).is_none());
        assert!(
            ray_triangle(
                down(0.5, 0.5),
                TRIANGLE,
                f64::INFINITY,
                TriangleCulling::BackFaces
            )
            .is_some()
        );
    }

    #[test]
    fn edges_and_vertices_are_inclusive() {
        for (x, y) in [
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (0.0, 0.0),
            (2.0, 0.0),
            (0.0, 2.0),
        ] {
            let hit = ray_triangle(down(x, y), TRIANGLE, f64::INFINITY, TriangleCulling::None);
            assert!(hit.is_some(), "boundary point ({x}, {y}) must hit");
        }
        let vertex = ray_triangle(down(2.0, 0.0), TRIANGLE, 10.0, TriangleCulling::None).unwrap();
        assert_eq!(vertex.barycentric, [0.0, 1.0, 0.0]);
        for (x, y) in [(-0.001, 0.5), (0.5, -0.001), (1.001, 1.0), (2.001, 0.0)] {
            assert!(
                ray_triangle(down(x, y), TRIANGLE, f64::INFINITY, TriangleCulling::None).is_none(),
                "outside point ({x}, {y}) must miss"
            );
        }
    }

    #[test]
    fn parallel_coplanar_and_degenerate_cases_miss() {
        let parallel = Ray3::new([-1.0, 0.5, 1.0], [1.0, 0.0, 0.0]);
        assert!(ray_triangle(parallel, TRIANGLE, f64::INFINITY, TriangleCulling::None).is_none());
        let coplanar = Ray3::new([-1.0, 0.5, 0.0], [1.0, 0.0, 0.0]);
        assert!(ray_triangle(coplanar, TRIANGLE, f64::INFINITY, TriangleCulling::None).is_none());
        for degenerate in [
            [[0.0; 3], [0.0; 3], [0.0; 3]],
            [[0.0; 3], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0; 3], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
        ] {
            assert!(
                ray_triangle(
                    down(0.5, 0.5),
                    degenerate,
                    f64::INFINITY,
                    TriangleCulling::None
                )
                .is_none()
            );
        }
    }

    #[test]
    fn distance_bounds_are_inclusive_and_behind_the_origin_misses() {
        assert!(ray_triangle(down(0.5, 0.5), TRIANGLE, 3.0, TriangleCulling::None).is_some());
        assert!(ray_triangle(down(0.5, 0.5), TRIANGLE, 2.999, TriangleCulling::None).is_none());
        let away = Ray3::new([0.5, 0.5, 3.0], [0.0, 0.0, 1.0]);
        assert!(ray_triangle(away, TRIANGLE, f64::INFINITY, TriangleCulling::None).is_none());
        // Origin on the triangle: distance zero.
        let on = Ray3::new([0.5, 0.5, 0.0], [0.0, 0.0, -1.0]);
        let hit = ray_triangle(on, TRIANGLE, 0.0, TriangleCulling::None).unwrap();
        assert_eq!(hit.distance, 0.0);
    }

    #[test]
    fn segments_report_fractions_and_include_their_endpoints() {
        let hit = segment_triangle(
            [0.5, 0.5, 1.0],
            [0.5, 0.5, -3.0],
            TRIANGLE,
            TriangleCulling::None,
        )
        .unwrap();
        assert_eq!(hit.fraction, 0.25);
        assert_eq!(hit.facing, TriangleFacing::Front);
        let ends_on = segment_triangle(
            [0.5, 0.5, 1.0],
            [0.5, 0.5, 0.0],
            TRIANGLE,
            TriangleCulling::None,
        )
        .unwrap();
        assert_eq!(ends_on.fraction, 1.0);
        let starts_on = segment_triangle(
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 1.0],
            TRIANGLE,
            TriangleCulling::None,
        )
        .unwrap();
        assert_eq!(
            (starts_on.fraction, starts_on.facing),
            (0.0, TriangleFacing::Back)
        );
        assert!(
            segment_triangle(
                [0.5, 0.5, 1.0],
                [0.5, 0.5, 0.5],
                TRIANGLE,
                TriangleCulling::None
            )
            .is_none()
        );
        assert!(
            segment_triangle(
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                TRIANGLE,
                TriangleCulling::None
            )
            .is_none()
        );
    }

    #[test]
    fn origins_on_vertices_and_segment_ends_on_the_plane_hit() {
        // Rays starting exactly on each vertex, in generic directions.
        let triangle = [[0.3, -1.7, 2.9], [4.1, 0.2, -0.6], [-2.2, 3.3, 1.4]];
        for corner in triangle {
            for direction in [[0.3, 0.9, -0.2], [-0.7, 0.1, 0.5], [0.2, -0.4, -0.9]] {
                let ray = Ray3::new(corner, direction);
                let hit = ray_triangle(ray, triangle, f64::INFINITY, TriangleCulling::None);
                let hit = hit.unwrap_or_else(|| panic!("vertex {corner:?} {direction:?}"));
                assert_eq!(hit.distance, 0.0);
                assert!(hit.barycentric.iter().all(|w| (0.0..=1.0).contains(w)));
            }
        }
        // Segments ending exactly on the triangle's plane, at generic points.
        let mut state = 0x2545_f491_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1_u32 << 24) as f32
        };
        let flat = [[-3.0, -3.0, 0.75], [5.0, -3.0, 0.75], [-3.0, 5.0, 0.75]];
        for _ in 0..512 {
            let end = [next() * 2.0 - 1.0, next() * 2.0 - 1.0, 0.75];
            let start = [next() * 6.0 - 3.0, next() * 6.0 - 3.0, next() * 4.0 + 1.0];
            let hit = segment_triangle(start, end, flat, TriangleCulling::None)
                .unwrap_or_else(|| panic!("segment {start:?} -> {end:?}"));
            assert!((hit.fraction - 1.0).abs() <= 1.0e-12, "{}", hit.fraction);
        }
    }

    /// Independent oracle: intersect the supporting plane, then classify the
    /// point with signed sub-triangle areas. Returns the signed plane
    /// parameter and the signed, unclamped weights whether or not the point
    /// is inside, so disagreements can be judged from the oracle's own data
    /// rather than from the kernel's clamped output. `None` only for
    /// degenerate triangles and (near-)parallel rays.
    fn plane_area_oracle(ray: Ray3, triangle: [[f32; 3]; 3]) -> Option<(f64, [f64; 3], bool)> {
        let [a, b, c] = triangle.map(|corner| corner.map(f64::from));
        let normal = cross(sub(b, a), sub(c, a));
        let area = length_squared(normal).sqrt();
        if area == 0.0 {
            return None;
        }
        let along = dot(ray.direction(), normal);
        if along.abs() <= 1.0e-9 * area {
            return None;
        }
        let distance = dot(sub(a, ray.origin64()), normal) / along;
        let point = add(ray.origin64(), scale(ray.direction(), distance));
        let weight =
            |p: Vec3, q: Vec3| dot(cross(sub(q, p), sub(point, p)), normal) / (area * area);
        let weights = [weight(b, c), weight(c, a), weight(a, b)];
        Some((distance, weights, along < 0.0))
    }

    #[test]
    fn generated_queries_match_the_plane_area_oracle() {
        let mut state = 0x9e37_79b9_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1_u32 << 24) as f32
        };
        let mut hits = 0;
        for _ in 0..8192 {
            let triangle: [[f32; 3]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| next() * 8.0 - 4.0));
            let origin = std::array::from_fn(|_| next() * 16.0 - 8.0);
            let target: [f32; 3] = std::array::from_fn(|_| next() * 6.0 - 3.0);
            let direction: [f32; 3] = std::array::from_fn(|axis| target[axis] - origin[axis]);
            if direction == [0.0; 3] {
                continue;
            }
            let ray = Ray3::new(origin, direction);
            let oracle = plane_area_oracle(ray, triangle);
            let expected = oracle.filter(|(distance, weights, _)| {
                *distance >= 0.0 && weights.iter().all(|weight| *weight >= 0.0)
            });
            let actual = ray_triangle(ray, triangle, f64::INFINITY, TriangleCulling::None);
            // Inputs within rounding of an edge, of the origin or of parallel
            // may legitimately disagree. Judge that from the oracle's signed,
            // unclamped data: the kernel clamps its own weights and distance,
            // so they always look near-boundary.
            let near_boundary = |weights: [f64; 3]| weights.iter().any(|w| w.abs() < 1.0e-6);
            let barely_outside = |distance: f64, weights: [f64; 3]| {
                distance >= -1.0e-9 && weights.iter().all(|w| *w >= -1.0e-6)
            };
            match (actual, expected) {
                (Some(actual), Some((distance, weights, front))) => {
                    hits += 1;
                    assert!(
                        (actual.distance - distance).abs() <= 1.0e-9 * distance.max(1.0),
                        "{ray:?} {triangle:?}"
                    );
                    for (actual, expected) in actual.barycentric.iter().zip(weights) {
                        assert!((actual - expected).abs() <= 1.0e-9);
                    }
                    assert_eq!(actual.facing == TriangleFacing::Front, front);
                }
                (None, None) => {}
                (Some(_), None) => match oracle {
                    // Parallel within the oracle's coarser cutoff.
                    None => {}
                    Some((distance, weights, _)) => assert!(
                        barely_outside(distance, weights),
                        "kernel hit, oracle missed by {distance} {weights:?}: {ray:?} {triangle:?}"
                    ),
                },
                (None, Some((_, weights, _))) => assert!(
                    near_boundary(weights),
                    "oracle hit, kernel missed: {ray:?} {triangle:?}"
                ),
            }
        }
        assert!(hits > 500, "fixture must exercise hits, got {hits}");
    }

    #[test]
    fn direction_scale_and_triangle_translation_are_metamorphic() {
        let base = ray_triangle(
            down(0.4, 0.7),
            TRIANGLE,
            f64::INFINITY,
            TriangleCulling::None,
        )
        .unwrap();
        for magnitude in [1.0e-20_f32, 1.0, 1.0e20] {
            let ray = Ray3::new([0.4, 0.7, 3.0], [0.0, 0.0, -magnitude]);
            let hit = ray_triangle(ray, TRIANGLE, f64::INFINITY, TriangleCulling::None).unwrap();
            assert_eq!(hit, base);
        }
        let shifted = TRIANGLE.map(|[x, y, z]| [x + 8.0, y - 4.0, z + 2.0]);
        let ray = Ray3::new([8.4, -3.3, 5.0], [0.0, 0.0, -1.0]);
        let hit = ray_triangle(ray, shifted, f64::INFINITY, TriangleCulling::None).unwrap();
        assert!((hit.distance - base.distance).abs() <= 1.0e-12);
        for (left, right) in hit.barycentric.iter().zip(base.barycentric) {
            assert!((left - right).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn exact_triple_products_match_integer_arithmetic() {
        // y = 2^54 + small, given as the difference of 2^54 and -small, which
        // f64 cannot hold, so its low parts matter; z nearly parallel to y,
        // so the cross product cancels. i128 evaluates the same triple
        // product exactly.
        let mut state = 0x0141_c0de_u64;
        let mut next = |bits: u32| -> i64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as i64 >> (53 - bits)) - (1_i64 << (bits - 1))
        };
        let far = 1_i64 << 54;
        let mut rounded = 0;
        for round in 0..4096 {
            let x: [i64; 3] = std::array::from_fn(|_| next(30));
            let small: [i64; 3] = std::array::from_fn(|_| next(30) | 1);
            let tweak: [i64; 3] = std::array::from_fn(|_| next(3));
            let y: [i64; 3] = small.map(|value| far + value);
            let z: [i64; 3] = std::array::from_fn(|axis| (y[axis] >> 22) + tweak[axis]);
            let y_exact = exact_difference([far as f64; 3], small.map(|value| -value as f64));
            rounded += usize::from(y_exact.iter().any(|[_, low]| *low != 0.0));
            let z_exact = if round % 2 == 0 {
                exact_vector(z.map(|value| value as f64))
            } else {
                let high = z.map(|value| (value >> 10 << 10) as f64);
                let low = std::array::from_fn(|axis| (high[axis] as i64 - z[axis]) as f64);
                exact_difference(high, low)
            };
            let exact = exact_triple(exact_vector(x.map(|value| value as f64)), y_exact, z_exact)
                .estimate();
            let [x, y, z] = [x, y, z].map(|vector| vector.map(i128::from));
            let expected = x[0] * (y[1] * z[2] - y[2] * z[1])
                + x[1] * (y[2] * z[0] - y[0] * z[2])
                + x[2] * (y[0] * z[1] - y[1] * z[0]);
            assert_eq!(
                exact.signum() as i128 * i128::from(exact != 0.0),
                expected.signum()
            );
            assert!(
                (exact - expected as f64).abs() <= 1.0e-15 * (expected as f64).abs(),
                "{exact} {expected}"
            );
        }
        assert!(rounded > 4000, "fixture must need the low parts: {rounded}");
    }

    #[test]
    #[should_panic(expected = "maximum distance must be non-negative")]
    fn rejects_negative_bounds() {
        let _ = ray_triangle(down(0.5, 0.5), TRIANGLE, -1.0, TriangleCulling::None);
    }
}
