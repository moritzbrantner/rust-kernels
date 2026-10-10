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
//!   distance (segment endpoints) count. Comparisons are exact in f64 after
//!   widening the f32 inputs, so results are deterministic.
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
    let delta = sub(end.map(f64::from), origin);
    let length = length_squared(delta).sqrt();
    if length == 0.0 {
        return None;
    }
    let direction = scale(delta, length.recip());
    let [a, b, c] = triangle.map(|corner| corner.map(f64::from));
    let hit = ray_triangle_f64(origin, direction, [a, b, c], length, culling)?;
    Some(SegmentTriangleHit {
        fraction: (hit.distance / length).min(1.0),
        point: hit.point,
        barycentric: hit.barycentric,
        facing: hit.facing,
    })
}

fn ray_triangle_f64(
    origin: Vec3,
    direction: Vec3,
    [a, b, c]: [Vec3; 3],
    max_distance: f64,
    culling: TriangleCulling,
) -> Option<RayTriangleHit> {
    assert!(
        [a, b, c].iter().flatten().all(|value| value.is_finite()),
        "triangle corners must be finite"
    );
    // Möller–Trumbore with a unit direction.
    let edge_ab = sub(b, a);
    let edge_ac = sub(c, a);
    let normal = cross(edge_ab, edge_ac);
    let normal_length_squared = length_squared(normal);
    if normal_length_squared == 0.0 {
        return None;
    }
    let p = cross(direction, edge_ac);
    // determinant = -(direction · normal).
    let determinant = dot(edge_ab, p);
    if determinant.abs() <= PARALLEL_TOLERANCE * normal_length_squared.sqrt() {
        return None;
    }
    let facing = if determinant > 0.0 {
        TriangleFacing::Front
    } else {
        TriangleFacing::Back
    };
    if culling == TriangleCulling::BackFaces && facing == TriangleFacing::Back {
        return None;
    }
    let inverse = determinant.recip();
    let to_origin = sub(origin, a);
    let u = dot(to_origin, p) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(to_origin, edge_ab);
    let v = dot(direction, q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = dot(edge_ac, q) * inverse;
    if !(0.0..=max_distance).contains(&distance) {
        return None;
    }
    Some(RayTriangleHit {
        distance,
        point: add(origin, scale(direction, distance)),
        barycentric: [1.0 - u - v, u, v],
        facing,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIANGLE: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];

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

    /// Independent oracle: intersect the supporting plane, then classify the
    /// point with signed sub-triangle areas.
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
        if distance < 0.0 {
            return None;
        }
        let point = ray.at(distance);
        let weight =
            |p: Vec3, q: Vec3| dot(cross(sub(q, p), sub(point, p)), normal) / (area * area);
        let weights = [weight(b, c), weight(c, a), weight(a, b)];
        weights
            .iter()
            .all(|weight| *weight >= 0.0)
            .then_some((distance, weights, along < 0.0))
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
            let expected = plane_area_oracle(ray, triangle);
            let actual = ray_triangle(ray, triangle, f64::INFINITY, TriangleCulling::None);
            // Skip inputs within rounding of an edge or of parallel, where
            // the two formulations may legitimately disagree.
            let near_boundary = |weights: [f64; 3]| weights.iter().any(|w| w.abs() < 1.0e-6);
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
                (Some(actual), None) => assert!(
                    near_boundary(actual.barycentric),
                    "kernel hit, oracle missed: {ray:?} {triangle:?}"
                ),
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
    #[should_panic(expected = "maximum distance must be non-negative")]
    fn rejects_negative_bounds() {
        let _ = ray_triangle(down(0.5, 0.5), TRIANGLE, -1.0, TriangleCulling::None);
    }
}
