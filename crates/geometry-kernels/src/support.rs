use spatial_kernels::Aabb;

use crate::{
    Sphere,
    math3::{Vec3, add, dot, is_finite, neg, normalized, scale, sub},
    obb3::Obb3,
    primitives::{Capsule, Segment3},
};

pub trait SupportMap3 {
    #[must_use]
    fn support_point(&self, direction: Vec3) -> Vec3;
}

fn unit_or_x(direction: Vec3) -> Vec3 {
    assert!(is_finite(direction), "support direction must be finite");
    normalized(direction).unwrap_or([1.0, 0.0, 0.0])
}

impl SupportMap3 for Sphere {
    fn support_point(&self, direction: Vec3) -> Vec3 {
        let unit = unit_or_x(direction);
        add(self.center.map(f64::from), scale(unit, f64::from(self.radius)))
    }
}

impl SupportMap3 for Aabb {
    fn support_point(&self, direction: Vec3) -> Vec3 {
        assert!(is_finite(direction), "support direction must be finite");
        std::array::from_fn(|axis| {
            if direction[axis] >= 0.0 {
                f64::from(self.max[axis])
            } else {
                f64::from(self.min[axis])
            }
        })
    }
}

impl SupportMap3 for Obb3 {
    fn support_point(&self, direction: Vec3) -> Vec3 {
        assert!(is_finite(direction), "support direction must be finite");
        let axes = self.axes();
        let mut point = self.center.map(f64::from);
        for axis in 0..3 {
            let sign = if dot(axes[axis], direction) >= 0.0 {
                1.0
            } else {
                -1.0
            };
            point = add(
                point,
                scale(axes[axis], sign * f64::from(self.half_extents[axis])),
            );
        }
        point
    }
}

impl SupportMap3 for Segment3 {
    fn support_point(&self, direction: Vec3) -> Vec3 {
        assert!(is_finite(direction), "support direction must be finite");
        let start = self.start64();
        let end = self.end64();
        if dot(start, direction) >= dot(end, direction) {
            start
        } else {
            end
        }
    }
}

impl SupportMap3 for Capsule {
    fn support_point(&self, direction: Vec3) -> Vec3 {
        let unit = unit_or_x(direction);
        add(
            self.segment.support_point(direction),
            scale(unit, f64::from(self.radius)),
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConvexHull3<'a> {
    points: &'a [Vec3],
}

impl<'a> ConvexHull3<'a> {
    #[must_use]
    pub fn new(points: &'a [Vec3]) -> Self {
        assert!(!points.is_empty(), "convex hull support set must not be empty");
        assert!(
            points.iter().copied().all(is_finite),
            "convex hull support points must be finite"
        );
        Self { points }
    }

    #[must_use]
    pub fn points(self) -> &'a [Vec3] {
        self.points
    }
}

impl SupportMap3 for ConvexHull3<'_> {
    fn support_point(&self, direction: Vec3) -> Vec3 {
        assert!(is_finite(direction), "support direction must be finite");
        let mut best = self.points[0];
        let mut best_projection = dot(best, direction);
        for &point in &self.points[1..] {
            let projection = dot(point, direction);
            if projection > best_projection {
                best = point;
                best_projection = projection;
            }
        }
        best
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MinkowskiSupportPoint {
    /// Point in the Minkowski difference: `left - right`.
    pub point: Vec3,
    /// Witness point selected from the left shape.
    pub left: Vec3,
    /// Witness point selected from the right shape.
    pub right: Vec3,
}

#[must_use]
pub fn minkowski_support<L, R>(
    left: &L,
    right: &R,
    direction: Vec3,
) -> MinkowskiSupportPoint
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    let left_point = left.support_point(direction);
    let right_point = right.support_point(neg(direction));
    MinkowskiSupportPoint {
        point: sub(left_point, right_point),
        left: left_point,
        right: right_point,
    }
}

#[cfg(test)]
mod tests {
    use super::{ConvexHull3, SupportMap3, minkowski_support};
    use crate::{Sphere, obb3::Obb3, primitives::Capsule};
    use spatial_kernels::Aabb;

    #[test]
    fn sphere_support_uses_radius_in_requested_direction() {
        let sphere = Sphere::new([1.0, 2.0, 3.0], 2.0);
        assert_eq!(sphere.support_point([1.0, 0.0, 0.0]), [3.0, 2.0, 3.0]);
        assert_eq!(sphere.support_point([0.0; 3]), [3.0, 2.0, 3.0]);
    }

    #[test]
    fn aabb_support_selects_the_extreme_corner() {
        let aabb = Aabb::new([-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]);
        assert_eq!(aabb.support_point([1.0, -1.0, 0.5]), [4.0, -2.0, 6.0]);
    }

    #[test]
    fn zero_rotation_obb_support_matches_equivalent_aabb() {
        let obb = Obb3::new([1.0, 2.0, 3.0], [2.0, 1.0, 0.5], [0.0; 3]);
        let aabb = Aabb::from_center_half_extents([1.0, 2.0, 3.0], [2.0, 1.0, 0.5]);
        for direction in [[1.0, 2.0, 3.0], [-1.0, 4.0, -2.0], [0.0, -1.0, 0.0]] {
            assert_eq!(obb.support_point(direction), aabb.support_point(direction));
        }
    }

    #[test]
    fn capsule_support_selects_an_axis_endpoint_plus_radius() {
        let capsule = Capsule::new([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5);
        assert_eq!(capsule.support_point([1.0, 0.0, 0.0]), [1.5, 0.0, 0.0]);
        assert_eq!(capsule.support_point([-1.0, 0.0, 0.0]), [-1.5, 0.0, 0.0]);
    }

    #[test]
    fn convex_hull_support_is_deterministic_on_ties() {
        let points = [
            [-1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ];
        let hull = ConvexHull3::new(&points);
        assert_eq!(hull.support_point([1.0, 0.0, 0.0]), points[1]);
        assert_eq!(hull.points(), &points);
    }

    #[test]
    fn minkowski_support_preserves_both_witnesses() {
        let left = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let right = Sphere::new([3.0, 0.0, 0.0], 0.5);
        let support = minkowski_support(&left, &right, [1.0, 0.0, 0.0]);
        assert_eq!(support.left, [1.0, 0.0, 0.0]);
        assert_eq!(support.right, [2.5, 0.0, 0.0]);
        assert_eq!(support.point, [-1.5, 0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "convex hull support set must not be empty")]
    fn convex_hull_rejects_empty_point_sets() {
        let _ = ConvexHull3::new(&[]);
    }
}