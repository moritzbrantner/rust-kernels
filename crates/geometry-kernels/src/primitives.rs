use spatial_kernels::Aabb;

use crate::{
    Sphere,
    math3::{NORMALIZE_EPSILON_SQUARED, Vec3, add, clamp01, dot, length_squared, lerp, scale, sub},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment3 {
    pub start: [f32; 3],
    pub end: [f32; 3],
}

impl Segment3 {
    #[must_use]
    pub fn new(start: [f32; 3], end: [f32; 3]) -> Self {
        assert!(
            start
                .iter()
                .chain(end.iter())
                .all(|value| value.is_finite()),
            "segment endpoints must be finite"
        );
        Self { start, end }
    }

    #[must_use]
    pub fn start64(self) -> Vec3 {
        self.start.map(f64::from)
    }

    #[must_use]
    pub fn end64(self) -> Vec3 {
        self.end.map(f64::from)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule {
    pub segment: Segment3,
    pub radius: f32,
}

impl Capsule {
    #[must_use]
    pub fn new(start: [f32; 3], end: [f32; 3], radius: f32) -> Self {
        assert!(
            radius.is_finite() && radius >= 0.0,
            "capsule radius must be non-negative and finite"
        );
        Self {
            segment: Segment3::new(start, end),
            radius,
        }
    }

    #[must_use]
    pub fn from_segment(segment: Segment3, radius: f32) -> Self {
        Self::new(segment.start, segment.end, radius)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointSegmentRelation {
    pub closest_point: Vec3,
    pub parameter: f64,
    pub distance_squared: f64,
    pub distance: f64,
}

#[must_use]
pub fn point_segment(point: Vec3, segment: Segment3) -> PointSegmentRelation {
    assert!(
        point.into_iter().all(f64::is_finite),
        "point must be finite"
    );
    let start = segment.start64();
    let end = segment.end64();
    let direction = sub(end, start);
    let direction_length_squared = length_squared(direction);
    let parameter = if direction_length_squared <= NORMALIZE_EPSILON_SQUARED {
        0.0
    } else {
        clamp01(dot(sub(point, start), direction) / direction_length_squared)
    };
    let closest_point = lerp(start, end, parameter);
    let delta = sub(point, closest_point);
    let distance_squared = length_squared(delta);

    PointSegmentRelation {
        closest_point,
        parameter,
        distance_squared,
        distance: distance_squared.sqrt(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentSegmentRelation {
    pub left_point: Vec3,
    pub right_point: Vec3,
    pub left_parameter: f64,
    pub right_parameter: f64,
    pub distance_squared: f64,
    pub distance: f64,
}

#[must_use]
pub fn segment_segment(left: Segment3, right: Segment3) -> SegmentSegmentRelation {
    let left_start = left.start64();
    let right_start = right.start64();
    let left_direction = sub(left.end64(), left_start);
    let right_direction = sub(right.end64(), right_start);
    let between_starts = sub(left_start, right_start);
    let left_length_squared = dot(left_direction, left_direction);
    let right_length_squared = dot(right_direction, right_direction);

    let (left_parameter, right_parameter) = if left_length_squared <= NORMALIZE_EPSILON_SQUARED
        && right_length_squared <= NORMALIZE_EPSILON_SQUARED
    {
        (0.0, 0.0)
    } else if left_length_squared <= NORMALIZE_EPSILON_SQUARED {
        (
            0.0,
            clamp01(dot(right_direction, between_starts) / right_length_squared),
        )
    } else {
        let left_projection = dot(left_direction, between_starts);
        if right_length_squared <= NORMALIZE_EPSILON_SQUARED {
            (clamp01(-left_projection / left_length_squared), 0.0)
        } else {
            let cross_projection = dot(left_direction, right_direction);
            let right_projection = dot(right_direction, between_starts);
            let denominator =
                left_length_squared * right_length_squared - cross_projection * cross_projection;
            let mut left_parameter = if denominator.abs() > NORMALIZE_EPSILON_SQUARED {
                clamp01(
                    (cross_projection * right_projection - left_projection * right_length_squared)
                        / denominator,
                )
            } else {
                0.0
            };
            let mut right_parameter =
                (cross_projection * left_parameter + right_projection) / right_length_squared;

            if right_parameter < 0.0 {
                right_parameter = 0.0;
                left_parameter = clamp01(-left_projection / left_length_squared);
            } else if right_parameter > 1.0 {
                right_parameter = 1.0;
                left_parameter =
                    clamp01((cross_projection - left_projection) / left_length_squared);
            }
            (left_parameter, right_parameter)
        }
    };

    let left_point = add(left_start, scale(left_direction, left_parameter));
    let right_point = add(right_start, scale(right_direction, right_parameter));
    let delta = sub(left_point, right_point);
    let distance_squared = length_squared(delta);

    SegmentSegmentRelation {
        left_point,
        right_point,
        left_parameter,
        right_parameter,
        distance_squared,
        distance: distance_squared.sqrt(),
    }
}

#[must_use]
pub fn closest_point_on_aabb(point: Vec3, aabb: Aabb) -> Vec3 {
    assert!(
        point.into_iter().all(f64::is_finite),
        "point must be finite"
    );
    std::array::from_fn(|axis| {
        point[axis].clamp(f64::from(aabb.min[axis]), f64::from(aabb.max[axis]))
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereAabbRelation {
    pub closest_point: Vec3,
    pub center_distance_squared: f64,
    pub center_distance: f64,
    pub signed_separation: f64,
    pub overlaps: bool,
}

#[must_use]
pub fn sphere_aabb(sphere: Sphere, aabb: Aabb) -> SphereAabbRelation {
    let center = sphere.center.map(f64::from);
    let closest_point = closest_point_on_aabb(center, aabb);
    let center_distance_squared = length_squared(sub(center, closest_point));
    let center_distance = center_distance_squared.sqrt();
    let signed_separation = center_distance - f64::from(sphere.radius);

    SphereAabbRelation {
        closest_point,
        center_distance_squared,
        center_distance,
        signed_separation,
        overlaps: center_distance_squared <= f64::from(sphere.radius).powi(2),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereCapsuleRelation {
    pub capsule_axis_point: Vec3,
    pub capsule_parameter: f64,
    pub center_distance_squared: f64,
    pub center_distance: f64,
    pub radius_sum: f64,
    pub signed_separation: f64,
    pub overlaps: bool,
}

#[must_use]
pub fn sphere_capsule(sphere: Sphere, capsule: Capsule) -> SphereCapsuleRelation {
    let axis_relation = point_segment(sphere.center.map(f64::from), capsule.segment);
    let radius_sum = f64::from(sphere.radius) + f64::from(capsule.radius);
    let signed_separation = axis_relation.distance - radius_sum;

    SphereCapsuleRelation {
        capsule_axis_point: axis_relation.closest_point,
        capsule_parameter: axis_relation.parameter,
        center_distance_squared: axis_relation.distance_squared,
        center_distance: axis_relation.distance,
        radius_sum,
        signed_separation,
        overlaps: axis_relation.distance_squared <= radius_sum * radius_sum,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapsuleCapsuleRelation {
    pub left_axis_point: Vec3,
    pub right_axis_point: Vec3,
    pub left_parameter: f64,
    pub right_parameter: f64,
    pub axis_distance_squared: f64,
    pub axis_distance: f64,
    pub radius_sum: f64,
    pub signed_separation: f64,
    pub overlaps: bool,
}

#[must_use]
pub fn capsule_capsule(left: Capsule, right: Capsule) -> CapsuleCapsuleRelation {
    let axis_relation = segment_segment(left.segment, right.segment);
    let radius_sum = f64::from(left.radius) + f64::from(right.radius);
    let signed_separation = axis_relation.distance - radius_sum;

    CapsuleCapsuleRelation {
        left_axis_point: axis_relation.left_point,
        right_axis_point: axis_relation.right_point,
        left_parameter: axis_relation.left_parameter,
        right_parameter: axis_relation.right_parameter,
        axis_distance_squared: axis_relation.distance_squared,
        axis_distance: axis_relation.distance,
        radius_sum,
        signed_separation,
        overlaps: axis_relation.distance_squared <= radius_sum * radius_sum,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Capsule, Segment3, capsule_capsule, closest_point_on_aabb, point_segment, segment_segment,
        sphere_aabb, sphere_capsule,
    };
    use crate::Sphere;
    use spatial_kernels::Aabb;

    #[test]
    fn point_segment_clamps_to_the_finite_segment() {
        let segment = Segment3::new([0.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        let middle = point_segment([0.5, 1.0, 0.0], segment);
        assert_eq!(middle.closest_point, [0.5, 0.0, 0.0]);
        assert_eq!(middle.parameter, 0.25);
        assert_eq!(middle.distance, 1.0);

        let beyond = point_segment([3.0, 0.0, 0.0], segment);
        assert_eq!(beyond.closest_point, [2.0, 0.0, 0.0]);
        assert_eq!(beyond.parameter, 1.0);
    }

    #[test]
    fn degenerate_segment_behaves_as_a_point() {
        let segment = Segment3::new([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]);
        let relation = point_segment([4.0, 6.0, 3.0], segment);
        assert_eq!(relation.closest_point, [1.0, 2.0, 3.0]);
        assert_eq!(relation.parameter, 0.0);
        assert_eq!(relation.distance, 5.0);
    }

    #[test]
    fn segment_segment_handles_crossing_parallel_and_degenerate_cases() {
        let crossing = segment_segment(
            Segment3::new([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            Segment3::new([0.0, -1.0, 0.0], [0.0, 1.0, 0.0]),
        );
        assert!(crossing.distance < 1.0e-12);
        assert_eq!(crossing.left_parameter, 0.5);
        assert_eq!(crossing.right_parameter, 0.5);

        let parallel = segment_segment(
            Segment3::new([0.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
            Segment3::new([0.0, 2.0, 0.0], [2.0, 2.0, 0.0]),
        );
        assert_eq!(parallel.distance, 2.0);

        let points = segment_segment(
            Segment3::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            Segment3::new([0.0, 3.0, 4.0], [0.0, 3.0, 4.0]),
        );
        assert_eq!(points.distance, 5.0);
    }

    #[test]
    fn closest_point_on_aabb_clamps_each_axis() {
        let aabb = Aabb::new([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]);
        assert_eq!(
            closest_point_on_aabb([4.0, 0.5, -5.0], aabb),
            [1.0, 0.5, -3.0]
        );
    }

    #[test]
    fn sphere_aabb_counts_touching_and_containment_as_overlap() {
        let aabb = Aabb::new([-1.0; 3], [1.0; 3]);
        let touching = sphere_aabb(Sphere::new([2.0, 0.0, 0.0], 1.0), aabb);
        assert!(touching.overlaps);
        assert_eq!(touching.signed_separation, 0.0);

        let inside = sphere_aabb(Sphere::new([0.0, 0.0, 0.0], 0.25), aabb);
        assert!(inside.overlaps);
        assert_eq!(inside.center_distance, 0.0);
    }

    #[test]
    fn sphere_capsule_reduces_to_point_segment_distance_plus_radii() {
        let capsule = Capsule::new([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5);
        let touching = sphere_capsule(Sphere::new([0.0, 1.5, 0.0], 1.0), capsule);
        assert!(touching.overlaps);
        assert!((touching.signed_separation).abs() < 1.0e-12);
        assert_eq!(touching.capsule_axis_point, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn capsule_capsule_uses_closest_axis_points() {
        let left = Capsule::new([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5);
        let touching = Capsule::new([0.0, 1.0, -1.0], [0.0, 1.0, 1.0], 0.5);
        let separated = Capsule::new([0.0, 1.1, -1.0], [0.0, 1.1, 1.0], 0.5);
        assert!(capsule_capsule(left, touching).overlaps);
        assert!(!capsule_capsule(left, separated).overlaps);
    }

    #[test]
    fn capsule_relation_is_symmetric() {
        let left = Capsule::new([-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.7);
        let right = Capsule::new([0.0, 1.0, -1.0], [0.0, 1.0, 2.0], 0.4);
        assert_eq!(
            capsule_capsule(left, right).overlaps,
            capsule_capsule(right, left).overlaps
        );
    }

    #[test]
    #[should_panic(expected = "capsule radius must be non-negative and finite")]
    fn capsule_rejects_invalid_radius() {
        let _ = Capsule::new([0.0; 3], [1.0, 0.0, 0.0], -1.0);
    }
}
