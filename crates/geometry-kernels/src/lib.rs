use spatial_kernels::Aabb;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
}

impl Sphere {
    #[must_use]
    pub fn new(center: [f32; 3], radius: f32) -> Self {
        assert!(
            center.iter().all(|coordinate| coordinate.is_finite()),
            "sphere center must be finite"
        );
        assert!(
            radius.is_finite() && radius >= 0.0,
            "sphere radius must be non-negative and finite"
        );
        Self { center, radius }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereSphereRelation {
    /// Squared center distance. The overlap decision uses this value so it does
    /// not require a square root.
    pub center_distance_squared: f64,
    /// Euclidean center distance, useful for diagnostics and teaching.
    pub center_distance: f64,
    pub radius_sum: f64,
    /// Positive when separated, zero when touching, negative when penetrating.
    pub signed_separation: f64,
    pub overlaps: bool,
}

#[must_use]
pub fn sphere_sphere(left: Sphere, right: Sphere) -> SphereSphereRelation {
    let delta = [
        f64::from(right.center[0]) - f64::from(left.center[0]),
        f64::from(right.center[1]) - f64::from(left.center[1]),
        f64::from(right.center[2]) - f64::from(left.center[2]),
    ];
    let center_distance_squared = delta.into_iter().map(|value| value * value).sum::<f64>();
    let center_distance = center_distance_squared.sqrt();
    let radius_sum = f64::from(left.radius) + f64::from(right.radius);
    let overlaps = center_distance_squared <= radius_sum * radius_sum;

    SphereSphereRelation {
        center_distance_squared,
        center_distance,
        radius_sum,
        signed_separation: center_distance - radius_sum,
        overlaps,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AabbAabbRelation {
    /// Signed interval overlap on X/Y/Z. Positive means penetration on that
    /// axis, zero means touching, and negative means an explicit separating gap.
    pub axis_overlap: [f64; 3],
    pub overlaps: bool,
}

#[must_use]
pub fn aabb_aabb(left: Aabb, right: Aabb) -> AabbAabbRelation {
    let axis_overlap = std::array::from_fn(|axis| {
        f64::from(left.max[axis].min(right.max[axis]))
            - f64::from(left.min[axis].max(right.min[axis]))
    });
    let overlaps = axis_overlap.iter().all(|overlap| *overlap >= 0.0);

    AabbAabbRelation {
        axis_overlap,
        overlaps,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obb2 {
    pub center: [f32; 2],
    pub half_extents: [f32; 2],
    pub rotation_radians: f32,
}

impl Obb2 {
    #[must_use]
    pub fn new(center: [f32; 2], half_extents: [f32; 2], rotation_radians: f32) -> Self {
        assert!(
            center.iter().all(|coordinate| coordinate.is_finite()),
            "OBB center must be finite"
        );
        assert!(
            half_extents
                .iter()
                .all(|extent| extent.is_finite() && *extent >= 0.0),
            "OBB half extents must be non-negative and finite"
        );
        assert!(rotation_radians.is_finite(), "OBB rotation must be finite");
        Self {
            center,
            half_extents,
            rotation_radians,
        }
    }

    #[must_use]
    pub fn axes(self) -> [[f64; 2]; 2] {
        let rotation = f64::from(self.rotation_radians);
        let cosine = rotation.cos();
        let sine = rotation.sin();
        [[cosine, sine], [-sine, cosine]]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SatAxis2 {
    /// Unit candidate axis used for the projection test.
    pub axis: [f64; 2],
    pub left_radius: f64,
    pub right_radius: f64,
    pub center_distance: f64,
    /// Positive means projected intervals overlap, zero means touching, and
    /// negative means this axis separates the rectangles.
    pub signed_overlap: f64,
    pub separating: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obb2SatRelation {
    /// SAT axes in deterministic order: left X/Y followed by right X/Y.
    pub axes: [SatAxis2; 4],
    pub overlaps: bool,
    /// Index of the axis with the smallest signed overlap. When separated, this
    /// identifies a separating axis; when overlapping, it is the shallowest
    /// penetration/touch axis.
    pub critical_axis: usize,
}

#[must_use]
pub fn obb2_sat(left: Obb2, right: Obb2) -> Obb2SatRelation {
    let left_axes = left.axes();
    let right_axes = right.axes();
    let candidate_axes = [left_axes[0], left_axes[1], right_axes[0], right_axes[1]];
    let center_delta = [
        f64::from(right.center[0]) - f64::from(left.center[0]),
        f64::from(right.center[1]) - f64::from(left.center[1]),
    ];

    let axes = candidate_axes.map(|axis| {
        let left_radius = projection_radius(left, axis);
        let right_radius = projection_radius(right, axis);
        let center_distance = dot(center_delta, axis).abs();
        let signed_overlap = left_radius + right_radius - center_distance;
        SatAxis2 {
            axis,
            left_radius,
            right_radius,
            center_distance,
            signed_overlap,
            separating: signed_overlap < 0.0,
        }
    });

    let critical_axis = axes
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| left.signed_overlap.total_cmp(&right.signed_overlap))
        .map_or(0, |(index, _)| index);

    Obb2SatRelation {
        overlaps: axes.iter().all(|axis| !axis.separating),
        axes,
        critical_axis,
    }
}

fn projection_radius(obb: Obb2, axis: [f64; 2]) -> f64 {
    let local_axes = obb.axes();
    f64::from(obb.half_extents[0]) * dot(local_axes[0], axis).abs()
        + f64::from(obb.half_extents[1]) * dot(local_axes[1], axis).abs()
}

fn dot(left: [f64; 2], right: [f64; 2]) -> f64 {
    left[0] * right[0] + left[1] * right[1]
}

#[cfg(test)]
mod tests {
    use super::{Obb2, Sphere, aabb_aabb, obb2_sat, sphere_sphere};
    use spatial_kernels::Aabb;

    #[test]
    fn sphere_overlap_is_symmetric() {
        let left = Sphere::new([0.0, 0.0, 0.0], 1.25);
        let right = Sphere::new([2.0, 0.0, 0.0], 1.0);
        assert_eq!(sphere_sphere(left, right), sphere_sphere(right, left));
    }

    #[test]
    fn sphere_touching_counts_as_overlap() {
        let left = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let right = Sphere::new([2.0, 0.0, 0.0], 1.0);
        let relation = sphere_sphere(left, right);
        assert!(relation.overlaps);
        assert_eq!(relation.center_distance_squared, 4.0);
        assert_eq!(relation.radius_sum, 2.0);
        assert_eq!(relation.signed_separation, 0.0);
    }

    #[test]
    fn sphere_separation_and_penetration_have_opposite_signs() {
        let base = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let separated = sphere_sphere(base, Sphere::new([3.0, 0.0, 0.0], 1.0));
        let penetrating = sphere_sphere(base, Sphere::new([1.25, 0.0, 0.0], 1.0));
        assert!(!separated.overlaps);
        assert!(separated.signed_separation > 0.0);
        assert!(penetrating.overlaps);
        assert!(penetrating.signed_separation < 0.0);
    }

    #[test]
    fn identical_spheres_have_full_radial_penetration() {
        let sphere = Sphere::new([4.0, -2.0, 1.0], 2.5);
        let relation = sphere_sphere(sphere, sphere);
        assert!(relation.overlaps);
        assert_eq!(relation.center_distance, 0.0);
        assert_eq!(relation.signed_separation, -5.0);
    }

    #[test]
    #[should_panic(expected = "sphere center must be finite")]
    fn sphere_rejects_non_finite_center() {
        let _ = Sphere::new([f32::NAN, 0.0, 0.0], 1.0);
    }

    #[test]
    #[should_panic(expected = "sphere radius must be non-negative and finite")]
    fn sphere_rejects_invalid_radius() {
        let _ = Sphere::new([0.0; 3], -1.0);
    }

    #[test]
    fn aabb_relation_matches_existing_overlap_oracle() {
        let fixtures = [
            (
                Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                Aabb::new([0.5, 0.5, 0.5], [2.0, 2.0, 2.0]),
            ),
            (
                Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                Aabb::new([1.0, 0.25, 0.25], [2.0, 0.75, 0.75]),
            ),
            (
                Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                Aabb::new([1.1, 0.25, 0.25], [2.0, 0.75, 0.75]),
            ),
        ];

        for (left, right) in fixtures {
            assert_eq!(aabb_aabb(left, right).overlaps, left.overlaps(right));
            assert_eq!(aabb_aabb(right, left).overlaps, right.overlaps(left));
            assert_eq!(aabb_aabb(left, right), aabb_aabb(right, left));
        }
    }

    #[test]
    fn aabb_axis_overlap_exposes_penetration_touch_and_gap() {
        let base = Aabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let penetrating = aabb_aabb(base, Aabb::new([1.0, 0.5, -1.0], [3.0, 1.5, 1.0]));
        assert_eq!(penetrating.axis_overlap, [1.0, 1.0, 1.0]);
        assert!(penetrating.overlaps);

        let touching = aabb_aabb(base, Aabb::new([2.0, 0.5, 0.5], [3.0, 1.5, 1.5]));
        assert_eq!(touching.axis_overlap[0], 0.0);
        assert!(touching.overlaps);

        let separated = aabb_aabb(base, Aabb::new([2.25, 0.5, 0.5], [3.0, 1.5, 1.5]));
        assert_eq!(separated.axis_overlap[0], -0.25);
        assert!(!separated.overlaps);
    }

    #[test]
    fn zero_rotation_obb_matches_aabb_decision() {
        let fixtures = [
            (Obb2::new([0.0, 0.0], [1.0, 2.0], 0.0), Obb2::new([1.5, 0.5], [1.0, 0.5], 0.0)),
            (Obb2::new([0.0, 0.0], [1.0, 2.0], 0.0), Obb2::new([2.0, 0.0], [1.0, 0.5], 0.0)),
            (Obb2::new([0.0, 0.0], [1.0, 2.0], 0.0), Obb2::new([2.1, 0.0], [1.0, 0.5], 0.0)),
        ];

        for (left, right) in fixtures {
            let left_aabb = Aabb::from_center_half_extents(
                [left.center[0], left.center[1], 0.0],
                [left.half_extents[0], left.half_extents[1], 0.5],
            );
            let right_aabb = Aabb::from_center_half_extents(
                [right.center[0], right.center[1], 0.0],
                [right.half_extents[0], right.half_extents[1], 0.5],
            );
            assert_eq!(obb2_sat(left, right).overlaps, left_aabb.overlaps(right_aabb));
        }
    }

    #[test]
    fn rotated_overlap_has_no_separating_axis() {
        let left = Obb2::new([0.0, 0.0], [1.4, 0.8], 0.35);
        let right = Obb2::new([1.3, 0.25], [1.0, 0.7], -0.6);
        let relation = obb2_sat(left, right);
        assert!(relation.overlaps);
        assert!(relation.axes.iter().all(|axis| !axis.separating));
        assert!(relation.axes.iter().all(|axis| axis.signed_overlap >= 0.0));
    }

    #[test]
    fn rotated_separation_finds_at_least_one_axis() {
        let left = Obb2::new([0.0, 0.0], [1.4, 0.8], 0.35);
        let right = Obb2::new([4.0, 0.4], [1.0, 0.7], -0.6);
        let relation = obb2_sat(left, right);
        assert!(!relation.overlaps);
        assert!(relation.axes.iter().any(|axis| axis.separating));
        assert!(relation.axes[relation.critical_axis].separating);
    }

    #[test]
    fn touching_obb_counts_as_overlap() {
        let left = Obb2::new([0.0, 0.0], [1.0, 1.0], 0.0);
        let right = Obb2::new([2.0, 0.0], [1.0, 0.5], 0.0);
        let relation = obb2_sat(left, right);
        assert!(relation.overlaps);
        assert!(relation.axes.iter().any(|axis| axis.signed_overlap.abs() <= f64::EPSILON));
    }

    #[test]
    fn obb_sat_decision_is_symmetric() {
        let left = Obb2::new([-0.5, 0.3], [1.3, 0.7], 0.42);
        let right = Obb2::new([1.2, -0.1], [0.8, 1.1], -0.73);
        assert_eq!(obb2_sat(left, right).overlaps, obb2_sat(right, left).overlaps);
    }

    #[test]
    fn sat_candidate_axes_are_unit_length() {
        let left = Obb2::new([0.0, 0.0], [1.0, 1.0], 0.77);
        let right = Obb2::new([1.0, 0.0], [1.0, 1.0], -0.31);
        for axis in obb2_sat(left, right).axes {
            let length = (axis.axis[0] * axis.axis[0] + axis.axis[1] * axis.axis[1]).sqrt();
            assert!((length - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn critical_axis_has_the_smallest_signed_overlap() {
        let relation = obb2_sat(
            Obb2::new([0.0, 0.0], [1.4, 0.8], 0.2),
            Obb2::new([2.7, 0.2], [1.0, 0.7], -0.5),
        );
        let critical = relation.axes[relation.critical_axis].signed_overlap;
        assert!(relation.axes.iter().all(|axis| critical <= axis.signed_overlap));
    }

    #[test]
    #[should_panic(expected = "OBB half extents must be non-negative and finite")]
    fn obb_rejects_invalid_half_extents() {
        let _ = Obb2::new([0.0, 0.0], [-1.0, 1.0], 0.0);
    }

    #[test]
    #[should_panic(expected = "OBB rotation must be finite")]
    fn obb_rejects_non_finite_rotation() {
        let _ = Obb2::new([0.0, 0.0], [1.0, 1.0], f32::NAN);
    }
}
