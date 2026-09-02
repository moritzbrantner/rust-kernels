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

#[cfg(test)]
mod tests {
    use super::{Sphere, aabb_aabb, sphere_sphere};
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
}
