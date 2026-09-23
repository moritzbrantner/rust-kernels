use spatial_kernels::Aabb;

use crate::{
    Sphere,
    math3::{Vec3, add, cross, dot, length_squared, scale, sub},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray3 {
    pub origin: [f32; 3],
    direction: Vec3,
    raw_direction: Vec3,
}

impl Ray3 {
    /// Builds a ray whose internal direction is unit length, so hit parameters
    /// are world-space distances. Direction magnitude does not affect results.
    #[must_use]
    pub fn new(origin: [f32; 3], direction: [f32; 3]) -> Self {
        assert!(
            origin
                .iter()
                .chain(direction.iter())
                .all(|value| value.is_finite()),
            "ray origin and direction must be finite"
        );
        let maximum = direction
            .iter()
            .map(|value| f64::from(*value).abs())
            .fold(0.0_f64, f64::max);
        assert!(maximum > 0.0, "ray direction must be non-zero");
        let scaled = direction.map(|value| f64::from(value) / maximum);
        let inverse_length = length_squared(scaled).sqrt().recip();
        Self {
            origin,
            direction: scale(scaled, inverse_length),
            raw_direction: direction.map(f64::from),
        }
    }

    #[must_use]
    pub const fn direction(self) -> Vec3 {
        self.direction
    }

    #[must_use]
    pub fn origin64(self) -> Vec3 {
        self.origin.map(f64::from)
    }

    fn raw_direction(self) -> Vec3 {
        self.raw_direction
    }

    #[must_use]
    pub fn at(self, distance: f64) -> Vec3 {
        assert!(distance.is_finite(), "ray distance must be finite");
        add(self.origin64(), scale(self.direction, distance))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayIntervalHit {
    /// First non-negative point occupied by the shape. This is zero when the
    /// ray starts inside or on the boundary.
    pub enter_distance: f64,
    /// Last non-negative point occupied by the shape.
    pub exit_distance: f64,
    pub enter_point: Vec3,
    pub exit_point: Vec3,
    pub starts_inside: bool,
}

#[must_use]
pub fn ray_aabb(ray: Ray3, aabb: Aabb) -> Option<RayIntervalHit> {
    let origin = ray.origin64();
    let direction = ray.direction();
    let starts_inside = (0..3).all(|axis| {
        origin[axis] >= f64::from(aabb.min[axis]) && origin[axis] <= f64::from(aabb.max[axis])
    });

    let mut enter = 0.0_f64;
    let mut exit = f64::INFINITY;
    for axis in 0..3 {
        let minimum = f64::from(aabb.min[axis]);
        let maximum = f64::from(aabb.max[axis]);
        let component = direction[axis];
        if component == 0.0 {
            if origin[axis] < minimum || origin[axis] > maximum {
                return None;
            }
            continue;
        }

        let first = (minimum - origin[axis]) / component;
        let second = (maximum - origin[axis]) / component;
        let (near, far) = if first <= second {
            (first, second)
        } else {
            (second, first)
        };
        enter = enter.max(near);
        exit = exit.min(far);
        if exit < enter {
            return None;
        }
    }

    if exit < 0.0 || !exit.is_finite() {
        return None;
    }
    enter = enter.max(0.0);
    Some(RayIntervalHit {
        enter_distance: enter,
        exit_distance: exit,
        enter_point: point_at_unchecked(ray, enter),
        exit_point: point_at_unchecked(ray, exit),
        starts_inside,
    })
}

#[must_use]
pub fn ray_sphere(ray: Ray3, sphere: Sphere) -> Option<RayIntervalHit> {
    ray_sphere_radius(ray, sphere.center.map(f64::from), f64::from(sphere.radius))
}

pub(crate) fn ray_sphere_radius(ray: Ray3, center: Vec3, radius: f64) -> Option<RayIntervalHit> {
    debug_assert!(radius.is_finite() && radius >= 0.0);
    let origin = ray.origin64();
    let direction = ray.direction();
    let raw_direction = ray.raw_direction();
    let to_center = sub(center, origin);
    let radius_squared = radius * radius;
    let starts_inside = length_squared(to_center) <= radius_squared;

    // Compute line distance from the original f32 direction widened to f64.
    // This preserves exact collinearity for point-sphere queries instead of
    // injecting a tiny cross product through unit-vector normalization.
    // Full-range f32 products remain finite in f64.
    let perpendicular_squared =
        length_squared(cross(to_center, raw_direction)) / length_squared(raw_direction);
    if perpendicular_squared > radius_squared {
        return None;
    }

    let center_distance = dot(to_center, direction);
    let half_chord = (radius_squared - perpendicular_squared).max(0.0).sqrt();
    let near = center_distance - half_chord;
    let far = center_distance + half_chord;
    if far < 0.0 {
        return None;
    }
    let enter = near.max(0.0);
    Some(RayIntervalHit {
        enter_distance: enter,
        exit_distance: far,
        enter_point: point_at_unchecked(ray, enter),
        exit_point: point_at_unchecked(ray, far),
        starts_inside,
    })
}

fn point_at_unchecked(ray: Ray3, distance: f64) -> Vec3 {
    add(ray.origin64(), scale(ray.direction(), distance))
}

#[cfg(test)]
mod tests {
    use super::{Ray3, ray_aabb, ray_sphere};
    use crate::Sphere;
    use spatial_kernels::Aabb;

    #[test]
    fn direction_scale_does_not_change_world_distance() {
        let origin = [-3.0, 0.0, 0.0];
        let sphere = Sphere::new([0.0; 3], 1.0);
        for magnitude in [f32::from_bits(1), 1.0e-20, 1.0, 1.0e20, f32::MAX] {
            let ray = Ray3::new(origin, [magnitude, 0.0, 0.0]);
            assert_eq!(ray.direction(), [1.0, 0.0, 0.0]);
            let hit = ray_sphere(ray, sphere).unwrap();
            assert_eq!(hit.enter_distance, 2.0);
            assert_eq!(hit.exit_distance, 4.0);
        }
    }

    #[test]
    fn ray_aabb_handles_inside_parallel_touch_and_miss() {
        let aabb = Aabb::new([-1.0; 3], [1.0; 3]);
        let hit = ray_aabb(Ray3::new([-3.0, 0.0, 0.0], [1.0, 0.0, 0.0]), aabb).unwrap();
        assert_eq!((hit.enter_distance, hit.exit_distance), (2.0, 4.0));
        assert_eq!(hit.enter_point, [-1.0, 0.0, 0.0]);
        assert!(!hit.starts_inside);

        let inside = ray_aabb(Ray3::new([0.0; 3], [0.0, 1.0, 0.0]), aabb).unwrap();
        assert_eq!(inside.enter_distance, 0.0);
        assert_eq!(inside.exit_distance, 1.0);
        assert!(inside.starts_inside);

        let touching = ray_aabb(Ray3::new([-3.0, 1.0, 0.0], [1.0, 0.0, 0.0]), aabb).unwrap();
        assert_eq!(touching.enter_distance, 2.0);
        assert!(ray_aabb(Ray3::new([-3.0, 2.0, 0.0], [1.0, 0.0, 0.0]), aabb).is_none());
    }

    #[test]
    fn point_sphere_preserves_exact_non_axis_aligned_collinearity() {
        let direction = [0.3847446, 9.175763, -3.0520046];
        let ray = Ray3::new([0.0; 3], direction);
        let hit = ray_sphere(ray, Sphere::new(direction, 0.0)).expect("collinear point must hit");
        assert!((hit.enter_distance - hit.exit_distance).abs() <= f64::EPSILON);
        assert_eq!(hit.enter_point, direction.map(f64::from));

        let mut off_line = direction;
        off_line[2] = f32::from_bits(off_line[2].to_bits() + 1);
        assert!(ray_sphere(ray, Sphere::new(off_line, 0.0)).is_none());
    }

    #[test]
    fn ray_sphere_handles_tangent_inside_and_behind() {
        let sphere = Sphere::new([0.0; 3], 1.0);
        let tangent = ray_sphere(Ray3::new([-2.0, 1.0, 0.0], [1.0, 0.0, 0.0]), sphere).unwrap();
        assert_eq!(tangent.enter_distance, 2.0);
        assert_eq!(tangent.exit_distance, 2.0);

        let inside = ray_sphere(Ray3::new([0.0; 3], [1.0, 0.0, 0.0]), sphere).unwrap();
        assert_eq!(inside.enter_distance, 0.0);
        assert_eq!(inside.exit_distance, 1.0);
        assert!(inside.starts_inside);

        assert!(ray_sphere(Ray3::new([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]), sphere).is_none());
    }

    fn face_oracle(ray: Ray3, aabb: Aabb) -> Option<(f64, f64)> {
        let origin = ray.origin64();
        let direction = ray.direction();
        let inside = (0..3).all(|axis| {
            origin[axis] >= f64::from(aabb.min[axis]) && origin[axis] <= f64::from(aabb.max[axis])
        });
        let mut distances = Vec::new();
        if inside {
            distances.push(0.0);
        }
        for axis in 0..3 {
            if direction[axis] == 0.0 {
                continue;
            }
            for plane in [aabb.min[axis], aabb.max[axis]] {
                let distance = (f64::from(plane) - origin[axis]) / direction[axis];
                if distance < 0.0 {
                    continue;
                }
                let point = ray.at(distance);
                let on_face = (0..3).all(|other| {
                    let tolerance = 1.0e-9_f64.max(distance.abs() * 1.0e-12);
                    point[other] >= f64::from(aabb.min[other]) - tolerance
                        && point[other] <= f64::from(aabb.max[other]) + tolerance
                });
                if on_face {
                    distances.push(distance);
                }
            }
        }
        if distances.is_empty() {
            None
        } else {
            distances.sort_by(f64::total_cmp);
            Some((*distances.first().unwrap(), *distances.last().unwrap()))
        }
    }

    #[test]
    fn aabb_slab_query_matches_independent_face_enumeration() {
        let mut state = 0x6d2b79f5_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1_u32 << 24) as f32
        };
        for _ in 0..2048 {
            let a: [f32; 3] = std::array::from_fn(|_| next() * 20.0 - 10.0);
            let size: [f32; 3] = std::array::from_fn(|_| next() * 3.0);
            let aabb = Aabb::new(a, std::array::from_fn(|axis| a[axis] + size[axis]));
            let origin = std::array::from_fn(|_| next() * 30.0 - 15.0);
            let mut direction = std::array::from_fn(|_| next() * 2.0 - 1.0);
            if direction == [0.0; 3] {
                direction[0] = 1.0;
            }
            let ray = Ray3::new(origin, direction);
            let expected = face_oracle(ray, aabb);
            let actual = ray_aabb(ray, aabb);
            assert_eq!(
                actual.is_some(),
                expected.is_some(),
                "ray={ray:?}, aabb={aabb:?}"
            );
            if let (Some(actual), Some((enter, exit))) = (actual, expected) {
                assert!((actual.enter_distance - enter).abs() <= 1.0e-8);
                assert!((actual.exit_distance - exit).abs() <= 1.0e-8);
            }
        }
    }

    fn sphere_quadratic_oracle(ray: Ray3, sphere: Sphere) -> Option<(f64, f64)> {
        let origin = ray.origin64();
        let direction = ray.direction();
        let center = sphere.center.map(f64::from);
        let oc: [f64; 3] = std::array::from_fn(|axis| origin[axis] - center[axis]);
        let b = 2.0 * (0..3).map(|axis| oc[axis] * direction[axis]).sum::<f64>();
        let c = oc.into_iter().map(|v| v * v).sum::<f64>() - f64::from(sphere.radius).powi(2);
        let discriminant = b * b - 4.0 * c;
        if discriminant < 0.0 {
            return None;
        }
        let root = discriminant.sqrt();
        let near = (-b - root) * 0.5;
        let far = (-b + root) * 0.5;
        (far >= 0.0).then_some((near.max(0.0), far))
    }

    #[test]
    fn sphere_query_matches_quadratic_oracle_on_generated_inputs() {
        let mut state = 0x1234abcd_u32;
        let mut next = || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 8) as f32 / (1_u32 << 24) as f32
        };
        for _ in 0..4096 {
            let sphere = Sphere::new(std::array::from_fn(|_| next() * 20.0 - 10.0), next() * 4.0);
            let origin = std::array::from_fn(|_| next() * 30.0 - 15.0);
            let mut direction = std::array::from_fn(|_| next() * 2.0 - 1.0);
            if direction == [0.0; 3] {
                direction[0] = 1.0;
            }
            let ray = Ray3::new(origin, direction);
            let expected = sphere_quadratic_oracle(ray, sphere);
            let actual = ray_sphere(ray, sphere);
            assert_eq!(actual.is_some(), expected.is_some());
            if let (Some(actual), Some((enter, exit))) = (actual, expected) {
                assert!((actual.enter_distance - enter).abs() <= 1.0e-8);
                assert!((actual.exit_distance - exit).abs() <= 1.0e-8);
            }
        }
    }

    #[test]
    #[should_panic(expected = "ray direction must be non-zero")]
    fn rejects_zero_direction() {
        let _ = Ray3::new([0.0; 3], [0.0; 3]);
    }
}
