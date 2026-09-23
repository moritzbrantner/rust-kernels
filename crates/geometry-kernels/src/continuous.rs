use crate::{
    Sphere,
    math3::{Vec3, length_squared, sub},
    ray::{Ray3, ray_sphere_radius},
    sphere_sphere,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereSweepHit {
    /// Time from the start of the sweep. Starts-overlapping returns zero.
    pub time: f64,
    /// World-space distance travelled by the moving sphere center.
    pub distance: f64,
    pub moving_center: Vec3,
}

/// Earliest contact between a linearly moving sphere and a stationary sphere
/// during `[0, max_time]`. Touching at the start counts as time zero.
///
/// Velocity is not normalized and may be zero. The kernel owns only time of
/// impact; response, integration and collision filtering remain caller policy.
#[must_use]
pub fn sphere_sphere_time_of_impact(
    moving: Sphere,
    velocity: [f32; 3],
    target: Sphere,
    max_time: f64,
) -> Option<SphereSweepHit> {
    assert!(
        velocity.iter().all(|value| value.is_finite()),
        "sphere sweep velocity must be finite"
    );
    assert!(
        max_time.is_finite() && max_time >= 0.0,
        "sphere sweep max_time must be non-negative and finite"
    );

    if sphere_sphere(moving, target).overlaps {
        return Some(SphereSweepHit {
            time: 0.0,
            distance: 0.0,
            moving_center: moving.center.map(f64::from),
        });
    }

    let speed = robust_length(velocity);
    if speed == 0.0 || max_time == 0.0 {
        return None;
    }

    let ray = Ray3::new(moving.center, velocity);
    let radius_sum = f64::from(moving.radius) + f64::from(target.radius);
    let hit = ray_sphere_radius(ray, target.center.map(f64::from), radius_sum)?;
    let time = hit.enter_distance / speed;
    if time > max_time {
        return None;
    }

    Some(SphereSweepHit {
        time,
        distance: hit.enter_distance,
        moving_center: hit.enter_point,
    })
}

fn robust_length(value: [f32; 3]) -> f64 {
    let maximum = value
        .iter()
        .map(|component| f64::from(*component).abs())
        .fold(0.0_f64, f64::max);
    if maximum == 0.0 {
        return 0.0;
    }
    let scaled = value.map(|component| f64::from(component) / maximum);
    maximum * length_squared(scaled).sqrt()
}

#[cfg(test)]
mod tests {
    use super::sphere_sphere_time_of_impact;
    use crate::Sphere;

    #[test]
    fn reports_contact_time_inside_window_and_respects_limit() {
        let moving = Sphere::new([-5.0, 0.0, 0.0], 1.0);
        let target = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let hit = sphere_sphere_time_of_impact(moving, [2.0, 0.0, 0.0], target, 10.0).unwrap();
        assert_eq!(hit.time, 1.5);
        assert_eq!(hit.distance, 3.0);
        assert_eq!(hit.moving_center, [-2.0, 0.0, 0.0]);
        assert!(sphere_sphere_time_of_impact(moving, [2.0, 0.0, 0.0], target, 1.0).is_none());
    }

    #[test]
    fn starts_overlapping_is_zero_even_without_motion() {
        let moving = Sphere::new([0.0, 0.0, 0.0], 2.0);
        let target = Sphere::new([1.0, 0.0, 0.0], 1.0);
        let hit = sphere_sphere_time_of_impact(moving, [0.0; 3], target, 0.0).unwrap();
        assert_eq!(hit.time, 0.0);
        assert_eq!(hit.distance, 0.0);
    }

    #[test]
    fn misses_for_stationary_receding_and_transverse_motion() {
        let moving = Sphere::new([-5.0, 0.0, 0.0], 1.0);
        let target = Sphere::new([0.0; 3], 1.0);
        assert!(sphere_sphere_time_of_impact(moving, [0.0; 3], target, 10.0).is_none());
        assert!(sphere_sphere_time_of_impact(moving, [-1.0, 0.0, 0.0], target, 10.0).is_none());
        assert!(sphere_sphere_time_of_impact(moving, [0.0, 1.0, 0.0], target, 10.0).is_none());
    }

    fn quadratic_oracle(
        moving: Sphere,
        velocity: [f32; 3],
        target: Sphere,
        max_time: f64,
    ) -> Option<f64> {
        let relative = sub(moving.center.map(f64::from), target.center.map(f64::from));
        let velocity = velocity.map(f64::from);
        let radius = f64::from(moving.radius) + f64::from(target.radius);
        let c = length_squared(relative) - radius * radius;
        if c <= 0.0 {
            return Some(0.0);
        }
        let a = length_squared(velocity);
        if a == 0.0 {
            return None;
        }
        let b = 2.0
            * (0..3)
                .map(|axis| relative[axis] * velocity[axis])
                .sum::<f64>();
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return None;
        }
        let time = (-b - discriminant.sqrt()) / (2.0 * a);
        (time >= 0.0 && time <= max_time).then_some(time)
    }

    #[test]
    fn generated_sweeps_match_direct_time_quadratic() {
        let mut state = 0x9e3779b9_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1_u32 << 24) as f32
        };
        for _ in 0..4096 {
            let moving = Sphere::new(std::array::from_fn(|_| next() * 20.0 - 10.0), next() * 2.0);
            let target = Sphere::new(std::array::from_fn(|_| next() * 20.0 - 10.0), next() * 2.0);
            let velocity = std::array::from_fn(|_| next() * 8.0 - 4.0);
            let max_time = 5.0;
            let expected = quadratic_oracle(moving, velocity, target, max_time);
            let actual = sphere_sphere_time_of_impact(moving, velocity, target, max_time);
            assert_eq!(
                actual.is_some(),
                expected.is_some(),
                "{moving:?} {velocity:?} {target:?}"
            );
            if let (Some(actual), Some(expected)) = (actual, expected) {
                assert!((actual.time - expected).abs() <= 1.0e-9);
            }
        }
    }

    #[test]
    fn velocity_scaling_changes_time_but_not_contact_position() {
        let moving = Sphere::new([-10.0, 2.0, 0.0], 1.0);
        let target = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let slow = sphere_sphere_time_of_impact(moving, [1.0, 0.0, 0.0], target, 20.0).unwrap();
        let fast = sphere_sphere_time_of_impact(moving, [4.0, 0.0, 0.0], target, 20.0).unwrap();
        assert!((slow.time / fast.time - 4.0).abs() <= 1.0e-12);
        assert_eq!(slow.moving_center, fast.moving_center);
    }
}
