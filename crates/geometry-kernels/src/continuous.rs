use crate::{
    Sphere,
    math3::{Vec3, length_squared},
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


/// Checked first-contact fraction for fixed-orientation spheres translated over one step.
///
/// Both center displacement and radius are scaled together before the line/sphere
/// intersection, avoiding f64 squared-distance overflow and preserving tiny sweeps.
/// This is the f64 primitive counterpart of the public sphere/ray CCD query.
/// No collision response or discrete-contact policy lives here.
pub(crate) fn try_sphere_sphere_fraction_f64(
    initial_delta: Vec3,
    displacement: Vec3,
    combined_radius: f64,
) -> Result<Option<f64>, ()> {
    if !crate::math3::is_finite(initial_delta)
        || !crate::math3::is_finite(displacement)
        || !combined_radius.is_finite()
        || combined_radius < 0.0
    {
        return Err(());
    }

    let magnitude = initial_delta
        .into_iter()
        .chain(displacement)
        .chain([combined_radius])
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if magnitude == 0.0 {
        return Ok(Some(0.0));
    }
    let delta = initial_delta.map(|value| value / magnitude);
    let travel = displacement.map(|value| value / magnitude);
    let radius = combined_radius / magnitude;
    let radius_squared = radius * radius;
    if length_squared(delta) <= radius_squared {
        return Ok(Some(0.0));
    }

    let travel_squared = length_squared(travel);
    if travel_squared == 0.0 {
        return Ok(None);
    }
    let approach = -crate::math3::dot(delta, travel);
    if approach <= 0.0 {
        return Ok(None);
    }

    // The cross-product line distance is stable for tangencies, unlike
    // subtracting nearly equal terms in a direct quadratic discriminant.
    let perpendicular_squared =
        length_squared(crate::math3::cross(delta, travel)) / travel_squared;
    let boundary_tolerance =
        4.0 * f64::EPSILON * perpendicular_squared.max(radius_squared);
    if perpendicular_squared > radius_squared + boundary_tolerance {
        return Ok(None);
    }
    let center_time = approach / travel_squared;
    let half_chord =
        ((radius_squared - perpendicular_squared).max(0.0) / travel_squared).sqrt();
    let enter = center_time - half_chord;
    if !enter.is_finite() || !half_chord.is_finite() {
        return Err(());
    }

    Ok((enter <= 1.0 && center_time + half_chord >= 0.0).then_some(enter.max(0.0)))
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
    use crate::{
        Sphere,
        math3::{length_squared, sub},
    };

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
    fn grazing_sweep_preserves_tangent_contact() {
        let moving = Sphere::new([0.0; 3], 0.0);
        let target = Sphere::new([-19199.0, 20393.0, 0.0], 19805.0);
        let hit = sphere_sphere_time_of_impact(moving, [597.0, 19796.0, 0.0], target, 2.0)
            .expect("exact grazing sweep must hit");
        assert!((hit.time - 1.0).abs() <= 4.0 * f64::EPSILON);

        let near_miss = Sphere::new([-19200.0, 20393.0, 0.0], 19805.0);
        assert!(
            sphere_sphere_time_of_impact(moving, [597.0, 19796.0, 0.0], near_miss, 2.0,).is_none()
        );
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
