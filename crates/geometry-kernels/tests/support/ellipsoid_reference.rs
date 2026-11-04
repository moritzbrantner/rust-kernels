//! Generic support-based ellipsoid pair time of impact (#190 reference).
//!
//! Conservative advancement over GJK distance on the analytic `SupportMap3`. It does not use
//! the shared transform, the sphere solver or any homothety, so it applies to any two
//! fixed-orientation ellipsoids. Under pure translation the separation is convex in time, so
//! each step `gap / closing` along the current witness normal never passes first contact.
//! Shared by the pair tests and the Divan benchmark.

use geometry_kernels::gjk_distance::{
    GjkDistanceConfig, GjkDistanceStatus, gjk_distance_with_config,
};
use geometry_kernels::math3::{Vec3, add, dot, scale, sub};
use geometry_kernels::primitive3::ellipsoid::EllipsoidBody3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReferenceToi {
    Overlapping,
    Hit(f64),
    Miss,
    /// The search did not conclude; never a miss.
    Unresolved,
}

fn moved(body: EllipsoidBody3, offset: Vec3) -> EllipsoidBody3 {
    EllipsoidBody3::new(
        body.shape,
        body.pose
            .try_with_center(add(body.pose.center(), offset))
            .expect("finite reference pose"),
    )
}

/// `tolerance` is the absolute world gap accepted as contact.
pub fn conservative_advancement(
    a: EllipsoidBody3,
    velocity_a: Vec3,
    b: EllipsoidBody3,
    velocity_b: Vec3,
    dt: f64,
    tolerance: f64,
    max_iterations: usize,
) -> ReferenceToi {
    let travel_a = scale(velocity_a, dt);
    let travel_b = scale(velocity_b, dt);
    let relative = sub(travel_b, travel_a);
    let config = GjkDistanceConfig {
        max_iterations: 128,
        relative_tolerance: 1.0e-14,
        epsilon: tolerance * 1.0e-3,
    };
    let mut time = 0.0_f64;
    for iteration in 0..max_iterations {
        let left = moved(a, scale(travel_a, time));
        let right = moved(b, scale(travel_b, time));
        let result = gjk_distance_with_config(&left, &right, config);
        let Some(closest) = result.closest else {
            return if result.status == GjkDistanceStatus::Intersecting {
                if iteration == 0 {
                    ReferenceToi::Overlapping
                } else {
                    ReferenceToi::Hit(time)
                }
            } else {
                ReferenceToi::Unresolved
            };
        };
        if closest.distance <= tolerance {
            return ReferenceToi::Hit(time);
        }
        // `normal` points from `left` toward `right`; `right` closes on `left` along -normal.
        let normal = closest.normal;
        let closing = -dot(relative, normal);
        if closing <= 0.0 {
            return ReferenceToi::Miss;
        }
        // The separating-plane gap along `normal` is a lower bound on the true distance,
        // whatever GJK's convergence, so advancing by it never passes first contact.
        let lower = dot(sub(right.pose.center(), left.pose.center()), normal)
            - support_radius(right, scale(normal, -1.0))
            - support_radius(left, normal);
        if lower <= 0.5 * tolerance {
            return ReferenceToi::Unresolved;
        }
        time += (lower - 0.5 * tolerance) / closing;
        if time > 1.0 {
            return ReferenceToi::Miss;
        }
    }
    ReferenceToi::Unresolved
}

fn support_radius(body: EllipsoidBody3, direction: Vec3) -> f64 {
    body.try_support_radius(direction)
        .expect("finite reference direction")
}
