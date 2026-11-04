//! Shared fixtures and the generic convex reference for the experimental platonic sweeps.
//!
//! The reference cast is conservative advancement over [`gjk_distance`] on the solids' world
//! vertex hulls ([`ConvexHull3`]). It shares no axis enumeration or projection code with the
//! swept-SAT kernel under test.

use geometry_kernels::gjk_distance::{GjkDistanceConfig, gjk_distance_with_config};
use geometry_kernels::math3::{Vec3, add, dot, scale, sub};
use geometry_kernels::primitive3::platonic::{PlatonicBody3, PlatonicShape3};
use geometry_kernels::support::ConvexHull3;

/// Separation at or below `margin + REFERENCE_DISTANCE_TOLERANCE * size` is contact.
pub const REFERENCE_DISTANCE_TOLERANCE: f64 = 1.0e-9;
/// Conservative-advancement iteration budget before the reference gives up.
pub const REFERENCE_MAX_ITERATIONS: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReferenceCast {
    Hit { time: f64, iterations: u32 },
    Miss { iterations: u32 },
    Unresolved,
}

/// Minkowski separation of the two bodies after both translate for `time` seconds.
/// Contact (GJK intersection or touching) reports zero.
pub fn separation(a: PlatonicBody3, b: PlatonicBody3, time: f64) -> Option<(f64, Vec3)> {
    let (left, left_len) = a.world_vertices(time);
    let (right, right_len) = b.world_vertices(time);
    let result = gjk_distance_with_config(
        &ConvexHull3::new(&left[..left_len]),
        &ConvexHull3::new(&right[..right_len]),
        GjkDistanceConfig {
            max_iterations: 64,
            ..GjkDistanceConfig::default()
        },
    );
    match result.closest {
        Some(closest) => Some((closest.distance, closest.normal)),
        None => match result.status {
            geometry_kernels::gjk_distance::GjkDistanceStatus::IterationLimit => None,
            _ => Some((0.0, [0.0; 3])),
        },
    }
}

/// Generic translational convex cast: conservative advancement with GJK distance.
/// `dt` is the window in seconds; the returned time is normalized to `[0, 1]`.
pub fn conservative_advancement(
    a: PlatonicBody3,
    b: PlatonicBody3,
    dt: f64,
    margin: f64,
) -> ReferenceCast {
    let size = a.shape.circumradius().max(b.shape.circumradius());
    let tolerance = REFERENCE_DISTANCE_TOLERANCE * size;
    let relative = scale(sub(b.velocity, a.velocity), dt);
    let mut time = 0.0;
    for iteration in 1..=REFERENCE_MAX_ITERATIONS {
        let Some((distance, normal)) = separation(a, b, time * dt) else {
            return ReferenceCast::Unresolved;
        };
        let gap = distance - margin;
        if gap <= tolerance {
            return ReferenceCast::Hit {
                time,
                iterations: iteration,
            };
        }
        let closing = -dot(relative, normal);
        if closing <= 0.0 {
            return ReferenceCast::Miss {
                iterations: iteration,
            };
        }
        time += gap / closing;
        if time > 1.0 {
            return ReferenceCast::Miss {
                iterations: iteration,
            };
        }
    }
    ReferenceCast::Unresolved
}

/// Deterministic 64-bit LCG with a 53-bit unit output.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }

    pub fn signed(&mut self) -> f64 {
        2.0 * self.unit() - 1.0
    }

    pub fn direction(&mut self) -> Vec3 {
        loop {
            let v = [self.signed(), self.signed(), self.signed()];
            let length = dot(v, v).sqrt();
            if length > 0.1 && length <= 1.0 {
                return scale(v, length.recip());
            }
        }
    }

    /// Orthonormal frame from a uniformly sampled unit quaternion.
    pub fn rotation(&mut self) -> [Vec3; 3] {
        loop {
            let q = [self.signed(), self.signed(), self.signed(), self.signed()];
            let norm = q.iter().map(|value| value * value).sum::<f64>().sqrt();
            if norm > 0.1 && norm <= 1.0 {
                let [w, x, y, z] = q.map(|value| value / norm);
                return quaternion_axes(w, x, y, z);
            }
        }
    }
}

pub fn quaternion_axes(w: f64, x: f64, y: f64, z: f64) -> [Vec3; 3] {
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        [
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ],
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Rotation by `angle` radians about the unit `axis`.
pub fn axis_angle(axis: Vec3, angle: f64) -> [Vec3; 3] {
    let (sin, cos) = (0.5 * angle).sin_cos();
    quaternion_axes(cos, axis[0] * sin, axis[1] * sin, axis[2] * sin)
}

pub const DT: f64 = 1.0 / 60.0;

/// Randomly oriented pairs with `S = R_a + R_b`. `B` starts `2 S` from `A` along a random
/// direction, offset sideways by `offset * S`, and moves `4 S` along `-direction` over the
/// window (`receding` reverses the motion).
pub fn random_pairs<const N: usize>(
    seed: u64,
    a_shape: PlatonicShape3,
    b_shape: PlatonicShape3,
    offset: (f64, f64),
    receding: bool,
) -> [[PlatonicBody3; 2]; N] {
    let mut rng = Lcg(seed);
    let size = a_shape.circumradius() + b_shape.circumradius();
    std::array::from_fn(|_| {
        let a_axes = rng.rotation();
        let b_axes = rng.rotation();
        let direction = rng.direction();
        let random = rng.direction();
        let side = sub(random, scale(direction, dot(random, direction)));
        let side_length = dot(side, side).sqrt();
        let side = if side_length > 1.0e-6 {
            scale(side, side_length.recip())
        } else {
            [direction[1], -direction[0], 0.0]
        };
        let amount = offset.0 + (offset.1 - offset.0) * rng.unit();
        let start = add(scale(direction, 2.0 * size), scale(side, amount * size));
        let speed = if receding { 4.0 } else { -4.0 } * size / DT;
        [
            PlatonicBody3::new(a_shape, [0.0; 3], a_axes, [0.0; 3]),
            PlatonicBody3::new(b_shape, start, b_axes, scale(direction, speed)),
        ]
    })
}
