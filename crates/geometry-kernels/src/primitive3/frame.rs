//! Scale-safe normalization and local/world frame conversions.

use crate::math3::{Vec3, add, dot, length_squared, scale};

pub(super) fn fallback_normal(delta: Vec3, fallback: Vec3) -> Vec3 {
    if length_squared(delta) > 1.0e-20 {
        unit_or_zero(delta)
    } else if length_squared(fallback) > 1.0e-20 {
        unit_or_zero(fallback)
    } else {
        [1.0, 0.0, 0.0]
    }
}

pub(super) fn unit_or_zero(vector: Vec3) -> Vec3 {
    let max_component = vector.into_iter().map(f64::abs).fold(0.0, f64::max);
    if max_component == 0.0 || !max_component.is_finite() {
        return [0.0; 3];
    }
    let scaled = [
        vector[0] / max_component,
        vector[1] / max_component,
        vector[2] / max_component,
    ];
    let length = length_squared(scaled).sqrt();
    if length == 0.0 || !length.is_finite() {
        [0.0; 3]
    } else {
        scale(scaled, length.recip())
    }
}

pub(super) fn abs(vector: Vec3) -> Vec3 {
    [vector[0].abs(), vector[1].abs(), vector[2].abs()]
}

pub(super) fn rotate_local(axes: [Vec3; 3], local: Vec3) -> Vec3 {
    add(
        add(scale(axes[0], local[0]), scale(axes[1], local[1])),
        scale(axes[2], local[2]),
    )
}

pub(super) fn inverse_rotate(axes: [Vec3; 3], world: Vec3) -> Vec3 {
    [
        dot(world, axes[0]),
        dot(world, axes[1]),
        dot(world, axes[2]),
    ]
}
