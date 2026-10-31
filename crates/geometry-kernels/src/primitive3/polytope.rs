//! Separating-axis contact for box and wedge pairs.

use super::frame::{rotate_local, unit_or_zero};
use super::query::{PrimitiveContact3, PrimitiveWork3, support_point};
use super::shape::{PrimitiveBody3, PrimitiveKind3};
use crate::math3::{Vec3, cross, dot, length_squared, neg, normalized};

pub(super) fn poly_poly(
    a: PrimitiveBody3,
    b: PrimitiveBody3,
    work: &mut PrimitiveWork3,
) -> PrimitiveContact3 {
    let (axes, len) = poly_axes(a, b);
    let mut best_separation = f64::NEG_INFINITY;
    let mut best_normal = [1.0, 0.0, 0.0];

    for &axis in &axes[..len] {
        work.axes_tested += 1;
        let (a_min, a_max) = projection_interval(a, axis, work);
        let (b_min, b_max) = projection_interval(b, axis, work);
        let forward = b_min - a_max;
        let backward = a_min - b_max;
        let (separation, normal) = if forward >= backward {
            (forward, axis)
        } else {
            (backward, neg(axis))
        };
        if separation > best_separation {
            best_separation = separation;
            best_normal = normal;
        }
    }

    PrimitiveContact3 {
        normal: best_normal,
        separation: best_separation,
        point_a: support_point(a, best_normal, work),
        point_b: support_point(b, neg(best_normal), work),
    }
}

pub(super) fn projection_interval(
    body: PrimitiveBody3,
    axis: Vec3,
    work: &mut PrimitiveWork3,
) -> (f64, f64) {
    let maximum = dot(support_point(body, axis, work), axis);
    let minimum = dot(support_point(body, neg(axis), work), axis);
    (minimum, maximum)
}

pub(super) fn poly_axes(a: PrimitiveBody3, b: PrimitiveBody3) -> ([Vec3; 32], usize) {
    let mut axes = [[0.0; 3]; 32];
    let mut len = 0;
    append_face_axes(a, &mut axes, &mut len);
    append_face_axes(b, &mut axes, &mut len);

    let (a_edges, a_len) = edge_axes(a);
    let (b_edges, b_len) = edge_axes(b);
    for &left in &a_edges[..a_len] {
        for &right in &b_edges[..b_len] {
            push_axis(&mut axes, &mut len, cross(left, right));
        }
    }
    (axes, len)
}

fn append_face_axes(body: PrimitiveBody3, axes: &mut [Vec3; 32], len: &mut usize) {
    match body.shape.kind {
        PrimitiveKind3::Box => {
            for axis in body.axes {
                push_axis(axes, len, axis);
            }
        }
        PrimitiveKind3::Wedge => {
            let half = body.shape.half_extents();
            for local in [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                unit_or_zero([half[1], half[0], 0.0]),
            ] {
                push_axis(axes, len, rotate_local(body.axes, local));
            }
        }
        _ => {}
    }
}

fn edge_axes(body: PrimitiveBody3) -> ([Vec3; 4], usize) {
    let mut out = [[0.0; 3]; 4];
    match body.shape.kind {
        PrimitiveKind3::Box => {
            out[..3].copy_from_slice(&body.axes);
            (out, 3)
        }
        PrimitiveKind3::Wedge => {
            let half = body.shape.half_extents();
            for (index, local) in [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                unit_or_zero([half[0], -half[1], 0.0]),
            ]
            .into_iter()
            .enumerate()
            {
                out[index] = rotate_local(body.axes, local);
            }
            (out, 4)
        }
        _ => (out, 0),
    }
}

fn push_axis(axes: &mut [Vec3; 32], len: &mut usize, axis: Vec3) {
    if length_squared(axis) <= 1.0e-12 {
        return;
    }
    let Some(axis) = normalized(axis) else {
        return;
    };
    if axes[..*len]
        .iter()
        .any(|existing| dot(*existing, axis).abs() >= 1.0 - 1.0e-10)
    {
        return;
    }
    axes[*len] = axis;
    *len += 1;
}
