//! Contact queries for pairs with a curved member.

use super::closest::{
    clamp_aabb, closest_point_segment, closest_segment_aabb, closest_segment_segment,
    nearest_box_face, segment_aabb_interval,
};
use super::frame::{fallback_normal, inverse_rotate, rotate_local, unit_or_zero};
use super::query::{PrimitiveContact3, PrimitiveWork3};
use super::shape::{PrimitiveBody3, PrimitiveKind3};
use super::wedge::{
    closest_point_wedge, closest_segment_wedge, segment_face_penetration, segment_wedge_interval,
    wedge_inside_depth, wedge_planes,
};
use crate::math3::{Vec3, add, cross, dot, length, neg, scale, sub};

pub(super) fn sphere_sphere(a: PrimitiveBody3, b: PrimitiveBody3) -> PrimitiveContact3 {
    let delta = sub(b.position, a.position);
    let distance = length(delta);
    let normal = fallback_normal(delta, sub(b.position, a.position));
    let a_radius = a.shape.sphere_radius();
    let b_radius = b.shape.sphere_radius();
    PrimitiveContact3 {
        normal,
        separation: distance - a_radius - b_radius,
        point_a: add(a.position, scale(normal, a_radius)),
        point_b: sub(b.position, scale(normal, b_radius)),
    }
}

pub(super) fn sphere_box(sphere: PrimitiveBody3, box_body: PrimitiveBody3) -> PrimitiveContact3 {
    let half = box_body.shape.half_extents();
    let local = inverse_rotate(box_body.axes, sub(sphere.position, box_body.position));
    let radius = sphere.shape.sphere_radius();
    if local
        .into_iter()
        .enumerate()
        .all(|(axis, coordinate)| coordinate >= -half[axis] && coordinate <= half[axis])
    {
        let (outward, depth) = nearest_box_face(local, half);
        let outward = rotate_local(box_body.axes, outward);
        let normal = neg(outward);
        return PrimitiveContact3 {
            normal,
            separation: -depth - radius,
            point_a: add(sphere.position, scale(normal, radius)),
            point_b: add(sphere.position, scale(outward, depth)),
        };
    }

    let local_box = clamp_aabb(local, half);
    let point_b = add(box_body.position, rotate_local(box_body.axes, local_box));
    let delta = sub(point_b, sphere.position);
    let distance = length(delta);
    let normal = fallback_normal(delta, sub(box_body.position, sphere.position));
    PrimitiveContact3 {
        normal,
        separation: distance - radius,
        point_a: add(sphere.position, scale(normal, radius)),
        point_b,
    }
}

fn capsule_segment(body: PrimitiveBody3) -> (Vec3, Vec3) {
    let (half_segment, _) = body.shape.capsule_parts();
    let axis = scale(body.axes[1], half_segment);
    (sub(body.position, axis), add(body.position, axis))
}

pub(super) fn capsule_sphere(
    capsule: PrimitiveBody3,
    sphere: PrimitiveBody3,
    work: &mut PrimitiveWork3,
) -> PrimitiveContact3 {
    let (a, b) = capsule_segment(capsule);
    let (core, _) = closest_segment_segment(a, b, sphere.position, sphere.position, work);
    let delta = sub(sphere.position, core);
    let (_, capsule_radius) = capsule.shape.capsule_parts();
    let sphere_radius = sphere.shape.sphere_radius();
    let (normal, normal_distance) = capsule_normal(delta, capsule, sphere);
    PrimitiveContact3 {
        normal,
        separation: normal_distance - capsule_radius - sphere_radius,
        point_a: add(core, scale(normal, capsule_radius)),
        point_b: sub(sphere.position, scale(normal, sphere_radius)),
    }
}

pub(super) fn capsule_capsule(
    a: PrimitiveBody3,
    b: PrimitiveBody3,
    work: &mut PrimitiveWork3,
) -> PrimitiveContact3 {
    let (a0, a1) = capsule_segment(a);
    let (b0, b1) = capsule_segment(b);
    let (point_a, point_b) = closest_segment_segment(a0, a1, b0, b1, work);
    let delta = sub(point_b, point_a);
    let (_, a_radius) = a.shape.capsule_parts();
    let (_, b_radius) = b.shape.capsule_parts();
    let (normal, normal_distance) = capsule_normal(delta, a, b);
    PrimitiveContact3 {
        normal,
        separation: normal_distance - a_radius - b_radius,
        point_a: add(point_a, scale(normal, a_radius)),
        point_b: sub(point_b, scale(normal, b_radius)),
    }
}

// At a skeleton intersection, center-to-center motion can run along a skeleton
// and leave the solids penetrating. Choose a direction perpendicular to every
// nonzero skeleton instead. Near-zero closest-point deltas are classified with
// a scale-aware budget including pose-coordinate roundoff. Projection keeps
// witnesses and separation consistent with the chosen direction rather than
// understating its clearance.
fn capsule_normal(delta: Vec3, left: PrimitiveBody3, right: PrimitiveBody3) -> (Vec3, f64) {
    let left_axis = capsule_axis(left);
    let right_axis = capsule_axis(right);
    let distance = length(delta);
    let coordinate_span = left
        .position
        .into_iter()
        .chain(right.position)
        .map(f64::abs)
        .fold(0.0, f64::max);
    let characteristic_length = coordinate_span
        .max(left.shape.radius())
        .max(right.shape.radius());
    let tolerance = 64.0 * f64::EPSILON * characteristic_length;
    let radius_sum = left.shape.half_extents()[0] + right.shape.half_extents()[0];
    // A normal tie must never turn a resolved separation into a contact. For
    // two collapsed skeletons the center delta is already the complete geometry.
    if distance >= radius_sum
        || distance > tolerance
        || (left_axis.is_none() && right_axis.is_none() && distance > 0.0)
    {
        return (unit_or_zero(delta), distance);
    }
    let mut normal = match (left_axis, right_axis) {
        (Some(left), Some(right)) => {
            // Subtract nearly parallel axes before crossing: this avoids the
            // cancellation of order-one products at tiny relative angles.
            let relative = if dot(left, right) >= 0.0 {
                sub(right, left)
            } else {
                add(right, left)
            };
            let perpendicular = unit_or_zero(cross(left, relative));
            if perpendicular == [0.0; 3] {
                perpendicular_to(left)
            } else {
                perpendicular
            }
        }
        (Some(axis), None) | (None, Some(axis)) => perpendicular_to(axis),
        (None, None) => [1.0, 0.0, 0.0],
    };
    let projection = dot(normal, delta);
    if projection < 0.0
        || (projection == 0.0 && dot(normal, sub(right.position, left.position)) < 0.0)
    {
        normal = neg(normal);
    }
    (normal, dot(normal, delta))
}

fn capsule_axis(body: PrimitiveBody3) -> Option<Vec3> {
    (body.shape.kind() == PrimitiveKind3::Capsule && body.shape.capsule_parts().0 > 0.0)
        .then_some(body.axes[1])
}

fn perpendicular_to(axis: Vec3) -> Vec3 {
    let mut least_aligned = 0;
    for index in 1..3 {
        if axis[index].abs() < axis[least_aligned].abs() {
            least_aligned = index;
        }
    }
    let mut basis = [0.0; 3];
    basis[least_aligned] = 1.0;
    unit_or_zero(cross(axis, basis))
}

pub(super) fn capsule_box(
    capsule: PrimitiveBody3,
    box_body: PrimitiveBody3,
    work: &mut PrimitiveWork3,
) -> PrimitiveContact3 {
    let half = box_body.shape.half_extents();
    let (_, radius) = capsule.shape.capsule_parts();
    let (world_a, world_b) = capsule_segment(capsule);
    let local_a = inverse_rotate(box_body.axes, sub(world_a, box_body.position));
    let local_b = inverse_rotate(box_body.axes, sub(world_b, box_body.position));

    if segment_aabb_interval(local_a, local_b, half, work).is_some() {
        let capsule_axis = unit_or_zero(sub(local_b, local_a));
        let candidates = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            cross(capsule_axis, [1.0, 0.0, 0.0]),
            cross(capsule_axis, [0.0, 1.0, 0.0]),
            cross(capsule_axis, [0.0, 0.0, 1.0]),
        ];
        let mut best = ([1.0, 0.0, 0.0], f64::INFINITY, local_a, 0.0);
        for candidate in candidates {
            let axis = unit_or_zero(candidate);
            if axis == [0.0; 3] {
                continue;
            }
            work.axes_tested += 1;
            let box_radius =
                half[0] * axis[0].abs() + half[1] * axis[1].abs() + half[2] * axis[2].abs();
            let a_projection = dot(axis, local_a);
            let b_projection = dot(axis, local_b);
            let segment_min = a_projection.min(b_projection);
            let segment_max = a_projection.max(b_projection);
            let positive = box_radius - (segment_min - radius);
            let negative = segment_max + radius + box_radius;
            let (outward, penetration) = if positive <= negative {
                (axis, positive)
            } else {
                (neg(axis), negative)
            };
            if penetration >= best.1 {
                continue;
            }

            let outward_a = dot(outward, local_a);
            let outward_b = dot(outward, local_b);
            let projection_scale = outward_a.abs().max(outward_b.abs()).max(box_radius);
            let projection_tolerance =
                64.0 * f64::EPSILON * projection_scale.max(f64::MIN_POSITIVE);
            let box_support = [
                if outward[0] < 0.0 { -half[0] } else { half[0] },
                if outward[1] < 0.0 { -half[1] } else { half[1] },
                if outward[2] < 0.0 { -half[2] } else { half[2] },
            ];
            let local_core = if (outward_a - outward_b).abs() <= projection_tolerance {
                closest_point_segment(box_support, local_a, local_b, work)
            } else if outward_a < outward_b {
                local_a
            } else {
                local_b
            };
            let core_depth = (box_radius - dot(outward, local_core)).max(0.0);
            best = (outward, penetration.max(0.0), local_core, core_depth);
        }

        let (outward, penetration, local_core, core_depth) = best;
        let core = add(box_body.position, rotate_local(box_body.axes, local_core));
        let outward = rotate_local(box_body.axes, outward);
        let normal = neg(outward);
        return PrimitiveContact3 {
            normal,
            separation: -penetration,
            point_a: add(core, scale(normal, radius)),
            point_b: add(core, scale(outward, core_depth)),
        };
    }

    let (local_core, local_box) = closest_segment_aabb(local_a, local_b, half, work);
    let core = add(box_body.position, rotate_local(box_body.axes, local_core));
    let point_b = add(box_body.position, rotate_local(box_body.axes, local_box));
    let delta = sub(point_b, core);
    let distance = length(delta);
    let normal = fallback_normal(delta, sub(box_body.position, capsule.position));
    PrimitiveContact3 {
        normal,
        separation: distance - radius,
        point_a: add(core, scale(normal, radius)),
        point_b,
    }
}

pub(super) fn sphere_wedge(sphere: PrimitiveBody3, wedge: PrimitiveBody3) -> PrimitiveContact3 {
    let half = wedge.shape.half_extents();
    let radius = sphere.shape.sphere_radius();
    let local = inverse_rotate(wedge.axes, sub(sphere.position, wedge.position));
    if let Some((outward, depth)) = wedge_inside_depth(local, half) {
        let outward_world = rotate_local(wedge.axes, outward);
        let normal = neg(outward_world);
        return PrimitiveContact3 {
            normal,
            separation: -depth - radius,
            point_a: add(sphere.position, scale(normal, radius)),
            point_b: add(sphere.position, scale(outward_world, depth)),
        };
    }

    let local_wedge = closest_point_wedge(local, half);
    let point_b = add(wedge.position, rotate_local(wedge.axes, local_wedge));
    let delta = sub(point_b, sphere.position);
    let distance = length(delta);
    let normal = fallback_normal(delta, sub(wedge.position, sphere.position));
    PrimitiveContact3 {
        normal,
        separation: distance - radius,
        point_a: add(sphere.position, scale(normal, radius)),
        point_b,
    }
}

pub(super) fn capsule_wedge(
    capsule: PrimitiveBody3,
    wedge: PrimitiveBody3,
    work: &mut PrimitiveWork3,
) -> PrimitiveContact3 {
    let half = wedge.shape.half_extents();
    let (_, radius) = capsule.shape.capsule_parts();
    let (world_a, world_b) = capsule_segment(capsule);
    let local_a = inverse_rotate(wedge.axes, sub(world_a, wedge.position));
    let local_b = inverse_rotate(wedge.axes, sub(world_b, wedge.position));

    if segment_wedge_interval(local_a, local_b, half).is_some() {
        let (outward, penetration, local_core, core_depth) =
            segment_face_penetration(local_a, local_b, &wedge_planes(half), radius);
        let core = add(wedge.position, rotate_local(wedge.axes, local_core));
        let outward = rotate_local(wedge.axes, outward);
        let normal = neg(outward);
        return PrimitiveContact3 {
            normal,
            separation: -penetration,
            point_a: add(core, scale(normal, radius)),
            point_b: add(core, scale(outward, core_depth)),
        };
    }

    let (local_core, local_wedge) = closest_segment_wedge(local_a, local_b, half, work);
    let core = add(wedge.position, rotate_local(wedge.axes, local_core));
    let point_b = add(wedge.position, rotate_local(wedge.axes, local_wedge));
    let delta = sub(point_b, core);
    let distance = length(delta);
    let normal = fallback_normal(delta, sub(wedge.position, capsule.position));
    PrimitiveContact3 {
        normal,
        separation: distance - radius,
        point_a: add(core, scale(normal, radius)),
        point_b,
    }
}
