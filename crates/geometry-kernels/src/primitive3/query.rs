//! Work counters, support mapping and canonical pair dispatch.

use super::frame::{abs, rotate_local, unit_or_zero};
use super::pairs::{
    capsule_box, capsule_capsule, capsule_sphere, capsule_wedge, sphere_box, sphere_sphere,
    sphere_wedge,
};
use super::polytope::poly_poly;
use super::shape::{PrimitiveBody3, PrimitiveKind3, PrimitivePair3};
use super::wedge::wedge_vertices;
use crate::math3::{Vec3, add, dot, neg, scale};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrimitiveWork3 {
    pub pair_dispatches: [u64; 10],
    pub support_evaluations: u64,
    pub axes_tested: u64,
    pub vertex_tests: u64,
    pub sweep_iterations: u64,
    /// Closest segment/point, segment/segment or segment/box problems evaluated.
    /// Includes unsuccessful queries; does not count individual scalar operations.
    pub segment_distance_evaluations: u64,
    /// Segment endpoint regions, box slabs/breakpoints and distance candidates visited.
    /// Triangle-only feature work and numerical validation checks are excluded.
    pub segment_feature_tests: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrimitiveContact3 {
    pub normal: Vec3,
    pub separation: f64,
    pub point_a: Vec3,
    pub point_b: Vec3,
}

#[must_use]
pub fn bounds_extents(body: PrimitiveBody3) -> Vec3 {
    match body.shape.kind {
        PrimitiveKind3::Sphere => {
            let radius = body.shape.sphere_radius();
            [radius, radius, radius]
        }
        PrimitiveKind3::Box | PrimitiveKind3::Wedge => {
            let half = body.shape.half_extents();
            add(
                add(
                    scale(abs(body.axes[0]), half[0]),
                    scale(abs(body.axes[1]), half[1]),
                ),
                scale(abs(body.axes[2]), half[2]),
            )
        }
        PrimitiveKind3::Capsule => {
            let (half_segment, radius) = body.shape.capsule_parts();
            add([radius; 3], scale(abs(body.axes[1]), half_segment))
        }
    }
}

#[must_use]
pub fn support_point(body: PrimitiveBody3, direction: Vec3, work: &mut PrimitiveWork3) -> Vec3 {
    work.support_evaluations += 1;
    support_impl(body, direction, Some(work))
}

fn support_impl(body: PrimitiveBody3, direction: Vec3, work: Option<&mut PrimitiveWork3>) -> Vec3 {
    let unit = unit_or_zero(direction);
    match body.shape.kind {
        PrimitiveKind3::Sphere => add(body.position, scale(unit, body.shape.sphere_radius())),
        PrimitiveKind3::Box => {
            let half = body.shape.half_extents();
            let mut point = body.position;
            for (index, axis) in body.axes.into_iter().enumerate() {
                let projection = dot(direction, axis);
                let sign = if projection < 0.0 { -1.0 } else { 1.0 };
                point = add(point, scale(axis, half[index] * sign));
            }
            point
        }
        PrimitiveKind3::Capsule => {
            let (half_segment, radius) = body.shape.capsule_parts();
            let axis = body.axes[1];
            let end = scale(
                axis,
                half_segment
                    * if dot(direction, axis) < 0.0 {
                        -1.0
                    } else {
                        1.0
                    },
            );
            add(add(body.position, end), scale(unit, radius))
        }
        PrimitiveKind3::Wedge => {
            let vertices = wedge_vertices(body.shape.half_extents());
            if let Some(work) = work {
                work.vertex_tests += vertices.len() as u64;
            }
            let mut best = vertices[0];
            let mut best_projection = dot(direction, rotate_local(body.axes, best));
            for &vertex in &vertices[1..] {
                let projection = dot(direction, rotate_local(body.axes, vertex));
                if projection > best_projection {
                    best = vertex;
                    best_projection = projection;
                }
            }
            add(body.position, rotate_local(body.axes, best))
        }
    }
}

#[must_use]
pub fn query(a: PrimitiveBody3, b: PrimitiveBody3, work: &mut PrimitiveWork3) -> PrimitiveContact3 {
    let (pair, reversed) = PrimitivePair3::canonical(a.shape.kind, b.shape.kind);
    let (left, right) = if reversed { (b, a) } else { (a, b) };
    let contact = query_canonical(pair, left, right, work);
    if reversed { flip(contact) } else { contact }
}

#[must_use]
pub fn query_canonical(
    pair: PrimitivePair3,
    a: PrimitiveBody3,
    b: PrimitiveBody3,
    work: &mut PrimitiveWork3,
) -> PrimitiveContact3 {
    work.pair_dispatches[pair.index()] += 1;
    match pair {
        PrimitivePair3::SphereSphere => sphere_sphere(a, b),
        PrimitivePair3::SphereBox => sphere_box(a, b),
        PrimitivePair3::SphereCapsule => flip(capsule_sphere(b, a, work)),
        PrimitivePair3::SphereWedge => sphere_wedge(a, b),
        PrimitivePair3::BoxBox | PrimitivePair3::BoxWedge | PrimitivePair3::WedgeWedge => {
            poly_poly(a, b, work)
        }
        PrimitivePair3::BoxCapsule => flip(capsule_box(b, a, work)),
        PrimitivePair3::CapsuleCapsule => capsule_capsule(a, b, work),
        PrimitivePair3::CapsuleWedge => capsule_wedge(a, b, work),
    }
}

pub(super) fn flip(contact: PrimitiveContact3) -> PrimitiveContact3 {
    PrimitiveContact3 {
        normal: neg(contact.normal),
        separation: contact.separation,
        point_a: contact.point_b,
        point_b: contact.point_a,
    }
}
