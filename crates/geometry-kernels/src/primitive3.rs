//! Reusable f64 primitive contact geometry for fixed-orientation 3D shapes.
//!
//! This module owns shape math only. Simulation policy, contact persistence, response,
//! collision filtering, sleeping, and CCD admission remain consumer responsibilities.

use crate::math3::{Vec3, add, cross, dot, length, length_squared, neg, normalized, scale, sub};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PrimitiveKind3 {
    Sphere,
    Box,
    Capsule,
    Wedge,
}

impl PrimitiveKind3 {
    pub const ALL: [Self; 4] = [Self::Sphere, Self::Box, Self::Capsule, Self::Wedge];
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrimitiveShape3 {
    kind: PrimitiveKind3,
    data: [f64; 4],
}

impl PrimitiveShape3 {
    #[must_use]
    pub fn sphere(radius: f64) -> Self {
        assert!(
            radius.is_finite() && radius > 0.0,
            "sphere radius must be positive and finite"
        );
        Self {
            kind: PrimitiveKind3::Sphere,
            data: [radius, 0.0, 0.0, 0.0],
        }
    }

    #[must_use]
    pub fn cuboid(half_extents: Vec3) -> Self {
        validate_half_extents(half_extents, "box");
        Self {
            kind: PrimitiveKind3::Box,
            data: [half_extents[0], half_extents[1], half_extents[2], 0.0],
        }
    }

    #[must_use]
    pub fn capsule(half_segment: f64, radius: f64) -> Self {
        assert!(
            half_segment.is_finite() && half_segment >= 0.0,
            "capsule half segment must be non-negative and finite"
        );
        assert!(
            radius.is_finite() && radius > 0.0,
            "capsule radius must be positive and finite"
        );
        Self {
            kind: PrimitiveKind3::Capsule,
            data: [half_segment, radius, 0.0, 0.0],
        }
    }

    #[must_use]
    pub fn wedge(half_extents: Vec3) -> Self {
        validate_half_extents(half_extents, "wedge");
        Self {
            kind: PrimitiveKind3::Wedge,
            data: [half_extents[0], half_extents[1], half_extents[2], 0.0],
        }
    }

    #[must_use]
    pub const fn kind(self) -> PrimitiveKind3 {
        self.kind
    }

    #[must_use]
    pub fn radius(self) -> f64 {
        match self.kind {
            PrimitiveKind3::Sphere => self.data[0],
            PrimitiveKind3::Box | PrimitiveKind3::Wedge => length(self.half_extents()),
            PrimitiveKind3::Capsule => self.data[0] + self.data[1],
        }
    }

    #[must_use]
    pub const fn half_extents(self) -> Vec3 {
        match self.kind {
            PrimitiveKind3::Sphere => [self.data[0], self.data[0], self.data[0]],
            PrimitiveKind3::Box | PrimitiveKind3::Wedge => {
                [self.data[0], self.data[1], self.data[2]]
            }
            PrimitiveKind3::Capsule => [self.data[1], self.data[0] + self.data[1], self.data[1]],
        }
    }

    const fn sphere_radius(self) -> f64 {
        self.data[0]
    }

    const fn capsule_parts(self) -> (f64, f64) {
        (self.data[0], self.data[1])
    }
}

fn validate_half_extents(half_extents: Vec3, label: &str) {
    assert!(
        half_extents
            .into_iter()
            .all(|value| value.is_finite() && value > 0.0),
        "{label} half extents must be positive and finite"
    );
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrimitiveBody3 {
    pub shape: PrimitiveShape3,
    pub position: Vec3,
    /// World-space local X/Y/Z axes. Callers own orientation representation.
    pub axes: [Vec3; 3],
    pub velocity: Vec3,
}

impl PrimitiveBody3 {
    #[must_use]
    pub fn new(shape: PrimitiveShape3, position: Vec3, axes: [Vec3; 3], velocity: Vec3) -> Self {
        debug_assert!(position.into_iter().all(f64::is_finite));
        debug_assert!(velocity.into_iter().all(f64::is_finite));
        debug_assert!(axes.into_iter().flatten().all(f64::is_finite));
        Self {
            shape,
            position,
            axes,
            velocity,
        }
    }

    #[must_use]
    pub fn axis_aligned(shape: PrimitiveShape3, position: Vec3, velocity: Vec3) -> Self {
        Self::new(
            shape,
            position,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            velocity,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrimitivePair3 {
    SphereSphere,
    SphereBox,
    SphereCapsule,
    SphereWedge,
    BoxBox,
    BoxCapsule,
    BoxWedge,
    CapsuleCapsule,
    CapsuleWedge,
    WedgeWedge,
}

impl PrimitivePair3 {
    pub const ALL: [Self; 10] = [
        Self::SphereSphere,
        Self::SphereBox,
        Self::SphereCapsule,
        Self::SphereWedge,
        Self::BoxBox,
        Self::BoxCapsule,
        Self::BoxWedge,
        Self::CapsuleCapsule,
        Self::CapsuleWedge,
        Self::WedgeWedge,
    ];

    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::SphereSphere => 0,
            Self::SphereBox => 1,
            Self::SphereCapsule => 2,
            Self::SphereWedge => 3,
            Self::BoxBox => 4,
            Self::BoxCapsule => 5,
            Self::BoxWedge => 6,
            Self::CapsuleCapsule => 7,
            Self::CapsuleWedge => 8,
            Self::WedgeWedge => 9,
        }
    }

    #[must_use]
    pub const fn kinds(self) -> (PrimitiveKind3, PrimitiveKind3) {
        match self {
            Self::SphereSphere => (PrimitiveKind3::Sphere, PrimitiveKind3::Sphere),
            Self::SphereBox => (PrimitiveKind3::Sphere, PrimitiveKind3::Box),
            Self::SphereCapsule => (PrimitiveKind3::Sphere, PrimitiveKind3::Capsule),
            Self::SphereWedge => (PrimitiveKind3::Sphere, PrimitiveKind3::Wedge),
            Self::BoxBox => (PrimitiveKind3::Box, PrimitiveKind3::Box),
            Self::BoxCapsule => (PrimitiveKind3::Box, PrimitiveKind3::Capsule),
            Self::BoxWedge => (PrimitiveKind3::Box, PrimitiveKind3::Wedge),
            Self::CapsuleCapsule => (PrimitiveKind3::Capsule, PrimitiveKind3::Capsule),
            Self::CapsuleWedge => (PrimitiveKind3::Capsule, PrimitiveKind3::Wedge),
            Self::WedgeWedge => (PrimitiveKind3::Wedge, PrimitiveKind3::Wedge),
        }
    }

    #[must_use]
    pub fn canonical(left: PrimitiveKind3, right: PrimitiveKind3) -> (Self, bool) {
        let reversed = left > right;
        let (first, second) = if reversed {
            (right, left)
        } else {
            (left, right)
        };
        let pair = match (first, second) {
            (PrimitiveKind3::Sphere, PrimitiveKind3::Sphere) => Self::SphereSphere,
            (PrimitiveKind3::Sphere, PrimitiveKind3::Box) => Self::SphereBox,
            (PrimitiveKind3::Sphere, PrimitiveKind3::Capsule) => Self::SphereCapsule,
            (PrimitiveKind3::Sphere, PrimitiveKind3::Wedge) => Self::SphereWedge,
            (PrimitiveKind3::Box, PrimitiveKind3::Box) => Self::BoxBox,
            (PrimitiveKind3::Box, PrimitiveKind3::Capsule) => Self::BoxCapsule,
            (PrimitiveKind3::Box, PrimitiveKind3::Wedge) => Self::BoxWedge,
            (PrimitiveKind3::Capsule, PrimitiveKind3::Capsule) => Self::CapsuleCapsule,
            (PrimitiveKind3::Capsule, PrimitiveKind3::Wedge) => Self::CapsuleWedge,
            (PrimitiveKind3::Wedge, PrimitiveKind3::Wedge) => Self::WedgeWedge,
            _ => unreachable!("primitive kind ordering is exhaustive"),
        };
        (pair, reversed)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrimitiveWork3 {
    pub pair_dispatches: [u64; 10],
    pub support_evaluations: u64,
    pub axes_tested: u64,
    pub vertex_tests: u64,
    pub sweep_iterations: u64,
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
        PrimitivePair3::SphereCapsule => flip(capsule_sphere(b, a)),
        PrimitivePair3::SphereWedge => sphere_wedge(a, b),
        PrimitivePair3::BoxBox | PrimitivePair3::BoxWedge | PrimitivePair3::WedgeWedge => {
            poly_poly(a, b, work)
        }
        PrimitivePair3::BoxCapsule => flip(capsule_box(b, a, work)),
        PrimitivePair3::CapsuleCapsule => capsule_capsule(a, b),
        PrimitivePair3::CapsuleWedge => capsule_wedge(a, b),
    }
}

#[must_use]
pub fn swept_time(
    a: PrimitiveBody3,
    b: PrimitiveBody3,
    dt: f64,
    margin: f64,
    work: &mut PrimitiveWork3,
) -> Option<f64> {
    let (pair, reversed) = PrimitivePair3::canonical(a.shape.kind, b.shape.kind);
    let (a, b) = if reversed { (b, a) } else { (a, b) };
    if matches!(
        pair,
        PrimitivePair3::BoxBox | PrimitivePair3::BoxWedge | PrimitivePair3::WedgeWedge
    ) {
        return poly_sweep_time(a, b, dt, margin, work);
    }

    let displacement = scale(sub(b.velocity, a.velocity), dt);
    if displacement == [0.0; 3] {
        let contact = query_canonical(pair, a, b, work);
        return (contact.separation <= margin).then_some(0.0);
    }
    let target = margin;
    let tolerance = 1.0e-9 * (1.0 + a.shape.radius() + b.shape.radius());
    let mut time = 0.0;
    let mut current_a = a;
    let mut current_b = b;

    for _ in 0..128 {
        work.sweep_iterations += 1;
        current_a.position = add(a.position, scale(a.velocity, dt * time));
        current_b.position = add(b.position, scale(b.velocity, dt * time));
        let contact = query_canonical(pair, current_a, current_b, work);
        let gap = contact.separation - target;
        if gap <= 0.0 {
            return Some(time);
        }
        let closing = -dot(displacement, contact.normal);
        if closing <= 0.0 {
            return None;
        }
        let remaining = 1.0 - time;
        if gap > closing * remaining + tolerance {
            return None;
        }
        let next = time + gap / closing;
        if next > 1.0 {
            return None;
        }
        if next <= time {
            let bumped = f64::from_bits(time.to_bits() + 1);
            if bumped > 1.0 {
                return None;
            }
            time = bumped;
        } else {
            time = next;
        }
    }
    None
}

fn flip(contact: PrimitiveContact3) -> PrimitiveContact3 {
    PrimitiveContact3 {
        normal: neg(contact.normal),
        separation: contact.separation,
        point_a: contact.point_b,
        point_b: contact.point_a,
    }
}

fn sphere_sphere(a: PrimitiveBody3, b: PrimitiveBody3) -> PrimitiveContact3 {
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

fn sphere_box(sphere: PrimitiveBody3, box_body: PrimitiveBody3) -> PrimitiveContact3 {
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

fn capsule_sphere(capsule: PrimitiveBody3, sphere: PrimitiveBody3) -> PrimitiveContact3 {
    let (a, b) = capsule_segment(capsule);
    let core = closest_point_segment(sphere.position, a, b);
    let delta = sub(sphere.position, core);
    let distance = length(delta);
    let normal = fallback_normal(delta, sub(sphere.position, capsule.position));
    let (_, capsule_radius) = capsule.shape.capsule_parts();
    let sphere_radius = sphere.shape.sphere_radius();
    PrimitiveContact3 {
        normal,
        separation: distance - capsule_radius - sphere_radius,
        point_a: add(core, scale(normal, capsule_radius)),
        point_b: sub(sphere.position, scale(normal, sphere_radius)),
    }
}

fn capsule_capsule(a: PrimitiveBody3, b: PrimitiveBody3) -> PrimitiveContact3 {
    let (a0, a1) = capsule_segment(a);
    let (b0, b1) = capsule_segment(b);
    let (point_a, point_b) = closest_segment_segment(a0, a1, b0, b1);
    let delta = sub(point_b, point_a);
    let distance = length(delta);
    let normal = fallback_normal(delta, sub(b.position, a.position));
    let (_, a_radius) = a.shape.capsule_parts();
    let (_, b_radius) = b.shape.capsule_parts();
    PrimitiveContact3 {
        normal,
        separation: distance - a_radius - b_radius,
        point_a: add(point_a, scale(normal, a_radius)),
        point_b: sub(point_b, scale(normal, b_radius)),
    }
}

fn capsule_box(
    capsule: PrimitiveBody3,
    box_body: PrimitiveBody3,
    work: &mut PrimitiveWork3,
) -> PrimitiveContact3 {
    let half = box_body.shape.half_extents();
    let (_, radius) = capsule.shape.capsule_parts();
    let (world_a, world_b) = capsule_segment(capsule);
    let local_a = inverse_rotate(box_body.axes, sub(world_a, box_body.position));
    let local_b = inverse_rotate(box_body.axes, sub(world_b, box_body.position));

    if segment_aabb_interval(local_a, local_b, half).is_some() {
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
                closest_point_segment(box_support, local_a, local_b)
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

    let (local_core, local_box) = closest_segment_aabb(local_a, local_b, half);
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

fn sphere_wedge(sphere: PrimitiveBody3, wedge: PrimitiveBody3) -> PrimitiveContact3 {
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

fn capsule_wedge(capsule: PrimitiveBody3, wedge: PrimitiveBody3) -> PrimitiveContact3 {
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

    let (local_core, local_wedge) = closest_segment_wedge(local_a, local_b, half);
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

fn poly_poly(a: PrimitiveBody3, b: PrimitiveBody3, work: &mut PrimitiveWork3) -> PrimitiveContact3 {
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

fn poly_sweep_time(
    a: PrimitiveBody3,
    b: PrimitiveBody3,
    dt: f64,
    margin: f64,
    work: &mut PrimitiveWork3,
) -> Option<f64> {
    let (axes, len) = poly_axes(a, b);
    let displacement = scale(sub(b.velocity, a.velocity), dt);
    let mut enter = 0.0_f64;
    let mut exit = 1.0_f64;

    for &axis in &axes[..len] {
        work.axes_tested += 1;
        work.sweep_iterations += 1;
        let (a_min, a_max) = projection_interval(a, axis, work);
        let (b_min, b_max) = projection_interval(b, axis, work);
        let velocity = dot(displacement, axis);
        if velocity == 0.0 {
            if b_min > a_max + margin || a_min > b_max + margin {
                return None;
            }
            continue;
        }
        let first = (a_min - margin - b_max) / velocity;
        let second = (a_max + margin - b_min) / velocity;
        enter = enter.max(first.min(second));
        exit = exit.min(first.max(second));
        if enter > exit {
            return None;
        }
    }

    (exit >= 0.0 && enter <= 1.0).then_some(enter.max(0.0))
}

fn projection_interval(body: PrimitiveBody3, axis: Vec3, work: &mut PrimitiveWork3) -> (f64, f64) {
    let maximum = dot(support_point(body, axis, work), axis);
    let minimum = dot(support_point(body, neg(axis), work), axis);
    (minimum, maximum)
}

fn poly_axes(a: PrimitiveBody3, b: PrimitiveBody3) -> ([Vec3; 32], usize) {
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

fn wedge_vertices(half: Vec3) -> [Vec3; 6] {
    [
        [-half[0], -half[1], -half[2]],
        [-half[0], half[1], -half[2]],
        [half[0], -half[1], -half[2]],
        [-half[0], -half[1], half[2]],
        [-half[0], half[1], half[2]],
        [half[0], -half[1], half[2]],
    ]
}

fn segment_face_penetration(
    a: Vec3,
    b: Vec3,
    planes: &[(Vec3, f64)],
    radius: f64,
) -> (Vec3, f64, Vec3, f64) {
    let mut best = ([1.0, 0.0, 0.0], f64::INFINITY, a, 0.0);
    for &(normal, limit) in planes {
        let a_depth = limit - dot(normal, a);
        let b_depth = limit - dot(normal, b);
        let (core, core_depth) = if a_depth >= b_depth {
            (a, a_depth)
        } else {
            (b, b_depth)
        };
        let penetration = core_depth.max(0.0) + radius;
        if penetration < best.1 {
            best = (normal, penetration, core, core_depth.max(0.0));
        }
    }
    best
}

fn wedge_planes(half: Vec3) -> [(Vec3, f64); 5] {
    [
        ([0.0, -1.0, 0.0], half[1]),
        ([-1.0, 0.0, 0.0], half[0]),
        (unit_or_zero([half[1], half[0], 0.0]), 0.0),
        ([0.0, 0.0, 1.0], half[2]),
        ([0.0, 0.0, -1.0], half[2]),
    ]
}

fn wedge_triangles(half: Vec3) -> [[Vec3; 3]; 8] {
    let vertices = wedge_vertices(half);
    [
        [vertices[0], vertices[2], vertices[1]],
        [vertices[3], vertices[4], vertices[5]],
        [vertices[0], vertices[3], vertices[5]],
        [vertices[0], vertices[5], vertices[2]],
        [vertices[0], vertices[1], vertices[4]],
        [vertices[0], vertices[4], vertices[3]],
        [vertices[1], vertices[2], vertices[5]],
        [vertices[1], vertices[5], vertices[4]],
    ]
}

fn wedge_inside_depth(point: Vec3, half: Vec3) -> Option<(Vec3, f64)> {
    let geometry_scale = half
        .into_iter()
        .chain(point)
        .map(f64::abs)
        .fold(0.0, f64::max);
    let tolerance = 64.0 * f64::EPSILON * geometry_scale.max(f64::MIN_POSITIVE);
    let mut best = ([0.0, 1.0, 0.0], f64::INFINITY);
    for (normal, limit) in wedge_planes(half) {
        let depth = limit - dot(normal, point);
        if depth < -tolerance {
            return None;
        }
        if depth < best.1 {
            best = (normal, depth.max(0.0));
        }
    }
    Some(best)
}

fn segment_wedge_interval(a: Vec3, b: Vec3, half: Vec3) -> Option<(f64, f64)> {
    let delta = sub(b, a);
    let mut enter = 0.0_f64;
    let mut exit = 1.0_f64;
    for (normal, limit) in wedge_planes(half) {
        let start = dot(normal, a) - limit;
        let velocity = dot(normal, delta);
        if velocity.abs() <= 1.0e-14 {
            if start > 0.0 {
                return None;
            }
            continue;
        }
        let time = -start / velocity;
        if velocity > 0.0 {
            exit = exit.min(time);
        } else {
            enter = enter.max(time);
        }
        if enter > exit {
            return None;
        }
    }
    (exit >= 0.0 && enter <= 1.0).then_some((enter.max(0.0), exit.min(1.0)))
}

fn closest_point_wedge(point: Vec3, half: Vec3) -> Vec3 {
    let mut best = wedge_vertices(half)[0];
    let mut best_distance = f64::INFINITY;
    for triangle in wedge_triangles(half) {
        let candidate = closest_point_triangle(point, triangle[0], triangle[1], triangle[2]);
        let distance = length_squared(sub(candidate, point));
        if distance < best_distance {
            best = candidate;
            best_distance = distance;
        }
    }
    best
}

fn closest_segment_wedge(a: Vec3, b: Vec3, half: Vec3) -> (Vec3, Vec3) {
    let mut best = (a, wedge_vertices(half)[0]);
    let mut best_distance = f64::INFINITY;
    for triangle in wedge_triangles(half) {
        for endpoint in [a, b] {
            let candidate = closest_point_triangle(endpoint, triangle[0], triangle[1], triangle[2]);
            update_pair(endpoint, candidate, &mut best, &mut best_distance);
        }
        for edge in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            let candidate = closest_segment_segment(a, b, edge.0, edge.1);
            update_pair(candidate.0, candidate.1, &mut best, &mut best_distance);
        }
    }
    best
}

fn update_pair(a: Vec3, b: Vec3, best: &mut (Vec3, Vec3), best_distance: &mut f64) {
    let distance = length_squared(sub(b, a));
    if distance < *best_distance {
        *best = (a, b);
        *best_distance = distance;
    }
}

fn closest_point_triangle(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(point, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = sub(point, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return add(a, scale(ab, d1 / (d1 - d3)));
    }

    let cp = sub(point, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return add(a, scale(ac, d2 / (d2 - d6)));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return add(b, scale(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }

    let denominator = (va + vb + vc).recip();
    add(
        add(a, scale(ab, vb * denominator)),
        scale(ac, vc * denominator),
    )
}

fn closest_point_segment(point: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let delta = sub(b, a);
    let denominator = length_squared(delta);
    if denominator <= 1.0e-20 {
        return a;
    }
    let time = (dot(sub(point, a), delta) / denominator).clamp(0.0, 1.0);
    add(a, scale(delta, time))
}

fn closest_segment_segment(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    let d1 = sub(q1, p1);
    let d2 = sub(q2, p2);
    let r = sub(p1, p2);
    let a = length_squared(d1);
    let e = length_squared(d2);
    let epsilon = 1.0e-20;

    if a <= epsilon && e <= epsilon {
        return (p1, p2);
    }
    if a <= epsilon {
        let t = (dot(d2, r) / e).clamp(0.0, 1.0);
        return (p1, add(p2, scale(d2, t)));
    }
    if e <= epsilon {
        let s = (-dot(d1, r) / a).clamp(0.0, 1.0);
        return (add(p1, scale(d1, s)), p2);
    }

    let b = dot(d1, d2);
    let c = dot(d1, r);
    let f = dot(d2, r);
    let denominator = a * e - b * b;
    let mut s = if denominator.abs() > epsilon {
        ((b * f - c * e) / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }
    (add(p1, scale(d1, s)), add(p2, scale(d2, t)))
}

fn segment_aabb_interval(a: Vec3, b: Vec3, half: Vec3) -> Option<(f64, f64)> {
    let delta = sub(b, a);
    let mut enter = 0.0_f64;
    let mut exit = 1.0_f64;
    for axis in 0..3 {
        let start = a[axis];
        let velocity = delta[axis];
        let extent = half[axis];
        if velocity.abs() <= 1.0e-14 {
            if start < -extent || start > extent {
                return None;
            }
            continue;
        }
        let first = (-extent - start) / velocity;
        let second = (extent - start) / velocity;
        enter = enter.max(first.min(second));
        exit = exit.min(first.max(second));
        if enter > exit {
            return None;
        }
    }
    (exit >= 0.0 && enter <= 1.0).then_some((enter.max(0.0), exit.min(1.0)))
}

fn closest_segment_aabb(a: Vec3, b: Vec3, half: Vec3) -> (Vec3, Vec3) {
    let delta = sub(b, a);
    let mut times = [0.0; 8];
    times[0] = 0.0;
    times[1] = 1.0;
    let mut len = 2;
    for axis in 0..3 {
        let velocity = delta[axis];
        if velocity.abs() <= 1.0e-14 {
            continue;
        }
        for bound in [-half[axis], half[axis]] {
            let time = (bound - a[axis]) / velocity;
            if time > 0.0 && time < 1.0 {
                times[len] = time;
                len += 1;
            }
        }
    }
    times[..len].sort_by(f64::total_cmp);
    let mut unique = 1;
    for index in 1..len {
        if (times[index] - times[unique - 1]).abs() > 1.0e-14 {
            times[unique] = times[index];
            unique += 1;
        }
    }
    len = unique;

    let mut best = (a, clamp_aabb(a, half));
    let mut best_distance = length_squared(sub(best.1, best.0));
    let mut evaluate = |time: f64| {
        let point = add(a, scale(delta, time.clamp(0.0, 1.0)));
        let clamped = clamp_aabb(point, half);
        update_pair(point, clamped, &mut best, &mut best_distance);
    };
    for &time in &times[..len] {
        evaluate(time);
    }
    for window in times[..len].windows(2) {
        let low = window[0];
        let high = window[1];
        let middle = (low + high) * 0.5;
        let point = add(a, scale(delta, middle));
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for axis in 0..3 {
            let coordinate = point[axis];
            let bound = if coordinate < -half[axis] {
                -half[axis]
            } else if coordinate > half[axis] {
                half[axis]
            } else {
                continue;
            };
            let velocity = delta[axis];
            numerator += velocity * (a[axis] - bound);
            denominator += velocity * velocity;
        }
        if denominator > 1.0e-20 {
            evaluate((-numerator / denominator).clamp(low, high));
        }
    }
    best
}

fn clamp_aabb(point: Vec3, half: Vec3) -> Vec3 {
    [
        point[0].clamp(-half[0], half[0]),
        point[1].clamp(-half[1], half[1]),
        point[2].clamp(-half[2], half[2]),
    ]
}

fn nearest_box_face(point: Vec3, half: Vec3) -> (Vec3, f64) {
    let mut axis = 0;
    let mut depth = half[0] - point[0].abs();
    for candidate in 1..3 {
        let candidate_depth = half[candidate] - point[candidate].abs();
        if candidate_depth < depth {
            axis = candidate;
            depth = candidate_depth;
        }
    }
    let sign = if point[axis] < 0.0 { -1.0 } else { 1.0 };
    let mut normal = [0.0; 3];
    normal[axis] = sign;
    (normal, depth.max(0.0))
}

fn fallback_normal(delta: Vec3, fallback: Vec3) -> Vec3 {
    if length_squared(delta) > 1.0e-20 {
        unit_or_zero(delta)
    } else if length_squared(fallback) > 1.0e-20 {
        unit_or_zero(fallback)
    } else {
        [1.0, 0.0, 0.0]
    }
}

fn unit_or_zero(vector: Vec3) -> Vec3 {
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

fn abs(vector: Vec3) -> Vec3 {
    [vector[0].abs(), vector[1].abs(), vector[2].abs()]
}

fn rotate_local(axes: [Vec3; 3], local: Vec3) -> Vec3 {
    add(
        add(scale(axes[0], local[0]), scale(axes[1], local[1])),
        scale(axes[2], local[2]),
    )
}

fn inverse_rotate(axes: [Vec3; 3], world: Vec3) -> Vec3 {
    [
        dot(world, axes[0]),
        dot(world, axes[1]),
        dot(world, axes[2]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(shape: PrimitiveShape3, position: Vec3) -> PrimitiveBody3 {
        PrimitiveBody3::axis_aligned(shape, position, [0.0; 3])
    }

    #[test]
    fn primitive_pair_matrix_is_complete_canonical_and_dense() {
        let mut seen = [false; 10];
        for left in PrimitiveKind3::ALL {
            for right in PrimitiveKind3::ALL {
                let (pair, reversed) = PrimitivePair3::canonical(left, right);
                let (first, second) = pair.kinds();
                assert!(first <= second);
                assert_eq!(pair, PrimitivePair3::ALL[pair.index()]);
                assert_eq!(reversed, left > right);
                seen[pair.index()] = true;
            }
        }
        assert!(seen.into_iter().all(|value| value));
    }

    fn shape_for_kind(kind: PrimitiveKind3) -> PrimitiveShape3 {
        match kind {
            PrimitiveKind3::Sphere => PrimitiveShape3::sphere(1.0),
            PrimitiveKind3::Box => PrimitiveShape3::cuboid([1.0, 0.8, 1.2]),
            PrimitiveKind3::Capsule => PrimitiveShape3::capsule(0.7, 0.6),
            PrimitiveKind3::Wedge => PrimitiveShape3::wedge([1.2, 0.9, 1.1]),
        }
    }

    #[test]
    fn every_primitive_pair_executes_its_registered_kernel() {
        for pair in PrimitivePair3::ALL {
            let (left, right) = pair.kinds();
            let a = body(shape_for_kind(left), [0.0; 3]);
            let b = body(shape_for_kind(right), [0.25, 0.1, -0.05]);
            let mut work = PrimitiveWork3::default();
            let contact = query(a, b, &mut work);
            assert!(contact.separation.is_finite(), "{pair:?}");
            assert!(contact.normal.into_iter().all(f64::is_finite), "{pair:?}");
            assert_eq!(work.pair_dispatches[pair.index()], 1, "{pair:?}");
            assert_eq!(work.pair_dispatches.into_iter().sum::<u64>(), 1, "{pair:?}");
        }
    }

    #[test]
    fn capsule_sphere_and_capsule_capsule_use_analytic_segment_distance() {
        let capsule = body(PrimitiveShape3::capsule(2.0, 0.5), [0.0; 3]);
        let sphere = body(PrimitiveShape3::sphere(0.5), [0.0, 3.0, 0.0]);
        let mut work = PrimitiveWork3::default();
        let contact = query(capsule, sphere, &mut work);
        assert!(contact.separation.abs() < 1.0e-12);
        assert_eq!(work.vertex_tests, 0);

        let other = body(PrimitiveShape3::capsule(2.0, 0.5), [0.9, 0.0, 0.0]);
        let contact = query(capsule, other, &mut work);
        assert!(contact.separation < 0.0);
        assert_eq!(work.vertex_tests, 0);
    }

    #[test]
    fn sphere_wedge_respects_sloped_face_instead_of_bounding_box() {
        let wedge = body(PrimitiveShape3::wedge([2.0; 3]), [0.0; 3]);
        let near = body(PrimitiveShape3::sphere(0.8), [1.0, 0.0, 0.0]);
        let far = body(PrimitiveShape3::sphere(0.2), [1.5, 1.5, 0.0]);
        let mut work = PrimitiveWork3::default();
        assert!(query(near, wedge, &mut work).separation < 0.0);
        assert!(query(far, wedge, &mut work).separation > 0.0);
    }

    #[test]
    fn capsule_wedge_is_symmetric_for_rotated_capsule() {
        let wedge = body(PrimitiveShape3::wedge([2.0; 3]), [0.0; 3]);
        let capsule = PrimitiveBody3::new(
            PrimitiveShape3::capsule(1.5, 0.4),
            [0.0, 0.5, 0.0],
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
        );
        let mut work = PrimitiveWork3::default();
        let forward = query(capsule, wedge, &mut work);
        let reverse = query(wedge, capsule, &mut work);
        assert!((forward.separation - reverse.separation).abs() < 1.0e-12);
        assert!(length(add(forward.normal, reverse.normal)) < 1.0e-12);
    }

    #[test]
    fn wedge_sat_has_a_fixed_small_work_ceiling() {
        let a = body(PrimitiveShape3::wedge([2.0, 1.0, 3.0]), [0.0; 3]);
        let angle = 30.0_f64.to_radians();
        let (sin, cos) = angle.sin_cos();
        let b = PrimitiveBody3::new(
            PrimitiveShape3::wedge([1.5, 2.0, 1.0]),
            [3.0, 0.0, 0.0],
            [[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
        );
        let mut work = PrimitiveWork3::default();
        let contact = query(a, b, &mut work);
        assert!(contact.separation.is_finite());
        assert!(work.axes_tested <= 24, "{work:?}");
        assert!(work.vertex_tests <= 600, "{work:?}");
    }

    #[test]
    fn partial_capsule_box_penetration_uses_the_whole_segment() {
        let capsule = PrimitiveBody3::new(
            PrimitiveShape3::capsule(1.0, 0.1),
            [1.5, 0.0, 0.0],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
        );
        let cuboid = body(PrimitiveShape3::cuboid([1.0; 3]), [0.0; 3]);
        let mut work = PrimitiveWork3::default();
        let contact = query(capsule, cuboid, &mut work);
        assert!((contact.separation + 0.6).abs() < 1.0e-12, "{contact:?}");
    }

    #[test]
    fn partial_capsule_wedge_penetration_uses_the_whole_segment() {
        let capsule = PrimitiveBody3::new(
            PrimitiveShape3::capsule(1.0, 0.1),
            [0.5, 0.0, 0.0],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
        );
        let wedge = body(PrimitiveShape3::wedge([1.0; 3]), [0.0; 3]);
        let mut work = PrimitiveWork3::default();
        let contact = query(capsule, wedge, &mut work);
        let expected = -(0.5_f64 / 2.0_f64.sqrt() + 0.1);
        assert!(
            (contact.separation - expected).abs() < 1.0e-12,
            "{contact:?}"
        );
    }

    #[test]
    fn stationary_overlap_reports_time_zero() {
        let left = body(PrimitiveShape3::sphere(1.0), [0.0; 3]);
        let right = body(PrimitiveShape3::sphere(1.0), [1.5, 0.0, 0.0]);
        let mut work = PrimitiveWork3::default();
        assert_eq!(swept_time(left, right, 1.0, 0.0, &mut work), Some(0.0));
    }

    #[test]
    fn zero_margin_does_not_expand_a_near_miss() {
        let moving = PrimitiveBody3::axis_aligned(
            PrimitiveShape3::sphere(1.0),
            [-3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        );
        let target = body(PrimitiveShape3::sphere(1.0), [0.0, 2.000_000_5, 0.0]);
        let mut work = PrimitiveWork3::default();
        assert_eq!(swept_time(moving, target, 1.0, 0.0, &mut work), None);
    }

    #[test]
    fn glancing_capsule_sweep_uses_normal_closing_speed() {
        let capsule = body(PrimitiveShape3::capsule(0.0, 0.1), [-400.0, 1.0, 0.0]);
        let capsule = PrimitiveBody3 {
            velocity: [800.0, -2.0, 0.0],
            ..capsule
        };
        let target = body(PrimitiveShape3::cuboid([500.0, 0.01, 10.0]), [0.0; 3]);
        let mut work = PrimitiveWork3::default();
        let time = swept_time(capsule, target, 1.0, 0.02, &mut work)
            .expect("glancing capsule sweep must hit");
        assert!((time - 0.435).abs() < 1.0e-9, "time={time}");
        assert!(work.sweep_iterations <= 3, "{work:?}");
    }

    #[test]
    fn fast_capsule_conservative_advancement_reaches_thin_wedge() {
        let capsule = PrimitiveBody3::axis_aligned(
            PrimitiveShape3::capsule(0.5, 0.1),
            [-1.0, -1.0, 10.0],
            [0.0, 0.0, -10_000.0],
        );
        let wedge = body(PrimitiveShape3::wedge([4.0, 4.0, 0.02]), [0.0; 3]);
        let mut work = PrimitiveWork3::default();
        let time = swept_time(capsule, wedge, 1.0 / 60.0, 0.02, &mut work)
            .expect("fast capsule must reach thin wedge");
        assert!((0.0..=1.0).contains(&time));
        assert!(work.sweep_iterations > 0);
        assert!(work.sweep_iterations < 128);
    }

    #[test]
    fn support_and_bounds_are_allocation_free_value_operations() {
        let wedge = PrimitiveBody3::axis_aligned(
            PrimitiveShape3::wedge([2.0, 1.0, 3.0]),
            [1.0, 2.0, 3.0],
            [0.0; 3],
        );
        let mut work = PrimitiveWork3::default();
        assert_eq!(bounds_extents(wedge), [2.0, 1.0, 3.0]);
        assert_eq!(support_point(wedge, [1.0, 0.0, 0.0], &mut work)[0], 3.0);
        assert_eq!(work.vertex_tests, 6);

        let cuboid =
            PrimitiveBody3::axis_aligned(PrimitiveShape3::cuboid([1.0; 3]), [0.0; 3], [0.0; 3]);
        let tiny = support_point(cuboid, [1.0e-13, 0.0, 0.0], &mut work);
        let scaled = support_point(cuboid, [1.0, 0.0, 0.0], &mut work);
        assert_eq!(tiny[0], 1.0);
        assert_eq!(tiny[0], scaled[0]);

        for shape in [
            PrimitiveShape3::sphere(1.0),
            PrimitiveShape3::capsule(0.7, 0.4),
        ] {
            let body = PrimitiveBody3::axis_aligned(shape, [0.0; 3], [0.0; 3]);
            let tiny = support_point(body, [1.0e-13, 0.0, 0.0], &mut work);
            let scaled = support_point(body, [1.0, 0.0, 0.0], &mut work);
            assert_eq!(tiny, scaled);
        }
    }

    #[test]
    fn capsule_box_penetration_includes_capsule_axis_edge_axes() {
        let inv_sqrt_two = 1.0 / 2.0_f64.sqrt();
        let capsule = PrimitiveBody3::new(
            PrimitiveShape3::capsule(0.9 * 2.0_f64.sqrt(), 0.1),
            [0.0; 3],
            [
                [inv_sqrt_two, inv_sqrt_two, 0.0],
                [inv_sqrt_two, -inv_sqrt_two, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [0.0; 3],
        );
        let cuboid = body(PrimitiveShape3::cuboid([1.0, 1.0, 100.0]), [0.0; 3]);
        let mut work = PrimitiveWork3::default();
        let contact = query(capsule, cuboid, &mut work);
        let expected = -(2.0_f64.sqrt() + 0.1);
        assert!(
            (contact.separation - expected).abs() < 1.0e-12,
            "{contact:?}"
        );
        assert!(work.axes_tested >= 4, "{work:?}");
    }

    #[test]
    fn wedge_inside_tolerance_scales_with_geometry() {
        let wedge = body(PrimitiveShape3::wedge([1.0e-9; 3]), [0.0; 3]);
        let sphere = body(PrimitiveShape3::sphere(1.0e-12), [-1.05e-9, -5.0e-10, 0.0]);
        let mut work = PrimitiveWork3::default();
        assert!(query(sphere, wedge, &mut work).separation > 0.0);
    }

    #[test]
    fn tiny_nonzero_sweep_displacement_is_not_stationary() {
        let left = body(PrimitiveShape3::sphere(1.0e-16), [0.0; 3]);
        let right = PrimitiveBody3::axis_aligned(
            PrimitiveShape3::sphere(1.0e-16),
            [5.0e-16, 0.0, 0.0],
            [-4.0e-16, 0.0, 0.0],
        );
        let mut work = PrimitiveWork3::default();
        let time = swept_time(left, right, 1.0, 0.0, &mut work)
            .expect("tiny but nonzero relative motion must still reach contact");
        assert!((time - 0.75).abs() < 1.0e-12, "time={time}");
    }
}
