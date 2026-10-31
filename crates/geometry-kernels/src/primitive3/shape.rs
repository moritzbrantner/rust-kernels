//! Shape, pose and pair vocabulary shared by every primitive query.

use crate::math3::{Vec3, length};

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
    pub(super) kind: PrimitiveKind3,
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

    pub(super) const fn sphere_radius(self) -> f64 {
        self.data[0]
    }

    pub(super) const fn capsule_parts(self) -> (f64, f64) {
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
