//! Uniform geometric volume integrals, independent of mass and simulation policy.

use super::{PrimitiveKind3, PrimitiveShape3};
use crate::math3::Vec3;

/// Geometric products in the shape's local frame for a uniform solid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrimitiveVolumeProperties3 {
    pub volume: f64,
    pub centroid: Vec3,
    /// `integral((x-centroid)(x-centroid)^T dV) / volume`, in squared length units.
    /// Rows and columns follow local X/Y/Z; the matrix is symmetric.
    pub normalized_second_moment: [[f64; 3]; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrimitiveVolumeError3 {
    /// The direct f64 products overflowed or a positive volume/moment underflowed.
    Unrepresentable,
}

/// Compute bounded, allocation-free geometric integrals for the existing primitives.
///
/// Dimensions must satisfy the shape constructor's contract. Extremely large/small
/// valid dimensions may exceed the direct f64 multiplication budget and return an
/// error; no rescaling, saturation, or arbitrary-magnitude guarantee is implied.
/// Capsules use a local-Y skeleton. Wedges rise towards negative local X, with
/// centroid `[-half_x/3, -half_y/3, 0]`; their origin is not their centroid.
pub fn try_volume_properties(
    shape: PrimitiveShape3,
) -> Result<PrimitiveVolumeProperties3, PrimitiveVolumeError3> {
    let mut centroid = [0.0; 3];
    let mut cross_xy = 0.0;
    let (volume, diagonal) = match shape.kind() {
        PrimitiveKind3::Sphere => {
            let r = shape.sphere_radius();
            (
                std::f64::consts::PI * r * r * (4.0 * r / 3.0),
                [r * r / 5.0; 3],
            )
        }
        PrimitiveKind3::Box => {
            let [x, y, z] = shape.half_extents();
            (8.0 * x * y * z, [x * x / 3.0, y * y / 3.0, z * z / 3.0])
        }
        PrimitiveKind3::Capsule => {
            let (h, r) = shape.capsule_parts();
            let cap_length = 2.0 * r / 3.0;
            let denominator = h + cap_length;
            let cylinder_weight = h / denominator;
            // Keep the cap weight explicit for long, thin capsules.
            let cap_weight = cap_length / denominator;
            let radial = cylinder_weight * (r * r / 4.0) + cap_weight * (r * r / 5.0);
            let axial = cylinder_weight * (h * h / 3.0)
                + cap_weight * (r * r / 5.0 + h * h + 3.0 * h * r / 4.0);
            (
                std::f64::consts::PI * r * r * (2.0 * h + 4.0 * r / 3.0),
                [radial, axial, radial],
            )
        }
        PrimitiveKind3::Wedge => {
            let [x, y, z] = shape.half_extents();
            centroid = [-x / 3.0, -y / 3.0, 0.0];
            cross_xy = -x * y / 9.0;
            (
                4.0 * x * y * z,
                [2.0 * x * x / 9.0, 2.0 * y * y / 9.0, z * z / 3.0],
            )
        }
    };
    let positive_products = [volume, diagonal[0], diagonal[1], diagonal[2]];
    if positive_products
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || !cross_xy.is_finite()
    {
        return Err(PrimitiveVolumeError3::Unrepresentable);
    }
    Ok(PrimitiveVolumeProperties3 {
        volume,
        centroid,
        normalized_second_moment: [
            [diagonal[0], cross_xy, 0.0],
            [cross_xy, diagonal[1], 0.0],
            [0.0, 0.0, diagonal[2]],
        ],
    })
}
