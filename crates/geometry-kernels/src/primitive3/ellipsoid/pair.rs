//! Experimental exact time of impact for translating homothetic ellipsoids (issue #190).
//!
//! Two ellipsoids `A = cA + sA L ball` and `B = cB + sB L ball` that share one shape transform
//! `L = R diag(a,b,c)` become two spheres of radii `sA` and `sB` under `x' = L⁻¹ x`, and an
//! invertible linear map preserves intersection. With fixed orientations and constant linear
//! velocities the relative center moves linearly in that space too, so first contact is the
//! sphere–sphere root of `‖p' + v' t‖ = sA + sB`, solved by the crate's checked sphere
//! fraction. Contact witnesses are mapped back with `L` and normals with `L⁻ᵀ`.
//!
//! The shared transform is a precondition, not an approximation: [`try_homothetic_pair_cast`]
//! accepts only bitwise-identical axes and exactly proportional semiaxes and otherwise returns
//! [`EllipsoidPairCast3::Unsupported`]. No generic (mixed-shape, mixed-orientation or rotating)
//! ellipsoid pair cast exists here; callers must not treat an unsupported pair as a miss.

use super::{Ellipsoid3, EllipsoidBody3, EllipsoidError3, EllipsoidPose3};
use crate::math3::{Vec3, add, cross, dot, is_finite, scale, sub};
use crate::primitive3::frame::unit_or_zero;

/// Why a pair is outside the shared-transform domain. None of these is a miss.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EllipsoidPairUnsupported3 {
    /// The pose axes differ in at least one bit.
    DifferentOrientation,
    /// The semiaxes are not exactly proportional.
    NonHomotheticShapes,
    /// Exact proportionality could not be decided in f64 (extreme semiaxis ratios).
    UnverifiableProportion,
}

/// World-space contact witness at the first-contact pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EllipsoidPairContact3 {
    /// Common boundary point of both ellipsoids.
    pub point: Vec3,
    /// Unit outward normal of the first ellipsoid at `point` (towards the second).
    pub normal: Vec3,
}

/// Outcome of a supported shared-transform cast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EllipsoidPairToi3 {
    /// The interiors overlap at the start; no first-contact witness exists.
    InitiallyOverlapping,
    /// First contact at `fraction` of the interval (`0` when touching at the start).
    Hit {
        fraction: f64,
        contact: EllipsoidPairContact3,
    },
    /// Proven separation over the whole interval.
    Miss,
}

/// Result of the checked pair entry point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EllipsoidPairCast3 {
    Supported(EllipsoidPairToi3),
    Unsupported(EllipsoidPairUnsupported3),
}

/// One ellipsoid `center + scale * L * ball` of a shared frame with constant `velocity`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaledEllipsoidMotion3 {
    pub center: Vec3,
    pub velocity: Vec3,
    pub scale: f64,
}

/// A shared shape transform `L = R diag(a,b,c)` with its precomputed inverse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EllipsoidSharedFrame3 {
    semiaxes: Vec3,
    axes: [Vec3; 3],
    /// Rows of `L⁻¹ = diag(1/a,1/b,1/c) R⁻¹`; `L⁻ᵀ y = Σ y_i rows_i`.
    inverse_rows: [Vec3; 3],
}

impl EllipsoidSharedFrame3 {
    /// `R⁻¹` is the exact adjugate inverse of the supplied (validated, near-orthonormal)
    /// axes, not `Rᵀ`, so the transform is consistent with `L` itself.
    ///
    /// # Errors
    /// As [`EllipsoidPose3::try_new`], or [`EllipsoidError3::Unrepresentable`].
    pub fn try_new(base: Ellipsoid3, axes: [Vec3; 3]) -> Result<Self, EllipsoidError3> {
        EllipsoidPose3::try_new([0.0; 3], axes)?;
        let [x, y, z] = axes;
        let adjugate = [cross(y, z), cross(z, x), cross(x, y)];
        let determinant = dot(x, adjugate[0]);
        let semiaxes = base.semiaxes();
        let inverse_rows: [Vec3; 3] = std::array::from_fn(|index| {
            adjugate[index].map(|value| value / (determinant * semiaxes[index]))
        });
        if !inverse_rows.into_iter().all(is_finite) {
            return Err(EllipsoidError3::Unrepresentable);
        }
        Ok(Self {
            semiaxes,
            axes,
            inverse_rows,
        })
    }

    #[must_use]
    pub const fn semiaxes(&self) -> Vec3 {
        self.semiaxes
    }

    #[must_use]
    pub const fn axes(&self) -> [Vec3; 3] {
        self.axes
    }

    /// Exact first time of impact over `dt` for fixed orientations and constant velocities,
    /// up to f64 rounding of the transform and the sphere solve. The sphere-space contact
    /// radius is widened by a bound on the transform's rounding (`16ε` times the absolute
    /// row/vector products), so grazing paths within rounding report contact slightly early
    /// instead of a miss.
    ///
    /// # Errors
    /// [`EllipsoidError3::NonFiniteInput`] or [`EllipsoidError3::InvalidParameter`] for a
    /// non-positive scale or negative `dt`, [`EllipsoidError3::Unrepresentable`] when an
    /// intermediate is not representable.
    pub fn try_cast(
        &self,
        a: ScaledEllipsoidMotion3,
        b: ScaledEllipsoidMotion3,
        dt: f64,
    ) -> Result<EllipsoidPairToi3, EllipsoidError3> {
        if [a, b].into_iter().any(|motion| {
            !is_finite(motion.center) || !is_finite(motion.velocity) || !motion.scale.is_finite()
        }) || !dt.is_finite()
        {
            return Err(EllipsoidError3::NonFiniteInput);
        }
        if a.scale <= 0.0 || b.scale <= 0.0 || dt < 0.0 {
            return Err(EllipsoidError3::InvalidParameter);
        }
        let radius = a.scale + b.scale;
        let travel_a = scale(a.velocity, dt);
        let delta = sub(b.center, a.center);
        let displacement = sub(scale(b.velocity, dt), travel_a);
        if !radius.is_finite()
            || !is_finite(travel_a)
            || !is_finite(delta)
            || !is_finite(displacement)
        {
            return Err(EllipsoidError3::Unrepresentable);
        }
        let start = self.world_to_sphere(delta)?;
        let travel = self.world_to_sphere(displacement)?;
        if strictly_inside(start, radius) {
            return Ok(EllipsoidPairToi3::InitiallyOverlapping);
        }
        // Widen the contact radius by a bound on the transform's rounding so a grazing path
        // that rounding pushes just outside is still reported: early, never missed.
        let slack = self.rounding_bound(delta, displacement, radius);
        let Some(fraction) =
            crate::continuous::try_sphere_sphere_fraction_f64(start, travel, radius + slack)
                .map_err(|_| EllipsoidError3::Unrepresentable)?
        else {
            return Ok(EllipsoidPairToi3::Miss);
        };
        let relative = add(start, scale(travel, fraction));
        let direction = unit(relative);
        if direction == [0.0; 3] {
            return Err(EllipsoidError3::Unrepresentable);
        }
        let center = add(a.center, scale(travel_a, fraction));
        let point = add(center, self.sphere_to_world(scale(direction, a.scale)));
        let normal = unit(self.sphere_normal_to_world(direction));
        if !is_finite(point) || normal == [0.0; 3] {
            return Err(EllipsoidError3::Unrepresentable);
        }
        Ok(EllipsoidPairToi3::Hit {
            fraction,
            contact: EllipsoidPairContact3 { point, normal },
        })
    }

    fn world_to_sphere(&self, world: Vec3) -> Result<Vec3, EllipsoidError3> {
        let local = self.inverse_rows.map(|row| dot(row, world));
        if is_finite(local) {
            Ok(local)
        } else {
            Err(EllipsoidError3::Unrepresentable)
        }
    }

    /// Absolute sphere-space rounding bound for `L⁻¹ delta` and `L⁻¹ displacement`: each
    /// component's dot product errs by at most `γ₃ Σ |row_ij| |x_j|`, the rows themselves are
    /// rounded to a few ulps, and the solver adds the radius sum's own rounding.
    fn rounding_bound(&self, delta: Vec3, displacement: Vec3, radius: f64) -> f64 {
        let magnitude = |world: Vec3| {
            let world = world.map(f64::abs);
            self.inverse_rows
                .map(|row| dot(row.map(f64::abs), world))
                .into_iter()
                .sum::<f64>()
        };
        16.0 * f64::EPSILON * (magnitude(delta) + magnitude(displacement) + radius)
    }

    /// `L x = Σ s_i x_i axis_i`.
    fn sphere_to_world(&self, local: Vec3) -> Vec3 {
        (0..3).fold([0.0; 3], |world, index| {
            add(
                world,
                scale(self.axes[index], self.semiaxes[index] * local[index]),
            )
        })
    }

    /// `L⁻ᵀ n'`; the gradient of `‖L⁻¹(x - c)‖²` is parallel to it.
    fn sphere_normal_to_world(&self, local: Vec3) -> Vec3 {
        (0..3).fold([0.0; 3], |world, index| {
            add(world, scale(self.inverse_rows[index], local[index]))
        })
    }
}

/// Checked shared-transform cast for two posed ellipsoids with constant velocities over `dt`.
///
/// Supported only when both poses have bitwise-identical axes and the semiaxes are exactly
/// proportional (`b_i = s a_i` in real arithmetic, decided with exact products). The first
/// body's shape is the frame's base and the second's scale is the correctly rounded ratio.
/// Every other pair is [`EllipsoidPairCast3::Unsupported`]: there is no fuzzy acceptance and no
/// generic ellipsoid fallback.
///
/// # Errors
/// As [`EllipsoidSharedFrame3::try_cast`].
pub fn try_homothetic_pair_cast(
    a: EllipsoidBody3,
    velocity_a: Vec3,
    b: EllipsoidBody3,
    velocity_b: Vec3,
    dt: f64,
) -> Result<EllipsoidPairCast3, EllipsoidError3> {
    if a.pose.axes() != b.pose.axes() {
        return Ok(EllipsoidPairCast3::Unsupported(
            EllipsoidPairUnsupported3::DifferentOrientation,
        ));
    }
    let base = a.shape.semiaxes();
    let other = b.shape.semiaxes();
    if let Err(reason) = exactly_proportional(base, other) {
        return Ok(EllipsoidPairCast3::Unsupported(reason));
    }
    let frame = EllipsoidSharedFrame3::try_new(a.shape, a.pose.axes())?;
    let scale_b = other[0] / base[0];
    if !scale_b.is_finite() || scale_b <= 0.0 {
        return Err(EllipsoidError3::Unrepresentable);
    }
    frame
        .try_cast(
            ScaledEllipsoidMotion3 {
                center: a.pose.center(),
                velocity: velocity_a,
                scale: 1.0,
            },
            ScaledEllipsoidMotion3 {
                center: b.pose.center(),
                velocity: velocity_b,
                scale: scale_b,
            },
            dt,
        )
        .map(EllipsoidPairCast3::Supported)
}

/// Unit vector, or zero when it cannot be normalized.
fn unit(vector: Vec3) -> Vec3 {
    let squared = dot(vector, vector);
    if squared.is_normal() {
        scale(vector, squared.sqrt().recip())
    } else {
        unit_or_zero(vector)
    }
}

/// `true` when `delta` lies strictly inside the sphere of `radius`, compared after scaling
/// both by their largest magnitude so neither square overflows nor underflows.
fn strictly_inside(delta: Vec3, radius: f64) -> bool {
    let squared = dot(delta, delta);
    let radius_squared = radius * radius;
    if squared.is_normal() && radius_squared.is_normal() {
        return squared < radius_squared;
    }
    let magnitude = delta.into_iter().map(f64::abs).fold(radius, f64::max);
    let reduced = delta.map(|value| value / magnitude);
    let reduced_radius = radius / magnitude;
    dot(reduced, reduced) < reduced_radius * reduced_radius
}

/// Decides `other = s * base` exactly: `other_i * base_j == other_j * base_i` for all pairs,
/// comparing exact double-length products. Inputs are first scaled by powers of two (exact)
/// so ordinary magnitudes cannot underflow the product's error term.
fn exactly_proportional(base: Vec3, other: Vec3) -> Result<(), EllipsoidPairUnsupported3> {
    let base =
        power_of_two_normalized(base).ok_or(EllipsoidPairUnsupported3::UnverifiableProportion)?;
    let other =
        power_of_two_normalized(other).ok_or(EllipsoidPairUnsupported3::UnverifiableProportion)?;
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let left = exact_product(other[i], base[j])?;
        let right = exact_product(other[j], base[i])?;
        if left != right {
            return Err(EllipsoidPairUnsupported3::NonHomotheticShapes);
        }
    }
    Ok(())
}

/// `(p, e)` with `p + e == x * y` exactly, or unverifiable if the error term may have
/// lost bits to underflow.
fn exact_product(x: f64, y: f64) -> Result<(f64, f64), EllipsoidPairUnsupported3> {
    let product = x * y;
    // Below 2^-960 the rounding error (|e| <= ulp(p)/2) could be subnormal and inexact.
    if !product.is_finite() || product < f64::from_bits(63_u64 << 52) {
        return Err(EllipsoidPairUnsupported3::UnverifiableProportion);
    }
    Ok((product, x.mul_add(y, -product)))
}

/// Scales positive finite values by one power of two so the largest lies in `[1, 2)`.
/// Returns `None` for a subnormal largest value.
fn power_of_two_normalized(values: Vec3) -> Option<Vec3> {
    let largest = values.into_iter().fold(0.0, f64::max);
    if !largest.is_normal() {
        return None;
    }
    let exponent = ((largest.to_bits() >> 52) & 0x7ff) as i64 - 1023;
    // 2^-exponent for exponent in -1022..=1023; 2^-1023 is the subnormal 2^-1022 / 2.
    let factor = if exponent == 1023 {
        f64::from_bits(1_u64 << 51)
    } else {
        f64::from_bits(((1023 - exponent) as u64) << 52)
    };
    Some(values.map(|value| value * factor))
}
