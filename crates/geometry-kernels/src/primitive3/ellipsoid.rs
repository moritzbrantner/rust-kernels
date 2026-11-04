//! Experimental analytic triaxial ellipsoid geometry (issue #189).
//!
//! An ellipsoid is `E = center + L * unit_ball` with `L = R * diag(a, b, c)`: the validated
//! [`Ellipsoid3`] owns the semiaxes and the separate [`EllipsoidPose3`] owns `center` and the
//! world-space local axes (the columns of `R`). Every query here is closed-form f64 shape math;
//! it is deliberately **not** part of [`super::PrimitiveShape3`]'s pair matrix, and no
//! simulation policy, response or CCD admission lives here.
//!
//! Supported queries: support mapping, directional support radius, exact oriented AABB
//! extents, conservative outer/inner radii, fixed-orientation halfspace contact and
//! translational sweep, material-point displacement certificates for rotating motion, and
//! uniform-solid volume and principal inertia. Arbitrary ellipsoid pairs have no quadratic
//! time of impact; [`try_homothetic_pair_cast`] (#190) covers only the translating pair that
//! shares one shape transform and reports every other pair as unsupported.

use super::frame::unit_or_zero;
use crate::math3::{Vec3, add, dot, is_finite, scale, sub};
use crate::support::SupportMap3;

mod pair;

pub use pair::{
    EllipsoidPairCast3, EllipsoidPairContact3, EllipsoidPairToi3, EllipsoidPairUnsupported3,
    EllipsoidSharedFrame3, ScaledEllipsoidMotion3, try_homothetic_pair_cast,
};

/// Largest accepted `|dot(axis_i, axis_j) - delta_ij|` for pose axes.
///
/// Pose-level radius bounds include the resulting slack, so they stay conservative for every
/// accepted pose; support, AABB and halfspace queries use the supplied axes exactly.
pub const ELLIPSOID_AXES_TOLERANCE: f64 = 1.0e-10;

/// Explicit input and arithmetic failures. No query clamps or silently repairs input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EllipsoidError3 {
    /// A semiaxis is NaN or infinite.
    NonFiniteSemiaxis,
    /// A semiaxis is zero or negative.
    NonPositiveSemiaxis,
    /// A center, axis, direction, displacement, plane, mass, angle or interval is not finite.
    NonFiniteInput,
    /// Pose axes are not orthonormal within [`ELLIPSOID_AXES_TOLERANCE`].
    NonOrthonormalAxes,
    /// A support direction or halfspace normal is the zero vector.
    ZeroDirection,
    /// A finite scalar parameter is outside its domain (non-positive mass or scale, negative
    /// angle or interval).
    InvalidParameter,
    /// An intermediate or result is not representable in f64 (overflow or underflow).
    Unrepresentable,
}

impl std::fmt::Display for EllipsoidError3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ellipsoid primitive: {self:?}")
    }
}

impl std::error::Error for EllipsoidError3 {}

/// Validated triaxial ellipsoid shape with local X/Y/Z semiaxes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ellipsoid3 {
    semiaxes: Vec3,
}

impl Ellipsoid3 {
    /// # Errors
    /// [`EllipsoidError3::NonFiniteSemiaxis`] or [`EllipsoidError3::NonPositiveSemiaxis`].
    pub fn try_new(semiaxes: Vec3) -> Result<Self, EllipsoidError3> {
        if !is_finite(semiaxes) {
            return Err(EllipsoidError3::NonFiniteSemiaxis);
        }
        if semiaxes.into_iter().any(|value| value <= 0.0) {
            return Err(EllipsoidError3::NonPositiveSemiaxis);
        }
        Ok(Self { semiaxes })
    }

    #[must_use]
    pub const fn semiaxes(self) -> Vec3 {
        self.semiaxes
    }

    /// `max(a, b, c)`: every point lies within this distance of the center.
    #[must_use]
    pub fn outer_radius(self) -> f64 {
        self.semiaxes.into_iter().fold(0.0, f64::max)
    }

    /// `min(a, b, c)`: the ball of this radius about the center lies inside the ellipsoid.
    #[must_use]
    pub fn inner_radius(self) -> f64 {
        self.semiaxes.into_iter().fold(f64::INFINITY, f64::min)
    }

    /// Uniform-solid volume `4*pi*a*b*c/3`.
    ///
    /// # Errors
    /// [`EllipsoidError3::Unrepresentable`] on overflow or underflow.
    pub fn try_volume(self) -> Result<f64, EllipsoidError3> {
        let [a, b, c] = self.semiaxes;
        let volume = 4.0 / 3.0 * std::f64::consts::PI * a * b * c;
        positive_finite(volume)
    }

    /// Principal (local-axis) inertia of a uniform solid of `mass`:
    /// `[m(b²+c²)/5, m(a²+c²)/5, m(a²+b²)/5]`. The products of inertia are zero in the local
    /// frame; this is never an isotropic scalar unless `a = b = c`.
    ///
    /// # Errors
    /// [`EllipsoidError3::NonFiniteInput`] or [`EllipsoidError3::InvalidParameter`] for the
    /// mass, [`EllipsoidError3::Unrepresentable`] on overflow or underflow.
    pub fn try_inertia_diagonal(self, mass: f64) -> Result<Vec3, EllipsoidError3> {
        if !mass.is_finite() {
            return Err(EllipsoidError3::NonFiniteInput);
        }
        if mass <= 0.0 {
            return Err(EllipsoidError3::InvalidParameter);
        }
        let [a, b, c] = self.semiaxes.map(|value| value * value);
        let moment = |first: f64, second: f64| positive_finite(mass * ((first + second) / 5.0));
        Ok([moment(b, c)?, moment(a, c)?, moment(a, b)?])
    }
}

/// Validated pose: center plus world-space local X/Y/Z axes (the columns of `R`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EllipsoidPose3 {
    center: Vec3,
    axes: [Vec3; 3],
}

impl EllipsoidPose3 {
    /// # Errors
    /// [`EllipsoidError3::NonFiniteInput`] or [`EllipsoidError3::NonOrthonormalAxes`].
    pub fn try_new(center: Vec3, axes: [Vec3; 3]) -> Result<Self, EllipsoidError3> {
        if !is_finite(center) || !axes.into_iter().all(is_finite) {
            return Err(EllipsoidError3::NonFiniteInput);
        }
        for row in 0..3 {
            for column in row..3 {
                let expected = if row == column { 1.0 } else { 0.0 };
                let error = (dot(axes[row], axes[column]) - expected).abs();
                // Finite axes give a finite error; anything above the tolerance rejects.
                if error > ELLIPSOID_AXES_TOLERANCE {
                    return Err(EllipsoidError3::NonOrthonormalAxes);
                }
            }
        }
        Ok(Self { center, axes })
    }

    /// # Errors
    /// [`EllipsoidError3::NonFiniteInput`].
    pub fn try_axis_aligned(center: Vec3) -> Result<Self, EllipsoidError3> {
        Self::try_new(center, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    #[must_use]
    pub const fn center(self) -> Vec3 {
        self.center
    }

    #[must_use]
    pub const fn axes(self) -> [Vec3; 3] {
        self.axes
    }

    /// Same orientation at another center.
    ///
    /// # Errors
    /// [`EllipsoidError3::NonFiniteInput`].
    pub fn try_with_center(self, center: Vec3) -> Result<Self, EllipsoidError3> {
        if !is_finite(center) {
            return Err(EllipsoidError3::NonFiniteInput);
        }
        Ok(Self { center, ..self })
    }
}

/// Contact of an ellipsoid with a solid halfspace `{x : dot(x - point, normal) <= 0}`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EllipsoidHalfspaceContact3 {
    /// Distance from the ellipsoid to the boundary plane; negative when overlapping.
    pub separation: f64,
    /// Deepest ellipsoid point along `-normal` (`support(-normal)`).
    pub point: Vec3,
}

/// First touch of a translating ellipsoid with a halfspace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EllipsoidHalfspaceHit3 {
    /// Fraction of the displacement in `0..=1`; 0 when it starts touching or overlapping.
    pub fraction: f64,
    /// Contact at `fraction`: separation 0 for a moving hit, the starting value otherwise.
    pub contact: EllipsoidHalfspaceContact3,
}

/// A posed ellipsoid. Both parts are validated by their constructors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EllipsoidBody3 {
    pub shape: Ellipsoid3,
    pub pose: EllipsoidPose3,
}

impl EllipsoidBody3 {
    #[must_use]
    pub const fn new(shape: Ellipsoid3, pose: EllipsoidPose3) -> Self {
        Self { shape, pose }
    }

    /// `center + L Lᵀ d / ‖Lᵀ d‖`, invariant under positive scaling of `direction`.
    ///
    /// # Errors
    /// [`EllipsoidError3::NonFiniteInput`], [`EllipsoidError3::ZeroDirection`] or
    /// [`EllipsoidError3::Unrepresentable`].
    #[inline]
    pub fn try_support_point(self, direction: Vec3) -> Result<Vec3, EllipsoidError3> {
        if !is_finite(direction) {
            return Err(EllipsoidError3::NonFiniteInput);
        }
        if direction == [0.0; 3] {
            return Err(EllipsoidError3::ZeroDirection);
        }
        // The support point is invariant under positive scaling, so `direction` need not be
        // normalized first; `support_offset` rescales tiny or huge `Lᵀ d` exactly.
        let (offset, _) = self.support_offset(direction)?;
        finite_vector(add(self.pose.center, offset))
    }

    /// `max_{x in E} dot(x - center, d̂) = ‖Lᵀ d̂‖` for the normalized `direction`.
    ///
    /// # Errors
    /// As [`Self::try_support_point`].
    #[inline]
    pub fn try_support_radius(self, direction: Vec3) -> Result<f64, EllipsoidError3> {
        let unit = unit_direction(direction)?;
        Ok(self.support_offset(unit)?.1)
    }

    /// Exact world AABB half extents: component `i` is `‖row_i(L)‖`, the support radius
    /// along world axis `i`.
    ///
    /// # Errors
    /// [`EllipsoidError3::Unrepresentable`] on overflow.
    #[inline]
    pub fn try_aabb_half_extents(self) -> Result<Vec3, EllipsoidError3> {
        let [a, b, c] = self.shape.semiaxes;
        let [x, y, z] = self.pose.axes;
        Ok([
            scaled_norm([a * x[0], b * y[0], c * z[0]])?,
            scaled_norm([a * x[1], b * y[1], c * z[1]])?,
            scaled_norm([a * x[2], b * y[2], c * z[2]])?,
        ])
    }

    /// World AABB as `(min, max)`.
    ///
    /// # Errors
    /// [`EllipsoidError3::Unrepresentable`] on overflow.
    pub fn try_aabb(self) -> Result<(Vec3, Vec3), EllipsoidError3> {
        let half = self.try_aabb_half_extents()?;
        Ok((
            finite_vector(sub(self.pose.center, half))?,
            finite_vector(add(self.pose.center, half))?,
        ))
    }

    /// Conservative outer radius for the validated pose: `max(a,b,c)` widened by the
    /// accepted axis non-orthonormality, so `‖x - center‖ <= outer` for every `x in E`.
    #[must_use]
    pub fn outer_radius(self) -> f64 {
        // ‖RᵀR - I‖₂ <= 3τ gives σ_max(R) <= 1 + 1.5τ; the extra margin covers rounding.
        self.shape.outer_radius() * (1.0 + 2.0 * ELLIPSOID_AXES_TOLERANCE)
    }

    /// Conservative inner radius for the validated pose: the ball of this radius about the
    /// center lies inside `E`.
    #[must_use]
    pub fn inner_radius(self) -> f64 {
        // σ_min(R) >= sqrt(1 - 3τ) >= 1 - 3τ; the extra margin covers rounding.
        self.shape.inner_radius() * (1.0 - 4.0 * ELLIPSOID_AXES_TOLERANCE)
    }

    /// Contact with the solid halfspace `{x : dot(x - point, normal) <= 0}`; `normal` is
    /// normalized here and points out of the solid.
    ///
    /// # Errors
    /// [`EllipsoidError3::NonFiniteInput`], [`EllipsoidError3::ZeroDirection`] or
    /// [`EllipsoidError3::Unrepresentable`].
    pub fn try_halfspace_contact(
        self,
        point: Vec3,
        normal: Vec3,
    ) -> Result<EllipsoidHalfspaceContact3, EllipsoidError3> {
        if !is_finite(point) {
            return Err(EllipsoidError3::NonFiniteInput);
        }
        let (normal, norm) = plane_normal(normal)?;
        let support = self.support_offset(scale(normal, -1.0))?;
        let gap = raw_gap(self.pose.center, support.1, point, normal)?;
        halfspace_contact(self.pose.center, support.0, gap / norm)
    }

    /// Fixed-orientation translational first touch with the halfspace, as a fraction of
    /// `displacement`. `Ok(None)` is a proven miss over the whole displacement.
    ///
    /// The separation is affine in the fraction because orientation is fixed, so this is
    /// exact up to f64 rounding. Rotating motion is not swept here; see
    /// [`try_pose_displacement_bound`] for a conservative-advancement certificate.
    ///
    /// # Errors
    /// As [`Self::try_halfspace_contact`], plus a non-finite `displacement`.
    pub fn try_halfspace_sweep(
        self,
        displacement: Vec3,
        point: Vec3,
        normal: Vec3,
    ) -> Result<Option<EllipsoidHalfspaceHit3>, EllipsoidError3> {
        if !is_finite(point) || !is_finite(displacement) {
            return Err(EllipsoidError3::NonFiniteInput);
        }
        // Gap and rate share the normal's scale, so the fraction needs no normalization.
        let (normal, norm) = plane_normal(normal)?;
        // Orientation is fixed, so the deepest point's offset is the same along the path.
        let (offset, radius) = self.support_offset(scale(normal, -1.0))?;
        let gap = raw_gap(self.pose.center, radius, point, normal)?;
        if gap <= 0.0 {
            return Ok(Some(EllipsoidHalfspaceHit3 {
                fraction: 0.0,
                contact: halfspace_contact(self.pose.center, offset, gap / norm)?,
            }));
        }
        let rate = dot(displacement, normal);
        if !rate.is_finite() {
            return Err(EllipsoidError3::Unrepresentable);
        }
        if rate >= 0.0 {
            return Ok(None);
        }
        let fraction = gap / -rate;
        if !fraction.is_finite() {
            return Err(EllipsoidError3::Unrepresentable);
        }
        if fraction > 1.0 {
            return Ok(None);
        }
        let center = add(self.pose.center, scale(displacement, fraction));
        Ok(Some(EllipsoidHalfspaceHit3 {
            fraction,
            contact: halfspace_contact(center, offset, 0.0)?,
        }))
    }

    /// Returns `(L Lᵀ d / ‖Lᵀ d‖, ‖Lᵀ d‖)` for a finite non-zero `d`.
    // Always inlined: returning the tuple through memory costs a store-forwarding stall.
    #[inline(always)]
    fn support_offset(self, direction: Vec3) -> Result<(Vec3, f64), EllipsoidError3> {
        let semiaxes = self.shape.semiaxes;
        let axes = self.pose.axes;
        let local: Vec3 =
            std::array::from_fn(|index| semiaxes[index] * dot(axes[index], direction));
        let squared = dot(local, local);
        if squared.is_normal() {
            // Ordinary magnitudes: one square root and one reciprocal.
            let radius = squared.sqrt();
            let inverse = radius.recip();
            let mut offset = [0.0; 3];
            for index in 0..3 {
                offset = add(
                    offset,
                    scale(axes[index], semiaxes[index] * local[index] * inverse),
                );
            }
            // |offset| <= max semiaxis and a normal square has a finite root.
            return Ok((offset, radius));
        }
        self.support_offset_rescaled(local)
    }

    /// Tiny, subnormal or overflowing `Lᵀ d`: rescale by its largest component first.
    #[cold]
    #[inline(never)]
    fn support_offset_rescaled(self, local: Vec3) -> Result<(Vec3, f64), EllipsoidError3> {
        let semiaxes = self.shape.semiaxes;
        let axes = self.pose.axes;
        let largest = local.into_iter().map(f64::abs).fold(0.0, f64::max);
        if largest == 0.0 || !largest.is_finite() {
            return Err(EllipsoidError3::Unrepresentable);
        }
        let reduced = local.map(|value| value / largest);
        let reduced_length = dot(reduced, reduced).sqrt();
        let direction = reduced.map(|value| value / reduced_length);
        let mut offset = [0.0; 3];
        for index in 0..3 {
            offset = add(
                offset,
                scale(axes[index], semiaxes[index] * direction[index]),
            );
        }
        let radius = largest * reduced_length;
        if !radius.is_finite() {
            return Err(EllipsoidError3::Unrepresentable);
        }
        Ok((finite_vector(offset)?, radius))
    }
}

/// Generic support mapping for GJK/EPA. A zero direction maps to local +X of the world, as
/// for the crate's other support maps; non-finite directions panic like theirs.
///
/// # Panics
///
/// Panics if `direction` is not finite, or if the support point overflows f64 (use
/// [`EllipsoidBody3::try_support_point`] to handle that as an error).
impl SupportMap3 for EllipsoidBody3 {
    fn support_point(&self, direction: Vec3) -> Vec3 {
        assert!(is_finite(direction), "support direction must be finite");
        let direction = if direction == [0.0; 3] {
            [1.0, 0.0, 0.0]
        } else {
            direction
        };
        match self.try_support_point(direction) {
            Ok(point) => point,
            Err(error) => panic!("ellipsoid support point is not representable: {error}"),
        }
    }
}

/// Upper bound on the displacement of every material point of `shape` between two poses:
/// `‖Δcenter‖ + ‖(R₁ - R₀) diag(a,b,c)‖_F`, rounded up.
///
/// A material point `c + R D u` with `‖u‖ <= 1` moves by at most `‖Δc‖ + ‖(R₁-R₀)D‖₂`, and the
/// Frobenius norm bounds the spectral norm, so the certificate is valid for any accepted
/// axes. Conservative advancement may move at most `gap / bound` of the interval.
///
/// # Errors
/// [`EllipsoidError3::Unrepresentable`] on overflow.
pub fn try_pose_displacement_bound(
    shape: Ellipsoid3,
    from: EllipsoidPose3,
    to: EllipsoidPose3,
) -> Result<f64, EllipsoidError3> {
    let translation = scaled_norm(sub(to.center, from.center))?;
    let columns: [Vec3; 3] = std::array::from_fn(|index| {
        scale(sub(to.axes[index], from.axes[index]), shape.semiaxes[index])
    });
    let columns = columns.as_flattened();
    let largest = columns.iter().copied().map(f64::abs).fold(0.0, f64::max);
    let rotation = if largest == 0.0 {
        0.0
    } else {
        largest
            * columns
                .iter()
                .map(|&value| (value / largest).powi(2))
                .sum::<f64>()
                .sqrt()
    };
    round_up(translation + rotation)
}

/// Upper bound on material-point displacement for a center displacement and a rotation of
/// at most `angle_bound` radians: `‖Δcenter‖ + min(angle_bound, 2) * outer_radius`, rounded up
/// (a rotation by θ moves a point at distance r by the chord `2 r sin(θ/2) <= min(θ, 2) r`).
///
/// # Errors
/// [`EllipsoidError3::NonFiniteInput`], [`EllipsoidError3::InvalidParameter`] for a negative
/// angle or [`EllipsoidError3::Unrepresentable`].
pub fn try_rotational_displacement_bound(
    shape: Ellipsoid3,
    center_displacement: Vec3,
    angle_bound: f64,
) -> Result<f64, EllipsoidError3> {
    if !is_finite(center_displacement) || !angle_bound.is_finite() {
        return Err(EllipsoidError3::NonFiniteInput);
    }
    if angle_bound < 0.0 {
        return Err(EllipsoidError3::InvalidParameter);
    }
    let radius = shape.outer_radius() * (1.0 + 2.0 * ELLIPSOID_AXES_TOLERANCE);
    let translation = scaled_norm(center_displacement)?;
    round_up(translation + angle_bound.min(2.0) * radius)
}

/// `dot(center - point, n) - ‖Lᵀ n‖`: the halfspace gap scaled by `‖n‖`.
#[inline]
fn raw_gap(center: Vec3, radius: f64, point: Vec3, normal: Vec3) -> Result<f64, EllipsoidError3> {
    let gap = dot(finite_vector(sub(center, point))?, normal) - radius;
    if gap.is_finite() {
        Ok(gap)
    } else {
        Err(EllipsoidError3::Unrepresentable)
    }
}

#[inline]
fn halfspace_contact(
    center: Vec3,
    offset: Vec3,
    separation: f64,
) -> Result<EllipsoidHalfspaceContact3, EllipsoidError3> {
    if !separation.is_finite() {
        return Err(EllipsoidError3::Unrepresentable);
    }
    Ok(EllipsoidHalfspaceContact3 {
        separation,
        point: finite_vector(add(center, offset))?,
    })
}

/// A halfspace normal of moderate magnitude is used as given with its length; others are
/// normalized first so the products below neither overflow nor underflow.
#[inline]
fn plane_normal(normal: Vec3) -> Result<(Vec3, f64), EllipsoidError3> {
    if !is_finite(normal) {
        return Err(EllipsoidError3::NonFiniteInput);
    }
    let squared = dot(normal, normal);
    if (1.0e-200..=1.0e200).contains(&squared) {
        return Ok((normal, squared.sqrt()));
    }
    Ok((unit_direction(normal)?, 1.0))
}

#[inline]
fn unit_direction(direction: Vec3) -> Result<Vec3, EllipsoidError3> {
    if !is_finite(direction) {
        return Err(EllipsoidError3::NonFiniteInput);
    }
    let squared = dot(direction, direction);
    if squared.is_normal() {
        return Ok(scale(direction, squared.sqrt().recip()));
    }
    let unit = unit_or_zero(direction);
    if unit == [0.0; 3] {
        return Err(EllipsoidError3::ZeroDirection);
    }
    Ok(unit)
}

fn scaled_norm(vector: Vec3) -> Result<f64, EllipsoidError3> {
    let squared = dot(vector, vector);
    if squared.is_normal() {
        return Ok(squared.sqrt());
    }
    let largest = vector.into_iter().map(f64::abs).fold(0.0, f64::max);
    if !largest.is_finite() {
        return Err(EllipsoidError3::Unrepresentable);
    }
    if largest == 0.0 {
        return Ok(0.0);
    }
    let reduced = vector.map(|value| value / largest);
    let norm = largest * dot(reduced, reduced).sqrt();
    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(EllipsoidError3::Unrepresentable)
    }
}

fn finite_vector(vector: Vec3) -> Result<Vec3, EllipsoidError3> {
    if is_finite(vector) {
        Ok(vector)
    } else {
        Err(EllipsoidError3::Unrepresentable)
    }
}

fn positive_finite(value: f64) -> Result<f64, EllipsoidError3> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(EllipsoidError3::Unrepresentable)
    }
}

/// Widens a sum of a few non-negative rounded terms past its accumulated rounding error.
fn round_up(value: f64) -> Result<f64, EllipsoidError3> {
    let widened = value * (1.0 + 16.0 * f64::EPSILON);
    if widened.is_finite() {
        Ok(widened)
    } else {
        Err(EllipsoidError3::Unrepresentable)
    }
}
