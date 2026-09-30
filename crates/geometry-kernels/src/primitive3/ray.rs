//! Bounded analytic segment rays against the existing primitive authority.
use super::{
    PrimitiveBody3, PrimitiveKind3, inverse_rotate, rotate_local, unit_or_zero, wedge_planes,
};
use crate::math3::{Vec3, add, cross, dot, is_finite, neg, scale, sub};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveRayError3 {
    InvalidInput,
    NonFiniteComputation,
}

/// Stable geometric identity within a primitive, independent of its world pose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveRayFeature3 {
    /// An interior origin has no surface witness; normal opposes the ray.
    Interior,
    Sphere,
    /// Box faces: -X, +X, -Y, +Y, -Z, +Z (indices 0..6).
    BoxFace(u8),
    CapsuleSide,
    CapsuleNegativeCap,
    CapsulePositiveCap,
    /// Wedge faces: bottom, -X, slope, +Z, -Z (indices 0..5).
    WedgeFace(u8),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrimitiveRayHit3 {
    pub fraction: f64,
    pub point: Vec3,
    /// Outward at a surface; opposite displacement for an interior origin.
    pub normal: Vec3,
    pub feature: PrimitiveRayFeature3,
    /// Strict interior beyond the scale-aware rounding band; touching is false.
    pub starts_inside: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrimitiveRayWork3 {
    pub queries: u64,
    pub planes_tested: u64,
    pub quadratic_tests: u64,
}

/// First occupied point across origin + displacement * [0, 1], inclusive.
/// Zero displacement is invalid. Target velocity is irrelevant to snapshot geometry.
/// Sphere/capsule queries use at most 1/3 quadratics; box/wedge use 6/5 planes.
/// Applicable axes must be orthonormal (capsule needs only its local Y axis).
/// Shape dimensions come from validated constructors. Rounding bands use 32 eps
/// at the normalized geometry scale; extreme ratios that underflow are errors.
/// No allocation, iterative search, angular sweep, or arbitrary-magnitude guarantee.
pub fn try_ray_cast(
    body: PrimitiveBody3,
    origin: Vec3,
    displacement: Vec3,
    work: &mut PrimitiveRayWork3,
) -> Result<Option<PrimitiveRayHit3>, PrimitiveRayError3> {
    use PrimitiveRayError3::{InvalidInput, NonFiniteComputation};
    if !is_finite(origin)
        || !is_finite(displacement)
        || displacement == [0.0; 3]
        || !is_finite(body.position)
        || !valid_axes(body)
    {
        return Err(InvalidInput);
    }
    work.queries += 1;
    let offset = sub(origin, body.position);
    if !is_finite(offset) || !is_finite(add(origin, displacement)) {
        return Err(NonFiniteComputation);
    }
    let half = body.shape.half_extents();
    if !is_finite(half) {
        return Err(NonFiniteComputation);
    }
    let magnitude = offset
        .into_iter()
        .chain(displacement)
        .chain(half)
        .map(f64::abs)
        .fold(0.0, f64::max);
    let o = offset.map(|v| v / magnitude);
    let d = displacement.map(|v| v / magnitude);
    let half = half.map(|v| v / magnitude);
    if half.into_iter().any(|v| v == 0.0) || dot(d, d) == 0.0 {
        return Err(NonFiniteComputation);
    }
    let candidate = match body.shape.kind() {
        PrimitiveKind3::Sphere => curved(o, d, 0.0, half[0], [0.0, 1.0, 0.0], work),
        PrimitiveKind3::Capsule => {
            let (h, r) = body.shape.capsule_parts();
            if h > 0.0 && h / magnitude == 0.0 {
                return Err(NonFiniteComputation);
            }
            curved(o, d, h / magnitude, r / magnitude, body.axes[1], work)
        }
        PrimitiveKind3::Box | PrimitiveKind3::Wedge => polyhedral(
            body,
            inverse_rotate(body.axes, o),
            inverse_rotate(body.axes, d),
            half,
            work,
        ),
    }?;
    let Some(mut hit) = candidate else {
        return Ok(None);
    };
    hit.point = add(origin, scale(displacement, hit.fraction));
    if !is_finite(hit.point)
        || !is_finite(hit.normal)
        || (dot(hit.normal, hit.normal) - 1.0).abs() > 1e-9
    {
        return Err(NonFiniteComputation);
    }
    Ok(Some(hit))
}

fn valid_axes(body: PrimitiveBody3) -> bool {
    let unit = |v: Vec3| is_finite(v) && (dot(v, v) - 1.0).abs() <= 1e-9;
    match body.shape.kind() {
        PrimitiveKind3::Sphere => true,
        PrimitiveKind3::Capsule => unit(body.axes[1]),
        _ => {
            body.axes.into_iter().all(unit)
                && dot(body.axes[0], body.axes[1]).abs() <= 1e-9
                && dot(body.axes[0], body.axes[2]).abs() <= 1e-9
                && dot(body.axes[1], body.axes[2]).abs() <= 1e-9
        }
    }
}

const ROUNDING: f64 = 32.0 * f64::EPSILON;
fn hit(
    fraction: f64,
    normal: Vec3,
    feature: PrimitiveRayFeature3,
    starts_inside: bool,
) -> PrimitiveRayHit3 {
    PrimitiveRayHit3 {
        fraction,
        point: [0.0; 3],
        normal,
        feature,
        starts_inside,
    }
}

fn curved(
    o: Vec3,
    d: Vec3,
    h: f64,
    r: f64,
    axis: Vec3,
    work: &mut PrimitiveRayWork3,
) -> Result<Option<PrimitiveRayHit3>, PrimitiveRayError3> {
    use PrimitiveRayFeature3::*;
    let axial = dot(o, axis);
    let nearest = scale(axis, axial.clamp(-h, h));
    let radial = sub(o, nearest);
    let radius2 = r * r;
    if radius2 == 0.0 {
        return Err(PrimitiveRayError3::NonFiniteComputation);
    }
    let distance2 = dot(radial, radial);
    let band = ROUNDING * radius2.max(distance2);
    if distance2 <= radius2 + band {
        let inside = distance2 < radius2 - band;
        let feature = if inside {
            Interior
        } else if h == 0.0 {
            Sphere
        } else if axial < -h {
            CapsuleNegativeCap
        } else if axial > h {
            CapsulePositiveCap
        } else {
            CapsuleSide
        };
        return Ok(Some(hit(
            0.0,
            if inside {
                neg(unit_or_zero(d))
            } else {
                unit_or_zero(radial)
            },
            feature,
            inside,
        )));
    }
    let mut best = None;
    // A capsule is the union of its finite cylinder and endpoint balls.
    // The earliest occupied point of this union is its true surface entry.
    for (center, feature) in if h == 0.0 {
        [(0.0, Sphere), (0.0, Sphere)]
    } else {
        [(-h, CapsuleNegativeCap), (h, CapsulePositiveCap)]
    } {
        let oc = sub(o, scale(axis, center));
        if let Some(t) = quadratic_entry(oc, d, r, work)? {
            retain(
                &mut best,
                hit(t, unit_or_zero(add(oc, scale(d, t))), feature, false),
            );
        }
        if h == 0.0 {
            break;
        }
    }
    if h > 0.0 {
        let radial_o = sub(o, scale(axis, axial));
        let radial_d = sub(d, scale(axis, dot(d, axis)));
        if let Some(t) = quadratic_entry(radial_o, radial_d, r, work)? {
            let y = axial + dot(d, axis) * t;
            if y >= -h && y <= h {
                retain(
                    &mut best,
                    hit(
                        t,
                        unit_or_zero(add(radial_o, scale(radial_d, t))),
                        CapsuleSide,
                        false,
                    ),
                );
            }
        }
    }
    Ok(best)
}

fn retain(best: &mut Option<PrimitiveRayHit3>, candidate: PrimitiveRayHit3) {
    // Exact ties retain the first declared feature.
    if best.is_none_or(|previous| candidate.fraction < previous.fraction) {
        *best = Some(candidate);
    }
}

fn quadratic_entry(
    o: Vec3,
    d: Vec3,
    r: f64,
    work: &mut PrimitiveRayWork3,
) -> Result<Option<f64>, PrimitiveRayError3> {
    work.quadratic_tests += 1;
    let a = dot(d, d);
    if a == 0.0 {
        return Ok(None);
    }
    let perpendicular = cross(o, d);
    let perpendicular2 = dot(perpendicular, perpendicular) / a;
    let radius2 = r * r;
    if !perpendicular2.is_finite() {
        return Err(PrimitiveRayError3::NonFiniteComputation);
    }
    if perpendicular2 > radius2 + ROUNDING * radius2.max(perpendicular2) {
        return Ok(None);
    }
    let center = -dot(o, d) / a;
    let chord = ((radius2 - perpendicular2).max(0.0) / a).sqrt();
    let near = center - chord;
    let far = center + chord;
    if !near.is_finite() || !far.is_finite() {
        return Err(PrimitiveRayError3::NonFiniteComputation);
    }
    Ok((far >= 0.0 && near <= 1.0).then_some(near.max(0.0)))
}

fn polyhedral(
    body: PrimitiveBody3,
    o: Vec3,
    d: Vec3,
    half: Vec3,
    work: &mut PrimitiveRayWork3,
) -> Result<Option<PrimitiveRayHit3>, PrimitiveRayError3> {
    let box_planes = [
        ([-1.0, 0.0, 0.0], half[0]),
        ([1.0, 0.0, 0.0], half[0]),
        ([0.0, -1.0, 0.0], half[1]),
        ([0.0, 1.0, 0.0], half[1]),
        ([0.0, 0.0, -1.0], half[2]),
        ([0.0, 0.0, 1.0], half[2]),
    ];
    let wedge = wedge_planes(half);
    let is_box = body.shape.kind() == PrimitiveKind3::Box;
    let planes: &[(Vec3, f64)] = if is_box { &box_planes } else { &wedge };
    let mut enter = 0.0_f64;
    let mut exit = 1.0_f64;
    let mut entry_face = 0;
    let mut nearest_face = 0;
    let mut nearest_depth = f64::INFINITY;
    let mut inside = true;
    let mut strict = true;
    for (index, &(normal, limit)) in planes.iter().enumerate() {
        work.planes_tested += 1;
        let projected = dot(normal, o);
        let start = projected - limit;
        let velocity = dot(normal, d);
        let band = ROUNDING * projected.abs().max(limit.abs());
        inside &= start <= band;
        strict &= start < -band;
        if -start < nearest_depth {
            nearest_depth = -start;
            nearest_face = index;
        }
        if velocity == 0.0 {
            if start > band {
                return Ok(None);
            }
        } else {
            let t = -start / velocity;
            if velocity < 0.0 && t > enter {
                enter = t;
                entry_face = index;
            }
            if velocity > 0.0 {
                exit = exit.min(t);
            }
        }
    }
    if inside {
        if strict {
            return Ok(Some(hit(
                0.0,
                rotate_local(body.axes, neg(unit_or_zero(d))),
                PrimitiveRayFeature3::Interior,
                true,
            )));
        }
        entry_face = nearest_face;
        enter = 0.0;
    } else if enter > exit || exit < 0.0 || enter > 1.0 {
        return Ok(None);
    }
    let normal = rotate_local(body.axes, planes[entry_face].0);
    let feature = if is_box {
        PrimitiveRayFeature3::BoxFace(entry_face as u8)
    } else {
        PrimitiveRayFeature3::WedgeFace(entry_face as u8)
    };
    Ok(Some(hit(enter, normal, feature, false)))
}

#[cfg(test)]
mod tests;
