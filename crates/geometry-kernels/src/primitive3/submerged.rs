//! Volume and centroid of a posed solid on the submerged side of a plane.
//!
//! The submerged side of the plane through `plane_point` with normal `plane_normal`
//! is `{x : dot(x - plane_point, plane_normal) <= 0}`; the normal points out of
//! the fluid. Results are geometric only: density, buoyancy, drag and the fluid
//! model belong to consumers.

use super::{
    PrimitiveKind3, PrimitiveShape3, inverse_rotate, rotate_local, try_volume_properties,
    unit_or_zero,
};
use crate::hull::{Hull3, HullDimension};
use crate::math3::{Vec3, add, cross, dot, is_finite, length, scale, sub};
use std::f64::consts::PI;

/// Submerged volume and its centroid in world coordinates.
///
/// An empty result has `volume == 0.0` and reports the pose position as its centroid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubmergedProperties3 {
    pub volume: f64,
    pub centroid: Vec3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmergedError3 {
    /// Non-finite pose or plane, non-orthonormal axes, a zero plane normal, or a
    /// hull that is not a closed polytope with valid triangle indices.
    InvalidInput,
    /// The direct f64 products overflowed for otherwise valid input.
    NonFiniteComputation,
}

/// Local-frame accumulation shared by every shape: volume and first moment.
#[derive(Clone, Copy, Debug, Default)]
struct Moment {
    volume: f64,
    first: Vec3,
}

impl Moment {
    fn add(&mut self, other: Self) {
        self.volume += other.volume;
        self.first = add(self.first, other.first);
    }
}

/// Plane in the body's local frame: submerged points satisfy `dot(normal, x) <= level`.
#[derive(Clone, Copy, Debug)]
struct LocalPlane {
    normal: Vec3,
    level: f64,
}

impl LocalPlane {
    fn distance(self, point: Vec3) -> f64 {
        dot(self.normal, point) - self.level
    }
}

const AXES_TOLERANCE: f64 = 1e-9;

fn local_plane(
    position: Vec3,
    axes: [Vec3; 3],
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Result<LocalPlane, SubmergedError3> {
    let unit = |v: Vec3| is_finite(v) && (dot(v, v) - 1.0).abs() <= AXES_TOLERANCE;
    if !is_finite(position)
        || !is_finite(plane_point)
        || !is_finite(plane_normal)
        || !axes.into_iter().all(unit)
        || dot(axes[0], axes[1]).abs() > AXES_TOLERANCE
        || dot(axes[0], axes[2]).abs() > AXES_TOLERANCE
        || dot(axes[1], axes[2]).abs() > AXES_TOLERANCE
    {
        return Err(SubmergedError3::InvalidInput);
    }
    let normal = unit_or_zero(plane_normal);
    if normal == [0.0; 3] {
        return Err(SubmergedError3::InvalidInput);
    }
    let level = dot(normal, sub(plane_point, position));
    if !level.is_finite() {
        return Err(SubmergedError3::NonFiniteComputation);
    }
    Ok(LocalPlane {
        normal: inverse_rotate(axes, normal),
        level,
    })
}

fn finish(
    moment: Moment,
    position: Vec3,
    axes: [Vec3; 3],
) -> Result<SubmergedProperties3, SubmergedError3> {
    if !moment.volume.is_finite() || !is_finite(moment.first) {
        return Err(SubmergedError3::NonFiniteComputation);
    }
    if moment.volume <= 0.0 {
        return Ok(empty(position));
    }
    let local = scale(moment.first, moment.volume.recip());
    let centroid = add(position, rotate_local(axes, local));
    if !is_finite(centroid) {
        return Err(SubmergedError3::NonFiniteComputation);
    }
    Ok(SubmergedProperties3 {
        volume: moment.volume,
        centroid,
    })
}

const fn empty(position: Vec3) -> SubmergedProperties3 {
    SubmergedProperties3 {
        volume: 0.0,
        centroid: position,
    }
}

fn full(
    shape: PrimitiveShape3,
    position: Vec3,
    axes: [Vec3; 3],
) -> Result<SubmergedProperties3, SubmergedError3> {
    let properties =
        try_volume_properties(shape).map_err(|_| SubmergedError3::NonFiniteComputation)?;
    let centroid = add(position, rotate_local(axes, properties.centroid));
    if !is_finite(centroid) {
        return Err(SubmergedError3::NonFiniteComputation);
    }
    Ok(SubmergedProperties3 {
        volume: properties.volume,
        centroid,
    })
}

/// Volume and centroid of a posed primitive on the submerged side of a plane.
///
/// `axes` are the world-space local X/Y/Z axes and must be orthonormal within
/// 1e-9 for every shape; `plane_normal` need not be unit length but must be
/// finite and non-zero. A fully submerged shape returns exactly the volume and
/// (posed) centroid of [`try_volume_properties`].
///
/// - Sphere: closed-form spherical cap.
/// - Box and wedge: exact plane clip of the polytope, summed as tetrahedra.
/// - Capsule: closed-form hemispheres (divergence theorem with the two-cap
///   spherical area); the cylinder integrates circular-segment slices with a
///   fixed 16-point Gauss–Legendre rule in the segment angle, split at the
///   slice-type breakpoints. The integrand is entire in that angle, so the
///   quadrature error is at rounding level: the mean slice area is within
///   1e-13 `r^2` of the closed form (see `docs/submerged-volume.md`).
///
/// Every path is a fixed sequence of scalar operations with no allocation.
///
/// # Errors
/// [`SubmergedError3::InvalidInput`] for non-finite inputs, non-orthonormal axes
/// or a zero normal; [`SubmergedError3::NonFiniteComputation`] when the f64
/// products overflow.
pub fn try_submerged_properties(
    shape: PrimitiveShape3,
    position: Vec3,
    axes: [Vec3; 3],
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Result<SubmergedProperties3, SubmergedError3> {
    let plane = local_plane(position, axes, plane_point, plane_normal)?;
    let moment = match shape.kind() {
        PrimitiveKind3::Sphere => {
            let r = shape.sphere_radius();
            // Centre height above the surface.
            let s = -plane.level;
            if s >= r {
                return Ok(empty(position));
            }
            if s <= -r {
                return full(shape, position, axes);
            }
            let (volume, depth_offset) = sphere_cap(r, plane.level);
            Moment {
                volume,
                first: scale(plane.normal, -depth_offset * volume),
            }
        }
        PrimitiveKind3::Box => {
            let [x, y, z] = shape.half_extents();
            let vertices: [Vec3; 8] = std::array::from_fn(|i| {
                [
                    if i & 1 == 0 { -x } else { x },
                    if i & 2 == 0 { -y } else { y },
                    if i & 4 == 0 { -z } else { z },
                ]
            });
            match clip_polytope(&vertices, BOX_TRIANGLES.iter().copied(), plane) {
                Clip::Empty => return Ok(empty(position)),
                Clip::Full => return full(shape, position, axes),
                Clip::Partial(moment) => moment,
            }
        }
        PrimitiveKind3::Wedge => {
            let [x, y, z] = shape.half_extents();
            let vertices = [
                [-x, -y, -z],
                [-x, y, -z],
                [x, -y, -z],
                [-x, -y, z],
                [-x, y, z],
                [x, -y, z],
            ];
            match clip_polytope(&vertices, WEDGE_TRIANGLES.iter().copied(), plane) {
                Clip::Empty => return Ok(empty(position)),
                Clip::Full => return full(shape, position, axes),
                Clip::Partial(moment) => moment,
            }
        }
        PrimitiveKind3::Capsule => {
            let (half_segment, r) = shape.capsule_parts();
            let reach = plane.normal[1].abs() * half_segment + r;
            let s = -plane.level;
            if s - reach >= 0.0 {
                return Ok(empty(position));
            }
            if s + reach <= 0.0 {
                return full(shape, position, axes);
            }
            capsule(half_segment, r, plane)
        }
    };
    finish(moment, position, axes)
}

/// Volume and centroid of a posed closed convex hull on the submerged side of a plane.
///
/// Hull vertices are in the local frame given by `position` and `axes` (same
/// contract as [`try_submerged_properties`]). The hull must have dimension
/// [`HullDimension::Polytope`] with outward-wound triangles, as produced by
/// [`crate::hull::quickhull3`]. Each triangle is clipped against the plane and
/// summed as a tetrahedron with an apex on the plane, so the cut face needs no
/// explicit polygon. Work is linear in vertices plus triangles, without allocation.
///
/// # Errors
/// [`SubmergedError3::InvalidInput`] for invalid pose/plane, a non-polytope hull,
/// non-finite vertices or out-of-range indices;
/// [`SubmergedError3::NonFiniteComputation`] on overflow.
pub fn try_hull_submerged_properties(
    hull: &Hull3,
    position: Vec3,
    axes: [Vec3; 3],
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Result<SubmergedProperties3, SubmergedError3> {
    let plane = local_plane(position, axes, plane_point, plane_normal)?;
    let count = hull.vertices.len();
    if hull.dimension != HullDimension::Polytope
        || hull.triangles.is_empty()
        || !hull.vertices.iter().copied().all(is_finite)
        || hull
            .triangles
            .iter()
            .flatten()
            .any(|&index| index as usize >= count)
    {
        return Err(SubmergedError3::InvalidInput);
    }
    match clip_polytope(&hull.vertices, hull.triangles.iter().copied(), plane) {
        Clip::Empty => Ok(empty(position)),
        Clip::Full => {
            let apex = hull.vertices[0];
            let mut moment = Moment::default();
            for triangle in &hull.triangles {
                let [a, b, c] = triangle.map(|index| hull.vertices[index as usize]);
                moment.add(tetrahedron(apex, a, b, c));
            }
            finish(moment, position, axes)
        }
        Clip::Partial(moment) => finish(moment, position, axes),
    }
}

/// Spherical cap below a plane at `level` (relative to the centre, along the
/// outward normal): its volume and the centroid's depth below the centre.
fn sphere_cap(r: f64, level: f64) -> (f64, f64) {
    // Submerged height and its complement, both formed without cancellation.
    let height = r + level;
    let remainder = r - level;
    let denominator = 2.0 * r - level;
    let volume = PI * height * height * denominator / 3.0;
    let depth = 3.0 * remainder * remainder / (4.0 * denominator);
    (volume, depth)
}

/// Outward, counter-clockwise box triangles; vertex `i` has `+x/+y/+z` for bits 0/1/2.
const BOX_TRIANGLES: [[u32; 3]; 12] = [
    [0, 4, 6],
    [0, 6, 2],
    [1, 3, 7],
    [1, 7, 5],
    [0, 1, 5],
    [0, 5, 4],
    [2, 6, 7],
    [2, 7, 3],
    [0, 2, 3],
    [0, 3, 1],
    [4, 5, 7],
    [4, 7, 6],
];

/// Outward wedge triangles over the shape's six vertices (rising towards -X).
const WEDGE_TRIANGLES: [[u32; 3]; 8] = [
    [0, 1, 2],
    [3, 5, 4],
    [0, 2, 5],
    [0, 5, 3],
    [0, 3, 4],
    [0, 4, 1],
    [1, 4, 5],
    [1, 5, 2],
];

enum Clip {
    Empty,
    Full,
    Partial(Moment),
}

fn clip_polytope<I>(vertices: &[Vec3], triangles: I, plane: LocalPlane) -> Clip
where
    I: Iterator<Item = [u32; 3]>,
{
    let mut lowest = f64::INFINITY;
    let mut highest = f64::NEG_INFINITY;
    for &vertex in vertices {
        let distance = plane.distance(vertex);
        lowest = lowest.min(distance);
        highest = highest.max(distance);
    }
    if lowest >= 0.0 {
        return Clip::Empty;
    }
    if highest <= 0.0 {
        return Clip::Full;
    }
    // An apex on the plane gives the cut face zero volume, so only clipped
    // boundary triangles contribute.
    let anchor = vertices[0];
    let apex = sub(anchor, scale(plane.normal, plane.distance(anchor)));
    let mut moment = Moment::default();
    for triangle in triangles {
        let points = triangle.map(|index| vertices[index as usize]);
        let distances = points.map(|point| plane.distance(point));
        let mut polygon = [[0.0; 3]; 4];
        let mut size = 0;
        for i in 0..3 {
            let j = (i + 1) % 3;
            if distances[i] <= 0.0 {
                polygon[size] = points[i];
                size += 1;
            }
            if (distances[i] <= 0.0) != (distances[j] <= 0.0) {
                // Interpolate from the submerged end so both triangles sharing
                // the edge produce the same point.
                let (inside, outside, d_in, d_out) = if distances[i] <= 0.0 {
                    (points[i], points[j], distances[i], distances[j])
                } else {
                    (points[j], points[i], distances[j], distances[i])
                };
                let t = d_in / (d_in - d_out);
                polygon[size] = add(inside, scale(sub(outside, inside), t));
                size += 1;
            }
        }
        for k in 1..size.saturating_sub(1) {
            moment.add(tetrahedron(apex, polygon[0], polygon[k], polygon[k + 1]));
        }
    }
    Clip::Partial(moment)
}

fn tetrahedron(apex: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Moment {
    let (a, b, c) = (sub(a, apex), sub(b, apex), sub(c, apex));
    let volume = dot(a, cross(b, c)) / 6.0;
    let centroid = add(apex, scale(add(add(a, b), c), 0.25));
    Moment {
        volume,
        first: scale(centroid, volume),
    }
}

/// `acos(u) - u*sqrt(1-u^2)`: the area, over `rho^2`, of `{v >= u*rho}` in a disc.
fn segment_area(u: f64) -> f64 {
    let u = u.clamp(-1.0, 1.0);
    u.acos() - u * ((1.0 - u) * (1.0 + u)).sqrt()
}

fn capsule(half_segment: f64, r: f64, plane: LocalPlane) -> Moment {
    let n = plane.normal;
    let axis = [0.0, 1.0, 0.0];
    let mut moment = Moment::default();
    for sign in [1.0, -1.0] {
        let centre = [0.0, sign * half_segment, 0.0];
        let level = plane.level - dot(n, centre);
        let mut part = hemisphere(r, scale(axis, sign), n, level);
        part.first = add(part.first, scale(centre, part.volume));
        moment.add(part);
    }
    if half_segment > 0.0 {
        moment.add(cylinder(half_segment, r, plane));
    }
    moment
}

/// Ball of radius `r` at the origin, restricted to `dot(x, direction) >= 0` and
/// `dot(x, n) <= level`. Exact via the divergence theorem: the base face passes
/// through the origin, the sphere contributes `r * area / 3`, and the spherical
/// area is the intersection of two caps (Tovchigrechko & Vakser, 2001).
fn hemisphere(r: f64, direction: Vec3, n: Vec3, level: f64) -> Moment {
    let half_ball = 2.0 * PI * r * r * r / 3.0;
    let half_moment = |sign: f64| Moment {
        volume: half_ball,
        first: scale(direction, sign * 0.375 * r * half_ball),
    };
    if level <= -r {
        return Moment::default();
    }
    if level >= r {
        return half_moment(1.0);
    }
    let cos = dot(n, direction);
    let tangent = sub(direction, scale(n, cos));
    let sin = length(tangent);
    let (cap_volume, cap_depth) = sphere_cap(r, level);
    let cap = Moment {
        volume: cap_volume,
        first: scale(n, -cap_depth * cap_volume),
    };
    if level.abs() >= r * sin {
        // The water disc misses the base plane, so the other half-ball lies
        // wholly on one side of the surface.
        return if level * cos >= 0.0 {
            if -cos * r * 0.5 - level <= 0.0 {
                let other = half_moment(-1.0);
                Moment {
                    volume: cap.volume - other.volume,
                    first: sub(cap.first, other.first),
                }
            } else {
                cap
            }
        } else if cos * r * 0.5 - level <= 0.0 {
            half_moment(1.0)
        } else {
            Moment::default()
        };
    }
    // Generic crossing: sin > 0 and |level| < r * sin.
    let in_water = scale(tangent, sin.recip());
    let rho_squared = (r - level) * (r + level);
    let rho = rho_squared.sqrt();
    // Water face: disc of radius rho at level*n, keeping v >= offset along in_water.
    let offset = -level * cos / sin;
    let ratio = (offset / rho).clamp(-1.0, 1.0);
    let water_area = rho_squared * segment_area(ratio);
    let chord = rho_squared * (1.0 - ratio) * (1.0 + ratio);
    let water_first = add(
        scale(n, level * water_area),
        scale(in_water, 2.0 * chord * chord.sqrt() / 3.0),
    );
    // Base face: disc of radius r with dot(x, n) <= level.
    let base_area = r * r * segment_area(-level / (r * sin));
    let eta = level / r;
    let sphere_area = r
        * r
        * (2.0 * PI - 2.0 * (-cos * r / rho).clamp(-1.0, 1.0).acos() + 2.0 * eta * ratio.acos());
    let volume = (r * sphere_area + level * water_area) / 3.0;
    let first = scale(
        add(
            scale(
                sub(scale(direction, base_area), scale(n, water_area)),
                r * r,
            ),
            scale(water_first, level),
        ),
        0.25,
    );
    Moment { volume, first }
}

/// Positive 16-point Gauss–Legendre nodes and weights on `[-1, 1]`.
const GAUSS_NODES: [f64; 8] = [
    0.095_012_509_837_637_44,
    0.281_603_550_779_258_9,
    0.458_016_777_657_227_37,
    0.617_876_244_402_643_8,
    0.755_404_408_355_003,
    0.865_631_202_387_831_8,
    0.944_575_023_073_232_6,
    0.989_400_934_991_649_9,
];
const GAUSS_WEIGHTS: [f64; 8] = [
    0.189_450_610_455_068_64,
    0.182_603_415_044_923_64,
    0.169_156_519_395_002_65,
    0.149_595_988_816_576_7,
    0.124_628_971_255_534_07,
    0.095_158_511_682_492_6,
    0.062_253_523_938_647_456,
    0.027_152_459_411_754_176,
];

/// Local-Y cylinder `|y| <= half_segment`, radius `r`, below the plane.
///
/// Each slice is a disc whose submerged part is a circular segment with
/// normalized offset `u = s(t) / (r * sin)`, linear in the axial coordinate `t`.
/// Cuts at the breakpoints `u = +-1` carry those values exactly; between them the
/// rule integrates in `psi = acos(|u|)`, where area, centroid and the Jacobian
/// `sin(psi)` are trigonometric polynomials.
fn cylinder(half_segment: f64, r: f64, plane: LocalPlane) -> Moment {
    let n = plane.normal;
    let cos = n[1];
    let tangent = [n[0], 0.0, n[2]];
    let sin = length(tangent);
    let in_water = if sin > 0.0 {
        scale(tangent, sin.recip())
    } else {
        [0.0; 3]
    };
    let rho = r * sin;
    let normalized = |t: f64| {
        let s = cos * t - plane.level;
        if rho > 0.0 {
            (s / rho).clamp(-1.0, 1.0)
        } else if s < 0.0 {
            -1.0
        } else {
            1.0
        }
    };
    // (axial coordinate, normalized offset) at each cut, in axial order.
    let end_cut = |t: f64| (t, normalized(t));
    let mut cuts = [
        end_cut(-half_segment),
        end_cut(half_segment),
        end_cut(half_segment),
        end_cut(half_segment),
    ];
    if cos != 0.0 {
        for (slot, edge) in [-1.0, 1.0].into_iter().enumerate() {
            let t = (edge * rho + plane.level) / cos;
            cuts[slot + 1] = if t.is_finite() && t.abs() < half_segment {
                (t, edge)
            } else {
                end_cut(if t.is_finite() {
                    t.clamp(-half_segment, half_segment)
                } else {
                    half_segment
                })
            };
        }
        cuts[1..3].sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    let disc = PI * r * r;
    let mut moment = Moment::default();
    for window in cuts.windows(2) {
        let ((start, mut u_start), (end, mut u_end)) = (window[0], window[1]);
        let length = end - start;
        if length <= 0.0 {
            continue;
        }
        if rho == 0.0 {
            // A plane normal to the axis: classify by the interval midpoint.
            let middle = normalized(0.5 * (start + end));
            (u_start, u_end) = (middle, middle);
        }
        if u_start <= -1.0 && u_end <= -1.0 {
            let volume = disc * length;
            moment.add(Moment {
                volume,
                first: [0.0, volume * 0.5 * (start + end), 0.0],
            });
            continue;
        }
        if u_start >= 1.0 && u_end >= 1.0 {
            continue;
        }
        let means = segment_means(u_start, u_end);
        let volume = r * r * length * means.area;
        let axial = r * r * length * (start * means.area + length * means.area_fraction);
        let side = -2.0 * r * r * r * length * means.chord_cubed / 3.0;
        moment.add(Moment {
            volume,
            first: add([0.0, axial, 0.0], scale(in_water, side)),
        });
    }
    moment
}

/// Means over an axial interval on which `u` runs linearly from `start` to `end`.
struct SegmentMeans {
    /// Of the segment area over `r^2`.
    area: f64,
    /// Of that area times the fraction of the interval travelled.
    area_fraction: f64,
    /// Of `(1 - u^2)^(3/2)`.
    chord_cubed: f64,
}

/// Gauss–Legendre in `psi = acos(|u|)` with weights `sin(psi)` (the `|du|` Jacobian).
/// Mirroring negative offsets keeps `psi` in `[0, pi/2]`, where `u = +-1` maps to
/// `psi = 0` exactly and small angle differences keep full relative precision.
fn segment_means(start: f64, end: f64) -> SegmentMeans {
    let mirrored = start + end < 0.0;
    let (start, end) = if mirrored {
        (-start, -end)
    } else {
        (start, end)
    };
    let (psi_start, psi_end) = (start.acos(), end.acos());
    let middle = 0.5 * (psi_start + psi_end);
    let half = 0.5 * (psi_end - psi_start);
    // cos(psi_start) - cos(psi_end), without cancellation.
    let span = 2.0 * middle.sin() * half.sin();
    let mut weight = 0.0;
    let mut area = 0.0;
    let mut area_fraction = 0.0;
    let mut fraction_sum = 0.0;
    let mut chord_cubed = 0.0;
    for (node, base_weight) in GAUSS_NODES.into_iter().zip(GAUSS_WEIGHTS) {
        for offset in [-node, node] {
            let psi = middle + half * offset;
            let (sine, cosine) = psi.sin_cos();
            let w = base_weight * sine;
            // Fraction of the axial interval: (cos(psi_start) - cos(psi)) / span.
            let fraction = if span == 0.0 {
                0.5
            } else {
                let step = half * (1.0 + offset);
                2.0 * (psi_start + 0.5 * step).sin() * (0.5 * step).sin() / span
            };
            let segment = psi - sine * cosine;
            weight += w;
            area += w * segment;
            area_fraction += w * segment * fraction;
            fraction_sum += w * fraction;
            chord_cubed += w * sine * sine * sine;
        }
    }
    if weight == 0.0 || half == 0.0 {
        // Constant offset over the interval.
        let sine = psi_start.sin();
        let segment = psi_start - sine * psi_start.cos();
        let segment = if mirrored { PI - segment } else { segment };
        return SegmentMeans {
            area: segment,
            area_fraction: 0.5 * segment,
            chord_cubed: sine * sine * sine,
        };
    }
    let (area, area_fraction) = (area / weight, area_fraction / weight);
    let fraction = fraction_sum / weight;
    if mirrored {
        // The submerged area at -u is pi minus the area at u.
        SegmentMeans {
            area: PI - area,
            area_fraction: PI * fraction - area_fraction,
            chord_cubed: chord_cubed / weight,
        }
    } else {
        SegmentMeans {
            area,
            area_fraction,
            chord_cubed: chord_cubed / weight,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Antiderivatives of the slice integrands in `u`: segment area, `u` times
    /// it, and `(1-u^2)^(3/2)`.
    fn closed_forms(u: f64) -> [f64; 3] {
        let root = ((1.0 - u) * (1.0 + u)).sqrt();
        let acos = u.acos();
        let area = u * acos - root + root * root * root / 3.0;
        // integral u*phi(u) du
        let moment = 0.5 * u * u * acos + 0.125 * u.asin()
            - 0.25 * u * root
            - 0.125 * u * (2.0 * u * u - 1.0) * root;
        let chord = 0.125 * (u * root * (5.0 - 2.0 * u * u) + 3.0 * u.asin());
        [area, moment, chord]
    }

    #[test]
    fn segment_quadrature_matches_closed_forms() {
        let mut worst = [0.0_f64; 3];
        let samples = 41;
        for i in 0..samples {
            for j in 0..samples {
                let a = -1.0 + 2.0 * f64::from(i) / f64::from(samples - 1);
                let b = -1.0 + 2.0 * f64::from(j) / f64::from(samples - 1);
                if (a - b).abs() < 0.04 {
                    continue;
                }
                let means = segment_means(a, b);
                let (fa, fb) = (closed_forms(a), closed_forms(b));
                let du = b - a;
                let area = (fb[0] - fa[0]) / du;
                // Fraction (u - a) / du: mean(phi * fraction) = (M(b)-M(a) - a*(A(b)-A(a))) / du^2.
                let area_fraction = (fb[1] - fa[1] - a * (fb[0] - fa[0])) / (du * du);
                let chord = (fb[2] - fa[2]) / du;
                worst[0] = worst[0].max((means.area - area).abs());
                worst[1] = worst[1].max((means.area_fraction - area_fraction).abs());
                worst[2] = worst[2].max((means.chord_cubed - chord).abs());
            }
        }
        // Area and chord means agree to rounding; the fraction reference divides
        // by du^2 >= 0.0016, which amplifies its own rounding.
        assert!(worst[0] <= 1e-13 && worst[2] <= 1e-13, "{worst:?}");
        assert!(worst[1] <= 1e-12, "{worst:?}");
    }
}
