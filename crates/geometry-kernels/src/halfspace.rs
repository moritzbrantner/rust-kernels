//! Closed forms for a solid halfspace `{x : dot(x - point, normal) <= 0}` with a unit `normal`
//! pointing out of the solid: capsule (sphere when the segment collapses) contact and
//! translational sweep, and segment casts.

use crate::math3::{Vec3, add, dot, scale, sub};

/// A capsule's contact with the halfspace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfspaceContact3 {
    /// Distance from the capsule surface to the boundary plane; negative when overlapping.
    pub separation: f64,
    /// Closest point on the boundary plane.
    pub point: Vec3,
}

/// Contact of the capsule `p`–`q` with `radius`.
#[must_use]
pub fn capsule_halfspace_contact(
    p: Vec3,
    q: Vec3,
    radius: f64,
    point: Vec3,
    normal: Vec3,
) -> HalfspaceContact3 {
    let (hp, hq) = (dot(sub(p, point), normal), dot(sub(q, point), normal));
    let (low, height) = if hq < hp { (q, hq) } else { (p, hp) };
    HalfspaceContact3 {
        separation: height - radius,
        point: sub(low, scale(normal, height)),
    }
}

/// First touch, as a fraction of `displacement` in `0..=1`, of the translating capsule; 0 when
/// it starts touching or overlapping, `None` when it never touches within the displacement.
#[must_use]
pub fn capsule_halfspace_sweep(
    p: Vec3,
    q: Vec3,
    radius: f64,
    displacement: Vec3,
    point: Vec3,
    normal: Vec3,
) -> Option<(f64, HalfspaceContact3)> {
    let start = capsule_halfspace_contact(p, q, radius, point, normal);
    if start.separation <= 0.0 {
        return Some((0.0, start));
    }
    let rate = dot(displacement, normal);
    if rate >= 0.0 {
        return None;
    }
    let fraction = start.separation / -rate;
    if fraction > 1.0 {
        return None;
    }
    let offset = scale(displacement, fraction);
    let contact = capsule_halfspace_contact(add(p, offset), add(q, offset), radius, point, normal);
    Some((
        fraction,
        HalfspaceContact3 {
            separation: 0.0,
            ..contact
        },
    ))
}

/// Entry fraction of the segment `start`–`end` into the solid halfspace; 0 when it starts inside.
#[must_use]
pub fn segment_halfspace(start: Vec3, end: Vec3, point: Vec3, normal: Vec3) -> Option<f64> {
    let h0 = dot(sub(start, point), normal);
    if h0 <= 0.0 {
        return Some(0.0);
    }
    let h1 = dot(sub(end, point), normal);
    if h1 > 0.0 {
        return None;
    }
    Some(h0 / (h0 - h1))
}

#[cfg(test)]
mod tests {
    use super::*;

    const UP: Vec3 = [0.0, 1.0, 0.0];

    #[test]
    fn contact_sweep_and_segment_on_a_floor() {
        let c = capsule_halfspace_contact([1.0, 1.3, 0.0], [1.0, 2.3, 0.0], 0.3, [0.0; 3], UP);
        assert!((c.separation - 1.0).abs() < 1e-12);
        assert_eq!(c.point, [1.0, 0.0, 0.0]);
        let (t, hit) = capsule_halfspace_sweep(
            [1.0, 1.3, 0.0],
            [1.0, 2.3, 0.0],
            0.3,
            [0.0, -2.0, 0.0],
            [0.0; 3],
            UP,
        )
        .unwrap();
        assert!((t - 0.5).abs() < 1e-12 && hit.separation == 0.0);
        assert!(
            capsule_halfspace_sweep(
                [0.0, 1.3, 0.0],
                [0.0, 2.3, 0.0],
                0.3,
                [1.0, 0.0, 0.0],
                [0.0; 3],
                UP
            )
            .is_none()
        );
        assert_eq!(
            segment_halfspace([0.0, 4.0, 0.0], [0.0, -4.0, 0.0], [0.0; 3], UP),
            Some(0.5)
        );
        assert_eq!(
            segment_halfspace([0.0, -1.0, 0.0], [0.0, 4.0, 0.0], [0.0; 3], UP),
            Some(0.0)
        );
        assert_eq!(
            segment_halfspace([0.0, 1.0, 0.0], [5.0, 2.0, 0.0], [0.0; 3], UP),
            None
        );
    }
}
