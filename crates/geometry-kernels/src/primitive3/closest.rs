//! Closest-point queries on triangles, segments and axis-aligned boxes.

use super::query::PrimitiveWork3;
use crate::math3::{Vec3, add, cross, dot, length_squared, scale, sub};

// `#[inline]` keeps these inlinable into `pairs` despite the codegen-unit split; measured.

#[inline]
pub(super) fn update_pair(a: Vec3, b: Vec3, best: &mut (Vec3, Vec3), best_distance: &mut f64) {
    let distance = length_squared(sub(b, a));
    if distance < *best_distance {
        *best = (a, b);
        *best_distance = distance;
    }
}

pub(crate) fn closest_point_triangle(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
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

#[inline]
pub(super) fn closest_point_segment(
    point: Vec3,
    a: Vec3,
    b: Vec3,
    work: &mut PrimitiveWork3,
) -> Vec3 {
    work.segment_distance_evaluations += 1;
    let delta = sub(b, a);
    let denominator = length_squared(delta);
    if denominator <= 1.0e-20 {
        return a;
    }
    work.segment_feature_tests += 1;
    let time = (dot(sub(point, a), delta) / denominator).clamp(0.0, 1.0);
    add(a, scale(delta, time))
}

/// Returns closest points on two closed segments, including zero-length segments.
///
/// Returns `None` for non-finite endpoints or arithmetic outside the finite f64
/// range. Parallel segments can have multiple equally close point pairs; the
/// first segment's start is preferred when it belongs to that set.
#[must_use]
pub fn closest_segment_points(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> Option<(Vec3, Vec3)> {
    if !p1
        .into_iter()
        .chain(q1)
        .chain(p2)
        .chain(q2)
        .all(f64::is_finite)
    {
        return None;
    }
    let points = closest_segment_segment(p1, q1, p2, q2, &mut PrimitiveWork3::default());
    points
        .0
        .into_iter()
        .chain(points.1)
        .all(f64::is_finite)
        .then_some(points)
}

#[inline]
pub(super) fn closest_segment_segment(
    p1: Vec3,
    q1: Vec3,
    p2: Vec3,
    q2: Vec3,
    work: &mut PrimitiveWork3,
) -> (Vec3, Vec3) {
    work.segment_distance_evaluations += 1;
    let unscaled_d1 = sub(q1, p1);
    let unscaled_d2 = sub(q2, p2);
    let unscaled_r = sub(p1, p2);
    let components = unscaled_d1.into_iter().chain(unscaled_d2).chain(unscaled_r);
    let mut span = 0.0_f64;
    for component in components {
        if !component.is_finite() {
            return ([f64::NAN; 3], [f64::NAN; 3]);
        }
        span = span.max(component.abs());
    }
    if span == 0.0 {
        return (p1, p2);
    }
    // Divide components individually: the reciprocal can overflow for subnormal spans.
    let d1 = unscaled_d1.map(|value| value / span);
    let d2 = unscaled_d2.map(|value| value / span);
    let r = unscaled_r.map(|value| value / span);
    let a = length_squared(d1);
    let e = length_squared(d2);

    if a == 0.0 && e == 0.0 {
        return (p1, p2);
    }
    if a == 0.0 {
        work.segment_feature_tests += 1;
        let t = (dot(d2, r) / e).clamp(0.0, 1.0);
        return (p1, add(p2, scale(unscaled_d2, t)));
    }
    if e == 0.0 {
        work.segment_feature_tests += 1;
        let s = (-dot(d1, r) / a).clamp(0.0, 1.0);
        return (add(p1, scale(unscaled_d1, s)), p2);
    }

    let b = dot(d1, d2);
    let c = dot(d1, r);
    let f = dot(d2, r);
    // The cross product avoids subtracting nearly equal squared dot products.
    let normal = cross(d1, d2);
    let denominator = length_squared(normal);
    let mut s = if denominator > 0.0 {
        work.segment_feature_tests += 1;
        (dot(cross(d2, r), normal) / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = (b * s + f) / e;
    work.segment_feature_tests += 1;
    if t < 0.0 {
        work.segment_feature_tests += 1;
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else {
        work.segment_feature_tests += 1;
        if t > 1.0 {
            work.segment_feature_tests += 1;
            t = 1.0;
            s = ((b - c) / a).clamp(0.0, 1.0);
        }
    }
    (
        add(p1, scale(unscaled_d1, s)),
        add(p2, scale(unscaled_d2, t)),
    )
}

#[inline]
pub(super) fn segment_aabb_interval(
    a: Vec3,
    b: Vec3,
    half: Vec3,
    work: &mut PrimitiveWork3,
) -> Option<(f64, f64)> {
    let delta = sub(b, a);
    let mut enter = 0.0_f64;
    let mut exit = 1.0_f64;
    for axis in 0..3 {
        work.segment_feature_tests += 1;
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

#[inline]
pub(super) fn closest_segment_aabb(
    a: Vec3,
    b: Vec3,
    half: Vec3,
    work: &mut PrimitiveWork3,
) -> (Vec3, Vec3) {
    work.segment_distance_evaluations += 1;
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
            work.segment_feature_tests += 1;
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
    work.segment_feature_tests += 1;
    let mut evaluate = |time: f64, work: &mut PrimitiveWork3| {
        work.segment_feature_tests += 1;
        let point = add(a, scale(delta, time.clamp(0.0, 1.0)));
        let clamped = clamp_aabb(point, half);
        update_pair(point, clamped, &mut best, &mut best_distance);
    };
    for &time in &times[..len] {
        evaluate(time, work);
    }
    for window in times[..len].windows(2) {
        let low = window[0];
        let high = window[1];
        let middle = (low + high) * 0.5;
        let point = add(a, scale(delta, middle));
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for axis in 0..3 {
            work.segment_feature_tests += 1;
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
            evaluate((-numerator / denominator).clamp(low, high), work);
        }
    }
    best
}

#[inline]
pub(super) fn clamp_aabb(point: Vec3, half: Vec3) -> Vec3 {
    [
        point[0].clamp(-half[0], half[0]),
        point[1].clamp(-half[1], half[1]),
        point[2].clamp(-half[2], half[2]),
    ]
}

#[inline]
pub(super) fn nearest_box_face(point: Vec3, half: Vec3) -> (Vec3, f64) {
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
