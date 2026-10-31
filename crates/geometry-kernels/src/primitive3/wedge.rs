//! Wedge faces, edges and closest-feature queries in the wedge's local frame.

use super::closest::{closest_point_triangle, closest_segment_segment, update_pair};
use super::frame::unit_or_zero;
use super::query::PrimitiveWork3;
use crate::math3::{Vec3, dot, length_squared, scale, sub};

pub(super) fn wedge_vertices(half: Vec3) -> [Vec3; 6] {
    [
        [-half[0], -half[1], -half[2]],
        [-half[0], half[1], -half[2]],
        [half[0], -half[1], -half[2]],
        [-half[0], -half[1], half[2]],
        [-half[0], half[1], half[2]],
        [half[0], -half[1], half[2]],
    ]
}

pub(super) fn segment_face_penetration(
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

pub(super) fn wedge_planes(half: Vec3) -> [(Vec3, f64); 5] {
    [
        ([0.0, -1.0, 0.0], half[1]),
        ([-1.0, 0.0, 0.0], half[0]),
        (unit_or_zero([half[1], half[0], 0.0]), 0.0),
        ([0.0, 0.0, 1.0], half[2]),
        ([0.0, 0.0, -1.0], half[2]),
    ]
}

pub(super) fn wedge_triangles(half: Vec3) -> [[Vec3; 3]; 8] {
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

pub(super) fn wedge_inside_depth(point: Vec3, half: Vec3) -> Option<(Vec3, f64)> {
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

pub(super) fn segment_wedge_interval(a: Vec3, b: Vec3, half: Vec3) -> Option<(f64, f64)> {
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

pub(super) fn closest_point_wedge(point: Vec3, half: Vec3) -> Vec3 {
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

pub(super) fn closest_segment_wedge(
    a: Vec3,
    b: Vec3,
    half: Vec3,
    work: &mut PrimitiveWork3,
) -> (Vec3, Vec3) {
    // The segment does not meet the wedge, so the closest pair is an endpoint against a face
    // interior or the segment against an edge. An edge's Voronoi region lies in front of one
    // of its two faces, so edges behind both faces along the whole segment are skipped.
    let vertices = wedge_vertices(half);
    let planes = wedge_planes(half);
    let heights = planes.map(|(normal, limit)| [dot(normal, a) - limit, dot(normal, b) - limit]);
    let in_front = heights.map(|[ha, hb]| ha > 0.0 || hb > 0.0);
    let mut best = (a, vertices[0]);
    let mut best_distance = f64::INFINITY;
    for (face, &(normal, _)) in planes.iter().enumerate() {
        for (endpoint, height) in [(a, heights[face][0]), (b, heights[face][1])] {
            if height <= 0.0 {
                continue;
            }
            let foot = sub(endpoint, scale(normal, height));
            let inside = planes
                .iter()
                .enumerate()
                .all(|(other, &(n, l))| other == face || dot(n, foot) <= l);
            if inside {
                update_pair(endpoint, foot, &mut best, &mut best_distance);
            }
        }
    }
    for (i, j, left, right) in WEDGE_EDGES {
        if in_front[left] || in_front[right] {
            let candidate = closest_segment_segment(a, b, vertices[i], vertices[j], work);
            update_pair(candidate.0, candidate.1, &mut best, &mut best_distance);
        }
    }
    best
}

/// The wedge's nine edges: indices into [`wedge_vertices`] and of their two faces in
/// [`wedge_planes`].
const WEDGE_EDGES: [(usize, usize, usize, usize); 9] = [
    (0, 2, 0, 4),
    (2, 5, 0, 2),
    (5, 3, 0, 3),
    (3, 0, 0, 1),
    (0, 1, 1, 4),
    (1, 4, 1, 2),
    (4, 3, 1, 3),
    (1, 2, 2, 4),
    (4, 5, 2, 3),
];
