//! Shared contact workloads for allocation ratchets, instruction-count
//! sentinels and Divan benches: a 32-vertex authored rock hull (box/hull
//! contacts) and a closed wobbling cave mesh (box/triangle-mesh contacts).
#![allow(
    dead_code,
    clippy::expect_used,
    reason = "each consumer uses a subset of the fixtures; a broken fixture must panic"
)]

use geometry_kernels::hull::{Hull3, QuickHullWork, quickhull3};
use geometry_kernels::segment_hull::PreparedSegmentHull;
use geometry_kernels::triangle_mesh::{PreparedTriangleMesh3, TriangleMeshData3};
use std::sync::Arc;

pub type Vec3 = [f64; 3];
pub type Axes = [Vec3; 3];
pub const IDENTITY: Axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// An authored rock proxy in centimetres, 32 vertices.
const ROCK: [[i32; 3]; 32] = [
    [-160, 5, 45],
    [-157, 1, 47],
    [-160, 72, 16],
    [-125, 109, 58],
    [-149, 79, 46],
    [-150, 1, -37],
    [-77, 214, -22],
    [-78, 216, -8],
    [-79, 193, 28],
    [-43, 220, 11],
    [-151, 5, -37],
    [-134, 84, -88],
    [-84, 1, -107],
    [-28, 80, -133],
    [-97, 197, -33],
    [-68, 161, -98],
    [-20, 199, -85],
    [21, 95, 122],
    [64, 2, 150],
    [66, 5, 147],
    [16, 126, 106],
    [18, 215, -10],
    [72, 183, -6],
    [54, 1, -150],
    [158, 90, -69],
    [152, 1, -89],
    [155, 0, -110],
    [160, 1, -107],
    [7, 200, -77],
    [45, 195, -66],
    [105, 174, -58],
    [154, 122, -53],
];

pub fn rock() -> Arc<PreparedSegmentHull> {
    let points: Vec<_> = ROCK
        .iter()
        .map(|p| [f64::from(p[0]), f64::from(p[1] - 110), f64::from(p[2])].map(|v| v * 0.01))
        .collect();
    let mut hull = Hull3::default();
    quickhull3(&points, &mut QuickHullWork::default(), &mut hull);
    Arc::new(PreparedSegmentHull::new(hull).expect("the rock is a closed polytope"))
}

/// 64 deterministic tumbling orientations, each centred against the rock's
/// support surface in a spread of directions: resting contacts with depths from
/// slight penetration to a small gap, and a few clear misses.
pub fn rock_queries() -> Vec<(Vec3, Axes)> {
    let points: Vec<Vec3> = ROCK
        .iter()
        .map(|p| [f64::from(p[0]), f64::from(p[1] - 110), f64::from(p[2])].map(|v| v * 0.01))
        .collect();
    (0..64)
        .map(|i| {
            let (a, b) = (f64::from(i) * 0.37, f64::from(i) * 0.91);
            let ((s, c), (t, d)) = (a.sin_cos(), b.sin_cos());
            let axes = [[c, 0.0, -s], [s * t, d, c * t], [s * d, -t, c * d]];
            // Fibonacci-sphere directions, upper hemisphere weighted like resting debris.
            let z = 1.0 - 1.6 * (f64::from(i) + 0.5) / 64.0;
            let (u, r) = (f64::from(i) * 2.39996, (1.0 - z * z).sqrt());
            let direction = [r * u.cos(), z.abs(), r * u.sin()];
            let support = points
                .iter()
                .map(|p| p[0] * direction[0] + p[1] * direction[1] + p[2] * direction[2])
                .fold(f64::NEG_INFINITY, f64::max);
            let depth = 0.3 + 0.1 * f64::from(i % 7) / 6.0 - if i % 16 == 15 { -0.6 } else { 0.0 };
            (direction.map(|v| v * (support + depth)), axes)
        })
        .collect()
}

/// The closed cave of the `triangle_mesh` sweep tests: a wobbling tube of 41 rings of 24
/// vertices along X with a flattened floor and capped ends (1968 triangles).
pub fn cave() -> TriangleMeshData3 {
    let (rings, around, size, wobble) = (41_usize, 24_usize, 3.0, 0.1);
    let mut vertices = Vec::new();
    let half = (rings - 1) as f64 * 0.5;
    for i in 0..rings {
        let x = i as f64 - half;
        for k in 0..around {
            let angle = std::f64::consts::TAU * k as f64 / around as f64;
            let r = size
                * (1.0
                    + wobble
                        * ((0.7 * x + 3.0 * angle).sin() + 0.5 * (1.3 * x - 2.0 * angle).cos()));
            let y = (r * angle.sin()).max(-0.6 * size + 0.05 * (0.9 * x).sin());
            vertices.push([x, y, r * angle.cos()]);
        }
    }
    let ring = |i: usize, k: usize| (i * around + k % around) as u32;
    let mut triangles = Vec::new();
    for i in 0..rings - 1 {
        for k in 0..around {
            triangles.push([ring(i, k), ring(i + 1, k), ring(i + 1, k + 1)]);
            triangles.push([ring(i, k), ring(i + 1, k + 1), ring(i, k + 1)]);
        }
    }
    for (i, x) in [(0, -half - 0.5), (rings - 1, half + 0.5)] {
        let centre = vertices.len() as u32;
        vertices.push([x, 0.0, 0.0]);
        for k in 0..around {
            triangles.push([centre, ring(i, k + 1), ring(i, k)]);
        }
    }
    TriangleMeshData3 {
        vertices,
        triangles,
    }
}

/// 64 box centres (half extents 0.4 x 0.3 x 0.5) resting on the cave floor.
pub fn cave_boxes(mesh: &PreparedTriangleMesh3) -> Vec<Vec3> {
    (0..64)
        .map(|i| {
            let (x, z) = (
                f64::from(i % 16) * 2.1 - 16.0,
                f64::from(i / 16) * 0.5 - 0.8,
            );
            let floor = mesh
                .segment_hit([x, 0.0, z], [x, -4.0, z])
                .expect("finite segment")
                .expect("the floor is below every centre")
                .point[1];
            [x, floor + 0.29, z]
        })
        .collect()
}

pub const CAVE_BOX_HALF: Vec3 = [0.4, 0.3, 0.5];

/// The tilted resting pose of the cave boxes.
pub fn tilted() -> Axes {
    let (s, c) = 0.21_f64.sin_cos();
    let (s2, c2) = 0.37_f64.sin_cos();
    [[c, s, 0.], [-s * c2, c * c2, s2], [s * s2, -c * s2, c2]]
}
