//! Closest points between a finite segment and a convex 3D hull when the
//! segment's centreline is strictly outside the solid.
//!
//! This is a useful narrow phase for capsules against *static* convex terrain:
//! the segment-to-hull distance can be reduced by the capsule radius. If the
//! centreline touches or crosses the solid, the caller must use a penetration
//! algorithm instead.

use crate::{
    hull::{Hull3, HullDimension},
    math3::{Vec3, cross, dot, length_squared, sub},
    primitive3::{closest_point_triangle, closest_segment_points},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentHullClosest {
    pub distance: f64,
    pub point_segment: Vec3,
    pub point_hull: Vec3,
}

/// Deterministic per-call work counters. These accumulate when reused.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentHullWork {
    pub separating_plane_tests: u64,
    pub triangles_considered: u64,
    pub triangles_pruned: u64,
}

/// Returns the exact closest witnesses for a segment outside a closed convex
/// hull. `None` means that the segment intersects the solid or that the input
/// is not a finite, 3D polytope; it is not a proof of separation.
#[must_use]
pub fn separated_segment_hull_closest(
    p: Vec3,
    q: Vec3,
    hull: &Hull3,
) -> Option<SegmentHullClosest> {
    separated_segment_hull_closest_with_work(p, q, hull, &mut SegmentHullWork::default())
}

/// As [`separated_segment_hull_closest`], also accumulating deterministic work.
#[must_use]
pub fn separated_segment_hull_closest_with_work(
    p: Vec3,
    q: Vec3,
    hull: &Hull3,
    work: &mut SegmentHullWork,
) -> Option<SegmentHullClosest> {
    if hull.dimension != HullDimension::Polytope || !p.into_iter().chain(q).all(f64::is_finite) {
        return None;
    }

    // Parametrically clip the segment against all closed hull halfspaces.
    // If any part remains, the centreline crosses or touches the solid.
    let (mut enter, mut exit) = (0.0_f64, 1.0_f64);
    for face in &hull.triangles {
        work.separating_plane_tests += 1;
        let [a, b, c] = face.map(|index| hull.vertices[index as usize]);
        let normal = cross(sub(b, a), sub(c, a));
        let (hp, hq) = (dot(normal, sub(p, a)), dot(normal, sub(q, a)));
        if hp > 0.0 && hq > 0.0 {
            enter = 1.0;
            exit = -1.0;
            break;
        }
        if hp > 0.0 {
            enter = enter.max(hp / (hp - hq));
        } else if hq > 0.0 {
            exit = exit.min(hp / (hp - hq));
        }
        if enter > exit {
            break;
        }
    }
    if enter <= exit {
        return None;
    }

    let segment_min: Vec3 = std::array::from_fn(|i| p[i].min(q[i]));
    let segment_max: Vec3 = std::array::from_fn(|i| p[i].max(q[i]));
    let mut best = (f64::INFINITY, p, p);
    for face in &hull.triangles {
        work.triangles_considered += 1;
        let [a, b, c] = face.map(|index| hull.vertices[index as usize]);
        let min_bound: Vec3 = std::array::from_fn(|i| a[i].min(b[i]).min(c[i]));
        let max_bound: Vec3 = std::array::from_fn(|i| a[i].max(b[i]).max(c[i]));
        let lower_bound_squared: f64 = (0..3)
            .map(|i| {
                let delta = (min_bound[i] - segment_max[i])
                    .max(segment_min[i] - max_bound[i])
                    .max(0.0);
                delta * delta
            })
            .sum();
        if lower_bound_squared >= best.0 {
            work.triangles_pruned += 1;
            continue;
        }
        for endpoint in [p, q] {
            let on_face = closest_point_triangle(endpoint, a, b, c);
            consider(endpoint, on_face, &mut best);
        }
        for (start, end) in [(a, b), (b, c), (c, a)] {
            let (on_segment, on_edge) = closest_segment_points(p, q, start, end)?;
            consider(on_segment, on_edge, &mut best);
        }
    }
    best.0.is_finite().then(|| SegmentHullClosest {
        distance: best.0.sqrt(),
        point_segment: best.1,
        point_hull: best.2,
    })
}

fn consider(a: Vec3, b: Vec3, best: &mut (f64, Vec3, Vec3)) {
    let d2 = length_squared(sub(a, b));
    if d2 < best.0 {
        *best = (d2, a, b);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SegmentHullWork, separated_segment_hull_closest, separated_segment_hull_closest_with_work,
    };
    use crate::hull::{Hull3, QuickHullWork, quickhull3};

    fn cube() -> Hull3 {
        let points: Vec<[f64; 3]> = (-1..=1)
            .step_by(2)
            .flat_map(|x| {
                (-1..=1).step_by(2).flat_map(move |y| {
                    (-1..=1)
                        .step_by(2)
                        .map(move |z| [x as f64, y as f64, z as f64])
                })
            })
            .collect();
        let mut hull = Hull3::default();
        quickhull3(&points, &mut QuickHullWork::default(), &mut hull);
        hull
    }

    #[test]
    fn outside_segment_has_exact_face_edge_and_corner_witnesses() {
        let hull = cube();
        let cases = [
            ([2.0, 0.0, 0.0], 1.0),
            ([2.0, 2.0, 0.0], 2.0_f64.sqrt()),
            ([2.0, 2.0, 2.0], 3.0_f64.sqrt()),
        ];
        for (point, expected) in cases {
            let hit = separated_segment_hull_closest(point, point, &hull).unwrap();
            assert!((hit.distance - expected).abs() < 1e-12);
            assert_eq!(hit.point_segment, point);
        }
    }

    #[test]
    fn centreline_inside_or_crossing_the_solid_defers_to_penetration_solver() {
        let hull = cube();
        for (p, q) in [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([0.0, -2.0, 0.0], [0.0, 2.0, 0.0]),
            ([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
        ] {
            assert!(separated_segment_hull_closest(p, q, &hull).is_none());
        }
    }

    #[test]
    fn diagonal_segment_can_miss_a_corner_without_one_face_separating_it() {
        let hull = cube();
        let p = [2.2, 0.0, 0.0];
        let q = [0.0, 2.2, 0.0];
        let hit = separated_segment_hull_closest(p, q, &hull).unwrap();
        assert!((hit.distance - 0.2 / 2.0_f64.sqrt()).abs() < 1e-12);
        assert_eq!(hit.point_hull, [1.0, 1.0, 0.0]);
        assert!(
            separated_segment_hull_closest([2.0, 0.0, 0.0], [0.0, 2.0, 0.0], &hull).is_none(),
            "corner touching must fall back to a penetration-capable solver"
        );
    }

    #[test]
    fn translation_and_work_counters_are_deterministic() {
        let hull = cube();
        let p = [2.0, 2.0, 0.0];
        let q = [2.0, 2.0, 0.8];
        let mut work = SegmentHullWork::default();
        let original = separated_segment_hull_closest_with_work(p, q, &hull, &mut work).unwrap();
        assert!(work.separating_plane_tests > 0);
        assert_eq!(work.triangles_considered as usize, hull.triangles.len());
        assert!(work.triangles_pruned > 0);

        let offset = [17.0, 31.0, -9.0];
        let mut shifted_hull = hull.clone();
        for vertex in &mut shifted_hull.vertices {
            *vertex = std::array::from_fn(|i| vertex[i] + offset[i]);
        }
        let shifted = separated_segment_hull_closest(
            std::array::from_fn(|i| p[i] + offset[i]),
            std::array::from_fn(|i| q[i] + offset[i]),
            &shifted_hull,
        )
        .unwrap();
        assert!((shifted.distance - original.distance).abs() < 1e-12);
        for i in 0..3 {
            assert!(
                (shifted.point_segment[i] - original.point_segment[i] - offset[i]).abs() < 1e-12
            );
            assert!((shifted.point_hull[i] - original.point_hull[i] - offset[i]).abs() < 1e-12);
        }
    }
}
