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
    pub edges_considered: u64,
    pub edges_pruned: u64,
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    min: Vec3,
    max: Vec3,
}

impl Bounds {
    fn from_points<const N: usize>(points: [Vec3; N]) -> Self {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for p in points {
            for (i, coordinate) in p.into_iter().enumerate() {
                min[i] = min[i].min(coordinate);
                max[i] = max[i].max(coordinate);
            }
        }
        Self { min, max }
    }

    fn lower_bound_squared(self, segment: Self) -> f64 {
        (0..3)
            .map(|i| {
                let delta = (self.min[i] - segment.max[i])
                    .max(segment.min[i] - self.max[i])
                    .max(0.0);
                delta * delta
            })
            .sum()
    }
}

#[derive(Clone, Copy, Debug)]
struct Face {
    normal: Vec3,
    offset: f64,
    bounds: Bounds,
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    ends: [u32; 2],
    bounds: Bounds,
}

/// Immutable geometry prepared once per hull. Face halfspaces and bounds are
/// retained; shared edges are visited exactly once, in canonical index order.
#[derive(Clone, Debug)]
pub struct PreparedSegmentHull {
    hull: Hull3,
    faces: Vec<Face>,
    edges: Vec<Edge>,
}

impl PreparedSegmentHull {
    /// Consumes a canonical, closed 3D hull. Degenerate or non-finite data is
    /// rejected; callers can retain their original hull on failure by cloning.
    #[must_use]
    pub fn new(hull: Hull3) -> Option<Self> {
        if hull.dimension != HullDimension::Polytope
            || hull.triangles.is_empty()
            || hull
                .vertices
                .iter()
                .flat_map(|v| v.iter())
                .any(|v| !v.is_finite())
        {
            return None;
        }

        let mut faces = Vec::with_capacity(hull.triangles.len());
        let mut edge_keys = Vec::with_capacity(3 * hull.triangles.len());
        for &triangle in &hull.triangles {
            let [a, b, c] = triangle.map(|index| hull.vertices.get(index as usize).copied());
            let [Some(a), Some(b), Some(c)] = [a, b, c] else {
                return None;
            };
            let normal = cross(sub(b, a), sub(c, a));
            let squared = length_squared(normal);
            if !squared.is_finite() || squared == 0.0 {
                return None;
            }
            faces.push(Face {
                normal,
                offset: dot(normal, a),
                bounds: Bounds::from_points([a, b, c]),
            });
            for [a, b] in [
                [triangle[0], triangle[1]],
                [triangle[1], triangle[2]],
                [triangle[2], triangle[0]],
            ] {
                edge_keys.push([a.min(b), a.max(b)]);
            }
        }
        edge_keys.sort_unstable();
        edge_keys.dedup();
        let edges = edge_keys
            .into_iter()
            .map(|ends| {
                let [a, b] = ends.map(|i| hull.vertices[i as usize]);
                Edge {
                    ends,
                    bounds: Bounds::from_points([a, b]),
                }
            })
            .collect();
        Some(Self { hull, faces, edges })
    }

    /// The canonical hull used for support mapping, persistence and reporting.
    #[must_use]
    pub fn hull(&self) -> &Hull3 {
        &self.hull
    }

    /// Returns exact witnesses if the segment's centreline is outside the
    /// closed convex solid. Returns `None` for touching/intersecting centrelines
    /// or invalid endpoints, requiring the caller's penetration-capable path.
    #[must_use]
    pub fn closest(&self, p: Vec3, q: Vec3) -> Option<SegmentHullClosest> {
        self.closest_with_work(p, q, &mut SegmentHullWork::default())
    }

    /// As [`Self::closest`], accumulating deterministic query-work counters.
    #[must_use]
    pub fn closest_with_work(
        &self,
        p: Vec3,
        q: Vec3,
        work: &mut SegmentHullWork,
    ) -> Option<SegmentHullClosest> {
        if !p.into_iter().chain(q).all(f64::is_finite) {
            return None;
        }

        // Clip against prepared halfspaces. If any interval remains, the
        // centreline touches or crosses the closed solid.
        let (mut enter, mut exit) = (0.0_f64, 1.0_f64);
        for face in &self.faces {
            work.separating_plane_tests += 1;
            let (hp, hq) = (
                dot(face.normal, p) - face.offset,
                dot(face.normal, q) - face.offset,
            );
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

        let segment_bounds = Bounds::from_points([p, q]);
        let mut best = (f64::INFINITY, p, p);
        for (&triangle, face) in self.hull.triangles.iter().zip(&self.faces) {
            work.triangles_considered += 1;
            if face.bounds.lower_bound_squared(segment_bounds) >= best.0 {
                work.triangles_pruned += 1;
                continue;
            }
            let [a, b, c] = triangle.map(|index| self.hull.vertices[index as usize]);
            for endpoint in [p, q] {
                let on_face = closest_point_triangle(endpoint, a, b, c);
                consider(endpoint, on_face, &mut best);
            }
        }

        // An ordinary closed triangle mesh stores every hull edge twice.
        // The prepared edge set removes those duplicate robust segment pairs.
        for edge in &self.edges {
            work.edges_considered += 1;
            if edge.bounds.lower_bound_squared(segment_bounds) >= best.0 {
                work.edges_pruned += 1;
                continue;
            }
            let [a, b] = edge.ends.map(|i| self.hull.vertices[i as usize]);
            let (on_segment, on_edge) = closest_segment_points(p, q, a, b)?;
            consider(on_segment, on_edge, &mut best);
        }

        best.0.is_finite().then(|| SegmentHullClosest {
            distance: best.0.sqrt(),
            point_segment: best.1,
            point_hull: best.2,
        })
    }
}

fn consider(a: Vec3, b: Vec3, best: &mut (f64, Vec3, Vec3)) {
    let d2 = length_squared(sub(a, b));
    if d2 < best.0 {
        *best = (d2, a, b);
    }
}

#[cfg(test)]
mod tests {
    use super::{PreparedSegmentHull, SegmentHullWork};
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
        let hull = PreparedSegmentHull::new(cube()).unwrap();
        let cases = [
            ([2.0, 0.0, 0.0], 1.0),
            ([2.0, 2.0, 0.0], 2.0_f64.sqrt()),
            ([2.0, 2.0, 2.0], 3.0_f64.sqrt()),
        ];
        for (point, expected) in cases {
            let hit = hull.closest(point, point).unwrap();
            assert!((hit.distance - expected).abs() < 1e-12);
            assert_eq!(hit.point_segment, point);
        }
    }

    #[test]
    fn centreline_inside_or_crossing_the_solid_defers_to_penetration_solver() {
        let hull = PreparedSegmentHull::new(cube()).unwrap();
        for (p, q) in [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([0.0, -2.0, 0.0], [0.0, 2.0, 0.0]),
            ([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
        ] {
            assert!(hull.closest(p, q).is_none());
        }
    }

    #[test]
    fn diagonal_segment_can_miss_a_corner_without_one_face_separating_it() {
        let hull = PreparedSegmentHull::new(cube()).unwrap();
        let p = [2.2, 0.0, 0.0];
        let q = [0.0, 2.2, 0.0];
        let hit = hull.closest(p, q).unwrap();
        assert!((hit.distance - 0.2 / 2.0_f64.sqrt()).abs() < 1e-12);
        assert_eq!(hit.point_hull, [1.0, 1.0, 0.0]);
        assert!(
            hull.closest([2.0, 0.0, 0.0], [0.0, 2.0, 0.0]).is_none(),
            "corner touching must fall back to a penetration-capable solver"
        );
    }

    #[test]
    fn translation_and_work_counters_are_deterministic() {
        let hull = PreparedSegmentHull::new(cube()).unwrap();
        let p = [2.0, 2.0, 0.0];
        let q = [2.0, 2.0, 0.8];
        let mut work = SegmentHullWork::default();
        let original = hull.closest_with_work(p, q, &mut work).unwrap();
        assert!(work.separating_plane_tests > 0);
        assert_eq!(
            work.triangles_considered as usize,
            hull.hull().triangles.len()
        );
        assert!(work.triangles_pruned > 0);
        assert_eq!(work.edges_considered as usize, hull.edges.len());
        assert!(work.edges_pruned > 0);

        let offset = [17.0, 31.0, -9.0];
        let mut shifted_hull = hull.hull().clone();
        for vertex in &mut shifted_hull.vertices {
            *vertex = std::array::from_fn(|i| vertex[i] + offset[i]);
        }
        let shifted = PreparedSegmentHull::new(shifted_hull)
            .unwrap()
            .closest(
                std::array::from_fn(|i| p[i] + offset[i]),
                std::array::from_fn(|i| q[i] + offset[i]),
            )
            .unwrap();
        assert!((shifted.distance - original.distance).abs() < 1e-12);
        for (i, coordinate) in offset.iter().enumerate() {
            assert!(
                (shifted.point_segment[i] - original.point_segment[i] - coordinate).abs() < 1e-12
            );
            assert!((shifted.point_hull[i] - original.point_hull[i] - coordinate).abs() < 1e-12);
        }
    }
}
