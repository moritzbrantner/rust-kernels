//! Differential, degeneracy and determinism evidence for QuickHull 2D/3D.
use geometry_kernels::{
    Sphere,
    gjk::gjk_intersection,
    hull::{Hull3, HullDimension, QuickHullWork, Vec2, quickhull2, quickhull3},
    math3::{Vec3, cross, dot, sub},
    support::{ConvexHull3, SupportMap3},
};
use std::collections::{BTreeSet, HashMap};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn int(&mut self, range: i64) -> f64 {
        (self.next() % (2 * range as u64 + 1)) as f64 - range as f64
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, (self.next() % (i as u64 + 1)) as usize);
        }
    }
}

fn hull2(points: &[Vec2]) -> Vec<Vec2> {
    let mut out = Vec::new();
    quickhull2(points, &mut QuickHullWork::default(), &mut out);
    out
}

fn hull3(points: &[Vec3]) -> Hull3 {
    let mut out = Hull3::default();
    quickhull3(points, &mut QuickHullWork::default(), &mut out);
    out
}

/// Exact monotone chain over integer-valued points: strictly convex vertices,
/// counter-clockwise from the lexicographic minimum.
fn monotone_chain(points: &[Vec2]) -> Vec<Vec2> {
    let mut sorted: Vec<(i64, i64)> = points.iter().map(|p| (p[0] as i64, p[1] as i64)).collect();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() <= 2 {
        return sorted.iter().map(|&(x, y)| [x as f64, y as f64]).collect();
    }
    let turn = |o: (i64, i64), a: (i64, i64), b: (i64, i64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut hull: Vec<(i64, i64)> = Vec::new();
    for pass in [sorted.clone(), sorted.iter().rev().copied().collect()] {
        let base = hull.len();
        for p in pass {
            while hull.len() >= base + 2 && turn(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull.iter().map(|&(x, y)| [x as f64, y as f64]).collect()
}

#[test]
fn hull2_matches_exact_monotone_chain_on_degenerate_integer_sets() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for case in 0..2000 {
        let n = 1 + case % 40;
        let range = 1 + (case % 7) as i64;
        let points: Vec<Vec2> = (0..n).map(|_| [rng.int(range), rng.int(range)]).collect();
        assert_eq!(
            hull2(&points),
            monotone_chain(&points),
            "case {case}: {points:?}"
        );
    }
}

#[test]
fn hull2_degenerate_contracts() {
    assert!(hull2(&[]).is_empty());
    assert_eq!(hull2(&[[1.0, 2.0]; 5]), vec![[1.0, 2.0]]);
    let line: Vec<Vec2> = (0..9)
        .map(|i| [f64::from(i), 2.0 * f64::from(i)])
        .rev()
        .collect();
    assert_eq!(hull2(&line), vec![[0.0, 0.0], [8.0, 16.0]]);
    let square = [
        [1.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ];
    assert_eq!(
        hull2(&square),
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    );
}

#[test]
fn hull2_is_order_and_multiplicity_independent() {
    let mut rng = Rng(7);
    let base: Vec<Vec2> = (0..500).map(|_| [rng.unit(), rng.unit()]).collect();
    let reference = hull2(&base);
    for _ in 0..8 {
        let mut points = base.clone();
        points.extend_from_slice(&base[..100]);
        rng.shuffle(&mut points);
        assert_eq!(hull2(&points), reference);
    }
}

/// Every directed edge appears once and its reverse once, Euler holds, and
/// every input point lies on or below every face plane.
fn assert_closed_convex(hull: &Hull3, points: &[Vec3]) {
    assert_eq!(hull.dimension, HullDimension::Polytope);
    let mut edges = HashMap::new();
    for t in &hull.triangles {
        assert!(t[0] < t[1] && t[0] < t[2], "canonical rotation {t:?}");
        for i in 0..3 {
            *edges.entry((t[i], t[(i + 1) % 3])).or_insert(0) += 1;
        }
    }
    for (&(a, b), &count) in &edges {
        assert_eq!(count, 1);
        assert_eq!(edges.get(&(b, a)), Some(&1), "open edge {a}->{b}");
    }
    let (v, e, f) = (hull.vertices.len(), edges.len() / 2, hull.triangles.len());
    assert_eq!(v + f, e + 2, "Euler characteristic");
    let used: BTreeSet<u32> = hull.triangles.iter().flatten().copied().collect();
    assert_eq!(used.len(), v);
    assert!(
        hull.vertices.windows(2).all(|w| w[0] < w[1]),
        "sorted vertices"
    );
    assert!(
        hull.triangles.windows(2).all(|w| w[0] < w[1]),
        "sorted triangles"
    );
    for t in &hull.triangles {
        let [a, b, c] = t.map(|i| hull.vertices[i as usize]);
        let normal = cross(sub(b, a), sub(c, a));
        let scale = dot(normal, normal).sqrt();
        for &p in points {
            assert!(
                dot(normal, sub(p, a)) / scale <= 1e-9,
                "point {p:?} outside face"
            );
        }
    }
}

/// Brute-force hull vertices for points in general position: members of a
/// triple whose plane has every other point strictly on one side.
fn brute_vertices(points: &[Vec3]) -> BTreeSet<[i64; 3]> {
    let key = |p: Vec3| p.map(|c| c as i64);
    let ints: Vec<[i128; 3]> = points.iter().map(|p| p.map(|c| c as i128)).collect();
    let mut out = BTreeSet::new();
    let n = ints.len();
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let (a, b, c) = (ints[i], ints[j], ints[k]);
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let normal = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let side = |p: [i128; 3]| {
                    (0..3)
                        .map(|axis| normal[axis] * (p[axis] - a[axis]))
                        .sum::<i128>()
                };
                let (mut above, mut below) = (false, false);
                for (m, &p) in ints.iter().enumerate() {
                    if m != i && m != j && m != k {
                        let s = side(p);
                        above |= s > 0;
                        below |= s < 0;
                    }
                }
                if !(above && below) {
                    out.extend([points[i], points[j], points[k]].map(key));
                }
            }
        }
    }
    out
}

#[test]
fn hull3_matches_brute_force_supporting_planes() {
    let mut rng = Rng(0xdead_beef);
    for case in 0..300 {
        let n = 4 + case % 28;
        let points: Vec<Vec3> = (0..n)
            .map(|_| [rng.int(1000), rng.int(1000), rng.int(1000)])
            .collect();
        let hull = hull3(&points);
        assert_closed_convex(&hull, &points);
        let got: BTreeSet<[i64; 3]> = hull.vertices.iter().map(|p| p.map(|c| c as i64)).collect();
        assert_eq!(got, brute_vertices(&points), "case {case}");
    }
}

fn grid(n: i32) -> Vec<Vec3> {
    let mut points = Vec::new();
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                points.push([f64::from(x), f64::from(y), f64::from(z)]);
            }
        }
    }
    points
}

#[test]
fn hull3_drops_duplicate_edge_and_face_points() {
    let mut points = grid(4);
    points.extend(grid(4));
    let hull = hull3(&points);
    assert_closed_convex(&hull, &points);
    let corners: Vec<Vec3> = (0..8)
        .map(|i| [i >> 2 & 1, i >> 1 & 1, i & 1].map(|b| f64::from(b) * 3.0))
        .collect();
    assert_eq!(hull.vertices, corners);
    assert_eq!(hull.triangles.len(), 12);
}

#[test]
fn hull3_degenerate_dimensions() {
    assert_eq!(hull3(&[]).dimension, HullDimension::Empty);

    let point = hull3(&[[1.0, 2.0, 3.0]; 4]);
    assert_eq!(point.dimension, HullDimension::Point);
    assert_eq!(point.vertices, vec![[1.0, 2.0, 3.0]]);

    let line: Vec<Vec3> = (0..10)
        .map(|i| [f64::from(i), 1.0, -f64::from(i)])
        .rev()
        .collect();
    let segment = hull3(&line);
    assert_eq!(segment.dimension, HullDimension::Segment);
    assert_eq!(segment.vertices, vec![[0.0, 1.0, 0.0], [9.0, 1.0, -9.0]]);
    assert!(segment.triangles.is_empty());

    let mut plane = Vec::new();
    for x in 0..4 {
        for y in 0..4 {
            plane.push([f64::from(x), f64::from(y), f64::from(x) + f64::from(y)]);
        }
    }
    let polygon = hull3(&plane);
    assert_eq!(polygon.dimension, HullDimension::Polygon);
    assert!(polygon.triangles.is_empty());
    let mut corners = polygon.vertices.clone();
    corners.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(
        corners,
        vec![
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 3.0],
            [3.0, 0.0, 3.0],
            [3.0, 3.0, 6.0]
        ]
    );
    // Boundary order: consecutive corners share a coordinate.
    for i in 0..4 {
        let (a, b) = (polygon.vertices[i], polygon.vertices[(i + 1) % 4]);
        assert!(a[0] == b[0] || a[1] == b[1], "{:?}", polygon.vertices);
    }
}

#[test]
fn hull3_is_order_and_multiplicity_independent() {
    let mut rng = Rng(42);
    // Sphere shell plus interior points and an exactly coplanar cap.
    let mut base: Vec<Vec3> = (0..400)
        .map(|_| {
            let p = [rng.unit(), rng.unit(), rng.unit()];
            let len = dot(p, p).sqrt().max(1e-3);
            p.map(|c| c / len)
        })
        .collect();
    base.extend((0..200).map(|_| [rng.unit() * 0.5, rng.unit() * 0.5, rng.unit() * 0.5]));
    base.extend(
        (0..5).flat_map(|x| (0..5).map(move |y| [f64::from(x) * 0.1, f64::from(y) * 0.1, 1.5])),
    );
    let reference = hull3(&base);
    assert_closed_convex(&reference, &base);
    let mut work = QuickHullWork::default();
    let mut out = Hull3::default();
    for _ in 0..8 {
        let mut points = base.clone();
        points.extend_from_slice(&base[..150]);
        rng.shuffle(&mut points);
        quickhull3(&points, &mut work, &mut out);
        assert_eq!(out, reference);
    }
}

#[test]
fn hull3_stays_closed_and_canonical_on_heavily_degenerate_integer_sets() {
    let mut rng = Rng(0x5eed);
    for case in 0..1500 {
        let n = 4 + case % 60;
        let range = 1 + (case % 4) as i64;
        let mut points: Vec<Vec3> = (0..n)
            .map(|_| [rng.int(range), rng.int(range), rng.int(range)])
            .collect();
        let reference = hull3(&points);
        if reference.dimension == HullDimension::Polytope {
            assert_closed_convex(&reference, &points);
            // No hull vertex may lie inside or on the hull of the others:
            // every vertex is the unique maximiser of the sum of its face normals.
            for (index, &v) in reference.vertices.iter().enumerate() {
                let mut direction = [0.0; 3];
                for t in reference
                    .triangles
                    .iter()
                    .filter(|t| t.contains(&(index as u32)))
                {
                    let [a, b, c] = t.map(|i| reference.vertices[i as usize]);
                    let normal = cross(sub(b, a), sub(c, a));
                    let len = dot(normal, normal).sqrt();
                    direction = std::array::from_fn(|k| direction[k] + normal[k] / len);
                }
                for &p in &points {
                    assert!(
                        p == v || dot(p, direction) < dot(v, direction),
                        "case {case}"
                    );
                }
            }
        }
        rng.shuffle(&mut points);
        assert_eq!(hull3(&points), reference, "case {case}");
    }
}

#[test]
fn hull3_feeds_gjk_support_map_directly() {
    let mut rng = Rng(3);
    let cloud: Vec<Vec3> = (0..300)
        .map(|_| [rng.unit(), rng.unit(), rng.unit()])
        .collect();
    let hull = hull3(&cloud);
    let support = hull.support_map();
    let raw = ConvexHull3::new(&cloud);
    for _ in 0..200 {
        let d = [rng.unit(), rng.unit(), rng.unit()];
        assert_eq!(
            dot(support.support_point(d), d),
            dot(raw.support_point(d), d)
        );
    }
    let near = Sphere {
        center: [1.2, 0.0, 0.0],
        radius: 0.5,
    };
    let far = Sphere {
        center: [3.0, 0.0, 0.0],
        radius: 0.5,
    };
    assert_eq!(gjk_intersection(&support, &near).intersection(), Some(true));
    assert_eq!(gjk_intersection(&support, &far).intersection(), Some(false));
}

#[test]
fn hull3_distance_tests_stay_near_linear_for_interior_heavy_input() {
    let mut rng = Rng(11);
    let mut work = QuickHullWork::default();
    let mut out = Hull3::default();
    for n in [1_000usize, 10_000, 100_000] {
        let points: Vec<Vec3> = (0..n)
            .map(|_| [rng.unit(), rng.unit(), rng.unit()])
            .collect();
        quickhull3(&points, &mut work, &mut out);
        let per_point = work.distance_tests() as f64 / n as f64;
        assert!(
            per_point < 12.0,
            "n={n}: {per_point:.2} distance tests per point"
        );
    }
}
