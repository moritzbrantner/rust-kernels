//! Invariant, differential and fixture evidence for 2D polygon triangulation
//! and Hertel–Mehlhorn convex decomposition.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration tests are not production code; a panic is the intended failure report"
)]

use geometry_kernels::polygon2::{
    ConvexDecomposition2, Polygon2Error, Polygon2Summary, Polygon2Work, SimplicityCheck, Vec2,
    Winding, convex_decompose_polygon2, triangulate_polygon2, validate_simple_polygon2,
};
use std::collections::BTreeSet;
use std::f64::consts::TAU;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn orient(a: Vec2, b: Vec2, p: Vec2) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

fn signed_area(points: &[Vec2]) -> f64 {
    let n = points.len();
    let o = points[0];
    (0..n)
        .map(|i| {
            let (p, q) = (points[i], points[(i + 1) % n]);
            (p[0] - o[0]) * (q[1] - o[1]) - (q[0] - o[0]) * (p[1] - o[1])
        })
        .sum::<f64>()
        * 0.5
}

fn ring_area(points: &[Vec2], ring: &[u32]) -> f64 {
    let ring: Vec<Vec2> = ring.iter().map(|&i| points[i as usize]).collect();
    signed_area(&ring)
}

/// Crossing-number point-in-polygon on the raw input boundary.
fn inside(points: &[Vec2], p: Vec2) -> bool {
    let n = points.len();
    let mut inside = false;
    for i in 0..n {
        let (a, b) = (points[i], points[(i + 1) % n]);
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if p[0] < x {
                inside = !inside;
            }
        }
    }
    inside
}

fn segment_distance(a: Vec2, b: Vec2, p: Vec2) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let len2 = d[0] * d[0] + d[1] * d[1];
    let t = if len2 > 0.0 {
        (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p[0] - a[0] - t * d[0]).hypot(p[1] - a[1] - t * d[1])
}

fn bbox(points: &[Vec2]) -> ([f64; 2], [f64; 2]) {
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in points {
        for axis in 0..2 {
            lo[axis] = lo[axis].min(p[axis]);
            hi[axis] = hi[axis].max(p[axis]);
        }
    }
    (lo, hi)
}

/// Samples points in the bounding box away from every ring edge and checks
/// that interior points lie in exactly one ring and exterior points in none.
fn assert_coverage(points: &[Vec2], rings: &[Vec<u32>], samples: usize, seed: u64) {
    let (lo, hi) = bbox(points);
    let scale = (hi[0] - lo[0]).max(hi[1] - lo[1]);
    let guard = 1e-9 * scale;
    let mut rng = Rng(seed | 1);
    let mut tested = 0;
    for _ in 0..samples {
        let p = [
            lo[0] + rng.unit() * (hi[0] - lo[0]),
            lo[1] + rng.unit() * (hi[1] - lo[1]),
        ];
        let near_edge = rings.iter().any(|ring| {
            (0..ring.len()).any(|i| {
                let a = points[ring[i] as usize];
                let b = points[ring[(i + 1) % ring.len()] as usize];
                segment_distance(a, b, p) <= guard
            })
        }) || (0..points.len())
            .any(|i| segment_distance(points[i], points[(i + 1) % points.len()], p) <= guard);
        if near_edge {
            continue;
        }
        tested += 1;
        let containing = rings
            .iter()
            .filter(|ring| {
                (0..ring.len()).all(|i| {
                    orient(
                        points[ring[i] as usize],
                        points[ring[(i + 1) % ring.len()] as usize],
                        p,
                    ) > 0.0
                })
            })
            .count();
        assert_eq!(
            containing,
            usize::from(inside(points, p)),
            "coverage mismatch at {p:?}"
        );
    }
    assert!(tested * 2 >= samples, "too few coverage samples survived");
}

/// Brute-force diagonal oracle: `i -> j` is a valid internal diagonal of the
/// simple input polygon.
fn is_diagonal(points: &[Vec2], i: usize, j: usize) -> bool {
    let n = points.len();
    let (a, b) = (points[i], points[j]);
    for k in 0..n {
        if k != i && k != j && segment_distance(a, b, points[k]) == 0.0 {
            return false;
        }
        let l = (k + 1) % n;
        if [k, l].contains(&i) || [k, l].contains(&j) {
            continue;
        }
        let (c, d) = (points[k], points[l]);
        let (d1, d2) = (orient(a, b, c), orient(a, b, d));
        let (d3, d4) = (orient(c, d, a), orient(c, d, b));
        if d1 * d2 < 0.0 && d3 * d4 < 0.0 {
            return false;
        }
    }
    inside(points, [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5])
}

fn boundary_edges(points: &[Vec2]) -> BTreeSet<(u32, u32)> {
    let n = points.len() as u32;
    (0..n)
        .map(|i| {
            let j = (i + 1) % n;
            (i.min(j), i.max(j))
        })
        .collect()
}

fn check_triangulation(points: &[Vec2], tris: &[[u32; 3]], summary: &Polygon2Summary) {
    assert_eq!(tris.len() + 2, summary.boundary_vertices);
    let input_area = signed_area(points);
    assert_eq!(
        summary.winding,
        if input_area > 0.0 {
            Winding::CounterClockwise
        } else {
            Winding::Clockwise
        }
    );
    let scale = {
        let (lo, hi) = bbox(points);
        (hi[0] - lo[0]).max(hi[1] - lo[1])
    };
    let area_tol = 1e-12 * scale * scale * points.len() as f64;
    let mut total = 0.0;
    for t in tris {
        assert!(t.iter().all(|&i| (i as usize) < points.len()));
        let a = ring_area(points, t);
        assert!(a > 0.0, "triangle {t:?} is not counter-clockwise");
        total += a;
    }
    assert!(
        (total - input_area.abs()).abs() <= area_tol,
        "{total} vs {input_area}"
    );
    assert!((summary.area - input_area.abs()).abs() <= area_tol);
}

fn check_decomposition(points: &[Vec2], pieces: &ConvexDecomposition2, summary: &Polygon2Summary) {
    let input_area = signed_area(points).abs();
    let scale = {
        let (lo, hi) = bbox(points);
        (hi[0] - lo[0]).max(hi[1] - lo[1])
    };
    let area_tol = 1e-12 * scale * scale * points.len() as f64;
    assert_eq!(pieces.offsets.first(), Some(&0));
    assert_eq!(
        *pieces.offsets.last().unwrap() as usize,
        pieces.indices.len()
    );
    let mut total = 0.0;
    for piece in pieces.pieces() {
        assert!(piece.len() >= 3);
        assert!(piece.iter().all(|&i| (i as usize) < points.len()));
        let k = piece.len();
        for i in 0..k {
            let [a, b, c] = [i, (i + 1) % k, (i + 2) % k].map(|j| points[piece[j] as usize]);
            assert!(
                orient(a, b, c) >= 0.0,
                "piece {piece:?} has a reflex vertex"
            );
        }
        let a = ring_area(points, piece);
        assert!(a > 0.0);
        total += a;
    }
    assert!((total - input_area).abs() <= area_tol);
    // Hertel–Mehlhorn bound: at most 2r + 1 pieces for r reflex vertices.
    let n = points.len();
    let reflex = (0..n)
        .filter(|&i| {
            let s = orient(points[(i + n - 1) % n], points[i], points[(i + 1) % n]);
            if signed_area(points) > 0.0 {
                s < 0.0
            } else {
                s > 0.0
            }
        })
        .count();
    assert!(
        pieces.len() <= 2 * reflex + 1,
        "{} pieces, {reflex} reflex",
        pieces.len()
    );
    assert!(pieces.len() <= summary.boundary_vertices - 2);
    // Maximality: every remaining shared edge is essential.
    let rings: Vec<&[u32]> = pieces.pieces().collect();
    for (pi, p) in rings.iter().enumerate() {
        for i in 0..p.len() {
            let (u, v) = (p[i], p[(i + 1) % p.len()]);
            for q in &rings[pi + 1..] {
                let Some(j) = (0..q.len()).find(|&j| q[j] == v && q[(j + 1) % q.len()] == u) else {
                    continue;
                };
                let pos = |x: u32| points[x as usize];
                let p_prev = p[(i + p.len() - 1) % p.len()];
                let p_next = p[(i + 2) % p.len()];
                let q_prev = q[(j + q.len() - 1) % q.len()];
                let q_next = q[(j + 2) % q.len()];
                let at_u = orient(pos(p_prev), pos(u), pos(q_next));
                let at_v = orient(pos(q_prev), pos(v), pos(p_next));
                assert!(
                    at_u < 0.0 || at_v < 0.0,
                    "inessential diagonal {u}-{v} kept"
                );
            }
        }
    }
}

fn triangulate(points: &[Vec2]) -> Result<(Vec<[u32; 3]>, Polygon2Summary), Polygon2Error> {
    let mut tris = Vec::new();
    let mut work = Polygon2Work::default();
    let summary = triangulate_polygon2(points, SimplicityCheck::Validate, &mut work, &mut tris)?;
    assert_eq!(work.stalls(), 0, "valid input needed a stall refresh");
    Ok((tris, summary))
}

fn decompose(points: &[Vec2]) -> Result<(ConvexDecomposition2, Polygon2Summary), Polygon2Error> {
    let mut out = ConvexDecomposition2::default();
    let summary = convex_decompose_polygon2(
        points,
        SimplicityCheck::Validate,
        &mut Polygon2Work::default(),
        &mut out,
    )?;
    Ok((out, summary))
}

/// Full evidence for an accepted polygon whose vertices are all kept.
fn check_all(points: &[Vec2], seed: u64) -> (Vec<[u32; 3]>, ConvexDecomposition2) {
    let (tris, summary) = triangulate(points).unwrap();
    check_triangulation(points, &tris, &summary);
    let (pieces, dsummary) = decompose(points).unwrap();
    assert_eq!(summary, dsummary);
    check_decomposition(points, &pieces, &dsummary);
    let tri_rings: Vec<Vec<u32>> = tris.iter().map(|t| t.to_vec()).collect();
    assert_coverage(points, &tri_rings, 400, seed);
    let piece_rings: Vec<Vec<u32>> = pieces.pieces().map(<[u32]>::to_vec).collect();
    assert_coverage(points, &piece_rings, 400, seed ^ 0x55);
    (tris, pieces)
}

fn star(rng: &mut Rng, n: usize, spikiness: f64) -> Vec<Vec2> {
    let mut angles: Vec<f64> = (0..n)
        .map(|i| (i as f64 + 0.1 + 0.8 * rng.unit()) / n as f64 * TAU)
        .collect();
    angles.sort_by(f64::total_cmp);
    angles
        .into_iter()
        .map(|t| {
            let r = 1.0 - spikiness * rng.unit();
            [r * t.cos(), r * t.sin()]
        })
        .collect()
}

/// Random simple polygon via 2-opt untangling of random points.
fn two_opt(rng: &mut Rng, n: usize) -> Vec<Vec2> {
    let mut p: Vec<Vec2> = (0..n).map(|_| [rng.unit(), rng.unit()]).collect();
    'outer: loop {
        for i in 0..n {
            for j in i + 2..n {
                if i == 0 && j == n - 1 {
                    continue;
                }
                let (a, b, c, d) = (p[i], p[i + 1], p[j], p[(j + 1) % n]);
                if orient(a, b, c) * orient(a, b, d) < 0.0
                    && orient(c, d, a) * orient(c, d, b) < 0.0
                {
                    p[i + 1..=j].reverse();
                    continue 'outer;
                }
            }
        }
        return p;
    }
}

fn regular(n: usize, radius: f64) -> Vec<Vec2> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64 * TAU;
            [radius * t.cos(), radius * t.sin()]
        })
        .collect()
}

fn comb(teeth: usize) -> Vec<Vec2> {
    let mut p = vec![[0.0, 0.0], [teeth as f64 * 2.0 - 1.0, 0.0]];
    for k in (0..teeth).rev() {
        let x = k as f64 * 2.0;
        p.push([x + 1.0, 3.0]);
        p.push([x, 3.0]);
        if k > 0 {
            p.push([x, 1.0]);
            p.push([x - 1.0, 1.0]);
        }
    }
    p
}

/// Naive exact ear clipping with the same canonical start and selection rule.
fn reference_ear_clip(points: &[Vec2]) -> Vec<[u32; 3]> {
    let mut ring: Vec<usize> = (0..points.len()).collect();
    if signed_area(points) < 0.0 {
        ring.reverse();
    }
    let start = (0..ring.len())
        .min_by(|&a, &b| {
            let (pa, pb) = (points[ring[a]], points[ring[b]]);
            pa[0].total_cmp(&pb[0]).then(pa[1].total_cmp(&pb[1]))
        })
        .unwrap();
    ring.rotate_left(start);
    let mut out = Vec::new();
    let mut cursor = 0;
    while ring.len() > 3 {
        let m = ring.len();
        let ear = (0..m).map(|k| (cursor + k) % m).find(|&v| {
            let (a, b, c) = (ring[(v + m - 1) % m], ring[v], ring[(v + 1) % m]);
            let (pa, pb, pc) = (points[a], points[b], points[c]);
            orient(pa, pb, pc) > 0.0
                && ring.iter().all(|&q| {
                    q == a
                        || q == b
                        || q == c
                        || !(orient(pa, pb, points[q]) >= 0.0
                            && orient(pb, pc, points[q]) >= 0.0
                            && orient(pc, pa, points[q]) >= 0.0)
                })
        });
        let v = ear.expect("reference found no ear");
        out.push([ring[(v + m - 1) % m], ring[v], ring[(v + 1) % m]].map(|i| i as u32));
        ring.remove(v);
        cursor = v % ring.len();
    }
    out.push([0, 1, 2].map(|k| ring[(cursor + k) % 3] as u32));
    out
}

#[test]
fn seeded_random_polygons_satisfy_every_invariant() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for case in 0..300 {
        let n = 3 + (rng.next() % 40) as usize;
        let mut points = match case % 3 {
            0 => star(&mut rng, n, 0.9),
            1 => star(&mut rng, n, 0.3),
            _ => two_opt(&mut rng, n.min(24)),
        };
        if case % 2 == 1 {
            points.reverse();
        }
        check_all(&points, case);
    }
}

#[test]
fn small_polygons_match_brute_force_diagonal_oracle() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for case in 0..1500 {
        let n = 3 + (rng.next() % 8) as usize;
        let points = if case % 2 == 0 {
            star(&mut rng, n, 0.85)
        } else {
            two_opt(&mut rng, n)
        };
        let (tris, _) = triangulate(&points).unwrap();
        let boundary = boundary_edges(&points);
        let mut diagonals = BTreeSet::new();
        for t in &tris {
            for k in 0..3 {
                let (u, v) = (t[k], t[(k + 1) % 3]);
                let key = (u.min(v), u.max(v));
                if !boundary.contains(&key) {
                    assert!(
                        is_diagonal(&points, u as usize, v as usize),
                        "case {case}: {u}-{v} is not a diagonal"
                    );
                    diagonals.insert(key);
                }
            }
        }
        // A triangulation of an n-gon has exactly n - 3 distinct diagonals.
        assert_eq!(diagonals.len(), n - 3, "case {case}");
        // Differential: an independent naive exact ear clipper agrees exactly.
        assert_eq!(tris, reference_ear_clip(&points), "case {case}");
    }
}

#[test]
fn output_is_invariant_under_rotation_and_reversal() {
    let mut rng = Rng(0xdead_beef_cafe_f00d);
    for case in 0..200 {
        let n = 4 + (rng.next() % 30) as usize;
        let points = if case % 2 == 0 {
            star(&mut rng, n, 0.8)
        } else {
            two_opt(&mut rng, n.min(20))
        };
        let coords = |p: &[Vec2], tris: &[[u32; 3]]| -> Vec<[Vec2; 3]> {
            tris.iter().map(|t| t.map(|i| p[i as usize])).collect()
        };
        let piece_coords = |p: &[Vec2], d: &ConvexDecomposition2| -> Vec<Vec<Vec2>> {
            d.pieces()
                .map(|r| r.iter().map(|&i| p[i as usize]).collect())
                .collect()
        };
        let (base, _) = triangulate(&points).unwrap();
        let (base_pieces, _) = decompose(&points).unwrap();
        let shift = (rng.next() % points.len() as u64) as usize;
        let mut variants = vec![points.clone(), points.clone()];
        variants[0].rotate_left(shift);
        variants[1].reverse();
        variants[1].rotate_left(shift);
        for variant in &variants {
            let (tris, _) = triangulate(variant).unwrap();
            assert_eq!(coords(variant, &tris), coords(&points, &base));
            let (pieces, _) = decompose(variant).unwrap();
            assert_eq!(
                piece_coords(variant, &pieces),
                piece_coords(&points, &base_pieces)
            );
        }
    }
}

#[test]
fn reused_work_and_assume_simple_match_fresh_validated_runs() {
    let mut rng = Rng(0x0bad_5eed_0bad_5eed);
    let mut work = Polygon2Work::default();
    let mut tris = Vec::new();
    let mut pieces = ConvexDecomposition2::default();
    for case in 0..100 {
        let points = star(&mut rng, 5 + case, 0.7);
        let (fresh, fresh_summary) = triangulate(&points).unwrap();
        let (fresh_pieces, _) = decompose(&points).unwrap();
        let summary =
            triangulate_polygon2(&points, SimplicityCheck::AssumeSimple, &mut work, &mut tris)
                .unwrap();
        assert_eq!(summary, fresh_summary);
        assert_eq!(tris, fresh);
        convex_decompose_polygon2(
            &points,
            SimplicityCheck::AssumeSimple,
            &mut work,
            &mut pieces,
        )
        .unwrap();
        assert_eq!(pieces, fresh_pieces);
        assert_eq!(
            validate_simple_polygon2(&points, &mut work),
            Ok(fresh_summary)
        );
    }
}

#[test]
fn concave_fixtures() {
    let l_shape = [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ];
    let (_, pieces) = check_all(&l_shape, 1);
    assert_eq!(pieces.len(), 2);
    let u_shape = [
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 3.0],
        [2.0, 3.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [1.0, 3.0],
        [0.0, 3.0],
    ];
    let (_, pieces) = check_all(&u_shape, 2);
    assert!(pieces.len() <= 5);
    let arrow = [[0.0, 0.0], [2.0, 1.0], [0.0, 2.0], [0.5, 1.0]];
    let (_, pieces) = check_all(&arrow, 3);
    assert_eq!(pieces.len(), 2);
    for teeth in [2, 5, 17] {
        let points = comb(teeth);
        let (tris, pieces) = check_all(&points, teeth as u64);
        assert_eq!(tris.len(), points.len() - 2);
        assert!(pieces.len() >= teeth);
    }
    // Clockwise input is accepted and normalized.
    let mut cw = u_shape.to_vec();
    cw.reverse();
    let (tris, summary) = triangulate(&cw).unwrap();
    assert_eq!(summary.winding, Winding::Clockwise);
    check_triangulation(&cw, &tris, &summary);
}

#[test]
fn collinear_vertices_are_removed_from_the_working_boundary() {
    // Square with midpoints on every edge, starting mid-edge so a straight
    // run spans the wrap-around.
    let square = [
        [1.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [2.0, 2.0],
        [1.0, 2.0],
        [0.0, 2.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ];
    let (tris, summary) = triangulate(&square).unwrap();
    assert_eq!(summary.boundary_vertices, 4);
    assert_eq!(summary.area, 4.0);
    let used: BTreeSet<u32> = tris.iter().flatten().copied().collect();
    assert_eq!(used, BTreeSet::from([1, 3, 5, 7]));
    let (pieces, _) = decompose(&square).unwrap();
    assert_eq!(pieces.len(), 1);
    assert_eq!(pieces.indices, vec![5, 7, 1, 3]);
    // Triangle with many points on one edge and an L with a straight vertex
    // next to a reflex one.
    let mut fan = vec![[0.0, 0.0]];
    fan.extend((1..10).map(|i| [f64::from(i), 0.0]));
    fan.extend([[10.0, 0.0], [5.0, 5.0]]);
    let (tris, summary) = triangulate(&fan).unwrap();
    assert_eq!((tris.len(), summary.area), (1, 25.0));
    let l_flat = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.5, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [0.0, 2.0],
        [0.0, 1.0],
    ];
    let (tris, summary) = triangulate(&l_flat).unwrap();
    assert_eq!(summary.boundary_vertices, 6);
    check_triangulation(&l_flat, &tris, &summary);
}

#[test]
fn repeated_points_collapse_to_the_first_of_each_run() {
    let points = [
        [0.0, 0.0],
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 0.0],
        [3.0, 0.0],
        [3.0, 3.0],
        [0.0, 3.0],
        [0.0, 0.0],
    ];
    let (tris, summary) = triangulate(&points).unwrap();
    assert_eq!(summary.boundary_vertices, 4);
    let used: BTreeSet<u32> = tris.iter().flatten().copied().collect();
    assert_eq!(used, BTreeSet::from([0, 2, 5, 6]));
    assert_eq!(summary.area, 9.0);
    // Near-duplicates within the scale tolerance also collapse.
    let near = [[0.0, 0.0], [1.0, 0.0], [1.0 + 1e-17, 1e-17], [0.0, 1.0]];
    assert_eq!(triangulate(&near).unwrap().1.boundary_vertices, 3);
}

#[test]
fn nearly_degenerate_polygons_still_tile() {
    // Needle triangle.
    let needle = [[0.0, 0.0], [1.0, 0.0], [0.5, 1e-12]];
    let (tris, summary) = triangulate(&needle).unwrap();
    check_triangulation(&needle, &tris, &summary);
    assert_eq!(decompose(&needle).unwrap().0.len(), 1);
    // Almost-straight vertex just beyond tolerance, both convex and reflex.
    for offset in [1e-13, -1e-13] {
        let points = [
            [0.0, 0.0],
            [1.0, offset],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
        ];
        let (tris, summary) = triangulate(&points).unwrap();
        assert_eq!(summary.boundary_vertices, 5);
        check_triangulation(&points, &tris, &summary);
        let (pieces, summary) = decompose(&points).unwrap();
        check_decomposition(&points, &pieces, &summary);
    }
    // Deep, thin notch.
    let notch = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1e-9], [0.0, 1.0]];
    check_all(&notch, 8);
    // Large translation and tiny scale.
    let mut rng = Rng(77);
    let base = star(&mut rng, 25, 0.8);
    let far: Vec<Vec2> = base.iter().map(|p| [p[0] + 1e6, p[1] - 3e6]).collect();
    check_all(&far, 9);
    let tiny: Vec<Vec2> = base.iter().map(|p| [p[0] * 1e-9, p[1] * 1e-9]).collect();
    check_all(&tiny, 10);
}

#[test]
fn convex_polygons_become_one_piece() {
    for n in [3, 4, 7, 64, 257] {
        let points = regular(n, 2.5);
        let (tris, pieces) = check_all(&points, n as u64);
        assert_eq!(tris.len(), n - 2);
        assert_eq!(pieces.len(), 1);
        let ring: BTreeSet<u32> = pieces.indices.iter().copied().collect();
        assert_eq!(ring.len(), n);
    }
}

#[test]
fn invalid_inputs_return_typed_errors() {
    let mut work = Polygon2Work::default();
    let mut tris = vec![[9, 9, 9]];
    let mut pieces = ConvexDecomposition2 {
        indices: vec![1],
        offsets: vec![0, 1],
    };
    let cases: Vec<(Vec<Vec2>, Polygon2Error)> = vec![
        (vec![], Polygon2Error::TooFewVertices { remaining: 0 }),
        (
            vec![[0.0, 0.0], [1.0, 0.0]],
            Polygon2Error::TooFewVertices { remaining: 2 },
        ),
        (
            vec![[1.0, 1.0]; 5],
            Polygon2Error::TooFewVertices { remaining: 1 },
        ),
        (
            vec![[0.0, 0.0], [f64::NAN, 0.0], [0.0, 1.0]],
            Polygon2Error::NonFiniteVertex { index: 1 },
        ),
        (
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, f64::INFINITY]],
            Polygon2Error::NonFiniteVertex { index: 2 },
        ),
        (
            vec![[-1e200, 0.0], [1e200, 0.0], [0.0, 1e200]],
            Polygon2Error::UnrepresentableExtent,
        ),
        (
            vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            Polygon2Error::ZeroArea,
        ),
        (
            vec![[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            Polygon2Error::ZeroArea,
        ),
        (
            // Bowtie with unequal lobes.
            vec![[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 3.0]],
            Polygon2Error::SelfIntersection {
                first: 0,
                second: 2,
            },
        ),
        (
            // Two squares touching at a repeated, non-consecutive vertex.
            vec![
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [2.0, 2.0],
                [1.0, 2.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            Polygon2Error::SelfIntersection {
                first: 1,
                second: 5,
            },
        ),
        (
            // Vertex touching a non-adjacent edge.
            vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [2.0, 0.0], [0.0, 4.0]],
            Polygon2Error::SelfIntersection {
                first: 0,
                second: 2,
            },
        ),
        (
            // Zero-width spike.
            vec![[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [2.0, 1.0], [0.0, 2.0]],
            Polygon2Error::SelfIntersection {
                first: 1,
                second: 2,
            },
        ),
    ];
    for (points, expected) in cases {
        assert_eq!(
            triangulate_polygon2(&points, SimplicityCheck::Validate, &mut work, &mut tris),
            Err(expected),
            "{points:?}"
        );
        assert!(tris.is_empty());
        assert_eq!(
            convex_decompose_polygon2(&points, SimplicityCheck::Validate, &mut work, &mut pieces),
            Err(expected)
        );
        assert!(pieces.is_empty() && pieces.indices.is_empty());
        assert_eq!(validate_simple_polygon2(&points, &mut work), Err(expected));
        tris.push([9, 9, 9]);
    }
}

#[test]
fn assume_simple_on_crossing_input_never_panics() {
    let mut rng = Rng(0xfeed);
    let mut work = Polygon2Work::default();
    let mut tris = Vec::new();
    let mut pieces = ConvexDecomposition2::default();
    for _ in 0..500 {
        let n = 4 + (rng.next() % 12) as usize;
        let points: Vec<Vec2> = (0..n).map(|_| [rng.unit(), rng.unit()]).collect();
        if let Ok(summary) =
            triangulate_polygon2(&points, SimplicityCheck::AssumeSimple, &mut work, &mut tris)
        {
            assert_eq!(tris.len() + 2, summary.boundary_vertices);
        } else {
            assert!(tris.is_empty());
        }
        let _ = convex_decompose_polygon2(
            &points,
            SimplicityCheck::AssumeSimple,
            &mut work,
            &mut pieces,
        );
    }
}

#[test]
fn ear_tests_are_bounded_quadratically() {
    for n in [64usize, 256, 1024] {
        let mut rng = Rng(n as u64 * 31 + 1);
        let points = star(&mut rng, n, 0.9);
        let mut work = Polygon2Work::default();
        let mut tris = Vec::new();
        triangulate_polygon2(&points, SimplicityCheck::Validate, &mut work, &mut tris).unwrap();
        // n initial classifications plus two re-tests per clip.
        assert!(
            work.ear_tests() <= 3 * n as u64,
            "{n}: {}",
            work.ear_tests()
        );
    }
}
