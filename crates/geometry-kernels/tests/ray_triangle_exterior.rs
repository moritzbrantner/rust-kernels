//! Acceptance for issue #141: ray/segment-vs-triangle must never report a hit
//! for a query that lies outside the triangle by more than rounding error,
//! while exact edge, vertex and distance-zero hits stay inclusive.
//!
//! The oracle here is exact: triangles lie in the plane `z = 0` with integer
//! corners that f32 represents exactly (|coordinate| < 2^24), and rays are
//! vertical through integer points. Containment is then a 2D orientation test
//! evaluated exactly in `i128`, and the distance from an exterior point to the
//! closed triangle is derived from exact integer numerators.
use geometry_kernels::{Ray3, TriangleCulling, TriangleFacing, ray_triangle, segment_triangle};

/// Largest coordinate magnitude used, so every value is an exact f32.
const LIMIT: i64 = (1 << 24) - 1;

/// An exterior hit is tolerated only when the query point is this close to the
/// closed triangle, relative to the coordinate scale. f64 rounding of the
/// widened inputs is about 1e-16 of that scale; this is far looser than that
/// and far tighter than any genuinely exterior lattice query.
const EXTERIOR_TOLERANCE: f64 = 1.0e-9;

type P = [i64; 2];

fn orient(p: P, q: P, r: P) -> i128 {
    let [px, py] = p.map(i128::from);
    let [qx, qy] = q.map(i128::from);
    let [rx, ry] = r.map(i128::from);
    (qx - px) * (ry - py) - (qy - py) * (rx - px)
}

/// Exact closed containment: `Some(true)` inside or on the boundary,
/// `Some(false)` strictly outside, `None` for a zero-area triangle.
fn contains(triangle: [P; 3], point: P) -> Option<bool> {
    let [a, b, c] = triangle;
    let area = orient(a, b, c);
    if area == 0 {
        return None;
    }
    let sign = area.signum();
    Some(
        [
            orient(a, b, point),
            orient(b, c, point),
            orient(c, a, point),
        ]
        .iter()
        .all(|value| value * sign >= 0),
    )
}

/// Euclidean distance from `point` to the closed segment `p..q`, from exact
/// integer numerators (only the final division and square root round).
fn segment_distance(p: P, q: P, point: P) -> f64 {
    let d = [i128::from(q[0] - p[0]), i128::from(q[1] - p[1])];
    let w = [i128::from(point[0] - p[0]), i128::from(point[1] - p[1])];
    let length_squared = d[0] * d[0] + d[1] * d[1];
    let projection = w[0] * d[0] + w[1] * d[1];
    let squared = |v: [i128; 2]| (v[0] * v[0] + v[1] * v[1]) as f64;
    if length_squared == 0 || projection <= 0 {
        return squared(w).sqrt();
    }
    if projection >= length_squared {
        let r = [i128::from(point[0] - q[0]), i128::from(point[1] - q[1])];
        return squared(r).sqrt();
    }
    let cross = (d[0] * w[1] - d[1] * w[0]).abs() as f64;
    cross / (length_squared as f64).sqrt()
}

/// Distance from a point strictly outside the triangle to the closed triangle.
fn exterior_distance([a, b, c]: [P; 3], point: P) -> f64 {
    segment_distance(a, b, point)
        .min(segment_distance(b, c, point))
        .min(segment_distance(c, a, point))
}

fn to_f32([x, y]: P, z: i64) -> [f32; 3] {
    let exact = |value: i64| {
        assert!(
            value.abs() <= LIMIT,
            "{value} is not an exact f32 lattice value"
        );
        value as f32
    };
    [exact(x), exact(y), exact(z)]
}

fn triangle_f32(triangle: [P; 3]) -> [[f32; 3]; 3] {
    triangle.map(|corner| to_f32(corner, 0))
}

fn scale_of(triangle: [P; 3], point: P) -> f64 {
    triangle
        .iter()
        .chain(std::iter::once(&point))
        .flatten()
        .map(|value| value.unsigned_abs() as f64)
        .fold(1.0, f64::max)
}

#[derive(Default)]
struct Tally {
    inside_hits: usize,
    boundary_hits: usize,
    exterior_misses: usize,
    tolerated_exterior_hits: usize,
}

/// Checks a vertical ray from `(point, -height)` and a vertical segment through
/// the plane against the exact oracle.
fn check(triangle: [P; 3], point: P, height: i64, tally: &mut Tally) {
    let Some(inside) = contains(triangle, point) else {
        return;
    };
    let corners = triangle_f32(triangle);
    let ray = Ray3::new(to_f32(point, -height), [0.0, 0.0, 1.0]);
    let hit = ray_triangle(ray, corners, f64::INFINITY, TriangleCulling::None);
    let segment = segment_triangle(
        to_f32(point, -height),
        to_f32(point, height.max(1)),
        corners,
        TriangleCulling::None,
    );
    let scale = scale_of(triangle, point);
    let context = format!("triangle {triangle:?}, point {point:?}, height {height}");
    if inside {
        let hit = hit.unwrap_or_else(|| panic!("closed triangle contains the point: {context}"));
        let segment =
            segment.unwrap_or_else(|| panic!("segment crosses the closed triangle: {context}"));
        // +z travels against the normal exactly when the corners turn clockwise.
        let front = orient(triangle[0], triangle[1], triangle[2]) < 0;
        assert_eq!(hit.facing == TriangleFacing::Front, front, "{context}");
        assert_eq!(segment.facing, hit.facing, "{context}");
        assert!(
            (hit.distance - height as f64).abs() <= 1.0e-9 * scale,
            "{} {context}",
            hit.distance
        );
        let [x, y] = point.map(|value| value as f64);
        let offset = (hit.point[0] - x)
            .hypot(hit.point[1] - y)
            .hypot(hit.point[2]);
        assert!(offset <= 1.0e-9 * scale, "{:?} {context}", hit.point);
        let on_boundary = [
            orient(triangle[0], triangle[1], point),
            orient(triangle[1], triangle[2], point),
            orient(triangle[2], triangle[0], point),
        ]
        .contains(&0);
        if on_boundary {
            tally.boundary_hits += 1;
        } else {
            tally.inside_hits += 1;
        }
        return;
    }
    let distance = exterior_distance(triangle, point);
    let tolerated = distance <= EXTERIOR_TOLERANCE * scale;
    for (kind, admitted) in [("ray", hit.is_some()), ("segment", segment.is_some())] {
        assert!(
            !admitted || tolerated,
            "{kind} hit a point {distance} outside the triangle: {context}"
        );
    }
    if hit.is_none() {
        tally.exterior_misses += 1;
    } else {
        tally.tolerated_exterior_hits += 1;
    }
}

#[test]
fn issue_141_cancellation_example_misses() {
    // det = 1 but the u numerator is exactly -1: the origin is about 3.4e7
    // outside this sliver, yet it was admitted as a distance-zero vertex hit.
    let triangle = [
        [0.0, 0.0, 0.0],
        [16_777_215.0, 16_777_214.0, 0.0],
        [16_777_214.0, 16_777_213.0, 0.0],
    ];
    let ray = Ray3::new([-16_777_215.0, -16_777_214.0, 0.0], [0.0, 0.0, 1.0]);
    for culling in [TriangleCulling::None, TriangleCulling::BackFaces] {
        let hit = ray_triangle(ray, triangle, f64::INFINITY, culling);
        assert!(hit.is_none(), "{culling:?}: {hit:?}");
        let reversed = [triangle[0], triangle[2], triangle[1]];
        let hit = ray_triangle(ray, reversed, f64::INFINITY, culling);
        assert!(hit.is_none(), "reversed {culling:?}: {hit:?}");
    }
    let segment = segment_triangle(
        [-16_777_215.0, -16_777_214.0, -1.0],
        [-16_777_215.0, -16_777_214.0, 1.0],
        triangle,
        TriangleCulling::None,
    );
    assert!(segment.is_none(), "{segment:?}");
    // Same configuration with the ray starting below the plane.
    let below = Ray3::new([-16_777_215.0, -16_777_214.0, -5.0], [0.0, 0.0, 1.0]);
    let hit = ray_triangle(below, triangle, f64::INFINITY, TriangleCulling::None);
    assert!(hit.is_none(), "{hit:?}");
}

#[test]
fn ill_conditioned_lattice_queries_match_exact_containment() {
    let mut state = 0x5eed_0141_u64;
    let mut next = move |bound: i64| -> i64 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as i64).rem_euclid(2 * bound + 1) - bound
    };
    let mut tally = Tally::default();
    let near: [P; 9] = [
        [0, 0],
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
        [1, 1],
        [-1, -1],
        [1, -1],
        [-1, 1],
    ];
    for round in 0..600 {
        // Slivers like the issue's: a long edge and a nearly parallel second
        // edge, so the determinant is tiny relative to the coordinates.
        let half = LIMIT / 2;
        let a: P = [next(half / 4), next(half / 4)];
        let direction: P = [next(half), next(half)];
        let wiggle: P = [next(2), next(2)];
        let shrink = 1 + next(3).abs();
        let b: P = [a[0] + direction[0], a[1] + direction[1]];
        let c: P = [
            a[0] + direction[0] - shrink * direction[0].signum() + wiggle[0],
            a[1] + direction[1] - shrink * direction[1].signum() + wiggle[1],
        ];
        let sliver = [a, b, c];
        let generic: [P; 3] = std::array::from_fn(|_| [next(LIMIT / 2), next(LIMIT / 2)]);
        for triangle in [sliver, generic] {
            let [a, b, c] = triangle;
            let mirrored = |p: P| [2 * a[0] - p[0], 2 * a[1] - p[1]];
            let mut points: Vec<P> = Vec::new();
            for corner in triangle {
                points.extend(near.iter().map(|o| [corner[0] + o[0], corner[1] + o[1]]));
                points.push(mirrored(corner));
            }
            for (p, q) in [(a, b), (b, c), (c, a)] {
                // Lattice points on and next to each edge.
                let steps = 1 + next(6).abs();
                let g = gcd(q[0] - p[0], q[1] - p[1]).max(1);
                let unit = [(q[0] - p[0]) / g, (q[1] - p[1]) / g];
                let k = next(g).abs().min(g);
                let on = [p[0] + k * unit[0], p[1] + k * unit[1]];
                points.extend(
                    near.iter()
                        .map(|o| [on[0] + steps * o[0], on[1] + steps * o[1]]),
                );
                points.push(on);
            }
            points.push([next(LIMIT / 2), next(LIMIT / 2)]);
            for point in points {
                if point.iter().any(|value| value.abs() > LIMIT) {
                    continue;
                }
                let height = if round % 3 == 0 {
                    0
                } else {
                    1 + next(1000).abs()
                };
                check(triangle, point, height, &mut tally);
            }
        }
    }
    assert!(
        tally.boundary_hits > 500,
        "boundary hits {}",
        tally.boundary_hits
    );
    assert!(
        tally.inside_hits > 100,
        "interior hits {}",
        tally.inside_hits
    );
    assert!(
        tally.exterior_misses > 2000,
        "exterior misses {}",
        tally.exterior_misses
    );
    // Every tolerated exterior hit was already checked against the distance
    // bound; the counter only documents that the fixture reaches that regime.
    let _ = tally.tolerated_exterior_hits;
}

fn gcd(mut x: i64, mut y: i64) -> i64 {
    x = x.abs();
    y = y.abs();
    while y != 0 {
        (x, y) = (y, x % y);
    }
    x
}
