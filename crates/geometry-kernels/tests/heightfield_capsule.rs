//! Capsule contact, capsule sweep and segment casts against heightfields, checked against
//! exhaustive references over every triangle.
use geometry_kernels::heightfield::{HeightfieldData3, HeightfieldDiagonal3, PreparedHeightfield3};

fn hills(diagonal: HeightfieldDiagonal3) -> PreparedHeightfield3 {
    let n = 17;
    let columns: Vec<f64> = (0..n).map(|i| i as f64 * 2.0 - 16.0).collect();
    let rows = columns.clone();
    let heights = (0..n * n)
        .map(|k| {
            let (x, z) = (columns[k % n], rows[k / n]);
            1.5 * (x * 0.3).sin() * (z * 0.25).cos() + 0.05 * x
        })
        .collect();
    PreparedHeightfield3::try_new(HeightfieldData3 {
        columns,
        rows,
        heights,
        active_cells: None,
        diagonal,
    })
    .unwrap()
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Exhaustive surface distance from a point (all triangles).
fn brute_distance(field: &PreparedHeightfield3, point: [f64; 3]) -> f64 {
    let mut best = f64::INFINITY;
    field.visit_triangles([-1e9, -1e9], [1e9, 1e9], |_, [a, b, c], _| {
        // Sample the triangle densely enough for a 1e-3 reference.
        for i in 0..=40 {
            for j in 0..=(40 - i) {
                let (u, v) = (i as f64 / 40.0, j as f64 / 40.0);
                let p: [f64; 3] =
                    std::array::from_fn(|k| a[k] + u * (b[k] - a[k]) + v * (c[k] - a[k]));
                let d = ((p[0] - point[0]).powi(2)
                    + (p[1] - point[1]).powi(2)
                    + (p[2] - point[2]).powi(2))
                .sqrt();
                best = best.min(d);
            }
        }
    });
    best
}

#[test]
fn fixed_diagonal_matches_parry_split() {
    let field = hills(HeightfieldDiagonal3::Fixed);
    for id in [0u32, 1, 2, 3, 40, 41] {
        let [a, b, c] = field.triangle_vertices(id).unwrap();
        // Both triangles of every cell share the +X/-Z to -X/+Z diagonal (B–C).
        let cell = id / 2;
        let (cx, cz) = (
            (cell % 16) as f64 * 2.0 - 16.0,
            (cell / 16) as f64 * 2.0 - 16.0,
        );
        let has = |p: [f64; 3], x: f64, z: f64| p[0] == x && p[2] == z;
        let corners = [a, b, c];
        assert!(corners.iter().any(|&p| has(p, cx + 2.0, cz)));
        assert!(corners.iter().any(|&p| has(p, cx, cz + 2.0)));
    }
}

#[test]
fn sphere_contact_matches_exhaustive_distance() {
    for diagonal in [
        HeightfieldDiagonal3::Fixed,
        HeightfieldDiagonal3::Checkerboard,
    ] {
        let field = hills(diagonal);
        let mut s = 7;
        for _ in 0..60 {
            let x = lcg(&mut s) * 12.0;
            let z = lcg(&mut s) * 12.0;
            let ground = field.sample(x, z).unwrap().surface.unwrap().point[1];
            let p = [x, ground + 0.3 + lcg(&mut s).abs() * 0.4, z];
            let contact = field
                .capsule_contact(p, p, 0.3, 1.0)
                .expect("near the surface");
            let reference = brute_distance(&field, p) - 0.3;
            assert!(
                (contact.separation - reference).abs() < 2e-3,
                "{contact:?} vs {reference}"
            );
            let n = contact.normal;
            assert!((n[0] * n[0] + n[1] * n[1] + n[2] * n[2] - 1.0).abs() < 1e-9 && n[1] > 0.0);
        }
    }
}

#[test]
fn capsule_sweep_stops_at_first_touch() {
    let field = hills(HeightfieldDiagonal3::Fixed);
    let mut s = 11;
    for _ in 0..60 {
        let x = lcg(&mut s) * 10.0;
        let z = lcg(&mut s) * 10.0;
        let ground = field.sample(x, z).unwrap().surface.unwrap().point[1];
        let p = [x, ground + 2.0, z];
        let q = [x, ground + 3.0, z];
        let d = [lcg(&mut s) * 2.0, -3.0, lcg(&mut s) * 2.0];
        let hit = field
            .capsule_sweep(p, q, 0.3, d, 256)
            .unwrap()
            .expect("falls onto the hill");
        let at = |t: f64| {
            field
                .capsule_contact(
                    [p[0] + d[0] * t, p[1] + d[1] * t, p[2] + d[2] * t],
                    [q[0] + d[0] * t, q[1] + d[1] * t, q[2] + d[2] * t],
                    0.3,
                    10.0,
                )
                .unwrap()
                .separation
        };
        assert!(at(hit.fraction).abs() < 1e-6, "touching at the hit");
        for k in 0..20 {
            let t = hit.fraction * k as f64 / 20.0;
            assert!(at(t) > -1e-9, "no earlier contact at {t}");
        }
    }
    // Moving away or missing the field reports nothing.
    assert!(
        field
            .capsule_sweep([0.0, 10.0, 0.0], [0.0, 11.0, 0.0], 0.3, [0.0, 1.0, 0.0], 64)
            .unwrap()
            .is_none()
    );
}

#[test]
fn segment_cast_matches_exhaustive_triangles() {
    let field = hills(HeightfieldDiagonal3::Fixed);
    let mut s = 3;
    for _ in 0..200 {
        let start = [lcg(&mut s) * 15.0, 4.0 + lcg(&mut s), lcg(&mut s) * 15.0];
        let end = [lcg(&mut s) * 15.0, -4.0 + lcg(&mut s), lcg(&mut s) * 15.0];
        let hit = field.cast_segment(start, end);
        let mut best: Option<f64> = None;
        field.visit_triangles([-1e9, -1e9], [1e9, 1e9], |_, [a, b, c], _| {
            // Möller–Trumbore reference.
            let e1: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
            let e2: [f64; 3] = std::array::from_fn(|k| c[k] - a[k]);
            let d: [f64; 3] = std::array::from_fn(|k| end[k] - start[k]);
            let cr = |u: [f64; 3], v: [f64; 3]| {
                [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ]
            };
            let dt = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
            let h = cr(d, e2);
            let det = dt(e1, h);
            if det.abs() < 1e-12 {
                return;
            }
            let o: [f64; 3] = std::array::from_fn(|k| start[k] - a[k]);
            let u = dt(o, h) / det;
            let qv = cr(o, e1);
            let v = dt(d, qv) / det;
            let t = dt(e2, qv) / det;
            if (0.0..=1.0).contains(&u) && v >= 0.0 && u + v <= 1.0 && (0.0..=1.0).contains(&t) {
                best = Some(best.map_or(t, |b: f64| b.min(t)));
            }
        });
        match (hit, best) {
            (Some(h), Some(b)) => assert!((h.fraction - b).abs() < 1e-9, "{} vs {b}", h.fraction),
            (None, None) => {}
            other => panic!("cast disagrees with the reference: {other:?}"),
        }
    }
}
