#![allow(clippy::unwrap_used, reason = "test fixtures and assertions")]
use geometry_kernels::{
    math3::{add, cross, dot, length, scale, sub},
    triangle_mesh::{
        PreparedTriangleMesh3, TriangleMeshBoxManifold3, TriangleMeshData3, TriangleMeshError3,
        TriangleMeshWork3,
    },
};
const AXES: [[f64; 3]; 3] = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
fn mesh(t: [[f64; 3]; 3]) -> PreparedTriangleMesh3 {
    PreparedTriangleMesh3::try_new(TriangleMeshData3 {
        vertices: t.to_vec(),
        triangles: vec![[0, 1, 2]],
    })
    .unwrap()
}
fn query(
    m: &PreparedTriangleMesh3,
    c: [f64; 3],
    axes: [[f64; 3]; 3],
    h: [f64; 3],
    margin: f64,
) -> Vec<TriangleMeshBoxManifold3> {
    let mut out = vec![];
    m.box_contacts_counted(c, axes, h, margin, &mut TriangleMeshWork3::default(), |v| {
        out.push(v)
    })
    .unwrap();
    out
}
fn floor() -> [[f64; 3]; 3] {
    [[-10., 0., -10.], [0., 0., 10.], [10., 0., -10.]]
}
fn witnesses(
    m: &TriangleMeshBoxManifold3,
    c: [f64; 3],
    axes: [[f64; 3]; 3],
    h: [f64; 3],
    t: [[f64; 3]; 3],
) {
    assert!((1..=4).contains(&m.len));
    assert!((length(m.normal) - 1.).abs() < 1e-10);
    for p in &m.points[..m.len] {
        let d = sub(p.point_on_box, p.point_on_triangle);
        assert!((dot(d, m.normal) - p.separation).abs() < 1e-9, "{m:?}");
        assert!(
            length(sub(d, scale(m.normal, p.separation))) < 1e-8,
            "{m:?}"
        );
        let local = axes.map(|a| dot(sub(p.point_on_box, c), a));
        assert!((0..3).all(|i| local[i].abs() <= h[i] + 1e-9), "{m:?}");
        assert!((0..3).any(|i| (local[i].abs() - h[i]).abs() < 1e-8));
        let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
        assert!(dot(sub(p.point_on_triangle, t[0]), n).abs() < 1e-7);
        for i in 0..3 {
            assert!(
                dot(
                    cross(sub(t[(i + 1) % 3], t[i]), sub(p.point_on_triangle, t[i])),
                    n
                ) >= -1e-7,
                "{m:?}"
            );
        }
    }
}
#[test]
fn face_patch_has_four_real_witnesses_on_both_sides_and_windings() {
    for sign in [-1., 1.] {
        for reversed in [false, true] {
            let mut t = floor();
            if reversed {
                t.swap(0, 2);
            }
            let c = [0., sign * 0.9, 0.];
            let out = query(&mesh(t), c, AXES, [1.; 3], 0.);
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].len, 4);
            assert_eq!(out[0].normal, [0., sign, 0.]);
            for p in &out[0].points {
                assert!((p.separation + 0.1).abs() < 1e-12);
            }
            witnesses(&out[0], c, AXES, [1.; 3], t);
        }
    }
}
#[test]
fn speculative_face_patch_reaches_margin_and_corner_uses_euclidean_distance() {
    let t = floor();
    let m = mesh(t);
    let c = [0., 1.02, 0.];
    assert!(query(&m, c, AXES, [1.; 3], 0.01).is_empty());
    let out = query(&m, c, AXES, [1.; 3], 0.03);
    assert_eq!(out[0].len, 4);
    witnesses(&out[0], c, AXES, [1.; 3], t);
    // Every coordinate gap fits .1, but the true corner distance exceeds it.
    let t = [[1.08, 1.08, 1.08], [2., 1.08, 1.08], [1.08, 1.08, 2.]];
    let m = mesh(t);
    assert!(query(&m, [0.; 3], AXES, [1.; 3], 0.1).is_empty());
    let out = query(&m, [0.; 3], AXES, [1.; 3], 0.15);
    assert_eq!(out[0].len, 1);
    assert!((out[0].points[0].separation - 0.08 * 3f64.sqrt()).abs() < 1e-12);
    witnesses(&out[0], [0.; 3], AXES, [1.; 3], t);
}
#[test]
fn rotated_boxes_keep_face_edge_and_vertex_witnesses_valid() {
    let t = floor();
    let m = mesh(t);
    for i in 0..80 {
        let angle = i as f64 * 0.039;
        let (s, c) = angle.sin_cos();
        let axes = [[c, s, 0.], [-s, c, 0.], [0., 0., 1.]];
        let center = [0., 0.8, 0.];
        let out = query(&m, center, axes, [1., 0.5, 0.7], 0.4);
        assert_eq!(out.len(), 1);
        witnesses(&out[0], center, axes, [1., 0.5, 0.7], t);
    }
    // Small triangles cut a box face, exercise box-reference clipping.
    for i in 0..30 {
        let z = i as f64 * 0.01;
        let t = [[0.8, -2., z], [0.8, 2., z], [0.8, 0., z + 0.4]];
        let out = query(&mesh(t), [0.; 3], AXES, [1.; 3], 0.);
        assert_eq!(out.len(), 1);
        witnesses(&out[0], [0.; 3], AXES, [1.; 3], t);
    }
}
#[test]
fn bvh_matches_unpruned_triangle_queries_and_local_work_is_bounded() {
    for count in [64usize, 128, 256, 512] {
        let mut data = TriangleMeshData3 {
            vertices: vec![],
            triangles: vec![],
        };
        for i in 0..count {
            let at = data.vertices.len() as u32;
            data.vertices
                .extend(floor().map(|p| add(p, [i as f64 * 30., 0., 0.])));
            data.triangles.push([at, at + 1, at + 2]);
        }
        let all = PreparedTriangleMesh3::try_new(data.clone()).unwrap();
        let center = [30., 0.9, 0.];
        let mut work = TriangleMeshWork3::default();
        let mut actual = vec![];
        all.box_contacts_counted(center, AXES, [1.; 3], 0.05, &mut work, |v| actual.push(v))
            .unwrap();
        let mut expected = vec![];
        for (id, indices) in data.triangles.iter().enumerate() {
            for mut m in query(
                &mesh(indices.map(|i| data.vertices[i as usize])),
                center,
                AXES,
                [1.; 3],
                0.05,
            ) {
                m.triangle = id as u32;
                expected.push(m);
            }
        }
        actual.sort_by_key(|m| m.triangle);
        assert_eq!(actual, expected);
        assert_eq!(work.triangles_tested, 1);
        assert!(work.nodes_visited <= 2 * (count.ilog2() as u64) + 1);
    }
}
#[test]
fn translation_and_winding_preserve_contacts_and_replays() {
    let t = [[0.8, -2., 0.], [0.8, 2., 0.], [0.8, 0., 0.4]];
    let c = [0.; 3];
    let base = query(&mesh(t), c, AXES, [1.; 3], 0.);
    for shift in [[0.; 3], [-1000., 300., -900.]] {
        let moved = t.map(|p| add(p, shift));
        let actual = query(&mesh(moved), shift, AXES, [1.; 3], 0.);
        assert_eq!(actual, query(&mesh(moved), shift, AXES, [1.; 3], 0.));
        assert_eq!(actual[0].len, base[0].len);
        assert_eq!(actual[0].normal, base[0].normal);
        witnesses(&actual[0], shift, AXES, [1.; 3], moved);
    }
}
#[test]
fn invalid_queries_and_overflow_are_errors_before_callback() {
    let m = mesh(floor());
    let mut called = false;
    for (c, axes, h, margin) in [
        ([f64::NAN, 0., 0.], AXES, [1.; 3], 0.),
        ([0.; 3], [[0.; 3]; 3], [1.; 3], 0.),
        ([0.; 3], AXES, [0.; 3], 0.),
        ([0.; 3], AXES, [1.; 3], -1.),
    ] {
        assert_eq!(
            m.box_contacts_counted(
                c,
                axes,
                h,
                margin,
                &mut TriangleMeshWork3::default(),
                |_| called = true
            ),
            Err(TriangleMeshError3::InvalidQuery)
        );
    }
    assert_eq!(
        m.box_contacts_counted(
            [f64::MAX; 3],
            AXES,
            [f64::MAX; 3],
            0.,
            &mut TriangleMeshWork3::default(),
            |_| called = true
        ),
        Err(TriangleMeshError3::NonFiniteComputation)
    );
    assert!(!called);
}

#[test]
fn skew_triangle_feature_witnesses_remain_on_both_shapes() {
    let mut state = 31u64;
    let mut random = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 32) as u32 as f64 / u32::MAX as f64) * 4. - 2.
    };
    for _ in 0..2000 {
        let t = std::array::from_fn(|_| std::array::from_fn(|_| random()));
        let m = mesh(t);
        for manifold in query(&m, [0.; 3], AXES, [0.7, 0.9, 0.6], 0.1) {
            witnesses(&manifold, [0.; 3], AXES, [0.7, 0.9, 0.6], t);
        }
    }
}

#[test]
fn face_patch_scales_with_small_and_large_geometry() {
    for size in [1e-20, 1e-8, 1., 1e8, 1e20] {
        let t = floor().map(|p| scale(p, size));
        let c = [0., 0.9 * size, 0.];
        let h = [size; 3];
        let out = query(&mesh(t), c, AXES, h, 0.);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len, 4, "size={size}");
        for p in &out[0].points[..out[0].len] {
            assert!((p.separation / size + 0.1).abs() < 1e-12);
            assert!(p.point_on_triangle[1].abs() / size < 1e-12);
        }
    }
}

#[test]
fn shared_diagonal_preserves_unique_witnesses_and_all_box_support_corners() {
    let vertices = [[-2., 0., -2.], [2., 0., -2.], [2., 0., 2.], [-2., 0., 2.]];
    for reversed in [false, true] {
        let mut triangles = [[0u32, 1, 2], [0, 2, 3]];
        if reversed {
            for triangle in &mut triangles {
                triangle.swap(1, 2);
            }
        }
        let m = PreparedTriangleMesh3::try_new(TriangleMeshData3 {
            vertices: vertices.to_vec(),
            triangles: triangles.to_vec(),
        })
        .unwrap();
        for side in [-1., 1.] {
            let center = [0., side * 0.49, 0.];
            let out = query(&m, center, AXES, [0.5; 3], 0.02);
            assert_eq!(out.len(), 2);
            for manifold in &out {
                witnesses(
                    manifold,
                    center,
                    AXES,
                    [0.5; 3],
                    triangles[manifold.triangle as usize].map(|i| vertices[i as usize]),
                );
                for (i, point) in manifold.points[..manifold.len].iter().enumerate() {
                    assert!(manifold.points[..i].iter().all(|prior| {
                        length(sub(prior.point_on_triangle, point.point_on_triangle)) > 1e-12
                    }));
                }
            }
            // Duplicate clipping intersections must not consume the bounded witness slots
            // and displace a genuine box corner at the shared triangle edge.
            for corner in [
                [-0.5, 0., -0.5],
                [-0.5, 0., 0.5],
                [0.5, 0., -0.5],
                [0.5, 0., 0.5],
            ] {
                assert!(out.iter().any(|manifold| {
                    manifold.points[..manifold.len]
                        .iter()
                        .any(|point| length(sub(point.point_on_triangle, corner)) < 1e-12)
                }));
            }
        }
    }
}
