use super::*;

fn slab(y: f64, reverse: bool) -> PreparedTriangleMesh3 {
    let mut triangles = vec![[0, 2, 1], [1, 2, 3]];
    if reverse {
        for t in &mut triangles {
            t.swap(0, 2);
        }
    }
    PreparedTriangleMesh3::try_new(TriangleMeshData3 {
        vertices: vec![[-2., y, -2.], [2., y, -2.], [-2., y, 2.], [2., y, 2.]],
        triangles,
    })
    .unwrap()
}
#[test]
fn two_sided_floor_ceiling_and_winding() {
    for reverse in [false, true] {
        let mesh = slab(0., reverse);
        for sign in [-1., 1.] {
            let p = [0., sign * 0.3, 0.];
            let c = mesh.capsule_contact(p, p, 0.4, 0.).unwrap().unwrap();
            assert!((c.separation + 0.1).abs() < 1e-12);
            assert_eq!(c.normal, [0., sign, 0.]);
            assert_eq!(c.triangle, 0);
            let hit = mesh
                .capsule_sweep(
                    [0., sign * 2., 0.],
                    [0., sign * 2., 0.],
                    0.25,
                    [0., -sign * 4., 0.],
                    64,
                )
                .unwrap()
                .unwrap();
            assert!((hit.fraction - 0.4375).abs() < 1e-12);
            assert_eq!(hit.contact.normal, [0., sign, 0.]);
            let ray = mesh
                .segment_hit([0., sign * 2., 0.], [0., -sign * 2., 0.])
                .unwrap()
                .unwrap();
            assert_eq!(ray.fraction, 0.5);
            assert_eq!(ray.normal, [0., sign, 0.]);
        }
    }
}
#[test]
fn holes_edges_vertices_and_leaving_initial_overlap() {
    let mesh = slab(0., false);
    assert!(
        mesh.capsule_sweep([3., 2., 0.], [3., 2., 0.], 0.25, [0., -4., 0.], 64)
            .unwrap()
            .is_none()
    );
    let c = mesh
        .capsule_contact([2.1, 0., 2.1], [2.1, 0., 2.1], 0.2, 0.)
        .unwrap()
        .unwrap();
    assert!((c.separation - (0.02_f64.sqrt() - 0.2)).abs() < 1e-12);
    assert!(
        mesh.capsule_sweep([0., 0.1, 0.], [0., 0.1, 0.], 0.2, [0., 1., 0.], 64)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        mesh.capsule_sweep([0., 0.1, 0.], [0., 0.1, 0.], 0.2, [0., -1., 0.], 64)
            .unwrap()
            .unwrap()
            .fraction,
        0.
    );
    assert!(
        mesh.segment_hit([0., 0., 0.], [1., 0., 0.])
            .unwrap()
            .is_none()
    );
}
#[test]
fn invalid_inputs_and_exhaustion_are_not_misses() {
    let data = TriangleMeshData3 {
        vertices: vec![[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]],
        triangles: vec![[0, 1, 2]],
    };
    let mut d = data.clone();
    d.triangles[0][0] = 9;
    assert_eq!(
        PreparedTriangleMesh3::try_new(d).unwrap_err(),
        TriangleMeshError3::InvalidIndex
    );
    let mut d = data.clone();
    d.vertices[0][0] = f64::NAN;
    assert_eq!(
        PreparedTriangleMesh3::try_new(d).unwrap_err(),
        TriangleMeshError3::InvalidCoordinates
    );
    let mut d = data;
    d.triangles[0] = [0, 0, 1];
    assert_eq!(
        PreparedTriangleMesh3::try_new(d).unwrap_err(),
        TriangleMeshError3::DegenerateTriangle
    );
    let mesh = slab(0., false);
    assert_eq!(
        mesh.capsule_contact([f64::INFINITY, 0., 0.], [0.; 3], 1., 0.),
        Err(TriangleMeshError3::InvalidQuery)
    );
    assert_eq!(
        mesh.capsule_sweep([0., 2., 0.], [0., 2., 0.], 0.2, [0., -4., 0.], 0),
        Err(TriangleMeshError3::IterationLimit)
    );
}
#[test]
fn bvh_matches_unpruned_reference_and_local_work_stays_bounded() {
    for count in [16usize, 64, 256, 1024] {
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for i in 0..count {
            let x = i as f64 * 5.;
            let first = vertices.len() as u32;
            vertices.extend([[x - 1., 0., -1.], [x + 1., 0., -1.], [x, 0., 1.]]);
            triangles.push([first, first + 2, first + 1]);
        }
        let mesh = PreparedTriangleMesh3::try_new(TriangleMeshData3 {
            vertices,
            triangles,
        })
        .unwrap();
        for i in 0..31 {
            let x = i as f64 * 1.37 - 2.;
            let p = [x, 0.2, 0.];
            let mut work = TriangleMeshWork3::default();
            let actual = mesh
                .capsule_contact_counted(p, p, 0.3, 0.1, &mut work)
                .unwrap();
            let reference = (0..count as u32)
                .filter_map(|id| {
                    let c = mesh.contact(id, p, p, 0.3).unwrap();
                    (c.separation <= 0.1).then_some(c)
                })
                .min_by(|a, b| {
                    a.separation
                        .total_cmp(&b.separation)
                        .then(a.triangle.cmp(&b.triangle))
                });
            assert_eq!(actual, reference);
            assert!(work.triangles_tested <= 1, "{work:?}");
            assert!(
                work.nodes_visited <= 2 * (count.ilog2() as u64) + 3,
                "{work:?}"
            );
        }
        assert_eq!(mesh.preparation().triangles_prepared, count as u64);
        assert_eq!(mesh.preparation().nodes_prepared, (count * 2 - 1) as u64);
    }
}
#[test]
fn capsule_crossing_and_translation_replay() {
    let mesh = slab(0., false);
    let c = mesh
        .capsule_contact([0., -0.2, 0.], [0., 0.8, 0.], 0.2, 0.)
        .unwrap()
        .unwrap();
    assert_eq!(c.separation, -0.4);
    assert_eq!(c.normal, [0., 1., 0.]);
    let p = [-0.3, 2., 0.];
    let q = [0.3, 2., 0.];
    let a = mesh.capsule_sweep(p, q, 0.1, [0., -4., 0.], 64).unwrap();
    assert_eq!(a, mesh.capsule_sweep(p, q, 0.1, [0., -4., 0.], 64).unwrap());
}

#[test]
fn sweeps_and_rays_match_all_triangle_traversal_at_seams() {
    let vertices = vec![
        [-2., 0., -2.],
        [2., 0., -2.],
        [-2., 0., 2.],
        [2., 0., 2.],
        [-2., 2., -2.],
        [2., 2., -2.],
        [-2., 2., 2.],
        [2., 2., 2.],
    ];
    let triangles = vec![[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6]];
    let mesh = PreparedTriangleMesh3::try_new(TriangleMeshData3 {
        vertices: vertices.clone(),
        triangles: triangles.clone(),
    })
    .unwrap();
    for i in 0..37 {
        let x = -2.5 + i as f64 / 7.;
        let p = [x, 1., 0.];
        let q = [x, 1.3, 0.];
        for motion in [[0., -3., 0.], [0., 3., 0.], [1., -3., 0.]] {
            let actual = mesh.capsule_sweep(p, q, 0.2, motion, 64).unwrap();
            let mut reference: Option<TriangleMeshSweepHit3> = None;
            let mut ray_reference: Option<TriangleMeshSegmentHit3> = None;
            for (id, t) in triangles.iter().enumerate() {
                let one = PreparedTriangleMesh3::try_new(TriangleMeshData3 {
                    vertices: vertices.clone(),
                    triangles: vec![*t],
                })
                .unwrap();
                if let Some(mut hit) = one.capsule_sweep(p, q, 0.2, motion, 64).unwrap() {
                    hit.contact.triangle = id as u32;
                    let rate = dot(motion, hit.contact.normal);
                    if reference.is_none_or(|b| {
                        hit.fraction < b.fraction
                            || (hit.fraction == b.fraction
                                && rate
                                    .total_cmp(&dot(motion, b.contact.normal))
                                    .then(hit.contact.triangle.cmp(&b.contact.triangle))
                                    .is_lt())
                    }) {
                        reference = Some(hit);
                    }
                }
                if let Some(mut hit) = one.segment_hit(p, add(p, motion)).unwrap() {
                    hit.triangle = id as u32;
                    if ray_reference.is_none_or(|b| {
                        hit.fraction < b.fraction
                            || (hit.fraction == b.fraction && hit.triangle < b.triangle)
                    }) {
                        ray_reference = Some(hit);
                    }
                }
            }
            assert_eq!(actual, reference, "p={p:?} motion={motion:?}");
            assert_eq!(mesh.segment_hit(p, add(p, motion)).unwrap(), ray_reference);
        }
    }
}

#[test]
fn polytope_overlap_and_large_coordinate_queries_remain_finite() {
    let mesh = slab(0., false);
    let vertices = [
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
    ];
    let axes = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    let poly = ConvexPolytope3 {
        vertices: &vertices,
        face_normals: &axes,
        edges: &axes,
    };
    let c = mesh
        .polytope_contact_counted(poly, &mut TriangleMeshWork3::default())
        .unwrap()
        .unwrap();
    assert_eq!(c.separation, -0.5);
    for offset in [0., 1e6, -1e6] {
        let vertices = mesh
            .data()
            .vertices
            .iter()
            .map(|p| add(*p, [offset; 3]))
            .collect();
        let shifted = PreparedTriangleMesh3::try_new(TriangleMeshData3 {
            vertices,
            triangles: mesh.data().triangles.clone(),
        })
        .unwrap();
        let p = add([0., 2., 0.], [offset; 3]);
        let hit = shifted
            .capsule_sweep(p, p, 0.25, [0., -4., 0.], 64)
            .unwrap()
            .unwrap();
        assert!((hit.fraction - 0.4375).abs() < 1e-8);
    }
}
