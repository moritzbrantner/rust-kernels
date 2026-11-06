//! Two-sided finite triangle/box manifolds over the retained mesh BVH.
use super::{Bounds, PreparedTriangleMesh3, TriangleMeshError3, TriangleMeshWork3};
use crate::{
    math3::{Vec3, add, cross, dot, length_squared, neg, normalized, scale, sub},
    primitive3::closest_segment_points,
    segment_hull::clip,
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TriangleMeshBoxPoint3 {
    pub point_box: Vec3,
    pub point_mesh: Vec3,
    pub separation: f64,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleMeshBoxContact3 {
    pub triangle: u32,
    /// Unit normal from the two-sided mesh toward the box.
    pub normal: Vec3,
    pub points: [TriangleMeshBoxPoint3; 4],
    pub len: usize,
}
#[derive(Clone, Copy)]
enum Feature {
    Triangle,
    Box(usize),
    Edge(usize, usize),
}

impl PreparedTriangleMesh3 {
    /// Visits candidate triangles in deterministic BVH order and emits up to four witnesses
    /// per triangle. This is a pose/prediction query, not a rotational or translation sweep.
    /// The box frame is right-handed and orthonormal, extents positive, margin nonnegative.
    /// Errors may follow earlier callback emissions; callers must discard them on error.
    pub fn box_contacts_counted(
        &self,
        centre: Vec3,
        half: Vec3,
        axes: [Vec3; 3],
        margin: f64,
        work: &mut TriangleMeshWork3,
        mut emit: impl FnMut(TriangleMeshBoxContact3),
    ) -> Result<(), TriangleMeshError3> {
        if !centre
            .into_iter()
            .chain(half)
            .chain(axes.into_iter().flatten())
            .chain([margin])
            .all(f64::is_finite)
            || half.into_iter().any(|v| v <= 0.0)
            || margin < 0.0
            || axes.iter().any(|&v| (length_squared(v) - 1.0).abs() > 1e-8)
            || dot(axes[0], axes[1]).abs() > 1e-8
            || dot(axes[0], axes[2]).abs() > 1e-8
            || dot(axes[1], axes[2]).abs() > 1e-8
            || dot(cross(axes[0], axes[1]), axes[2]) < 1.0 - 1e-8
        {
            return Err(TriangleMeshError3::InvalidQuery);
        }
        let extents: Vec3 =
            std::array::from_fn(|k| (0..3).map(|i| half[i] * axes[i][k].abs()).sum());
        let bounds = Bounds {
            min: sub(centre, extents),
            max: add(centre, extents),
        }
        .expanded(margin)?;
        work.queries += 1;
        self.visit(bounds, work, |id, work| {
            work.triangles_tested += 1;
            work.normal_reuses += 1;
            if let Some(mut contact) = triangle_contact(
                self.triangle(id),
                self.normals[id as usize],
                centre,
                half,
                axes,
                margin,
            )? {
                contact.triangle = id;
                emit(contact);
            }
            Ok(())
        })
    }
}

fn triangle_contact(
    triangle: [Vec3; 3],
    triangle_normal: Vec3,
    centre: Vec3,
    half: Vec3,
    axes: [Vec3; 3],
    margin: f64,
) -> Result<Option<TriangleMeshBoxContact3>, TriangleMeshError3> {
    let vertices = triangle.map(|p| sub(p, centre));
    let edges = [
        sub(vertices[1], vertices[0]),
        sub(vertices[2], vertices[1]),
        sub(vertices[0], vertices[2]),
    ];
    let mut best = f64::NEG_INFINITY;
    let mut normal = triangle_normal;
    let mut feature = Feature::Triangle;
    let radius = |n| (0..3).map(|i| half[i] * dot(axes[i], n).abs()).sum::<f64>();
    let plane = dot(vertices[0], triangle_normal);
    let face_normal = if plane >= 0.0 {
        neg(triangle_normal)
    } else {
        triangle_normal
    };
    let face_gap = plane.abs() - radius(triangle_normal);
    if !face_gap.is_finite() {
        return Err(TriangleMeshError3::NonFiniteComputation);
    }
    if face_gap > margin {
        return Ok(None);
    }
    if face_gap >= 0.0 {
        let witnesses = face_points(
            vertices,
            edges,
            triangle_normal,
            face_normal,
            half,
            axes,
            margin,
        )?;
        if witnesses
            .iter()
            .any(|p| (p.separation - face_gap).abs() <= 1e-10 * (1.0 + face_gap.abs()))
        {
            return finish(witnesses, face_normal, centre);
        }
    }
    let mut offer = |axis: Vec3, candidate| -> Result<bool, TriangleMeshError3> {
        let Some(axis) = normalized(axis) else {
            return Ok(true);
        };
        let r = radius(axis);
        let (lo, hi) = vertices
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &p| {
                let v = dot(p, axis);
                (lo.min(v), hi.max(v))
            });
        let (gap, n) = if -hi - r >= lo - r {
            (-hi - r, axis)
        } else {
            (lo - r, neg(axis))
        };
        if !gap.is_finite() {
            return Err(TriangleMeshError3::NonFiniteComputation);
        }
        if gap > margin {
            return Ok(false);
        }
        if gap > best {
            best = gap;
            normal = n;
            feature = candidate;
        }
        Ok(true)
    };
    if !offer(triangle_normal, Feature::Triangle)? {
        return Ok(None);
    }
    for (i, axis) in axes.into_iter().enumerate() {
        if !offer(axis, Feature::Box(i))? {
            return Ok(None);
        }
    }
    for (i, axis) in axes.into_iter().enumerate() {
        for (j, edge) in edges.into_iter().enumerate() {
            let crossed = cross(axis, edge);
            if !length_squared(crossed).is_finite() {
                return Err(TriangleMeshError3::NonFiniteComputation);
            }
            if length_squared(crossed) > 1e-24 && !offer(crossed, Feature::Edge(i, j))? {
                return Ok(None);
            }
        }
    }
    let support = |direction: Vec3| {
        (0..3).fold([0.0; 3], |p, i| {
            add(
                p,
                scale(
                    axes[i],
                    if dot(direction, axes[i]) >= 0.0 {
                        half[i]
                    } else {
                        -half[i]
                    },
                ),
            )
        })
    };
    let mut points = Vec::new();
    let mut scratch = Vec::new();
    match feature {
        Feature::Triangle => {
            points = face_points(vertices, edges, triangle_normal, normal, half, axes, margin)?;
        }
        Feature::Box(k) => {
            let mut polygon = vertices.to_vec();
            for i in (0..3).filter(|&i| i != k) {
                for sign in [-1.0, 1.0] {
                    clip(&mut polygon, &mut scratch, |p| {
                        sign * dot(p, axes[i]) - half[i]
                    });
                }
            }
            for p in polygon {
                let separation = -half[k] - dot(normal, p);
                if separation <= margin {
                    points.push(TriangleMeshBoxPoint3 {
                        point_box: add(p, scale(normal, separation)),
                        point_mesh: p,
                        separation,
                    });
                }
            }
        }
        Feature::Edge(k, e) => {
            let c = (0..3).filter(|&i| i != k).fold([0.0; 3], |p, i| {
                add(
                    p,
                    scale(
                        axes[i],
                        if dot(axes[i], normal) > 0.0 {
                            -half[i]
                        } else {
                            half[i]
                        },
                    ),
                )
            });
            let u = scale(axes[k], half[k]);
            if let Some((point_box, point_mesh)) =
                closest_segment_points(sub(c, u), add(c, u), vertices[e], vertices[(e + 1) % 3])
            {
                points.push(TriangleMeshBoxPoint3 {
                    point_box,
                    point_mesh,
                    separation: best,
                });
            }
        }
    }
    if points.is_empty() {
        let point_box = support(neg(normal));
        let point_mesh = *vertices
            .iter()
            .max_by(|a, b| dot(**a, normal).total_cmp(&dot(**b, normal)))
            .unwrap_or(&vertices[0]);
        points.push(TriangleMeshBoxPoint3 {
            point_box,
            point_mesh,
            separation: best,
        });
    }
    finish(points, normal, centre)
}

fn finish(
    mut points: Vec<TriangleMeshBoxPoint3>,
    normal: Vec3,
    centre: Vec3,
) -> Result<Option<TriangleMeshBoxContact3>, TriangleMeshError3> {
    // Boundary intersections may repeat a polygon vertex. Remove them before bounded
    // selection, otherwise duplicates can displace a genuine supporting corner.
    let mut unique = 0;
    for read in 0..points.len() {
        let point = points[read];
        if !points[..unique]
            .iter()
            .any(|p| p.point_box == point.point_box && p.point_mesh == point.point_mesh)
        {
            points[unique] = point;
            unique += 1;
        }
    }
    points.truncate(unique);
    let len = points.len().min(4);
    let mut selected = [TriangleMeshBoxPoint3::default(); 4];
    for (i, p) in selected[..len].iter_mut().enumerate() {
        let source = points[i * points.len() / len];
        *p = TriangleMeshBoxPoint3 {
            point_box: add(source.point_box, centre),
            point_mesh: add(source.point_mesh, centre),
            separation: source.separation,
        };
        if !p
            .point_box
            .into_iter()
            .chain(p.point_mesh)
            .chain([p.separation])
            .all(f64::is_finite)
        {
            return Err(TriangleMeshError3::NonFiniteComputation);
        }
    }
    Ok(Some(TriangleMeshBoxContact3 {
        triangle: 0,
        normal,
        points: selected,
        len,
    }))
}

fn face_points(
    vertices: [Vec3; 3],
    edges: [Vec3; 3],
    triangle_normal: Vec3,
    normal: Vec3,
    half: Vec3,
    axes: [Vec3; 3],
    margin: f64,
) -> Result<Vec<TriangleMeshBoxPoint3>, TriangleMeshError3> {
    let mut points = Vec::new();
    let mut scratch = Vec::new();
    let k = (0..3)
        .max_by(|&i, &j| {
            dot(normal, axes[i])
                .abs()
                .total_cmp(&dot(normal, axes[j]).abs())
        })
        .unwrap_or(0);
    let c = scale(axes[k], -half[k] * dot(normal, axes[k]).signum());
    let u = scale(axes[(k + 1) % 3], half[(k + 1) % 3]);
    let v = scale(axes[(k + 2) % 3], half[(k + 2) % 3]);
    let mut polygon = vec![
        add(add(c, u), v),
        add(sub(c, u), v),
        sub(sub(c, u), v),
        add(sub(c, v), u),
    ];
    for i in 0..3 {
        let a = vertices[i];
        let mut side = normalized(cross(edges[i], triangle_normal))
            .ok_or(TriangleMeshError3::NonFiniteComputation)?;
        if dot(side, sub(vertices[(i + 2) % 3], a)) > 0.0 {
            side = neg(side);
        }
        clip(&mut polygon, &mut scratch, |p| dot(side, sub(p, a)));
    }
    for p in polygon {
        let separation = dot(normal, sub(p, vertices[0]));
        if separation <= margin {
            points.push(TriangleMeshBoxPoint3 {
                point_box: p,
                point_mesh: sub(p, scale(normal, separation)),
                separation,
            });
        }
    }
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        heightfield::{ConvexPolytope3, polytope_triangle_contact},
        triangle_mesh::TriangleMeshData3,
    };
    const AXES: [Vec3; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    const TRIANGLE: [Vec3; 3] = [[-10.0, 0.0, -10.0], [10.0, 0.0, -10.0], [0.0, 0.0, 10.0]];
    fn mesh(triangle: [Vec3; 3]) -> PreparedTriangleMesh3 {
        PreparedTriangleMesh3::try_new(TriangleMeshData3 {
            vertices: triangle.to_vec(),
            triangles: vec![[0, 1, 2]],
        })
        .unwrap()
    }
    #[test]
    fn floor_contacts_have_four_finite_witnesses_from_either_side_and_winding() {
        for triangle in [TRIANGLE, [TRIANGLE[0], TRIANGLE[2], TRIANGLE[1]]] {
            for sign in [-1.0, 1.0] {
                let mut contacts = Vec::new();
                mesh(triangle)
                    .box_contacts_counted(
                        [0.0, sign * 0.49, 0.0],
                        [0.5; 3],
                        AXES,
                        0.0,
                        &mut TriangleMeshWork3::default(),
                        |c| contacts.push(c),
                    )
                    .unwrap();
                assert_eq!(contacts.len(), 1);
                let c = contacts[0];
                assert_eq!(c.len, 4);
                assert_eq!(c.normal, [0.0, sign, 0.0]);
                for p in &c.points[..c.len] {
                    assert!((p.separation + 0.01).abs() < 1e-12);
                    assert!(p.point_mesh[1].abs() < 1e-12);
                    assert!(
                        (dot(sub(p.point_box, p.point_mesh), c.normal) - p.separation).abs()
                            < 1e-12
                    );
                    assert!(
                        sub(p.point_box, [0.0, sign * 0.49, 0.0])
                            .into_iter()
                            .all(|v| v.abs() <= 0.5 + 1e-12)
                    );
                }
            }
        }
    }
    #[test]
    fn finite_triangle_prediction_and_input_errors_remain_explicit() {
        let mesh = mesh(TRIANGLE);
        let count = |centre, margin| {
            let mut count = 0;
            mesh.box_contacts_counted(
                centre,
                [0.5; 3],
                AXES,
                margin,
                &mut TriangleMeshWork3::default(),
                |_| count += 1,
            )
            .unwrap();
            count
        };
        assert_eq!(count([0.0, 0.51, 0.0], 0.02), 1);
        assert_eq!(count([0.0, 0.53, 0.0], 0.02), 0);
        assert_eq!(count([9.0, 0.0, 9.0], 0.0), 0); // Empty corner of triangle AABB.
        assert_eq!(
            mesh.box_contacts_counted(
                [f64::NAN, 0.0, 0.0],
                [0.5; 3],
                AXES,
                0.0,
                &mut TriangleMeshWork3::default(),
                |_| {}
            ),
            Err(TriangleMeshError3::InvalidQuery)
        );
    }
    #[test]
    fn rotated_boxes_match_independent_polytope_triangle_overlap() {
        let triangle = [[-2.0, -0.2, -1.0], [2.0, 0.3, -1.0], [0.2, 1.0, 2.0]];
        let n = normalized(cross(
            sub(triangle[1], triangle[0]),
            sub(triangle[2], triangle[0]),
        ))
        .unwrap();
        let mut seed = 7u64;
        let mut sample = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 32) as u32 as f64 / u32::MAX as f64) * 6.0 - 3.0
        };
        for _ in 0..512 {
            let centre = [sample(), sample(), sample()];
            let (s, c) = sample().sin_cos();
            let axes = [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]];
            let half = [0.3, 0.5, 0.7];
            let vertices: Vec<_> = [-1.0, 1.0]
                .into_iter()
                .flat_map(|x| {
                    [-1.0, 1.0].into_iter().flat_map(move |y| {
                        [-1.0, 1.0].into_iter().map(move |z| {
                            add(
                                centre,
                                add(
                                    add(scale(axes[0], half[0] * x), scale(axes[1], half[1] * y)),
                                    scale(axes[2], half[2] * z),
                                ),
                            )
                        })
                    })
                })
                .collect();
            let expected = polytope_triangle_contact(
                ConvexPolytope3 {
                    vertices: &vertices,
                    face_normals: &axes,
                    edges: &axes,
                },
                triangle,
                n,
            );
            let actual = triangle_contact(triangle, n, centre, half, axes, 0.0).unwrap();
            assert_eq!(actual.is_some(), expected.is_some(), "centre {centre:?}");
            if let (Some(actual), Some(expected)) = (actual, expected) {
                for point in &actual.points[..actual.len] {
                    assert!(
                        (dot(sub(point.point_box, point.point_mesh), actual.normal)
                            - point.separation)
                            .abs()
                            < 1e-8,
                        "{actual:?}"
                    );
                }
                let gap = actual.points[..actual.len]
                    .iter()
                    .map(|p| p.separation)
                    .fold(f64::INFINITY, f64::min);
                // Finite-face clipping may exclude the deepest infinite-plane vertex.
                // Its witnesses cannot penetrate farther than the independent SAT depth.
                assert!(
                    gap >= expected.separation - 1e-8 && gap <= 1e-8,
                    "{actual:?} vs {expected:?}"
                );
            }
        }
    }
    #[test]
    fn box_bvh_matches_unpruned_triangles_and_local_work_stays_bounded() {
        for patches in [32, 64, 128, 256] {
            let mut vertices = Vec::new();
            let mut triangles = Vec::new();
            for i in 0..patches {
                let x = i as f64 * 6.0;
                let start = vertices.len() as u32;
                vertices.extend([
                    [x - 2.0, 0.0, -2.0],
                    [x + 2.0, 0.0, -2.0],
                    [x + 2.0, 0.0, 2.0],
                    [x - 2.0, 0.0, 2.0],
                ]);
                triangles.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]]);
            }
            let mesh = PreparedTriangleMesh3::try_new(TriangleMeshData3 {
                vertices,
                triangles,
            })
            .unwrap();
            for centre in [[0.0, 0.49, 0.0], [6.1, 0.51, 0.2], [3.0, 0.0, 0.0]] {
                let mut actual = Vec::new();
                let mut work = TriangleMeshWork3::default();
                mesh.box_contacts_counted(centre, [0.5; 3], AXES, 0.02, &mut work, |c| {
                    actual.push(c)
                })
                .unwrap();
                let mut expected = Vec::new();
                for id in 0..mesh.data.triangles.len() as u32 {
                    if let Some(mut c) = triangle_contact(
                        mesh.triangle(id),
                        mesh.normals[id as usize],
                        centre,
                        [0.5; 3],
                        AXES,
                        0.02,
                    )
                    .unwrap()
                    {
                        c.triangle = id;
                        expected.push(c);
                    }
                }
                actual.sort_by_key(|c| c.triangle);
                assert_eq!(actual, expected);
                assert!(work.triangles_tested <= 4);
                assert!(work.nodes_visited <= 8 * (patches as u32).ilog2() as u64 + 8);
            }
        }
    }
    #[test]
    fn shared_diagonal_keeps_all_four_box_corners_without_duplicate_witnesses() {
        let mesh = PreparedTriangleMesh3::try_new(TriangleMeshData3 {
            vertices: vec![
                [-2.0, 0.0, -2.0],
                [2.0, 0.0, -2.0],
                [2.0, 0.0, 2.0],
                [-2.0, 0.0, 2.0],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        })
        .unwrap();
        let mut contacts = Vec::new();
        mesh.box_contacts_counted(
            [0.0, 0.49, 0.0],
            [0.5; 3],
            AXES,
            0.02,
            &mut TriangleMeshWork3::default(),
            |c| contacts.push(c),
        )
        .unwrap();
        assert_eq!(contacts.len(), 2);
        for c in &contacts {
            assert_eq!(c.len, 3);
            for i in 0..c.len {
                assert!(
                    !c.points[..i]
                        .iter()
                        .any(|p| p.point_box == c.points[i].point_box)
                );
            }
        }
        for p in [
            [-0.5, 0.0, -0.5],
            [-0.5, 0.0, 0.5],
            [0.5, 0.0, -0.5],
            [0.5, 0.0, 0.5],
        ] {
            assert!(
                contacts
                    .iter()
                    .any(|c| c.points[..c.len].iter().any(|point| point.point_mesh == p)),
                "missing {p:?}"
            );
        }
    }
}
