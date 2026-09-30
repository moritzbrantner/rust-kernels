use geometry_kernels::heightfield::{HeightfieldData3, HeightfieldError3, PreparedHeightfield3};

fn mesh(data: &HeightfieldData3) -> Vec<(u32, [[f64; 3]; 3])> {
    let nx = data.columns.len();
    let mut triangles = Vec::new();
    for z in 0..data.rows.len() - 1 {
        for x in 0..nx - 1 {
            let cell = z * (nx - 1) + x;
            if data.active_cells.as_ref().is_some_and(|mask| !mask[cell]) {
                continue;
            }
            let vertices = [
                [data.columns[x], data.heights[z * nx + x], data.rows[z]],
                [
                    data.columns[x + 1],
                    data.heights[z * nx + x + 1],
                    data.rows[z],
                ],
                [
                    data.columns[x],
                    data.heights[(z + 1) * nx + x],
                    data.rows[z + 1],
                ],
                [
                    data.columns[x + 1],
                    data.heights[(z + 1) * nx + x + 1],
                    data.rows[z + 1],
                ],
            ];
            let indices = if (x + z) % 2 == 0 {
                [[0, 2, 3], [0, 3, 1]]
            } else {
                [[0, 2, 1], [1, 2, 3]]
            };
            for (side, indices) in indices.into_iter().enumerate() {
                triangles.push((
                    (cell * 2 + side) as u32,
                    indices.map(|index| vertices[index]),
                ));
            }
        }
    }
    triangles
}

// Exhaust every independently materialized triangle, solving its world-XZ plane
// and barycentrics. Never calls the prepared sampler or its triangle accessor.
fn exhaustive(triangles: &[(u32, [[f64; 3]; 3])], x: f64, z: f64) -> Option<(u32, f64, [f64; 3])> {
    let mut best = None;
    for &(id, [a, b, c]) in triangles {
        let bx = b[0] - a[0];
        let bz = b[2] - a[2];
        let cx = c[0] - a[0];
        let cz = c[2] - a[2];
        let determinant = bx * cz - bz * cx;
        let wx = x - a[0];
        let wz = z - a[2];
        let u = (wx * cz - wz * cx) / determinant;
        let v = (bx * wz - bz * wx) / determinant;
        let band = 64.0 * f64::EPSILON;
        if u < -band || v < -band || u + v > 1.0 + band {
            continue;
        }
        let height = a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1]);
        let sx = ((b[1] - a[1]) * cz - (c[1] - a[1]) * bz) / determinant;
        let sz = (bx * (c[1] - a[1]) - cx * (b[1] - a[1])) / determinant;
        let length = sx.hypot(1.0).hypot(sz);
        let normal = [-sx / length, 1.0 / length, -sz / length];
        let priority = (id / 2, std::cmp::Reverse(id % 2));
        if best.as_ref().is_none_or(|(previous, _, _)| {
            priority > (*previous / 2, std::cmp::Reverse(*previous % 2))
        }) {
            best = Some((id, height, normal));
        }
    }
    best
}

fn fixture(size: usize, scale: f64) -> HeightfieldData3 {
    let columns = (0..size)
        .map(|index| scale * (-7.0 + index as f64 * (1.0 + 0.03 * index as f64)))
        .collect::<Vec<_>>();
    let rows = (0..size)
        .map(|index| scale * (-11.0 + index as f64 * (0.75 + 0.02 * index as f64)))
        .collect::<Vec<_>>();
    let heights = (0..size * size)
        .map(|index| scale * ((index as f64 * 0.73).sin() + 0.2 * (index as f64 * 0.41).cos()))
        .collect();
    HeightfieldData3 {
        columns,
        rows,
        heights,
        active_cells: None,
    }
}

#[test]
fn indexed_nonuniform_surface_matches_independent_all_triangle_reference() {
    let mut random = 0x7915_1931_u64;
    let mut checked = 0;
    for scale in [1e-6, 1.0, 1e6] {
        for size in [2, 3, 9, 65] {
            let data = fixture(size, scale);
            let triangles = mesh(&data);
            let prepared = PreparedHeightfield3::try_new(data.clone()).unwrap();
            assert_eq!(
                prepared.preparation().triangles_prepared as usize,
                triangles.len()
            );
            for _ in 0..128 {
                random = random
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let x = data.columns[0]
                    + (data.columns[size - 1] - data.columns[0])
                        * ((random >> 11) as f64 / ((1_u64 << 53) as f64));
                random = random
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let z = data.rows[0]
                    + (data.rows[size - 1] - data.rows[0])
                        * ((random >> 11) as f64 / ((1_u64 << 53) as f64));
                let expected = exhaustive(&triangles, x, z).unwrap();
                let result = prepared.sample(x, z).unwrap();
                let actual = result.surface.unwrap();
                assert_eq!(actual.triangle, expected.0);
                assert!((actual.point[1] - expected.1).abs() <= 1e-10 * scale);
                for axis in 0..3 {
                    assert!((actual.normal[axis] - expected.2[axis]).abs() <= 1e-10);
                }
                assert_eq!(result.work.triangles_tested, 1);
                assert_eq!(result.work.cells_visited, 1);
                assert_eq!(result.work.normal_reuses, 1);
                assert!(
                    result.work.coordinate_comparisons
                        <= 4 + 2 * (size - 1).next_power_of_two().ilog2() as u64
                );
                assert_eq!(
                    prepared.triangle_vertices(actual.triangle).unwrap(),
                    triangles[actual.triangle as usize].1
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 1536);
}

#[test]
fn flat_tilted_steep_shared_diagonals_vertices_and_outer_edges_are_stable() {
    for scale in [1e-6, 1.0, 1e6] {
        for slope in [0.0_f64, 0.3, 1000.0] {
            let columns = vec![-3.0 * scale, -scale, 2.0 * scale];
            let rows = vec![-5.0 * scale, scale, 4.0 * scale];
            let heights = rows
                .iter()
                .flat_map(|z| columns.iter().map(move |x| slope * x - 0.2 * z))
                .collect();
            let data = HeightfieldData3 {
                columns,
                rows,
                heights,
                active_cells: None,
            };
            let triangles = mesh(&data);
            let prepared = PreparedHeightfield3::try_new(data.clone()).unwrap();
            let points = [
                [-3.0 * scale, -5.0 * scale],
                [2.0 * scale, 4.0 * scale],
                [-scale, scale],
                [-2.0 * scale, -2.0 * scale],
                [0.5 * scale, -2.0 * scale],
                [-scale, -2.0 * scale],
                [-2.0 * scale, scale],
                [2.0 * scale, scale],
            ];
            for [x, z] in points {
                let first = prepared.sample(x, z).unwrap();
                assert_eq!(first, prepared.sample(x, z).unwrap());
                let actual = first.surface.unwrap();
                let expected = exhaustive(&triangles, x, z).unwrap();
                assert_eq!(actual.triangle, expected.0);
                let budget = 1e-10 * scale * (1.0 + slope);
                assert!((actual.point[1] - (slope * x - 0.2 * z)).abs() <= budget);
                let norm = slope.hypot(1.0).hypot(0.2);
                for (value, expected) in
                    actual
                        .normal
                        .into_iter()
                        .zip([-slope / norm, 1.0 / norm, 0.2 / norm])
                {
                    assert!((value - expected).abs() < 1e-12);
                }
                assert!(actual.normal[1] > 0.0);
            }
            assert!(
                prepared
                    .sample(f64::from_bits(data.columns[0].to_bits() + 1), 0.0)
                    .unwrap()
                    .surface
                    .is_none()
            );
            assert!(
                prepared
                    .sample(f64::from_bits(data.columns[2].to_bits() + 1), 0.0)
                    .unwrap()
                    .surface
                    .is_none()
            );
        }
    }
}

#[test]
fn omitted_cells_preserve_present_shared_boundary_and_stable_triangle_ids() {
    let data = HeightfieldData3 {
        columns: vec![0.0, 1.0, 2.0],
        rows: vec![0.0, 1.0, 2.0],
        heights: vec![0.0; 9],
        active_cells: Some(vec![true, true, true, false]),
    };
    let triangles = mesh(&data);
    let prepared = PreparedHeightfield3::try_new(data).unwrap();
    assert_eq!(prepared.preparation().triangles_prepared, 6);
    assert!(prepared.sample(1.5, 1.5).unwrap().surface.is_none());
    assert!(prepared.triangle_vertices(6).is_none());
    assert!(prepared.triangle_vertices(u32::MAX).is_none());
    for [x, z] in [[1.0, 1.0], [1.0, 1.5], [1.5, 1.0]] {
        let actual = prepared.sample(x, z).unwrap();
        let expected = exhaustive(&triangles, x, z).unwrap();
        assert_eq!(actual.surface.unwrap().triangle, expected.0);
        assert!(actual.work.cells_visited <= 4);
        assert_eq!(actual.work.triangles_tested, 1);
    }
    let none = PreparedHeightfield3::try_new(HeightfieldData3 {
        columns: vec![0.0, 1.0],
        rows: vec![0.0, 1.0],
        heights: vec![0.0; 4],
        active_cells: Some(vec![false]),
    })
    .unwrap();
    assert_eq!(none.preparation().triangles_prepared, 0);
    assert!(none.sample(0.5, 0.5).unwrap().surface.is_none());
}

#[test]
fn invalid_data_and_queries_fail_with_attempted_work_instead_of_clamping() {
    let good = HeightfieldData3 {
        columns: vec![0.0, 1.0],
        rows: vec![0.0, 1.0],
        heights: vec![0.0; 4],
        active_cells: None,
    };
    for (mut data, reason, change) in [
        (good.clone(), HeightfieldError3::InvalidDimensions, 0),
        (good.clone(), HeightfieldError3::InvalidCoordinates, 1),
        (good.clone(), HeightfieldError3::InvalidCoordinates, 2),
        (good.clone(), HeightfieldError3::InvalidHeights, 3),
        (good.clone(), HeightfieldError3::InvalidHeights, 4),
        (good.clone(), HeightfieldError3::InvalidCellMask, 5),
    ] {
        match change {
            0 => {
                data.columns.pop();
            }
            1 => data.rows[1] = 0.0,
            2 => data.columns[1] = f64::INFINITY,
            3 => {
                data.heights.pop();
            }
            4 => data.heights[0] = f64::NAN,
            _ => data.active_cells = Some(vec![]),
        }
        assert_eq!(PreparedHeightfield3::try_new(data).unwrap_err(), reason);
    }
    let prepared = PreparedHeightfield3::try_new(good).unwrap();
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let failure = prepared.sample(invalid, 0.5).unwrap_err();
        assert_eq!(failure.reason, HeightfieldError3::InvalidCoordinates);
        assert_eq!(failure.work.queries, 1);
        assert_eq!(failure.work.coordinate_comparisons, 0);
    }
    assert_eq!(prepared.preparation().triangles_prepared, 2);
}

#[test]
fn very_small_flat_cells_and_large_negative_coordinates_have_explicit_numerical_limits() {
    for scale in [1e-300, 1.0, 1e150] {
        let prepared = PreparedHeightfield3::try_new(HeightfieldData3 {
            columns: vec![0.0, scale],
            rows: vec![0.0, scale],
            heights: vec![scale; 4],
            active_cells: None,
        })
        .unwrap();
        let point = prepared
            .sample(scale / 2.0, scale / 2.0)
            .unwrap()
            .surface
            .unwrap();
        assert_eq!(point.point[1], scale);
        assert_eq!(point.normal, [0.0, 1.0, 0.0]);
    }
    let prepared = PreparedHeightfield3::try_new(HeightfieldData3 {
        columns: vec![-1e9, -1e9 + 0.25],
        rows: vec![-1e9, -1e9 + 0.5],
        heights: vec![3.0; 4],
        active_cells: None,
    })
    .unwrap();
    assert_eq!(
        prepared
            .sample(-1e9 + 0.125, -1e9 + 0.25)
            .unwrap()
            .surface
            .unwrap()
            .point[1],
        3.0
    );
    let invalid = HeightfieldData3 {
        columns: vec![0.0, 1e-300],
        rows: vec![0.0, 1e-300],
        heights: vec![0.0, 1e300, 0.0, 1e300],
        active_cells: None,
    };
    assert_eq!(
        PreparedHeightfield3::try_new(invalid).unwrap_err(),
        HeightfieldError3::NonFiniteComputation
    );
}
