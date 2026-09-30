use geometry_kernels::{
    math3::{cross, dot, sub},
    primitive3::{PrimitiveShape3, PrimitiveVolumeError3, try_volume_properties},
};

fn close(actual: f64, expected: f64, scale: f64) {
    assert!(
        (actual - expected).abs() <= 2e-13 * scale.max(expected.abs()),
        "{actual:e} != {expected:e}"
    );
}

// Three-point Gauss-Legendre integrates each circular-slice polynomial (degree <=4).
fn integrate_capsule(h: f64, r: f64) -> [f64; 3] {
    let mut integrals = [0.0; 3];
    for (lo, hi) in [(-h - r, -h), (-h, h), (h, h + r)] {
        let midpoint = (lo + hi) / 2.0;
        let half_width = (hi - lo) / 2.0;
        for (node, weight) in [
            (-(3.0_f64 / 5.0).sqrt(), 5.0 / 9.0),
            (0.0, 8.0 / 9.0),
            ((3.0_f64 / 5.0).sqrt(), 5.0 / 9.0),
        ] {
            let y = midpoint + half_width * node;
            let cap_offset = (y.abs() - h).max(0.0);
            let radius_squared = r * r - cap_offset * cap_offset;
            let area = std::f64::consts::PI * radius_squared;
            let factor = half_width * weight;
            integrals[0] += factor * area;
            integrals[1] += factor * area * radius_squared / 4.0;
            integrals[2] += factor * area * y * y;
        }
    }
    integrals
}

#[test]
fn capsules_match_independent_circular_slice_integrals() {
    for scale in [1e-6, 0.25, 1.0, 1e6, 1e12] {
        for ratio in [0.0, 0.01, 0.5, 1.0, 10.0, 1000.0] {
            let r = scale;
            let h = ratio * scale;
            let properties = try_volume_properties(PrimitiveShape3::capsule(h, r)).unwrap();
            let [volume, x_integral, y_integral] = integrate_capsule(h, r);
            close(properties.volume, volume, volume);
            close(
                properties.normalized_second_moment[0][0],
                x_integral / volume,
                r * r,
            );
            close(
                properties.normalized_second_moment[1][1],
                y_integral / volume,
                (h + r) * (h + r),
            );
            assert_eq!(
                properties.normalized_second_moment[0][0],
                properties.normalized_second_moment[2][2]
            );
            assert_eq!(properties.centroid, [0.0; 3]);
        }
    }
}

// Integrate a triangular prism using three tetrahedra, independently of the wedge formula.
fn integrate_wedge([x, y, z]: [f64; 3]) -> (f64, [f64; 3], [[f64; 3]; 3]) {
    let a = [-x, -y, -z];
    let b = [-x, y, -z];
    let c = [x, -y, -z];
    let ap = [-x, -y, z];
    let bp = [-x, y, z];
    let cp = [x, -y, z];
    let mut volume = 0.0;
    let mut first = [0.0; 3];
    let mut second = [[0.0; 3]; 3];
    for vertices in [[a, b, c, cp], [a, b, bp, cp], [a, ap, bp, cp]] {
        let tetra_volume = dot(
            sub(vertices[1], vertices[0]),
            cross(sub(vertices[2], vertices[0]), sub(vertices[3], vertices[0])),
        )
        .abs()
            / 6.0;
        volume += tetra_volume;
        let sums: [f64; 3] = std::array::from_fn(|axis| vertices.iter().map(|p| p[axis]).sum());
        for i in 0..3 {
            first[i] += tetra_volume * sums[i] / 4.0;
            for j in 0..3 {
                let diagonal_sum: f64 = vertices.iter().map(|p| p[i] * p[j]).sum();
                second[i][j] += tetra_volume * (sums[i] * sums[j] + diagonal_sum) / 20.0;
            }
        }
    }
    let centroid = first.map(|value| value / volume);
    for i in 0..3 {
        for j in 0..3 {
            second[i][j] = second[i][j] / volume - centroid[i] * centroid[j];
        }
    }
    (volume, centroid, second)
}

#[test]
fn wedges_match_independent_tetrahedron_integrals() {
    for scale in [1e-6, 1.0, 1e6, 1e12] {
        for extents in [[1.0, 1.0, 1.0], [0.1, 3.0, 8.0], [13.0, 0.25, 2.0]] {
            let extents = extents.map(|v| v * scale);
            let actual = try_volume_properties(PrimitiveShape3::wedge(extents)).unwrap();
            let (volume, centroid, moment) = integrate_wedge(extents);
            close(actual.volume, volume, volume);
            for i in 0..3 {
                close(actual.centroid[i], centroid[i], scale);
                for j in 0..3 {
                    close(
                        actual.normalized_second_moment[i][j],
                        moment[i][j],
                        extents[i] * extents[j],
                    );
                }
            }
            assert!(actual.normalized_second_moment[0][1] < 0.0);
        }
    }
}

#[test]
fn sphere_limit_box_and_scale_covariance() {
    let sphere = try_volume_properties(PrimitiveShape3::sphere(2.0)).unwrap();
    assert_eq!(
        sphere,
        try_volume_properties(PrimitiveShape3::capsule(0.0, 2.0)).unwrap()
    );
    let cuboid = try_volume_properties(PrimitiveShape3::cuboid([1.0, 2.0, 3.0])).unwrap();
    assert_eq!(cuboid.volume, 48.0);
    assert_eq!(
        cuboid.normalized_second_moment,
        [
            [1.0 / 3.0, 0.0, 0.0],
            [0.0, 4.0 / 3.0, 0.0],
            [0.0, 0.0, 3.0]
        ]
    );
    for shape in [
        PrimitiveShape3::sphere(2.0),
        PrimitiveShape3::cuboid([1.0, 2.0, 3.0]),
        PrimitiveShape3::capsule(3.0, 2.0),
        PrimitiveShape3::wedge([1.0, 2.0, 3.0]),
    ] {
        let base = try_volume_properties(shape).unwrap();
        for scale in [1e-6, 10.0, 1e12] {
            let scaled = match shape.kind() {
                geometry_kernels::primitive3::PrimitiveKind3::Sphere => {
                    PrimitiveShape3::sphere(2.0 * scale)
                }
                geometry_kernels::primitive3::PrimitiveKind3::Box => {
                    PrimitiveShape3::cuboid([scale, 2.0 * scale, 3.0 * scale])
                }
                geometry_kernels::primitive3::PrimitiveKind3::Capsule => {
                    PrimitiveShape3::capsule(3.0 * scale, 2.0 * scale)
                }
                geometry_kernels::primitive3::PrimitiveKind3::Wedge => {
                    PrimitiveShape3::wedge([scale, 2.0 * scale, 3.0 * scale])
                }
            };
            let actual = try_volume_properties(scaled).unwrap();
            close(actual.volume, base.volume * scale.powi(3), actual.volume);
            for i in 0..3 {
                close(actual.centroid[i], base.centroid[i] * scale, scale);
                for j in 0..3 {
                    close(
                        actual.normalized_second_moment[i][j],
                        base.normalized_second_moment[i][j] * scale * scale,
                        scale * scale,
                    );
                }
            }
        }
    }
}

#[test]
fn unrepresentable_products_are_explicit_errors() {
    for dimension in [1e-200, 1e200] {
        for shape in [
            PrimitiveShape3::sphere(dimension),
            PrimitiveShape3::cuboid([dimension; 3]),
            PrimitiveShape3::capsule(dimension, dimension),
            PrimitiveShape3::wedge([dimension; 3]),
        ] {
            assert_eq!(
                try_volume_properties(shape),
                Err(PrimitiveVolumeError3::Unrepresentable)
            );
        }
    }
    assert_eq!(
        std::mem::size_of::<geometry_kernels::primitive3::PrimitiveVolumeProperties3>(),
        104
    );
}
