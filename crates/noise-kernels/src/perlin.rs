use crate::Permutation;

const GRADIENTS_3D: [[f64; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

/// Samples deterministic two-dimensional gradient Perlin noise.
///
/// Coordinates must be finite. The field repeats every 256 lattice cells on
/// each axis because the shared permutation table has 256 entries.
#[must_use]
pub fn perlin2(permutation: &Permutation, x: f64, y: f64) -> f64 {
    assert!(
        x.is_finite() && y.is_finite(),
        "Perlin coordinates must be finite"
    );

    let (cell_x, local_x) = lattice_coordinate(x);
    let (cell_y, local_y) = lattice_coordinate(y);
    let local_x1 = local_x - 1.0;
    let local_y1 = local_y - 1.0;

    let n00 = dot2(permutation.hash2(cell_x, cell_y), local_x, local_y);
    let n10 = dot2(
        permutation.hash2(cell_x + 1, cell_y),
        local_x1,
        local_y,
    );
    let n01 = dot2(
        permutation.hash2(cell_x, cell_y + 1),
        local_x,
        local_y1,
    );
    let n11 = dot2(
        permutation.hash2(cell_x + 1, cell_y + 1),
        local_x1,
        local_y1,
    );

    let u = fade(local_x);
    let v = fade(local_y);
    lerp(lerp(n00, n10, u), lerp(n01, n11, u), v)
}

/// Samples deterministic three-dimensional gradient Perlin noise.
///
/// Coordinates must be finite. The field repeats every 256 lattice cells on
/// each axis because the shared permutation table has 256 entries.
#[must_use]
pub fn perlin3(permutation: &Permutation, x: f64, y: f64, z: f64) -> f64 {
    assert!(
        x.is_finite() && y.is_finite() && z.is_finite(),
        "Perlin coordinates must be finite"
    );

    let (cell_x, local_x) = lattice_coordinate(x);
    let (cell_y, local_y) = lattice_coordinate(y);
    let (cell_z, local_z) = lattice_coordinate(z);
    let local_x1 = local_x - 1.0;
    let local_y1 = local_y - 1.0;
    let local_z1 = local_z - 1.0;

    let n000 = dot3(
        permutation.hash3(cell_x, cell_y, cell_z),
        local_x,
        local_y,
        local_z,
    );
    let n100 = dot3(
        permutation.hash3(cell_x + 1, cell_y, cell_z),
        local_x1,
        local_y,
        local_z,
    );
    let n010 = dot3(
        permutation.hash3(cell_x, cell_y + 1, cell_z),
        local_x,
        local_y1,
        local_z,
    );
    let n110 = dot3(
        permutation.hash3(cell_x + 1, cell_y + 1, cell_z),
        local_x1,
        local_y1,
        local_z,
    );
    let n001 = dot3(
        permutation.hash3(cell_x, cell_y, cell_z + 1),
        local_x,
        local_y,
        local_z1,
    );
    let n101 = dot3(
        permutation.hash3(cell_x + 1, cell_y, cell_z + 1),
        local_x1,
        local_y,
        local_z1,
    );
    let n011 = dot3(
        permutation.hash3(cell_x, cell_y + 1, cell_z + 1),
        local_x,
        local_y1,
        local_z1,
    );
    let n111 = dot3(
        permutation.hash3(cell_x + 1, cell_y + 1, cell_z + 1),
        local_x1,
        local_y1,
        local_z1,
    );

    let u = fade(local_x);
    let v = fade(local_y);
    let w = fade(local_z);

    let z0 = lerp(lerp(n000, n100, u), lerp(n010, n110, u), v);
    let z1 = lerp(lerp(n001, n101, u), lerp(n011, n111, u), v);
    lerp(z0, z1, w)
}

#[inline]
fn lattice_coordinate(value: f64) -> (usize, f64) {
    let floor = value.floor();
    (floor.rem_euclid(256.0) as usize, value - floor)
}

#[inline]
fn dot2(hash: u8, x: f64, y: f64) -> f64 {
    let gradient = GRADIENTS_3D[usize::from(hash) % GRADIENTS_3D.len()];
    gradient[0] * x + gradient[1] * y
}

#[inline]
fn dot3(hash: u8, x: f64, y: f64, z: f64) -> f64 {
    let gradient = GRADIENTS_3D[usize::from(hash) % GRADIENTS_3D.len()];
    gradient[0] * x + gradient[1] * y + gradient[2] * z
}

#[inline]
fn fade(value: f64) -> f64 {
    value * value * value * (value * (value * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(left: f64, right: f64, amount: f64) -> f64 {
    left + amount * (right - left)
}

#[cfg(test)]
mod tests {
    use super::{perlin2, perlin3};
    use crate::Permutation;

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 1e-12;
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn lattice_points_are_zero() {
        let permutation = Permutation::from_seed(17);

        for x in -3..=3 {
            for y in -3..=3 {
                assert_close(perlin2(&permutation, f64::from(x), f64::from(y)), 0.0);
                for z in -2..=2 {
                    assert_close(
                        perlin3(
                            &permutation,
                            f64::from(x),
                            f64::from(y),
                            f64::from(z),
                        ),
                        0.0,
                    );
                }
            }
        }
    }

    #[test]
    fn fields_repeat_after_256_lattice_cells() {
        let permutation = Permutation::from_seed(99);
        let point2 = [-13.375, 28.625];
        let point3 = [17.25, -4.875, 91.5];

        let base2 = perlin2(&permutation, point2[0], point2[1]);
        assert_close(perlin2(&permutation, point2[0] + 256.0, point2[1]), base2);
        assert_close(perlin2(&permutation, point2[0], point2[1] - 256.0), base2);

        let base3 = perlin3(&permutation, point3[0], point3[1], point3[2]);
        assert_close(
            perlin3(&permutation, point3[0] + 256.0, point3[1], point3[2]),
            base3,
        );
        assert_close(
            perlin3(&permutation, point3[0], point3[1] - 256.0, point3[2]),
            base3,
        );
        assert_close(
            perlin3(&permutation, point3[0], point3[1], point3[2] + 256.0),
            base3,
        );
    }

    #[test]
    fn stable_seeded_fixtures_cover_negative_and_large_coordinates() {
        let permutation = Permutation::from_seed(0xdead_beef_cafe_babe);

        let fixtures2 = [
            ((0.25, 0.75), 0.076_884_269_714_355_47),
            ((-1.125, 2.5), -0.557_483_673_095_703_1),
            ((12_345.25, -6_789.75), -0.092_047_691_345_214_84),
            ((1_000_000_000_000.25, -999_999_999_999.5), -0.288_818_359_375),
        ];
        for ((x, y), expected) in fixtures2 {
            assert_close(perlin2(&permutation, x, y), expected);
        }

        let fixtures3 = [
            ((0.25, 0.75, 0.5), -0.013_315_677_642_822_266),
            ((-1.125, 2.5, -3.75), 0.600_070_789_456_367_5),
            ((123.25, -67.75, 0.125), 0.246_532_306_075_096_13),
            (
                (1_000_000_000_000.25, -999_999_999_999.5, 42.75),
                0.333_878_993_988_037_1,
            ),
        ];
        for ((x, y, z), expected) in fixtures3 {
            assert_close(perlin3(&permutation, x, y, z), expected);
        }
    }

    #[test]
    #[should_panic(expected = "Perlin coordinates must be finite")]
    fn perlin2_rejects_non_finite_coordinates() {
        let permutation = Permutation::default();
        let _ = perlin2(&permutation, f64::NAN, 0.0);
    }

    #[test]
    #[should_panic(expected = "Perlin coordinates must be finite")]
    fn perlin3_rejects_non_finite_coordinates() {
        let permutation = Permutation::default();
        let _ = perlin3(&permutation, 0.0, f64::INFINITY, 0.0);
    }
}
