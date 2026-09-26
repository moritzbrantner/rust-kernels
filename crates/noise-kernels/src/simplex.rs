use crate::Permutation;

const F2: f64 = 0.366_025_403_784_438_65;
const G2: f64 = 0.211_324_865_405_187_13;
const F3: f64 = 1.0 / 3.0;
const G3: f64 = 1.0 / 6.0;
const F4: f64 = 0.309_016_994_374_947_45;
const G4: f64 = 0.138_196_601_125_010_5;
const LATTICE_PERIOD: f64 = 256.0;
// Above 2^40, direct skew/unskew subtraction starts losing meaningful local
// offsets even though the coordinates themselves are still finite.
const SKEW_REDUCTION_THRESHOLD: f64 = 1_099_511_627_776.0;

#[inline]
fn periodic_product(value: f64, factor: f64) -> f64 {
    let product = value * factor;
    let residual = value.mul_add(factor, -product);
    (product.rem_euclid(LATTICE_PERIOD) + residual.rem_euclid(LATTICE_PERIOD))
        .rem_euclid(LATTICE_PERIOD)
}

#[inline]
fn reduced_skew_lattice<const N: usize>(
    coordinates: [f64; N],
    skew_factor: f64,
    unskew_factor: f64,
) -> ([usize; N], [f64; N]) {
    let skew_terms = coordinates.map(|coordinate| periodic_product(coordinate, skew_factor));
    let mut cells = [0_usize; N];
    let mut fractions = [0.0; N];

    for axis in 0..N {
        let mut reduced = coordinates[axis].rem_euclid(LATTICE_PERIOD);
        for &term in &skew_terms {
            reduced = (reduced + term).rem_euclid(LATTICE_PERIOD);
        }
        let cell = reduced.floor();
        cells[axis] = cell as usize;
        fractions[axis] = reduced - cell;
    }

    let fraction_sum = fractions.into_iter().sum::<f64>();
    let origin = fractions.map(|fraction| fraction - fraction_sum * unskew_factor);
    (cells, origin)
}

#[inline]
fn needs_skew_reduction<const N: usize>(coordinates: [f64; N]) -> bool {
    coordinates
        .into_iter()
        .map(f64::abs)
        .fold(0.0, f64::max)
        > SKEW_REDUCTION_THRESHOLD
}

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

const GRADIENTS_4D: [[f64; 4]; 32] = [
    [0.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0, -1.0],
    [0.0, 1.0, -1.0, 1.0],
    [0.0, 1.0, -1.0, -1.0],
    [0.0, -1.0, 1.0, 1.0],
    [0.0, -1.0, 1.0, -1.0],
    [0.0, -1.0, -1.0, 1.0],
    [0.0, -1.0, -1.0, -1.0],
    [1.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, -1.0],
    [1.0, 0.0, -1.0, 1.0],
    [1.0, 0.0, -1.0, -1.0],
    [-1.0, 0.0, 1.0, 1.0],
    [-1.0, 0.0, 1.0, -1.0],
    [-1.0, 0.0, -1.0, 1.0],
    [-1.0, 0.0, -1.0, -1.0],
    [1.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, -1.0],
    [1.0, -1.0, 0.0, 1.0],
    [1.0, -1.0, 0.0, -1.0],
    [-1.0, 1.0, 0.0, 1.0],
    [-1.0, 1.0, 0.0, -1.0],
    [-1.0, -1.0, 0.0, 1.0],
    [-1.0, -1.0, 0.0, -1.0],
    [1.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, -1.0, 0.0],
    [1.0, -1.0, 1.0, 0.0],
    [1.0, -1.0, -1.0, 0.0],
    [-1.0, 1.0, 1.0, 0.0],
    [-1.0, 1.0, -1.0, 0.0],
    [-1.0, -1.0, 1.0, 0.0],
    [-1.0, -1.0, -1.0, 0.0],
];

/// Samples deterministic two-dimensional simplex noise.
///
/// Coordinates must remain finite. Large finite coordinates are reduced in the
/// periodic skew lattice before local offsets are formed, avoiding cancellation.
/// The hot sampling path performs no allocation or mutation.
#[must_use]
pub fn simplex2(permutation: &Permutation, x: f64, y: f64) -> f64 {
    assert!(
        x.is_finite() && y.is_finite(),
        "Simplex coordinates must be finite"
    );

    let (ii, jj, x0, y0) = if needs_skew_reduction([x, y]) {
        let (cells, origin) = reduced_skew_lattice([x, y], F2, G2);
        (cells[0], cells[1], origin[0], origin[1])
    } else {
        let sum = x + y;
        assert!(sum.is_finite(), "Simplex skew transform must remain finite");
        let skew = sum * F2;
        let skewed_x = x + skew;
        let skewed_y = y + skew;
        assert!(
            skewed_x.is_finite() && skewed_y.is_finite(),
            "Simplex skew transform must remain finite"
        );
        let i = skewed_x.floor();
        let j = skewed_y.floor();
        let lattice_sum = i + j;
        assert!(
            lattice_sum.is_finite(),
            "Simplex skew transform must remain finite"
        );
        let unskew = lattice_sum * G2;
        (
            lattice_index(i),
            lattice_index(j),
            x - i + unskew,
            y - j + unskew,
        )
    };

    let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };
    let x1 = x0 - i1 as f64 + G2;
    let y1 = y0 - j1 as f64 + G2;
    let x2 = x0 - 1.0 + 2.0 * G2;
    let y2 = y0 - 1.0 + 2.0 * G2;

    let n0 = contribution2(permutation.hash2(ii, jj), x0, y0);
    let n1 = contribution2(permutation.hash2(ii + i1, jj + j1), x1, y1);
    let n2 = contribution2(permutation.hash2(ii + 1, jj + 1), x2, y2);

    70.0 * (n0 + n1 + n2)
}

/// Samples deterministic three-dimensional simplex noise.
///
/// Coordinates must remain finite. Large finite coordinates are reduced in the
/// periodic skew lattice before local offsets are formed, avoiding cancellation.
/// The hot sampling path performs no allocation or mutation.
#[must_use]
pub fn simplex3(permutation: &Permutation, x: f64, y: f64, z: f64) -> f64 {
    assert!(
        x.is_finite() && y.is_finite() && z.is_finite(),
        "Simplex coordinates must be finite"
    );

    let (ii, jj, kk, x0, y0, z0) = if needs_skew_reduction([x, y, z]) {
        let (cells, origin) = reduced_skew_lattice([x, y, z], F3, G3);
        (
            cells[0], cells[1], cells[2], origin[0], origin[1], origin[2],
        )
    } else {
        let sum = x + y + z;
        assert!(sum.is_finite(), "Simplex skew transform must remain finite");
        let skew = sum * F3;
        let skewed_x = x + skew;
        let skewed_y = y + skew;
        let skewed_z = z + skew;
        assert!(
            skewed_x.is_finite() && skewed_y.is_finite() && skewed_z.is_finite(),
            "Simplex skew transform must remain finite"
        );
        let i = skewed_x.floor();
        let j = skewed_y.floor();
        let k = skewed_z.floor();
        let lattice_sum = i + j + k;
        assert!(
            lattice_sum.is_finite(),
            "Simplex skew transform must remain finite"
        );
        let unskew = lattice_sum * G3;
        (
            lattice_index(i),
            lattice_index(j),
            lattice_index(k),
            x - i + unskew,
            y - j + unskew,
            z - k + unskew,
        )
    };

    let ((i1, j1, k1), (i2, j2, k2)) = simplex3_corner_steps(x0, y0, z0);

    let x1 = x0 - i1 as f64 + G3;
    let y1 = y0 - j1 as f64 + G3;
    let z1 = z0 - k1 as f64 + G3;
    let x2 = x0 - i2 as f64 + 2.0 * G3;
    let y2 = y0 - j2 as f64 + 2.0 * G3;
    let z2 = z0 - k2 as f64 + 2.0 * G3;
    let x3 = x0 - 1.0 + 3.0 * G3;
    let y3 = y0 - 1.0 + 3.0 * G3;
    let z3 = z0 - 1.0 + 3.0 * G3;

    let n0 = contribution3(permutation.hash3(ii, jj, kk), x0, y0, z0);
    let n1 = contribution3(permutation.hash3(ii + i1, jj + j1, kk + k1), x1, y1, z1);
    let n2 = contribution3(permutation.hash3(ii + i2, jj + j2, kk + k2), x2, y2, z2);
    let n3 = contribution3(permutation.hash3(ii + 1, jj + 1, kk + 1), x3, y3, z3);

    32.0 * (n0 + n1 + n2 + n3)
}

/// Samples deterministic four-dimensional simplex noise.
///
/// The fourth coordinate is suitable for time-varying or looping procedural
/// fields. Coordinates must remain finite; large values use periodic skew-lattice
/// reduction before local offsets are formed.
#[must_use]
pub fn simplex4(permutation: &Permutation, x: f64, y: f64, z: f64, w: f64) -> f64 {
    assert!(
        x.is_finite() && y.is_finite() && z.is_finite() && w.is_finite(),
        "Simplex coordinates must be finite"
    );

    let (ii, jj, kk, ll, x0, y0, z0, w0) =
        if needs_skew_reduction([x, y, z, w]) {
            let (cells, origin) = reduced_skew_lattice([x, y, z, w], F4, G4);
            (
                cells[0], cells[1], cells[2], cells[3], origin[0], origin[1], origin[2], origin[3],
            )
        } else {
            let sum = x + y + z + w;
            assert!(sum.is_finite(), "Simplex skew transform must remain finite");
            let skew = sum * F4;
            let skewed_x = x + skew;
            let skewed_y = y + skew;
            let skewed_z = z + skew;
            let skewed_w = w + skew;
            assert!(
                skewed_x.is_finite()
                    && skewed_y.is_finite()
                    && skewed_z.is_finite()
                    && skewed_w.is_finite(),
                "Simplex skew transform must remain finite"
            );
            let i = skewed_x.floor();
            let j = skewed_y.floor();
            let k = skewed_z.floor();
            let l = skewed_w.floor();
            let lattice_sum = i + j + k + l;
            assert!(
                lattice_sum.is_finite(),
                "Simplex skew transform must remain finite"
            );
            let unskew = lattice_sum * G4;
            (
                lattice_index(i),
                lattice_index(j),
                lattice_index(k),
                lattice_index(l),
                x - i + unskew,
                y - j + unskew,
                z - k + unskew,
                w - l + unskew,
            )
        };

    let [rank_x, rank_y, rank_z, rank_w] = simplex4_ranks(x0, y0, z0, w0);
    let first = [
        step_for_rank(rank_x, 3),
        step_for_rank(rank_y, 3),
        step_for_rank(rank_z, 3),
        step_for_rank(rank_w, 3),
    ];
    let second = [
        step_for_rank(rank_x, 2),
        step_for_rank(rank_y, 2),
        step_for_rank(rank_z, 2),
        step_for_rank(rank_w, 2),
    ];
    let third = [
        step_for_rank(rank_x, 1),
        step_for_rank(rank_y, 1),
        step_for_rank(rank_z, 1),
        step_for_rank(rank_w, 1),
    ];

    let n0 = contribution4(permutation.hash4(ii, jj, kk, ll), x0, y0, z0, w0);
    let n1 = contribution4(
        permutation.hash4(ii + first[0], jj + first[1], kk + first[2], ll + first[3]),
        x0 - first[0] as f64 + G4,
        y0 - first[1] as f64 + G4,
        z0 - first[2] as f64 + G4,
        w0 - first[3] as f64 + G4,
    );
    let n2 = contribution4(
        permutation.hash4(
            ii + second[0],
            jj + second[1],
            kk + second[2],
            ll + second[3],
        ),
        x0 - second[0] as f64 + 2.0 * G4,
        y0 - second[1] as f64 + 2.0 * G4,
        z0 - second[2] as f64 + 2.0 * G4,
        w0 - second[3] as f64 + 2.0 * G4,
    );
    let n3 = contribution4(
        permutation.hash4(ii + third[0], jj + third[1], kk + third[2], ll + third[3]),
        x0 - third[0] as f64 + 3.0 * G4,
        y0 - third[1] as f64 + 3.0 * G4,
        z0 - third[2] as f64 + 3.0 * G4,
        w0 - third[3] as f64 + 3.0 * G4,
    );
    let n4 = contribution4(
        permutation.hash4(ii + 1, jj + 1, kk + 1, ll + 1),
        x0 - 1.0 + 4.0 * G4,
        y0 - 1.0 + 4.0 * G4,
        z0 - 1.0 + 4.0 * G4,
        w0 - 1.0 + 4.0 * G4,
    );

    27.0 * (n0 + n1 + n2 + n3 + n4)
}

#[inline]
fn simplex3_corner_steps(x: f64, y: f64, z: f64) -> ((usize, usize, usize), (usize, usize, usize)) {
    if x >= y {
        if y >= z {
            ((1, 0, 0), (1, 1, 0))
        } else if x >= z {
            ((1, 0, 0), (1, 0, 1))
        } else {
            ((0, 0, 1), (1, 0, 1))
        }
    } else if y < z {
        ((0, 0, 1), (0, 1, 1))
    } else if x < z {
        ((0, 1, 0), (0, 1, 1))
    } else {
        ((0, 1, 0), (1, 1, 0))
    }
}

#[inline]
fn simplex4_ranks(x: f64, y: f64, z: f64, w: f64) -> [u8; 4] {
    let mut ranks = [0_u8; 4];

    rank_pair(x, y, &mut ranks, 0, 1);
    rank_pair(x, z, &mut ranks, 0, 2);
    rank_pair(x, w, &mut ranks, 0, 3);
    rank_pair(y, z, &mut ranks, 1, 2);
    rank_pair(y, w, &mut ranks, 1, 3);
    rank_pair(z, w, &mut ranks, 2, 3);

    ranks
}

#[inline]
fn rank_pair(left: f64, right: f64, ranks: &mut [u8; 4], left_index: usize, right_index: usize) {
    if left > right {
        ranks[left_index] += 1;
    } else {
        ranks[right_index] += 1;
    }
}

#[inline]
fn step_for_rank(rank: u8, threshold: u8) -> usize {
    if rank >= threshold { 1 } else { 0 }
}

#[inline]
fn lattice_index(value: f64) -> usize {
    value.rem_euclid(256.0) as usize
}

#[inline]
fn contribution2(hash: u8, x: f64, y: f64) -> f64 {
    let attenuation = 0.5 - x * x - y * y;
    if attenuation <= 0.0 {
        return 0.0;
    }

    let squared = attenuation * attenuation;
    let gradient = GRADIENTS_3D[usize::from(hash) % GRADIENTS_3D.len()];
    squared * squared * (gradient[0] * x + gradient[1] * y)
}

#[inline]
fn contribution3(hash: u8, x: f64, y: f64, z: f64) -> f64 {
    let attenuation = 0.6 - x * x - y * y - z * z;
    if attenuation <= 0.0 {
        return 0.0;
    }

    let squared = attenuation * attenuation;
    let gradient = GRADIENTS_3D[usize::from(hash) % GRADIENTS_3D.len()];
    squared * squared * (gradient[0] * x + gradient[1] * y + gradient[2] * z)
}

#[inline]
fn contribution4(hash: u8, x: f64, y: f64, z: f64, w: f64) -> f64 {
    let attenuation = 0.6 - x * x - y * y - z * z - w * w;
    if attenuation <= 0.0 {
        return 0.0;
    }

    let squared = attenuation * attenuation;
    let gradient = GRADIENTS_4D[usize::from(hash) % GRADIENTS_4D.len()];
    squared * squared * (gradient[0] * x + gradient[1] * y + gradient[2] * z + gradient[3] * w)
}

#[cfg(test)]
mod tests {
    use super::{simplex2, simplex3, simplex4};
    use crate::Permutation;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn origin_is_zero_in_every_dimension() {
        let permutation = Permutation::from_seed(17);

        assert_close(simplex2(&permutation, 0.0, 0.0), 0.0);
        assert_close(simplex3(&permutation, 0.0, 0.0, 0.0), 0.0);
        assert_close(simplex4(&permutation, 0.0, 0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn stable_seeded_fixtures_cover_negative_and_large_coordinates() {
        let permutation = Permutation::from_seed(0xdead_beef_cafe_babe);

        let fixtures2 = [
            ((0.25, 0.75), 0.418_292_475_209_184_67),
            ((-1.125, 2.5), -0.114_673_213_353_439_8),
            ((12_345.25, -6_789.75), -0.886_326_184_974_423_1),
            (
                (1_000_000_000_000.25, -999_999_999_999.5),
                -0.130_403_632_877_460_68,
            ),
        ];
        for ((x, y), expected) in fixtures2 {
            assert_close(simplex2(&permutation, x, y), expected);
        }

        let fixtures3 = [
            ((0.25, 0.75, 0.5), -0.006_834_374_999_999_987),
            ((-1.125, 2.5, -3.75), -0.142_179_808_848_188_7),
            ((123.25, -67.75, 0.125), 0.017_322_885_031_368_664),
            (
                (1_000_000_000_000.25, -999_999_999_999.5, 42.75),
                0.345_743_750_000_004_73,
            ),
        ];
        for ((x, y, z), expected) in fixtures3 {
            assert_close(simplex3(&permutation, x, y, z), expected);
        }

        let fixtures4 = [
            ((0.25, 0.75, 0.5, 0.125), 0.290_693_094_118_438_13),
            ((-1.125, 2.5, -3.75, 0.625), -0.306_166_824_293_997_05),
            ((123.25, -67.75, 0.125, 9.5), 0.340_544_162_211_615_5),
            (
                (1_000_000_000_000.25, -999_999_999_999.5, 42.75, -17.125),
                -0.078_358_236_168_974_83,
            ),
        ];
        for ((x, y, z, w), expected) in fixtures4 {
            assert_close(simplex4(&permutation, x, y, z, w), expected);
        }
    }

    #[test]
    fn unbalanced_large_coordinates_preserve_local_simplex_detail() {
        let permutation = Permutation::from_seed(42);

        // Independently derived by reducing the exact f64 skew-lattice
        // coordinates modulo the 256-entry permutation period and sampling
        // the equivalent moderate-coordinate representatives.
        assert_close(simplex2(&permutation, 1.0e17, 0.25), 0.005_264_632_844_718_684);
        assert_close(
            simplex3(&permutation, 1.0e17, 0.25, -0.5),
            -0.038_847_381_465_779_415,
        );
        assert_close(
            simplex4(&permutation, 1.0e17, 0.25, -0.5, 0.125),
            0.397_704_364_220_138_2,
        );

        let shifted = simplex2(&permutation, 1.0e17, 0.75);
        assert!(
            (shifted - simplex2(&permutation, 1.0e17, 0.25)).abs() > 1.0e-12,
            "small representable axes must not be erased by a large coordinate"
        );
    }

    #[test]
    fn fourth_axis_changes_four_dimensional_field() {
        let permutation = Permutation::from_seed(9);
        let first = simplex4(&permutation, 0.125, -0.5, 2.25, 0.0);
        let second = simplex4(&permutation, 0.125, -0.5, 2.25, 0.5);

        assert!((first - second).abs() > 1e-12);
    }

    #[test]
    #[should_panic(expected = "Simplex coordinates must be finite")]
    fn rejects_non_finite_coordinates() {
        let permutation = Permutation::default();
        let _ = simplex3(&permutation, 0.0, f64::NAN, 0.0);
    }
}
