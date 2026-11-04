//! Brute-force Worley oracle.
//!
//! The oracle re-derives every feature point independently from the public
//! permutation table and scans a fixed neighbourhood that is provably wide
//! enough: with one feature per unit cell, `f2` is at most the distance to a
//! feature in an adjacent cell (`2 * sqrt(N)`), so offsets up to 4 cover every
//! candidate in 2D and 3D.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration tests are not production code; a panic is the intended failure report"
)]

use noise_kernels::{
    Permutation, WorleySample, worley2, worley2_feature, worley3, worley3_feature,
};

const ORACLE_RADIUS: i64 = 4;

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Independent restatement of the documented feature derivation.
fn reference_feature(table: &[u8; 256], cell: &[i64]) -> (Vec<f64>, u64) {
    let wrapped: Vec<usize> = cell.iter().map(|&c| c.rem_euclid(256) as usize).collect();
    let chain = |salt: usize| {
        let mut hash = usize::from(table[salt]);
        for &coordinate in wrapped.iter().rev() {
            hash = usize::from(table[(coordinate + hash) % 256]);
        }
        hash as u64
    };
    let mut key = (chain(0) << 24) | (chain(1) << 32);
    for (axis, &coordinate) in wrapped.iter().enumerate() {
        key |= (coordinate as u64) << (8 * axis);
    }
    let id = mix64(key);
    let jitter = (0..cell.len())
        .map(|axis| ((id >> (21 * axis)) & ((1 << 21) - 1)) as f64 / f64::from(1_u32 << 21))
        .collect();
    (jitter, id)
}

struct Oracle {
    f1_squared: f64,
    f2_squared: f64,
    nearest: Vec<f64>,
    feature_id: u64,
    /// Largest Chebyshev cell offset among the two nearest features.
    reach: i64,
}

fn brute_force(permutation: &Permutation, coordinates: &[f64]) -> Oracle {
    let table = permutation.table();
    let dimensions = coordinates.len();
    let floor: Vec<f64> = coordinates.iter().map(|c| c.floor()).collect();
    let local: Vec<f64> = coordinates.iter().zip(&floor).map(|(c, f)| c - f).collect();
    let base: Vec<i64> = floor.iter().map(|f| f.rem_euclid(256.0) as i64).collect();

    let side = (2 * ORACLE_RADIUS + 1) as usize;
    let mut candidates = Vec::new();
    for index in 0..side.pow(dimensions as u32) {
        let mut rest = index;
        let mut offset = Vec::with_capacity(dimensions);
        for _ in 0..dimensions {
            offset.push((rest % side) as i64 - ORACLE_RADIUS);
            rest /= side;
        }
        let cell: Vec<i64> = base.iter().zip(&offset).map(|(b, o)| b + o).collect();
        let (jitter, id) = reference_feature(table, &cell);
        let delta: Vec<f64> = (0..dimensions)
            .map(|axis| (offset[axis] as f64 + jitter[axis]) - local[axis])
            .collect();
        let distance_squared = delta.iter().map(|d| d * d).sum::<f64>();
        let reach = offset.iter().map(|o| o.abs()).max().unwrap_or(0);
        candidates.push((distance_squared, id, delta, reach));
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

    Oracle {
        f1_squared: candidates[0].0,
        f2_squared: candidates[1].0,
        nearest: candidates[0].2.clone(),
        feature_id: candidates[0].1,
        reach: candidates[0].3.max(candidates[1].3),
    }
}

fn assert_matches<const N: usize>(sample: WorleySample<N>, oracle: &Oracle, context: &str) {
    assert_eq!(
        sample.feature_id, oracle.feature_id,
        "{context}: nearest id"
    );
    assert_eq!(sample.f1, oracle.f1_squared.sqrt(), "{context}: f1");
    assert_eq!(sample.f2, oracle.f2_squared.sqrt(), "{context}: f2");
    assert_eq!(
        sample.nearest.as_slice(),
        oracle.nearest.as_slice(),
        "{context}: nearest"
    );
}

/// Deterministic coordinate generator covering negative, boundary-adjacent and
/// large finite inputs.
fn coordinate(index: u64, salt: u64) -> f64 {
    let mixed = mix64(index.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ salt);
    let magnitude = match mixed % 8 {
        0 => 1_000_000_000.0,
        1 => 1_000_000_000_000.0,
        2 => 1.0,
        _ => 300.0,
    };
    let unit = (mixed >> 11) as f64 / (1_u64 << 53) as f64;
    let value = (unit * 2.0 - 1.0) * magnitude;
    match (mixed >> 3) % 16 {
        // Exactly on, or one ulp around, a cell boundary.
        0 => value.round(),
        1 => value.round() + 1e-9,
        // Near a cell corner, where the nearest features are most often
        // outside the 3^N neighbourhood.
        2..=9 => value.floor() + (unit - 0.5) * 0.08,
        _ => value,
    }
}

#[test]
fn worley2_matches_brute_force_neighbourhood_oracle() {
    let mut beyond_first_ring = 0;
    for seed in [0, 1, 0x1234_5678_9abc_def0, u64::MAX] {
        let permutation = Permutation::from_seed(seed);
        for index in 0..2_000 {
            let x = coordinate(index, 0xa1 ^ seed);
            let y = coordinate(index, 0xb2 ^ seed);
            let oracle = brute_force(&permutation, &[x, y]);
            beyond_first_ring += usize::from(oracle.reach > 1);
            assert_matches(
                worley2(&permutation, x, y),
                &oracle,
                &format!("seed {seed} ({x}, {y})"),
            );
        }
    }
    // The fixtures must exercise the exact ring expansion, where a plain 3x3
    // search would report a wrong nearest feature or f2.
    assert!(beyond_first_ring > 0, "no 2D fixture needed ring 2");
}

#[test]
fn worley3_matches_brute_force_neighbourhood_oracle() {
    for seed in [0, 7, 0xdead_beef_cafe_babe] {
        let permutation = Permutation::from_seed(seed);
        for index in 0..600 {
            let x = coordinate(index, 0xc3 ^ seed);
            let y = coordinate(index, 0xd4 ^ seed);
            let z = coordinate(index, 0xe5 ^ seed);
            let oracle = brute_force(&permutation, &[x, y, z]);
            assert_matches(
                worley3(&permutation, x, y, z),
                &oracle,
                &format!("seed {seed} ({x}, {y}, {z})"),
            );
        }
    }
}

/// Samples just inside lattice corners, where every feature of the `3^N`
/// neighbourhood can be farther than a feature two cells away. Samples whose
/// optimized `f2` is below `0.95` cannot involve ring 2 (every ring-2 cell is
/// at least `1 - 1e-6` away), so only the rest pay for the brute-force oracle.
fn corner_scan<const N: usize>(
    permutation: &Permutation,
    cells: i64,
    sample: impl Fn([f64; N]) -> WorleySample<N>,
) -> (usize, usize) {
    let mut checked = 0;
    let mut beyond_first_ring = 0;
    for index in 0..cells.pow(N as u32) {
        let mut rest = index;
        let mut point = [0.0; N];
        for (axis, value) in point.iter_mut().enumerate() {
            let sign = if (index >> axis) & 1 == 0 {
                1e-6
            } else {
                -1e-6
            };
            *value = (rest % cells) as f64 + sign;
            rest /= cells;
        }
        let result = sample(point);
        if result.f2 < 0.95 {
            continue;
        }
        let oracle = brute_force(permutation, &point);
        beyond_first_ring += usize::from(oracle.reach > 1);
        checked += 1;
        assert_matches(result, &oracle, &format!("corner {point:?}"));
    }
    (checked, beyond_first_ring)
}

#[test]
fn ring_expansion_matches_oracle_at_adversarial_corners() {
    let permutation = Permutation::from_seed(0x0bad_cafe);
    let (checked2, beyond2) = corner_scan(&permutation, 160, |[x, y]| worley2(&permutation, x, y));
    let (checked3, beyond3) =
        corner_scan(&permutation, 40, |[x, y, z]| worley3(&permutation, x, y, z));
    println!("2D: {beyond2}/{checked2} checked corners need ring 2+");
    println!("3D: {beyond3}/{checked3} checked corners need ring 2+");
    assert!(beyond2 >= 30, "only {beyond2} 2D corners needed ring 2");
    assert!(beyond3 >= 30, "only {beyond3} 3D corners needed ring 2");
}

#[test]
fn public_feature_points_match_reference_derivation() {
    let permutation = Permutation::from_seed(42);
    let table = permutation.table();
    for cell_x in -300..300 {
        for cell_y in [-1_000_000_007, -1, 0, 255, 256, 9_999] {
            let (position, id) = worley2_feature(&permutation, cell_x, cell_y);
            let (jitter, expected_id) = reference_feature(table, &[cell_x, cell_y]);
            assert_eq!(id, expected_id);
            assert_eq!(
                position,
                [cell_x as f64 + jitter[0], cell_y as f64 + jitter[1]]
            );

            let (position, id) = worley3_feature(&permutation, cell_y, cell_x, 3);
            let (jitter, expected_id) = reference_feature(table, &[cell_y, cell_x, 3]);
            assert_eq!(id, expected_id);
            assert_eq!(
                position,
                [
                    cell_y as f64 + jitter[0],
                    cell_x as f64 + jitter[1],
                    3.0 + jitter[2]
                ]
            );
        }
    }
}

#[test]
fn stable_seeded_fixtures() {
    let permutation = Permutation::from_seed(0xdead_beef_cafe_babe);
    let close = |actual: f64, expected: f64| {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "expected {expected}, got {actual}"
        );
    };

    let fixtures2 = [
        (
            (0.25, 0.75),
            0.287_168_284_324_339_05,
            1.027_044_881_141_319_8,
            0x8430_2f4c_98d0_df46,
        ),
        (
            (-1.125, 2.5),
            0.297_183_650_288_856_3,
            0.639_483_376_684_801_8,
            0x48d5_7514_b042_07bf,
        ),
        (
            (12_345.25, -6_789.75),
            0.570_943_037_051_566_2,
            0.640_793_563_361_802_7,
            0x72b7_cf43_2b1c_24e6,
        ),
    ];
    for ((x, y), f1, f2, id) in fixtures2 {
        let sample = worley2(&permutation, x, y);
        close(sample.f1, f1);
        close(sample.f2, f2);
        assert_eq!(sample.feature_id, id);
    }

    let fixtures3 = [
        (
            (0.25, 0.75, 0.5),
            0.131_023_571_247_805_34,
            0.640_984_950_199_195_9,
            0x4baf_2f57_c929_3bb5,
        ),
        (
            (-1.125, 2.5, -3.75),
            0.726_243_940_065_198_7,
            0.740_872_255_673_79,
            0xf734_33db_048d_5011,
        ),
        (
            (123.25, -67.75, 0.125),
            0.280_304_736_032_132,
            0.575_757_181_827_954_4,
            0x0b13_bdd0_17a1_f21b,
        ),
    ];
    for ((x, y, z), f1, f2, id) in fixtures3 {
        let sample = worley3(&permutation, x, y, z);
        close(sample.f1, f1);
        close(sample.f2, f2);
        assert_eq!(sample.feature_id, id);
    }
}
