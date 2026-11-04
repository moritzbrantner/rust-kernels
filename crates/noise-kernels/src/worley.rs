//! Deterministic Worley (cellular) noise over a jittered unit-cell lattice.
//!
//! Every integer lattice cell owns exactly one feature point. Its position
//! inside the cell, and its [`WorleySample::feature_id`], are derived only from
//! the borrowed [`Permutation`] and the cell coordinates wrapped to the shared
//! 256-cell period, so there is no hidden global random state. A sample returns
//! the Euclidean distances to the nearest (`f1`) and second-nearest (`f2`)
//! feature points, the offset to the nearest one and its stable id. Turning
//! these into terrain, material or texture rules stays with the caller.
//!
//! The search first visits the `3^N` neighbourhood of the sample cell and then
//! widens one Chebyshev ring at a time only while a ring's conservative lower
//! distance bound can still beat the current `f2`, so results are exact rather
//! than the usual "nearest 3^N cells" approximation. Sampling uses only fixed
//! size stack state and performs no allocation.

use crate::Permutation;

const LATTICE_PERIOD: f64 = 256.0;
const POSITION_BITS: u32 = 21;
const POSITION_MASK: u64 = (1 << POSITION_BITS) - 1;
const POSITION_SCALE: f64 = 1.0 / (1_u64 << POSITION_BITS) as f64;
// Absolute slack applied to lattice lower bounds before pruning. Local offsets
// live in [0, 1) and candidate distances are a few cells at most, so this keeps
// pruning conservative against rounding without visiting extra rings in
// practice.
const PRUNE_SLACK: f64 = 1e-9;

/// Result of one Worley sample in `N` dimensions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorleySample<const N: usize> {
    /// Euclidean distance to the nearest feature point.
    pub f1: f64,
    /// Euclidean distance to the second-nearest feature point (`f2 >= f1`).
    pub f2: f64,
    /// Vector from the sample position to the nearest feature point.
    pub nearest: [f64; N],
    /// Stable id of the nearest feature point.
    ///
    /// Ids are distinct for distinct cells inside one 256-cell period and repeat
    /// with the field. When two feature points are exactly equidistant the one
    /// with the smaller id is reported as nearest.
    pub feature_id: u64,
}

/// Two-dimensional Worley sample.
pub type WorleySample2 = WorleySample<2>;
/// Three-dimensional Worley sample.
pub type WorleySample3 = WorleySample<3>;

/// Samples deterministic two-dimensional Worley noise.
///
/// Coordinates must be finite. The field repeats every 256 lattice cells on
/// each axis because feature points are derived from the shared permutation.
#[must_use]
pub fn worley2(permutation: &Permutation, x: f64, y: f64) -> WorleySample2 {
    assert!(
        x.is_finite() && y.is_finite(),
        "Worley coordinates must be finite"
    );
    sample(permutation, [x, y], first_pass2)
}

/// Samples deterministic three-dimensional Worley noise.
///
/// Coordinates must be finite. The field repeats every 256 lattice cells on
/// each axis because feature points are derived from the shared permutation.
#[must_use]
pub fn worley3(permutation: &Permutation, x: f64, y: f64, z: f64) -> WorleySample3 {
    assert!(
        x.is_finite() && y.is_finite() && z.is_finite(),
        "Worley coordinates must be finite"
    );
    sample(permutation, [x, y, z], first_pass3)
}

/// Returns the feature point of the 2D lattice cell `(cell_x, cell_y)`.
///
/// The position is in absolute lattice coordinates inside
/// `[cell, cell + 1)` on each axis; the id matches
/// [`WorleySample::feature_id`]. Intended for reproducibility, visualisation
/// and oracles.
#[must_use]
pub fn worley2_feature(permutation: &Permutation, cell_x: i64, cell_y: i64) -> ([f64; 2], u64) {
    absolute_feature(permutation, [cell_x, cell_y])
}

/// Returns the feature point of the 3D lattice cell `(cell_x, cell_y, cell_z)`.
///
/// See [`worley2_feature`].
#[must_use]
pub fn worley3_feature(
    permutation: &Permutation,
    cell_x: i64,
    cell_y: i64,
    cell_z: i64,
) -> ([f64; 3], u64) {
    absolute_feature(permutation, [cell_x, cell_y, cell_z])
}

fn absolute_feature<const N: usize>(permutation: &Permutation, cell: [i64; N]) -> ([f64; N], u64) {
    let wrapped = cell.map(|value| value.rem_euclid(256) as usize);
    let (jitter, id) = cell_feature(permutation, wrapped);
    let mut position = [0.0; N];
    for axis in 0..N {
        position[axis] = cell[axis] as f64 + jitter[axis];
    }
    (position, id)
}

/// Feature point jitter in `[0, 1)^N` and id for a cell wrapped to `0..256`.
#[inline]
fn cell_feature<const N: usize>(permutation: &Permutation, wrapped: [usize; N]) -> ([f64; N], u64) {
    feature_from_hashes(salted_hashes(permutation, wrapped), wrapped)
}

/// Two salted permutation chains, giving 16 seed-dependent bits per cell.
#[inline]
fn salted_hashes<const N: usize>(permutation: &Permutation, wrapped: [usize; N]) -> [usize; 2] {
    let table = permutation.table();
    let mut salted = [0_usize; 2];
    for (salt, value) in salted.iter_mut().enumerate() {
        let mut hash = usize::from(table[salt]);
        for &coordinate in wrapped.iter().rev() {
            hash = usize::from(table[(coordinate + hash) & 255]);
        }
        *value = hash;
    }
    salted
}

/// Completes [`cell_feature`] from the two salted chain hashes.
#[inline]
fn feature_from_hashes<const N: usize>(salted: [usize; 2], wrapped: [usize; N]) -> ([f64; N], u64) {
    // Packing the wrapped cell makes keys, and therefore ids, distinct for
    // distinct cells in one period; `mix64` is a bijection.
    let mut key = ((salted[0] as u64) << 24) | ((salted[1] as u64) << 32);
    for (axis, &coordinate) in wrapped.iter().enumerate() {
        key |= (coordinate as u64) << (8 * axis);
    }
    let id = mix64(key);

    let mut jitter = [0.0; N];
    for (axis, value) in jitter.iter_mut().enumerate() {
        let bits = (id >> (POSITION_BITS as usize * axis)) & POSITION_MASK;
        *value = bits as f64 * POSITION_SCALE;
    }
    (jitter, id)
}

#[inline]
fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct Nearest<const N: usize> {
    f1_squared: f64,
    f2_squared: f64,
    nearest: [f64; N],
    feature_id: u64,
}

impl<const N: usize> Nearest<N> {
    #[inline]
    fn insert(&mut self, distance_squared: f64, offset: [f64; N], feature_id: u64) {
        if distance_squared < self.f1_squared
            || (distance_squared == self.f1_squared && feature_id < self.feature_id)
        {
            self.f2_squared = self.f1_squared;
            self.f1_squared = distance_squared;
            self.nearest = offset;
            self.feature_id = feature_id;
        } else if distance_squared < self.f2_squared {
            self.f2_squared = distance_squared;
        }
    }
}

#[inline(always)]
fn sample<const N: usize>(
    permutation: &Permutation,
    coordinates: [f64; N],
    first_pass: impl FnOnce(&Permutation, [usize; N], [f64; N], &mut Nearest<N>),
) -> WorleySample<N> {
    let mut cell = [0_usize; N];
    let mut local = [0.0; N];
    for axis in 0..N {
        let floor = coordinates[axis].floor();
        cell[axis] = floor.rem_euclid(LATTICE_PERIOD) as usize;
        local[axis] = coordinates[axis] - floor;
    }

    let mut best = Nearest {
        f1_squared: f64::INFINITY,
        f2_squared: f64::INFINITY,
        nearest: [0.0; N],
        feature_id: u64::MAX,
    };

    first_pass(permutation, cell, local, &mut best);

    // Any cell on Chebyshev ring `r >= 2` lies at least `r - 1 + margin` away,
    // where `margin` is the distance from the sample to its own cell boundary.
    let margin = local
        .iter()
        .map(|&fraction| fraction.min(1.0 - fraction))
        .fold(1.0, f64::min);
    let mut ring = 2_isize;
    loop {
        let bound = ((ring - 1) as f64 + margin - PRUNE_SLACK).max(0.0);
        if bound * bound > best.f2_squared {
            break;
        }
        visit_ring(permutation, cell, local, ring, &mut best);
        ring += 1;
    }

    WorleySample {
        f1: best.f1_squared.sqrt(),
        f2: best.f2_squared.sqrt(),
        nearest: best.nearest,
        feature_id: best.feature_id,
    }
}

// Offsets of the first pass, centre first so the running `f2` tightens early
// and more of the outer cells are pruned.
const FIRST_PASS_OFFSETS: [isize; 3] = [0, -1, 1];

/// Conservative per-axis lower distance bounds for first-pass offsets, indexed
/// like [`FIRST_PASS_OFFSETS`].
#[inline(always)]
fn first_pass_gaps(fraction: f64) -> [f64; 3] {
    let below = (fraction - PRUNE_SLACK).max(0.0);
    let above = (1.0 - fraction - PRUNE_SLACK).max(0.0);
    [0.0, below * below, above * above]
}

#[inline(always)]
fn insert_feature<const N: usize>(
    salted: [usize; 2],
    wrapped: [usize; N],
    offset: [isize; N],
    local: [f64; N],
    best: &mut Nearest<N>,
) {
    let (jitter, feature_id) = feature_from_hashes(salted, wrapped);
    let mut delta = [0.0; N];
    let mut distance_squared = 0.0;
    for axis in 0..N {
        delta[axis] = (offset[axis] as f64 + jitter[axis]) - local[axis];
        distance_squared += delta[axis] * delta[axis];
    }
    best.insert(distance_squared, delta, feature_id);
}

/// Visits the 3x3 neighbourhood, hoisting the shared permutation-chain prefix
/// of each row.
fn first_pass2(
    permutation: &Permutation,
    cell: [usize; 2],
    local: [f64; 2],
    best: &mut Nearest<2>,
) {
    let table = permutation.table();
    let salts = [usize::from(table[0]), usize::from(table[1])];
    let gaps_x = first_pass_gaps(local[0]);
    let gaps_y = first_pass_gaps(local[1]);
    for (iy, &dy) in FIRST_PASS_OFFSETS.iter().enumerate() {
        let wy = ((cell[1] as isize + dy) & 255) as usize;
        let row = salts.map(|salt| usize::from(table[(wy + salt) & 255]));
        for (ix, &dx) in FIRST_PASS_OFFSETS.iter().enumerate() {
            if gaps_y[iy] + gaps_x[ix] > best.f2_squared {
                continue;
            }
            let wx = ((cell[0] as isize + dx) & 255) as usize;
            let salted = row.map(|prefix| usize::from(table[(wx + prefix) & 255]));
            insert_feature(salted, [wx, wy], [dx, dy], local, best);
        }
    }
}

/// Visits the 3x3x3 neighbourhood, hoisting shared permutation-chain prefixes.
fn first_pass3(
    permutation: &Permutation,
    cell: [usize; 3],
    local: [f64; 3],
    best: &mut Nearest<3>,
) {
    let table = permutation.table();
    let salts = [usize::from(table[0]), usize::from(table[1])];
    let gaps_x = first_pass_gaps(local[0]);
    let gaps_y = first_pass_gaps(local[1]);
    let gaps_z = first_pass_gaps(local[2]);
    for (iz, &dz) in FIRST_PASS_OFFSETS.iter().enumerate() {
        let wz = ((cell[2] as isize + dz) & 255) as usize;
        let slab = salts.map(|salt| usize::from(table[(wz + salt) & 255]));
        for (iy, &dy) in FIRST_PASS_OFFSETS.iter().enumerate() {
            let gap_yz = gaps_z[iz] + gaps_y[iy];
            if gap_yz > best.f2_squared {
                continue;
            }
            let wy = ((cell[1] as isize + dy) & 255) as usize;
            let row = slab.map(|prefix| usize::from(table[(wy + prefix) & 255]));
            for (ix, &dx) in FIRST_PASS_OFFSETS.iter().enumerate() {
                if gap_yz + gaps_x[ix] > best.f2_squared {
                    continue;
                }
                let wx = ((cell[0] as isize + dx) & 255) as usize;
                let salted = row.map(|prefix| usize::from(table[(wx + prefix) & 255]));
                insert_feature(salted, [wx, wy, wz], [dx, dy, dz], local, best);
            }
        }
    }
}

/// Visits every cell whose Chebyshev offset is exactly `radius`, skipping
/// cells whose lattice lower bound cannot beat the current second-nearest
/// distance.
#[inline(never)]
fn visit_ring<const N: usize>(
    permutation: &Permutation,
    cell: [usize; N],
    local: [f64; N],
    radius: isize,
    best: &mut Nearest<N>,
) {
    let mut offset = [-radius; N];
    loop {
        if offset.iter().any(|&delta| delta.abs() == radius) {
            visit_cell(permutation, cell, local, offset, best);
        }

        let mut axis = 0;
        loop {
            if axis == N {
                return;
            }
            offset[axis] += 1;
            if offset[axis] <= radius {
                break;
            }
            offset[axis] = -radius;
            axis += 1;
        }
    }
}

#[inline]
fn visit_cell<const N: usize>(
    permutation: &Permutation,
    cell: [usize; N],
    local: [f64; N],
    offset: [isize; N],
    best: &mut Nearest<N>,
) {
    let mut lower_squared = 0.0;
    for axis in 0..N {
        let gap = match offset[axis] {
            0 => 0.0,
            delta if delta > 0 => delta as f64 - local[axis],
            delta => local[axis] + (-delta - 1) as f64,
        };
        let gap = (gap - PRUNE_SLACK).max(0.0);
        lower_squared += gap * gap;
    }
    if lower_squared > best.f2_squared {
        return;
    }

    let mut wrapped = [0_usize; N];
    for axis in 0..N {
        wrapped[axis] = ((cell[axis] as isize + offset[axis]) & 255) as usize;
    }
    insert_feature(
        salted_hashes(permutation, wrapped),
        wrapped,
        offset,
        local,
        best,
    );
}

#[cfg(test)]
mod tests {
    use super::{worley2, worley2_feature, worley3, worley3_feature};
    use crate::Permutation;

    #[test]
    fn same_seed_and_coordinates_are_stable_and_seed_changes_the_field() {
        let first = Permutation::from_seed(7);
        let second = Permutation::from_seed(7);
        let other = Permutation::from_seed(8);

        let mut differs = false;
        for index in 0..64 {
            let x = f64::from(index) * 0.37 - 11.0;
            let y = f64::from(index) * -0.21 + 3.0;
            assert_eq!(worley2(&first, x, y), worley2(&second, x, y));
            assert_eq!(worley3(&first, x, y, 0.5), worley3(&second, x, y, 0.5));
            differs |= worley2(&first, x, y) != worley2(&other, x, y);
        }
        assert!(differs);
    }

    #[test]
    fn distances_are_ordered_and_nearest_offset_matches_f1() {
        let permutation = Permutation::from_seed(0x5eed);
        for index in 0..256 {
            let t = f64::from(index);
            let sample = worley2(&permutation, t * 0.173 - 20.0, t * 0.091 + 7.5);
            assert!(0.0 <= sample.f1 && sample.f1 <= sample.f2);
            let length = sample.nearest[0].hypot(sample.nearest[1]);
            assert!((length - sample.f1).abs() <= 1e-12);

            let sample = worley3(&permutation, t * 0.173, -t * 0.05, t * 0.311 - 9.0);
            assert!(0.0 <= sample.f1 && sample.f1 <= sample.f2);
            let length = sample.nearest.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((length - sample.f1).abs() <= 1e-12);
        }
    }

    #[test]
    fn sampling_at_a_feature_point_returns_zero_f1_and_its_id() {
        let permutation = Permutation::from_seed(99);
        let (point, id) = worley2_feature(&permutation, -3, 12);
        let sample = worley2(&permutation, point[0], point[1]);
        assert_eq!(sample.f1, 0.0);
        assert_eq!(sample.feature_id, id);

        let (point, id) = worley3_feature(&permutation, 5, -1, 200);
        let sample = worley3(&permutation, point[0], point[1], point[2]);
        assert_eq!(sample.f1, 0.0);
        assert_eq!(sample.feature_id, id);
    }

    #[test]
    fn features_stay_inside_their_cell_and_ids_are_unique_per_period() {
        let permutation = Permutation::from_seed(1234);
        let mut ids = std::collections::BTreeSet::new();
        for cell_x in 0..256 {
            for cell_y in 0..256 {
                let (point, id) = worley2_feature(&permutation, cell_x, cell_y);
                assert!((cell_x as f64..(cell_x + 1) as f64).contains(&point[0]));
                assert!((cell_y as f64..(cell_y + 1) as f64).contains(&point[1]));
                assert!(ids.insert(id), "duplicate feature id {id}");
            }
        }
    }

    #[test]
    fn fields_repeat_after_256_lattice_cells() {
        let permutation = Permutation::from_seed(3);
        let base2 = worley2(&permutation, -13.375, 28.625);
        let shifted2 = worley2(&permutation, -13.375 + 256.0, 28.625 - 512.0);
        assert_eq!(base2.feature_id, shifted2.feature_id);
        assert!((base2.f1 - shifted2.f1).abs() <= 1e-12);
        assert!((base2.f2 - shifted2.f2).abs() <= 1e-12);

        let base3 = worley3(&permutation, 17.25, -4.875, 91.5);
        let shifted3 = worley3(&permutation, 17.25, -4.875 + 256.0, 91.5 - 256.0);
        assert_eq!(base3.feature_id, shifted3.feature_id);
        assert!((base3.f1 - shifted3.f1).abs() <= 1e-12);
    }

    #[test]
    fn large_finite_coordinates_keep_local_detail() {
        let permutation = Permutation::from_seed(0xdead_beef);
        let base = 1_000_000_000_000.0;
        let near = worley2(&permutation, base + 0.25, -base + 0.5);
        let reduced = worley2(&permutation, 0.25 + base % 256.0, 0.5 - base % 256.0);
        assert_eq!(near.feature_id, reduced.feature_id);
        assert!((near.f1 - reduced.f1).abs() <= 1e-12);
        assert!((near.f2 - reduced.f2).abs() <= 1e-12);
    }

    #[test]
    #[should_panic(expected = "Worley coordinates must be finite")]
    fn worley2_rejects_non_finite_coordinates() {
        let _ = worley2(&Permutation::default(), f64::NAN, 0.0);
    }

    #[test]
    #[should_panic(expected = "Worley coordinates must be finite")]
    fn worley3_rejects_non_finite_coordinates() {
        let _ = worley3(&Permutation::default(), 0.0, f64::INFINITY, 0.0);
    }
}
