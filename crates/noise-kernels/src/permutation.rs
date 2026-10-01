/// Deterministic 256-entry permutation shared by procedural noise kernels.
///
/// The table is generated once from an explicit seed. Sampling kernels borrow
/// the table and perform no allocation or mutation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Permutation {
    table: [u8; 256],
}

impl Permutation {
    /// Builds a deterministic permutation from a 64-bit seed.
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        let mut table = [0_u8; 256];
        for (index, value) in table.iter_mut().enumerate() {
            *value = index as u8;
        }

        let mut state = seed;
        for index in (1..table.len()).rev() {
            let random = splitmix64(&mut state);
            let swap_with = bounded_index(random, index + 1);
            table.swap(index, swap_with);
        }

        Self { table }
    }

    /// Returns the canonical 256-entry table for reproducibility and oracles.
    #[must_use]
    pub const fn table(&self) -> &[u8; 256] {
        &self.table
    }

    #[inline]
    pub(crate) fn hash2(&self, x: usize, y: usize) -> u8 {
        let y_hash = usize::from(self.table[y & 255]);
        self.table[(x + y_hash) & 255]
    }

    #[inline]
    pub(crate) fn hash3(&self, x: usize, y: usize, z: usize) -> u8 {
        let z_hash = usize::from(self.table[z & 255]);
        let yz_hash = usize::from(self.table[(y + z_hash) & 255]);
        self.table[(x + yz_hash) & 255]
    }

    #[inline]
    pub(crate) fn hash4(&self, x: usize, y: usize, z: usize, w: usize) -> u8 {
        let w_hash = usize::from(self.table[w & 255]);
        let zw_hash = usize::from(self.table[(z + w_hash) & 255]);
        let yzw_hash = usize::from(self.table[(y + zw_hash) & 255]);
        self.table[(x + yzw_hash) & 255]
    }
}

impl Default for Permutation {
    fn default() -> Self {
        Self::from_seed(0)
    }
}

#[inline]
fn bounded_index(random: u64, upper_exclusive: usize) -> usize {
    ((u128::from(random) * upper_exclusive as u128) >> 64) as usize
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::Permutation;

    #[test]
    fn same_seed_produces_same_table_and_different_seed_changes_it() {
        let first = Permutation::from_seed(0x1234_5678_9abc_def0);
        let second = Permutation::from_seed(0x1234_5678_9abc_def0);
        let different = Permutation::from_seed(0x1234_5678_9abc_def1);

        assert_eq!(first, second);
        assert_ne!(first, different);
    }

    #[test]
    fn generated_table_contains_every_byte_once() {
        let permutation = Permutation::from_seed(42);
        let mut seen = [false; 256];

        for &value in permutation.table() {
            let slot = &mut seen[usize::from(value)];
            assert!(!*slot, "permutation contains duplicate value {value}");
            *slot = true;
        }

        assert!(seen.into_iter().all(|value| value));
    }
}
