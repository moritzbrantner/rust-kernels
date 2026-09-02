use collection_kernels::BitSet;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const SECOND_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

/// Deterministic Bloom filter for byte-oriented keys.
///
/// `might_contain` can return false positives but never false negatives for
/// values inserted since construction or the last `clear`. Hashing is stable
/// across processes and intentionally non-cryptographic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BloomFilter {
    bits: BitSet,
    hash_functions: u32,
    insertions: u64,
}

impl BloomFilter {
    #[must_use]
    pub fn new(bit_count: usize, hash_functions: u32) -> Self {
        assert!(bit_count > 0, "Bloom filter must have at least one bit");
        assert!(
            hash_functions > 0,
            "Bloom filter must use at least one hash function"
        );
        Self {
            bits: BitSet::new(bit_count),
            hash_functions,
            insertions: 0,
        }
    }

    #[must_use]
    pub const fn bit_count(&self) -> usize {
        self.bits.len()
    }

    #[must_use]
    pub const fn hash_functions(&self) -> u32 {
        self.hash_functions
    }

    /// Number of insertion calls since construction or `clear`.
    #[must_use]
    pub const fn insertions(&self) -> u64 {
        self.insertions
    }

    pub fn insert(&mut self, value: impl AsRef<[u8]>) {
        let (first, second) = base_hashes(value.as_ref());
        for round in 0..self.hash_functions {
            let index = bloom_index(first, second, round, self.bits.len());
            let _ = self.bits.insert(index);
        }
        self.insertions = self.insertions.saturating_add(1);
    }

    #[must_use]
    pub fn might_contain(&self, value: impl AsRef<[u8]>) -> bool {
        let (first, second) = base_hashes(value.as_ref());
        (0..self.hash_functions).all(|round| {
            let index = bloom_index(first, second, round, self.bits.len());
            self.bits.contains(index)
        })
    }

    pub fn clear(&mut self) {
        self.bits.clear();
        self.insertions = 0;
    }
}

fn base_hashes(bytes: &[u8]) -> (u64, u64) {
    let first = fnv1a(bytes, 0);
    let second = fnv1a(bytes, SECOND_SEED) | 1;
    (first, second)
}

fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = FNV_OFFSET ^ seed;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn bloom_index(first: u64, second: u64, round: u32, bit_count: usize) -> usize {
    let combined = first.wrapping_add(u64::from(round).wrapping_mul(second));
    (combined % bit_count as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::BloomFilter;

    #[test]
    fn inserted_values_never_become_false_negatives() {
        let values = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        ];
        let mut filter = BloomFilter::new(512, 5);
        for value in values {
            filter.insert(value);
        }

        for value in values {
            assert!(filter.might_contain(value), "false negative for {value}");
        }
        assert_eq!(filter.insertions(), values.len() as u64);
    }

    #[test]
    fn tiny_filters_demonstrate_false_positive_semantics() {
        let mut filter = BloomFilter::new(1, 1);
        filter.insert("present");
        assert!(filter.might_contain("present"));
        assert!(filter.might_contain("never inserted"));
    }

    #[test]
    fn clear_removes_all_previous_evidence_and_resets_count() {
        let mut filter = BloomFilter::new(256, 4);
        filter.insert("alpha");
        filter.insert("beta");
        filter.clear();

        assert_eq!(filter.insertions(), 0);
        assert!(!filter.might_contain("alpha"));
        assert!(!filter.might_contain("beta"));
    }

    #[test]
    fn configuration_is_reported_exactly() {
        let filter = BloomFilter::new(1000, 7);
        assert_eq!(filter.bit_count(), 1000);
        assert_eq!(filter.hash_functions(), 7);
    }
}
