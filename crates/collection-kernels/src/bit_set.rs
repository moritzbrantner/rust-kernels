/// Fixed-universe bit set backed by packed `u64` words.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BitSet {
    words: Vec<u64>,
    len: usize,
    count: usize,
}

impl BitSet {
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(64)],
            len,
            count: 0,
        }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[must_use]
    pub const fn count_ones(&self) -> usize {
        self.count
    }

    #[must_use]
    pub fn contains(&self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }
        let (word, mask) = word_and_mask(index);
        self.words[word] & mask != 0
    }

    /// Sets a bit, returning whether the set changed.
    pub fn insert(&mut self, index: usize) -> bool {
        self.assert_index(index);
        let (word, mask) = word_and_mask(index);
        if self.words[word] & mask != 0 {
            return false;
        }
        self.words[word] |= mask;
        self.count += 1;
        true
    }

    /// Clears a bit, returning whether the set changed.
    pub fn remove(&mut self, index: usize) -> bool {
        self.assert_index(index);
        let (word, mask) = word_and_mask(index);
        if self.words[word] & mask == 0 {
            return false;
        }
        self.words[word] &= !mask;
        self.count -= 1;
        true
    }

    pub fn clear(&mut self) {
        self.words.fill(0);
        self.count = 0;
    }

    pub fn iter_ones(&self) -> BitSetIter<'_> {
        BitSetIter {
            words: &self.words,
            len: self.len,
            word_index: 0,
            current: self.words.first().copied().unwrap_or(0),
        }
    }

    fn assert_index(&self, index: usize) {
        assert!(
            index < self.len,
            "bit-set index {index} out of bounds for length {}",
            self.len
        );
    }
}

fn word_and_mask(index: usize) -> (usize, u64) {
    (index / 64, 1_u64 << (index % 64))
}

pub struct BitSetIter<'a> {
    words: &'a [u64],
    len: usize,
    word_index: usize,
    current: u64,
}

impl Iterator for BitSetIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            while self.current != 0 {
                let bit = self.current.trailing_zeros() as usize;
                self.current &= self.current - 1;
                let index = self.word_index * 64 + bit;
                if index < self.len {
                    return Some(index);
                }
            }

            self.word_index += 1;
            self.current = self.words.get(self.word_index).copied()?;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::BitSet;

    #[test]
    fn tracks_bits_across_word_boundaries() {
        let mut set = BitSet::new(130);
        for index in [0, 63, 64, 65, 129] {
            assert!(set.insert(index));
        }

        assert_eq!(set.count_ones(), 5);
        assert_eq!(
            set.iter_ones().collect::<Vec<_>>(),
            vec![0, 63, 64, 65, 129]
        );
        assert!(set.contains(64));
        assert!(!set.contains(130));
        assert!(set.remove(64));
        assert!(!set.remove(64));
        assert_eq!(set.count_ones(), 4);
    }

    #[test]
    fn matches_a_standard_set_oracle() {
        let operations = [
            (true, 2),
            (true, 70),
            (true, 127),
            (false, 2),
            (true, 5),
            (false, 99),
            (true, 70),
        ];
        let mut bit_set = BitSet::new(128);
        let mut oracle = BTreeSet::new();

        for (insert, index) in operations {
            if insert {
                assert_eq!(bit_set.insert(index), oracle.insert(index));
            } else {
                assert_eq!(bit_set.remove(index), oracle.remove(&index));
            }
            assert_eq!(bit_set.count_ones(), oracle.len());
            assert_eq!(
                bit_set.iter_ones().collect::<Vec<_>>(),
                oracle.iter().copied().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn all_small_subsets_match_a_standard_set_oracle() {
        for len in 0_usize..=8 {
            for mask in 0_usize..(1_usize << len) {
                let mut bit_set = BitSet::new(len);
                let mut oracle = BTreeSet::new();

                for index in 0..len {
                    if mask & (1_usize << index) != 0 {
                        assert!(bit_set.insert(index));
                        assert!(oracle.insert(index));
                    }
                }

                assert_eq!(bit_set.len(), len);
                assert_eq!(bit_set.is_empty(), oracle.is_empty());
                assert_eq!(bit_set.count_ones(), oracle.len());
                assert_eq!(
                    bit_set.iter_ones().collect::<Vec<_>>(),
                    oracle.iter().copied().collect::<Vec<_>>()
                );
                for index in 0..=len {
                    assert_eq!(bit_set.contains(index), oracle.contains(&index));
                }

                for index in (0..len).rev() {
                    assert_eq!(bit_set.remove(index), oracle.remove(&index));
                    assert_eq!(bit_set.count_ones(), oracle.len());
                    assert_eq!(
                        bit_set.iter_ones().collect::<Vec<_>>(),
                        oracle.iter().copied().collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    #[test]
    fn clear_keeps_capacity_semantics_and_removes_all_members() {
        let mut set = BitSet::new(80);
        assert!(set.insert(1));
        assert!(set.insert(79));
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.iter_ones().next(), None);
        assert!(set.insert(79));
        assert!(set.contains(79));
    }
}
