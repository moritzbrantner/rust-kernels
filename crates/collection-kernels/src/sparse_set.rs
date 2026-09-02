const ABSENT: usize = usize::MAX;

/// Integer-key set with O(1) membership, insertion, and removal.
///
/// Iteration follows dense storage order. Removal uses swap-remove, so deleting
/// an element can change the iteration position of the last dense element.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SparseSet {
    sparse: Vec<usize>,
    dense: Vec<usize>,
}

impl SparseSet {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            sparse: Vec::new(),
            dense: Vec::with_capacity(capacity),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    #[must_use]
    pub fn contains(&self, key: usize) -> bool {
        self.dense_index(key).is_some()
    }

    #[must_use]
    pub fn dense_index(&self, key: usize) -> Option<usize> {
        let index = self.sparse.get(key).copied()?;
        if index == ABSENT || self.dense.get(index).copied() != Some(key) {
            None
        } else {
            Some(index)
        }
    }

    /// Inserts `key`, returning whether the set changed.
    pub fn insert(&mut self, key: usize) -> bool {
        if self.contains(key) {
            return false;
        }

        self.ensure_sparse_slot(key);
        let index = self.dense.len();
        self.dense.push(key);
        self.sparse[key] = index;
        true
    }

    /// Removes `key`, returning whether the set changed.
    pub fn remove(&mut self, key: usize) -> bool {
        let Some(index) = self.dense_index(key) else {
            return false;
        };

        let last_index = self.dense.len() - 1;
        self.dense.swap_remove(index);
        self.sparse[key] = ABSENT;

        if index != last_index {
            let moved_key = self.dense[index];
            self.sparse[moved_key] = index;
        }

        true
    }

    pub fn clear(&mut self) {
        for &key in &self.dense {
            self.sparse[key] = ABSENT;
        }
        self.dense.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.dense.iter().copied()
    }

    fn ensure_sparse_slot(&mut self, key: usize) {
        let Some(required_len) = key.checked_add(1) else {
            panic!("sparse-set key must be less than usize::MAX");
        };
        if self.sparse.len() < required_len {
            self.sparse.resize(required_len, ABSENT);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::SparseSet;

    #[test]
    fn insert_contains_and_remove_are_constant_shape_operations() {
        let mut set = SparseSet::new();
        assert!(set.insert(7));
        assert!(set.insert(2));
        assert!(!set.insert(7));
        assert!(set.contains(7));
        assert!(set.contains(2));
        assert!(!set.contains(3));
        assert_eq!(set.len(), 2);

        assert!(set.remove(7));
        assert!(!set.remove(7));
        assert!(!set.contains(7));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn swap_remove_repairs_the_moved_dense_index() {
        let mut set = SparseSet::new();
        for key in [10, 20, 30, 40] {
            assert!(set.insert(key));
        }

        assert!(set.remove(20));
        assert_eq!(set.dense_index(40), Some(1));
        assert!(set.contains(10));
        assert!(set.contains(30));
        assert!(set.contains(40));
    }

    #[test]
    fn supports_sparse_large_keys_without_dense_padding() {
        let mut set = SparseSet::with_capacity(2);
        assert!(set.insert(10_000));
        assert!(set.insert(3));
        assert_eq!(set.len(), 2);
        assert_eq!(set.iter().collect::<Vec<_>>(), vec![10_000, 3]);
    }

    #[test]
    fn matches_a_standard_set_oracle_for_a_mixed_operation_sequence() {
        let operations = [
            (true, 5),
            (true, 1),
            (true, 9),
            (false, 1),
            (true, 3),
            (false, 7),
            (true, 5),
            (false, 9),
        ];
        let mut sparse = SparseSet::new();
        let mut oracle = BTreeSet::new();

        for (insert, key) in operations {
            if insert {
                assert_eq!(sparse.insert(key), oracle.insert(key));
            } else {
                assert_eq!(sparse.remove(key), oracle.remove(&key));
            }

            for candidate in 0..12 {
                assert_eq!(
                    sparse.contains(candidate),
                    oracle.contains(&candidate),
                    "membership mismatch for key {candidate}"
                );
            }
        }
    }

    #[test]
    fn clear_preserves_reusable_sparse_storage_semantics() {
        let mut set = SparseSet::new();
        assert!(set.insert(100));
        assert!(set.insert(2));
        set.clear();
        assert!(set.is_empty());
        assert!(!set.contains(100));
        assert!(set.insert(100));
        assert!(set.contains(100));
    }
}
