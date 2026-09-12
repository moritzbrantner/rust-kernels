const ABSENT: usize = usize::MAX;

/// Integer-key sparse map with dense value storage.
///
/// Lookup, insertion, and removal are O(1) for keys that fit in `usize`. Iteration follows dense
/// storage order. Removal uses swap-remove, so deleting an entry can change the iteration position
/// of the last dense entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseMap<T> {
    sparse: Vec<usize>,
    dense_keys: Vec<usize>,
    dense_values: Vec<T>,
}

impl<T> Default for SparseMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SparseMap<T> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense_keys: Vec::new(),
            dense_values: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            sparse: Vec::new(),
            dense_keys: Vec::with_capacity(capacity),
            dense_values: Vec::with_capacity(capacity),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.dense_values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dense_values.is_empty()
    }

    #[must_use]
    pub fn contains_key(&self, key: usize) -> bool {
        self.dense_index(key).is_some()
    }

    #[must_use]
    pub fn dense_index(&self, key: usize) -> Option<usize> {
        let index = self.sparse.get(key).copied()?;
        if index == ABSENT || self.dense_keys.get(index).copied() != Some(key) {
            None
        } else {
            Some(index)
        }
    }

    #[must_use]
    pub fn get(&self, key: usize) -> Option<&T> {
        let index = self.dense_index(key)?;
        self.dense_values.get(index)
    }

    pub fn get_mut(&mut self, key: usize) -> Option<&mut T> {
        let index = self.dense_index(key)?;
        self.dense_values.get_mut(index)
    }

    /// Inserts `value` for `key`, returning the previous value when the key already existed.
    pub fn insert(&mut self, key: usize, value: T) -> Option<T> {
        if let Some(index) = self.dense_index(key) {
            return Some(std::mem::replace(&mut self.dense_values[index], value));
        }

        self.ensure_sparse_slot(key);
        let index = self.dense_values.len();
        self.dense_keys.push(key);
        self.dense_values.push(value);
        self.sparse[key] = index;
        None
    }

    /// Removes `key`, returning the stored value when present.
    pub fn remove(&mut self, key: usize) -> Option<T> {
        let index = self.dense_index(key)?;
        let last_index = self.dense_values.len() - 1;

        self.dense_keys.swap_remove(index);
        let removed = self.dense_values.swap_remove(index);
        self.sparse[key] = ABSENT;

        if index != last_index {
            let moved_key = self.dense_keys[index];
            self.sparse[moved_key] = index;
        }

        Some(removed)
    }

    pub fn clear(&mut self) {
        for &key in &self.dense_keys {
            self.sparse[key] = ABSENT;
        }
        self.dense_keys.clear();
        self.dense_values.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> + '_ {
        self.dense_keys
            .iter()
            .copied()
            .zip(self.dense_values.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut T)> + '_ {
        self.dense_keys
            .iter()
            .copied()
            .zip(self.dense_values.iter_mut())
    }

    pub fn keys(&self) -> impl Iterator<Item = usize> + '_ {
        self.dense_keys.iter().copied()
    }

    pub fn values(&self) -> impl Iterator<Item = &T> + '_ {
        self.dense_values.iter()
    }

    fn ensure_sparse_slot(&mut self, key: usize) {
        let Some(required_len) = key.checked_add(1) else {
            panic!("sparse-map key must be less than usize::MAX");
        };
        if self.sparse.len() < required_len {
            self.sparse.resize(required_len, ABSENT);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::SparseMap;

    #[test]
    fn insert_get_replace_and_remove_preserve_values() {
        let mut map = SparseMap::new();
        assert_eq!(map.insert(7, "archer"), None);
        assert_eq!(map.insert(2, "tower"), None);
        assert_eq!(map.get(7), Some(&"archer"));
        assert_eq!(map.insert(7, "raider"), Some("archer"));
        assert_eq!(map.get(7), Some(&"raider"));
        assert_eq!(map.remove(7), Some("raider"));
        assert_eq!(map.remove(7), None);
        assert_eq!(map.get(7), None);
        assert_eq!(map.get(2), Some(&"tower"));
    }

    #[test]
    fn swap_remove_repairs_moved_entry_lookup() {
        let mut map = SparseMap::new();
        for key in [10, 20, 30, 40] {
            assert_eq!(map.insert(key, key * 2), None);
        }

        assert_eq!(map.remove(20), Some(40));
        assert_eq!(map.dense_index(40), Some(1));
        assert_eq!(map.get(40), Some(&80));
        assert_eq!(map.get(30), Some(&60));
    }

    #[test]
    fn mutable_iteration_updates_dense_values_without_losing_keys() {
        let mut map = SparseMap::new();
        assert_eq!(map.insert(4, 10), None);
        assert_eq!(map.insert(9, 20), None);

        for (key, value) in map.iter_mut() {
            *value += key;
        }

        assert_eq!(map.get(4), Some(&14));
        assert_eq!(map.get(9), Some(&29));
    }

    #[test]
    fn supports_sparse_large_keys_without_dense_value_padding() {
        let mut map = SparseMap::with_capacity(2);
        assert_eq!(map.insert(10_000, 1), None);
        assert_eq!(map.insert(3, 2), None);
        assert_eq!(map.len(), 2);
        assert_eq!(
            map.iter()
                .map(|(key, value)| (key, *value))
                .collect::<Vec<_>>(),
            vec![(10_000, 1), (3, 2)]
        );
    }

    #[test]
    fn mixed_operations_match_a_standard_map_oracle() {
        let operations = [
            (true, 5, 50),
            (true, 1, 10),
            (true, 9, 90),
            (false, 1, 0),
            (true, 3, 30),
            (true, 5, 55),
            (false, 9, 0),
        ];
        let mut sparse = SparseMap::new();
        let mut oracle = BTreeMap::new();

        for (insert, key, value) in operations {
            if insert {
                assert_eq!(sparse.insert(key, value), oracle.insert(key, value));
            } else {
                assert_eq!(sparse.remove(key), oracle.remove(&key));
            }

            for candidate in 0..12 {
                assert_eq!(sparse.get(candidate), oracle.get(&candidate));
            }
        }
    }

    #[test]
    fn clear_preserves_reusable_sparse_storage_semantics() {
        let mut map = SparseMap::new();
        assert_eq!(map.insert(100, 7), None);
        assert_eq!(map.insert(2, 9), None);
        map.clear();
        assert!(map.is_empty());
        assert!(!map.contains_key(100));
        assert_eq!(map.insert(100, 11), None);
        assert_eq!(map.get(100), Some(&11));
    }
}
