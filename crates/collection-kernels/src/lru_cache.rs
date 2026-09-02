use std::collections::HashMap;
use std::hash::Hash;

#[derive(Clone, Debug)]
struct Node<K, V> {
    key: K,
    value: V,
    previous: Option<usize>,
    next: Option<usize>,
}

/// Fixed-capacity least-recently-used cache.
///
/// Lookup, insert/update, removal, and recency promotion are O(1) expected.
/// Iteration runs from most-recently-used to least-recently-used.
#[derive(Clone, Debug)]
pub struct LruCache<K, V> {
    capacity: usize,
    indices: HashMap<K, usize>,
    nodes: Vec<Option<Node<K, V>>>,
    free: Vec<usize>,
    most_recent: Option<usize>,
    least_recent: Option<usize>,
}

impl<K, V> LruCache<K, V>
where
    K: Clone + Eq + Hash,
{
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            indices: HashMap::with_capacity(capacity),
            nodes: Vec::with_capacity(capacity),
            free: Vec::new(),
            most_recent: None,
            least_recent: None,
        }
    }

    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.indices.contains_key(key)
    }

    /// Reads without changing recency.
    #[must_use]
    pub fn peek(&self, key: &K) -> Option<&V> {
        let index = *self.indices.get(key)?;
        self.nodes[index].as_ref().map(|node| &node.value)
    }

    /// Reads and promotes the entry to most-recently-used.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let index = *self.indices.get(key)?;
        self.promote(index);
        self.nodes[index].as_ref().map(|node| &node.value)
    }

    /// Mutably reads and promotes the entry to most-recently-used.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let index = *self.indices.get(key)?;
        self.promote(index);
        self.nodes[index].as_mut().map(|node| &mut node.value)
    }

    /// Inserts or updates an entry.
    ///
    /// Updating returns the previous value. Inserting at full capacity evicts
    /// the least-recently-used entry. Capacity zero accepts no entries.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.capacity == 0 {
            return None;
        }

        if let Some(&index) = self.indices.get(&key) {
            self.promote(index);
            let node = self.nodes[index]
                .as_mut()
                .expect("LRU index must be occupied");
            return Some(std::mem::replace(&mut node.value, value));
        }

        if self.len() == self.capacity {
            self.evict_least_recent();
        }

        let index = if let Some(index) = self.free.pop() {
            index
        } else {
            self.nodes.push(None);
            self.nodes.len() - 1
        };

        self.nodes[index] = Some(Node {
            key: key.clone(),
            value,
            previous: None,
            next: None,
        });
        self.indices.insert(key, index);
        self.attach_front(index);
        None
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let index = self.indices.remove(key)?;
        self.detach(index);
        let node = self.nodes[index]
            .take()
            .expect("LRU index must be occupied");
        self.free.push(index);
        Some(node.value)
    }

    pub fn clear(&mut self) {
        self.indices.clear();
        self.nodes.clear();
        self.free.clear();
        self.most_recent = None;
        self.least_recent = None;
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        LruIter {
            nodes: &self.nodes,
            current: self.most_recent,
        }
    }

    fn promote(&mut self, index: usize) {
        if self.most_recent == Some(index) {
            return;
        }
        self.detach(index);
        self.attach_front(index);
    }

    fn attach_front(&mut self, index: usize) {
        let old_front = self.most_recent;
        {
            let node = self.nodes[index]
                .as_mut()
                .expect("LRU index must be occupied");
            node.previous = None;
            node.next = old_front;
        }

        if let Some(old_front) = old_front {
            self.nodes[old_front]
                .as_mut()
                .expect("LRU index must be occupied")
                .previous = Some(index);
        } else {
            self.least_recent = Some(index);
        }
        self.most_recent = Some(index);
    }

    fn detach(&mut self, index: usize) {
        let (previous, next) = {
            let node = self.nodes[index]
                .as_ref()
                .expect("LRU index must be occupied");
            (node.previous, node.next)
        };

        if let Some(previous) = previous {
            self.nodes[previous]
                .as_mut()
                .expect("LRU index must be occupied")
                .next = next;
        } else {
            self.most_recent = next;
        }

        if let Some(next) = next {
            self.nodes[next]
                .as_mut()
                .expect("LRU index must be occupied")
                .previous = previous;
        } else {
            self.least_recent = previous;
        }

        let node = self.nodes[index]
            .as_mut()
            .expect("LRU index must be occupied");
        node.previous = None;
        node.next = None;
    }

    fn evict_least_recent(&mut self) {
        let Some(index) = self.least_recent else {
            return;
        };
        self.detach(index);
        let node = self.nodes[index]
            .take()
            .expect("LRU index must be occupied");
        self.indices.remove(&node.key);
        self.free.push(index);
    }
}

struct LruIter<'a, K, V> {
    nodes: &'a [Option<Node<K, V>>],
    current: Option<usize>,
}

impl<'a, K, V> Iterator for LruIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.current?;
        let node = self.nodes[index].as_ref()?;
        self.current = node.next;
        Some((&node.key, &node.value))
    }
}

#[cfg(test)]
mod tests {
    use super::LruCache;

    #[derive(Default)]
    struct Model {
        capacity: usize,
        entries: Vec<(u8, i32)>,
    }

    impl Model {
        fn new(capacity: usize) -> Self {
            Self {
                capacity,
                entries: Vec::new(),
            }
        }

        fn insert(&mut self, key: u8, value: i32) -> Option<i32> {
            if self.capacity == 0 {
                return None;
            }
            if let Some(index) = self.entries.iter().position(|entry| entry.0 == key) {
                let old = self.entries.remove(index).1;
                self.entries.insert(0, (key, value));
                return Some(old);
            }
            self.entries.insert(0, (key, value));
            self.entries.truncate(self.capacity);
            None
        }

        fn get(&mut self, key: u8) -> Option<i32> {
            let index = self.entries.iter().position(|entry| entry.0 == key)?;
            let entry = self.entries.remove(index);
            let value = entry.1;
            self.entries.insert(0, entry);
            Some(value)
        }

        fn remove(&mut self, key: u8) -> Option<i32> {
            let index = self.entries.iter().position(|entry| entry.0 == key)?;
            Some(self.entries.remove(index).1)
        }
    }

    fn snapshot(cache: &LruCache<u8, i32>) -> Vec<(u8, i32)> {
        cache.iter().map(|(key, value)| (*key, *value)).collect()
    }

    fn assert_matches_model(cache: &LruCache<u8, i32>, model: &Model) {
        assert_eq!(snapshot(cache), model.entries);
        assert_eq!(cache.len(), model.entries.len());
        assert_eq!(cache.is_empty(), model.entries.is_empty());
        assert_eq!(cache.capacity(), model.capacity);

        for key in 0_u8..=1 {
            let expected = model
                .entries
                .iter()
                .find(|entry| entry.0 == key)
                .map(|entry| entry.1);
            assert_eq!(cache.contains_key(&key), expected.is_some());
            assert_eq!(cache.peek(&key).copied(), expected);
        }
    }

    #[test]
    fn eviction_and_promotion_follow_lru_order() {
        let mut cache = LruCache::new(3);
        assert_eq!(cache.insert(1, 10), None);
        assert_eq!(cache.insert(2, 20), None);
        assert_eq!(cache.insert(3, 30), None);
        assert_eq!(snapshot(&cache), vec![(3, 30), (2, 20), (1, 10)]);

        assert_eq!(cache.get(&1), Some(&10));
        assert_eq!(snapshot(&cache), vec![(1, 10), (3, 30), (2, 20)]);

        assert_eq!(cache.insert(4, 40), None);
        assert!(!cache.contains_key(&2));
        assert_eq!(snapshot(&cache), vec![(4, 40), (1, 10), (3, 30)]);
    }

    #[test]
    fn updates_and_mutable_reads_promote_without_growing() {
        let mut cache = LruCache::new(2);
        assert_eq!(cache.insert("a", 1), None);
        assert_eq!(cache.insert("b", 2), None);
        assert_eq!(cache.insert("a", 3), Some(1));
        assert_eq!(cache.len(), 2);
        *cache.get_mut(&"b").expect("entry exists") += 10;
        assert_eq!(cache.peek(&"b"), Some(&12));
        assert_eq!(
            cache.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
            vec!["b", "a"]
        );
    }

    #[test]
    fn mixed_operations_match_a_simple_reference_model() {
        let mut cache = LruCache::new(4);
        let mut model = Model::new(4);

        for step in 0..80_u8 {
            let key = (step.wrapping_mul(7).wrapping_add(3)) % 9;
            match step % 4 {
                0 | 1 => {
                    let value = i32::from(step) * 11 - 17;
                    assert_eq!(cache.insert(key, value), model.insert(key, value));
                }
                2 => {
                    assert_eq!(cache.get(&key).copied(), model.get(key));
                }
                _ => {
                    assert_eq!(cache.remove(&key), model.remove(key));
                }
            }
            assert_eq!(snapshot(&cache), model.entries);
            assert_eq!(cache.len(), model.entries.len());
        }
    }

    #[test]
    fn exhaustive_short_sequences_match_reference_model() {
        for capacity in 0_usize..=3 {
            for case in 0_usize..6_usize.pow(5) {
                let mut cache = LruCache::new(capacity);
                let mut model = Model::new(capacity);
                let mut encoded = case;

                for step in 0_usize..5 {
                    let action = encoded % 6;
                    encoded /= 6;
                    let key = (action % 2) as u8;

                    match action / 2 {
                        0 => {
                            let value = (case * 7 + step) as i32;
                            assert_eq!(cache.insert(key, value), model.insert(key, value));
                        }
                        1 => assert_eq!(cache.get(&key).copied(), model.get(key)),
                        2 => assert_eq!(cache.remove(&key), model.remove(key)),
                        _ => unreachable!(),
                    }

                    assert_matches_model(&cache, &model);
                }
            }
        }
    }

    #[test]
    fn zero_capacity_never_retains_entries() {
        let mut cache = LruCache::new(0);
        assert_eq!(cache.insert(1, 10), None);
        assert!(cache.is_empty());
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.iter().next(), None);
    }
}
