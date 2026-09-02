use std::ops::{AddAssign, Range, Sub};

/// Fenwick tree (binary indexed tree) for additive prefix and range queries.
///
/// `T::default()` is treated as the additive identity. Point updates add a
/// delta to an existing position. Prefix and range queries use half-open
/// semantics: `prefix_sum(end)` covers `0..end`, and `range_sum(a..b)` covers
/// `a..b`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FenwickTree<T> {
    tree: Vec<T>,
}

impl<T> FenwickTree<T>
where
    T: Copy + Default + AddAssign + Sub<Output = T>,
{
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self {
            tree: vec![T::default(); len + 1],
        }
    }

    #[must_use]
    pub fn from_slice(values: &[T]) -> Self {
        let mut tree = Self::new(values.len());
        for (index, &value) in values.iter().enumerate() {
            tree.add(index, value);
        }
        tree
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tree.len() - 1
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds `delta` to one position in O(log n).
    pub fn add(&mut self, index: usize, delta: T) {
        assert!(index < self.len(), "Fenwick index out of bounds");
        let mut cursor = index + 1;
        while cursor < self.tree.len() {
            self.tree[cursor] += delta;
            cursor += cursor & (!cursor + 1);
        }
    }

    /// Returns the sum over `0..end` in O(log n).
    #[must_use]
    pub fn prefix_sum(&self, end: usize) -> T {
        assert!(end <= self.len(), "Fenwick prefix end out of bounds");
        let mut cursor = end;
        let mut sum = T::default();
        while cursor > 0 {
            sum += self.tree[cursor];
            cursor &= cursor - 1;
        }
        sum
    }

    /// Returns the sum over the half-open range in O(log n).
    #[must_use]
    pub fn range_sum(&self, range: Range<usize>) -> T {
        assert!(range.start <= range.end, "Fenwick range must be ordered");
        assert!(range.end <= self.len(), "Fenwick range end out of bounds");
        self.prefix_sum(range.end) - self.prefix_sum(range.start)
    }
}

#[cfg(test)]
mod tests {
    use super::FenwickTree;

    #[test]
    fn builds_and_queries_half_open_ranges() {
        let tree = FenwickTree::from_slice(&[3_i64, -1, 4, 1, 5]);
        assert_eq!(tree.prefix_sum(0), 0);
        assert_eq!(tree.prefix_sum(3), 6);
        assert_eq!(tree.range_sum(1..4), 4);
        assert_eq!(tree.range_sum(0..5), 12);
        assert_eq!(tree.range_sum(5..5), 0);
    }

    #[test]
    fn mixed_updates_match_a_naive_vector_oracle() {
        let mut values = vec![0_i64; 32];
        let mut tree = FenwickTree::new(values.len());

        let updates = [
            (0, 5),
            (31, 7),
            (12, -3),
            (7, 11),
            (12, 8),
            (1, -4),
            (30, 2),
            (7, -6),
        ];

        for (index, delta) in updates {
            values[index] += delta;
            tree.add(index, delta);

            for end in 0..=values.len() {
                assert_eq!(tree.prefix_sum(end), values[..end].iter().sum::<i64>());
            }

            for start in [0, 1, 7, 12, 30, 31, 32] {
                for end in [start, values.len()] {
                    assert_eq!(
                        tree.range_sum(start..end),
                        values[start..end].iter().sum::<i64>()
                    );
                }
            }
        }
    }

    #[test]
    fn empty_tree_supports_empty_queries() {
        let tree = FenwickTree::<i64>::new(0);
        assert!(tree.is_empty());
        assert_eq!(tree.prefix_sum(0), 0);
        assert_eq!(tree.range_sum(0..0), 0);
    }

    #[test]
    #[should_panic(expected = "Fenwick index out of bounds")]
    fn rejects_out_of_range_updates() {
        FenwickTree::<i64>::new(2).add(2, 1);
    }
}
