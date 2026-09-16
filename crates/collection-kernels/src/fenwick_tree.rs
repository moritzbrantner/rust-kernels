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

    /// Builds the tree in the same addition order as repeated point updates.
    ///
    /// This preserves the historical construction semantics for values whose
    /// addition is not associative, including floating-point values. The
    /// construction cost is O(n log n). Use [`Self::from_slice_linear`] only
    /// when regrouping additions is acceptable for the value type and caller.
    #[must_use]
    pub fn from_slice(values: &[T]) -> Self {
        let mut tree = Self::new(values.len());
        for (index, &value) in values.iter().enumerate() {
            tree.add(index, value);
        }
        tree
    }

    /// Builds the tree in O(n) by regrouping additions along Fenwick parents.
    ///
    /// Regrouping can change observable results for non-associative arithmetic
    /// such as floating-point addition. Callers that require the exact ordered
    /// point-update semantics should use [`Self::from_slice`] instead.
    #[must_use]
    pub fn from_slice_linear(values: &[T]) -> Self {
        let mut tree = vec![T::default(); values.len() + 1];
        for (zero_index, &value) in values.iter().enumerate() {
            let index = zero_index + 1;
            tree[index] += value;

            let parent = index + lowbit(index);
            if parent < tree.len() {
                let subtotal = tree[index];
                tree[parent] += subtotal;
            }
        }

        Self { tree }
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
            cursor += lowbit(cursor);
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

fn lowbit(index: usize) -> usize {
    debug_assert!(index > 0);
    index & index.wrapping_neg()
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        ops::{AddAssign, Sub},
    };

    use super::{FenwickTree, lowbit};

    thread_local! {
        static ADD_ASSIGN_COUNT: Cell<usize> = const { Cell::new(0) };
    }

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    struct CountingValue(i64);

    impl AddAssign for CountingValue {
        fn add_assign(&mut self, rhs: Self) {
            ADD_ASSIGN_COUNT.with(|count| count.set(count.get() + 1));
            self.0 += rhs.0;
        }
    }

    impl Sub for CountingValue {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 - rhs.0)
        }
    }

    fn reset_add_assign_count() {
        ADD_ASSIGN_COUNT.with(|count| count.set(0));
    }

    fn add_assign_count() -> usize {
        ADD_ASSIGN_COUNT.with(Cell::get)
    }

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
    fn construction_and_ranges_match_naive_oracle_for_all_small_ternary_inputs() {
        for len in 0_usize..=6 {
            for case in 0..3_usize.pow(len as u32) {
                let mut encoded = case;
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push((encoded % 3) as i64 - 1);
                    encoded /= 3;
                }

                let tree = FenwickTree::from_slice(&values);
                assert_eq!(tree.len(), len, "len={len}, case={case}");

                for end in 0..=len {
                    let expected_prefix = values.iter().take(end).sum::<i64>();
                    assert_eq!(
                        tree.prefix_sum(end),
                        expected_prefix,
                        "prefix len={len}, case={case}, end={end}, values={values:?}"
                    );

                    for start in 0..=end {
                        let expected_range =
                            values.iter().skip(start).take(end - start).sum::<i64>();
                        assert_eq!(
                            tree.range_sum(start..end),
                            expected_range,
                            "range len={len}, case={case}, start={start}, end={end}, values={values:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn linear_construction_matches_ordered_updates_for_associative_fixture() {
        let values = (0_i64..257)
            .map(|value| (value * 17) % 31 - 15)
            .collect::<Vec<_>>();
        let ordered = FenwickTree::from_slice(&values);
        let linear = FenwickTree::from_slice_linear(&values);

        assert_eq!(linear, ordered);
    }

    #[test]
    fn ordered_construction_preserves_floating_point_update_order() {
        let values = [1e20_f64, 0.0, -1e20, 1.0];
        let ordered = FenwickTree::from_slice(&values);
        let mut incremental = FenwickTree::new(values.len());
        for (index, value) in values.into_iter().enumerate() {
            incremental.add(index, value);
        }

        assert_eq!(ordered.prefix_sum(values.len()), 1.0);
        assert_eq!(
            ordered.prefix_sum(values.len()),
            incremental.prefix_sum(values.len())
        );
    }

    #[test]
    fn linear_construction_uses_bounded_addition_work() {
        const LEN: usize = 64;
        let values = vec![CountingValue(1); LEN];

        reset_add_assign_count();
        let linear = FenwickTree::from_slice_linear(&values);
        let linear_adds = add_assign_count();

        reset_add_assign_count();
        let ordered = FenwickTree::from_slice(&values);
        let ordered_adds = add_assign_count();

        let expected_ordered_adds = (1..=LEN)
            .map(|mut cursor| {
                let mut count = 0;
                while cursor <= LEN {
                    count += 1;
                    cursor += lowbit(cursor);
                }
                count
            })
            .sum::<usize>();

        assert_eq!(linear, ordered);
        assert_eq!(linear_adds, 2 * LEN - 1);
        assert_eq!(ordered_adds, expected_ordered_adds);
        assert!(linear_adds < ordered_adds);
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
