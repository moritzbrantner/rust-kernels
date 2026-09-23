/// Reusable banded edit-distance scratch space. It retains no input values or
/// references, and can be reused across element types and different limits.
#[derive(Debug, Default)]
pub struct LevenshteinWorkspace {
    previous: Vec<usize>,
    current: Vec<usize>,
}

impl LevenshteinWorkspace {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            previous: Vec::new(),
            current: Vec::new(),
        }
    }

    /// Exact unit-cost Levenshtein distance when it is at most `limit`, otherwise
    /// `None`. An exceeded limit is not an approximate distance or a metric.
    ///
    /// Length rejection and common affix trimming happen before allocating.
    /// After trimming, evaluates only the diagonal band of width `2*limit+1`,
    /// stops when no row can meet the limit, and reuses two O(min(n, limit)) rows.
    /// Time is O(n+m + max(n,m)*min(min(n,m), limit+1)); no input is cloned.
    /// Retained capacity follows the largest previous query, not the latest one.
    /// A limit of `usize::MAX` means unrestricted exact distance, not overflow.
    pub fn distance<T: Eq>(&mut self, left: &[T], right: &[T], limit: usize) -> Option<usize> {
        if left.len().abs_diff(right.len()) > limit {
            return None;
        }
        let prefix = left.iter().zip(right).take_while(|(a, b)| a == b).count();
        let (left, right) = (&left[prefix..], &right[prefix..]);
        let suffix = left
            .iter()
            .rev()
            .zip(right.iter().rev())
            .take_while(|(a, b)| a == b)
            .count();
        let (left, right) = (&left[..left.len() - suffix], &right[..right.len() - suffix]);
        let (short, long) = if left.len() <= right.len() {
            (left, right)
        } else {
            (right, left)
        };
        if short.is_empty() {
            return (long.len() <= limit).then_some(long.len());
        }
        if limit == 0 {
            return None; // The common prefix did not cover both inputs.
        }
        let k = limit.min(long.len());
        let infinity = k.saturating_add(1);
        self.previous.clear();
        self.previous.extend(0..=short.len().min(k));
        let mut previous_start = 0;
        for (row, value) in long.iter().enumerate() {
            let i = row + 1;
            let start = i.saturating_sub(k);
            let end = short.len().min(i.saturating_add(k));
            if start > end {
                return None;
            }
            self.current.resize(end - start + 1, infinity);
            let mut minimum = infinity;
            for j in start..=end {
                let at = |column: usize| {
                    column
                        .checked_sub(previous_start)
                        .and_then(|index| self.previous.get(index))
                        .copied()
                        .unwrap_or(infinity)
                };
                let cost = if j == 0 {
                    i
                } else {
                    let substitution =
                        at(j - 1).saturating_add(usize::from(value != &short[j - 1]));
                    let deletion = at(j).saturating_add(1);
                    let insertion = if j > start {
                        self.current[j - start - 1].saturating_add(1)
                    } else {
                        infinity
                    };
                    substitution.min(deletion).min(insertion).min(infinity)
                };
                self.current[j - start] = cost;
                minimum = minimum.min(cost);
            }
            if minimum > k {
                return None;
            }
            std::mem::swap(&mut self.previous, &mut self.current);
            previous_start = start;
        }
        let distance = self.previous[short.len() - previous_start];
        (distance <= limit).then_some(distance)
    }
}

/// Exact edit distance up to a caller-selected limit. See
/// [`LevenshteinWorkspace::distance`] for complexity and scratch reuse.
///
/// ```
/// use similarity_kernels::{levenshtein_bounded, LevenshteinWorkspace};
/// assert_eq!(levenshtein_bounded(b"kitten", b"sitting", 2), None);
/// let mut workspace = LevenshteinWorkspace::new();
/// assert_eq!(workspace.distance(b"kitten", b"sitting", 3), Some(3));
/// ```
#[must_use]
pub fn levenshtein_bounded<T: Eq>(left: &[T], right: &[T], limit: usize) -> Option<usize> {
    LevenshteinWorkspace::new().distance(left, right, limit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::levenshtein;

    fn words() -> Vec<Vec<u8>> {
        (0..=5)
            .flat_map(|n| {
                (0..3_usize.pow(n)).map(move |mut value| {
                    (0..n)
                        .map(|_| {
                            let digit = (value % 3) as u8;
                            value /= 3;
                            digit
                        })
                        .collect()
                })
            })
            .collect()
    }

    #[test]
    fn exhaustive_thresholds_match_unrestricted_oracle_with_reused_dirty_rows() {
        let mut scratch = LevenshteinWorkspace::new();
        let words = words();
        for a in &words {
            for b in &words {
                let exact = levenshtein(a, b);
                for k in [0, 1, 2, 4, usize::MAX] {
                    assert_eq!(
                        scratch.distance(a, b, k),
                        (exact <= k).then_some(exact),
                        "a={a:?}, b={b:?}, k={k}"
                    );
                }
            }
        }
    }

    #[test]
    fn band_boundaries_long_affixes_and_generic_tokens() {
        for (a, b) in [
            ("kitten", "sitting"),
            ("", "abc"),
            ("a", "b"),
            ("abcdef", "bcdefa"),
        ] {
            for k in 0..=7 {
                let d = levenshtein(a.as_bytes(), b.as_bytes());
                assert_eq!(
                    levenshtein_bounded(a.as_bytes(), b.as_bytes(), k),
                    (d <= k).then_some(d)
                );
            }
        }
        let mut scratch = LevenshteinWorkspace::new();
        assert_eq!(
            scratch.distance(&["foo", "bar", "baz"], &["foo", "baz"], 1),
            Some(1)
        );
        let a = [vec![0; 4096], vec![1, 2, 3], vec![0; 4096]].concat();
        let b = [vec![0; 4096], vec![4, 2, 3], vec![0; 4096]].concat();
        assert_eq!(scratch.distance(&a, &b, 1), Some(1));
        assert_eq!(scratch.distance(&a, &b, 0), None);
        assert_eq!(scratch.distance(&b, &a, usize::MAX), Some(1));
    }

    #[test]
    fn narrow_band_work_is_linear_and_reuse_does_not_grow_storage() {
        use std::cell::Cell;
        struct Counted<'a>(usize, &'a Cell<usize>);
        impl PartialEq for Counted<'_> {
            fn eq(&self, other: &Self) -> bool {
                self.1.set(self.1.get() + 1);
                self.0 == other.0
            }
        }
        impl Eq for Counted<'_> {}
        let count = Cell::new(0);
        let mut scratch = LevenshteinWorkspace::new();
        for n in [128, 1024, 8192] {
            let a: Vec<_> = (0..n).map(|i| Counted(i % 23, &count)).collect();
            let b: Vec<_> = (0..n).map(|i| Counted((i + 1) % 23, &count)).collect();
            count.set(0);
            assert_eq!(scratch.distance(&a, &b, 2), Some(2));
            assert!(
                count.get() <= 5 * n + 4,
                "n={n}, equality calls={}",
                count.get()
            );
            assert!(scratch.previous.capacity() + scratch.current.capacity() <= 32);
            let pointers = [scratch.previous.as_ptr(), scratch.current.as_ptr()];
            assert_eq!(scratch.distance(&a, &b, 2), Some(2));
            assert!(pointers.contains(&scratch.previous.as_ptr()));
            assert!(pointers.contains(&scratch.current.as_ptr()));
        }
        count.set(0);
        let a = [Counted(1, &count)];
        let b: Vec<_> = (0..128).map(|i| Counted(i, &count)).collect();
        assert_eq!(scratch.distance(&a, &b, 2), None);
        assert_eq!(count.get(), 0, "length rejection must precede comparisons");
    }
}
