use std::collections::{BTreeMap, VecDeque};
use std::fmt;

use crate::levenshtein;

const MYERS_MAX_PATTERN_LEN: usize = u64::BITS as usize;

/// Metric BK-tree over caller-owned values.
///
/// Construction and search must use the same distance semantics. Correct
/// pruning additionally requires a true metric: identity, symmetry, and the
/// triangle inequality. The tree deliberately does not own a distance closure
/// so storage remains independent from application policy.
#[derive(Clone, Debug)]
pub struct BkTree<T> {
    nodes: Vec<BkNode<T>>,
}

#[derive(Clone, Debug)]
struct BkNode<T> {
    value: T,
    children: BTreeMap<usize, usize>,
}

impl<T> Default for BkTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> BkTree<T> {
    /// Creates an empty BK-tree.
    #[must_use]
    pub const fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Number of values stored in insertion order.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns whether the tree contains no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Inserts one value using the caller-supplied metric distance.
    ///
    /// Duplicate values are allowed. They form deterministic zero-distance
    /// descendants rather than being silently deduplicated.
    pub fn insert<Distance>(&mut self, value: T, mut distance: Distance)
    where
        Distance: FnMut(&T, &T) -> usize,
    {
        if self.nodes.is_empty() {
            self.nodes.push(BkNode {
                value,
                children: BTreeMap::new(),
            });
            return;
        }

        let mut current = 0_usize;
        loop {
            let edge = distance(&value, &self.nodes[current].value);
            if let Some(next) = self.nodes[current].children.get(&edge).copied() {
                current = next;
                continue;
            }

            let child = self.nodes.len();
            self.nodes.push(BkNode {
                value,
                children: BTreeMap::new(),
            });
            self.nodes[current].children.insert(edge, child);
            return;
        }
    }

    /// Performs radius search with metric-distance pruning.
    ///
    /// Matches are ordered by ascending distance, then insertion order. The
    /// report exposes distance-evaluation count separately from metric cost so
    /// consumers can measure index pruning without pretending all metrics cost
    /// the same amount.
    #[must_use]
    pub fn search<'a, Distance>(
        &'a self,
        query: &T,
        radius: usize,
        mut distance: Distance,
    ) -> BkSearchReport<'a, T>
    where
        Distance: FnMut(&T, &T) -> usize,
    {
        if self.nodes.is_empty() {
            return BkSearchReport {
                matches: Vec::new(),
                distance_evaluations: 0,
            };
        }

        let mut queue = VecDeque::from([0_usize]);
        let mut matched = Vec::new();
        let mut distance_evaluations = 0_usize;

        while let Some(index) = queue.pop_front() {
            let node = &self.nodes[index];
            let query_distance = distance(query, &node.value);
            distance_evaluations += 1;

            if query_distance <= radius {
                matched.push((query_distance, index));
            }

            let lower = query_distance.saturating_sub(radius);
            let upper = query_distance.saturating_add(radius);
            for child in node.children.range(lower..=upper).map(|(_, child)| *child) {
                queue.push_back(child);
            }
        }

        matched.sort_unstable();
        let matches = matched
            .into_iter()
            .map(|(distance, index)| BkMatch {
                distance,
                value: &self.nodes[index].value,
            })
            .collect();

        BkSearchReport {
            matches,
            distance_evaluations,
        }
    }
}

/// One deterministic BK-tree radius-search match.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BkMatch<'a, T> {
    /// Exact caller-supplied metric distance from the query.
    pub distance: usize,
    /// Borrowed value stored in the tree.
    pub value: &'a T,
}

/// Radius-search result plus explicit pruning evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BkSearchReport<'a, T> {
    /// Matches ordered by distance, then insertion order.
    pub matches: Vec<BkMatch<'a, T>>,
    /// Number of stored values for which the metric was actually evaluated.
    pub distance_evaluations: usize,
}

/// Error returned when the bit-parallel Myers boundary is unsupported.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MyersError {
    /// The pattern is longer than the single-word implementation can represent.
    PatternTooLong { length: usize, maximum: usize },
}

impl fmt::Display for MyersError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PatternTooLong { length, maximum } => write!(
                formatter,
                "Myers byte pattern length {length} exceeds the supported maximum {maximum}"
            ),
        }
    }
}

impl std::error::Error for MyersError {}

/// Computes byte-oriented Levenshtein distance with Myers' bit-parallel method.
///
/// Patterns up to 64 bytes are supported. Longer patterns return an explicit
/// error instead of silently falling back to another algorithm. This function
/// operates on raw bytes; Unicode normalization, grapheme segmentation, and
/// tokenization remain caller policy. Use [`levenshtein`] when a generic
/// sequence boundary or an unrestricted pattern length is required.
pub fn myers_levenshtein_bytes(pattern: &[u8], text: &[u8]) -> Result<usize, MyersError> {
    if pattern.len() > MYERS_MAX_PATTERN_LEN {
        return Err(MyersError::PatternTooLong {
            length: pattern.len(),
            maximum: MYERS_MAX_PATTERN_LEN,
        });
    }
    if pattern.is_empty() {
        return Ok(text.len());
    }

    let mut equality = [0_u64; 256];
    for (index, byte) in pattern.iter().copied().enumerate() {
        equality[usize::from(byte)] |= 1_u64 << index;
    }

    let mut positive_vertical = !0_u64;
    let mut negative_vertical = 0_u64;
    let mut score = pattern.len();
    let high_bit = 1_u64 << (pattern.len() - 1);

    for byte in text.iter().copied() {
        let equal = equality[usize::from(byte)];
        let vertical_or_equal = equal | negative_vertical;
        let horizontal = (((equal & positive_vertical).wrapping_add(positive_vertical))
            ^ positive_vertical)
            | equal;
        let positive_horizontal = negative_vertical | !(horizontal | positive_vertical);
        let negative_horizontal = positive_vertical & horizontal;

        if positive_horizontal & high_bit != 0 {
            score += 1;
        }
        if negative_horizontal & high_bit != 0 {
            score = score
                .checked_sub(1)
                .expect("Myers distance invariant prevents a negative score");
        }

        let shifted_positive = (positive_horizontal << 1) | 1;
        let shifted_negative = negative_horizontal << 1;
        positive_vertical = shifted_negative | !(vertical_or_equal | shifted_positive);
        negative_vertical = shifted_positive & vertical_or_equal;
    }

    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::{BkTree, MyersError, myers_levenshtein_bytes};
    use crate::levenshtein;

    #[test]
    fn bk_tree_matches_exhaustive_linear_radius_search() {
        let values = all_fixed_width_sequences(4);
        let mut tree = BkTree::new();
        for value in values.iter().cloned() {
            tree.insert(value, |left, right| levenshtein(left, right));
        }

        for query in &values {
            for radius in 0..=2 {
                let report = tree.search(query, radius, |left, right| levenshtein(left, right));
                let mut expected = values
                    .iter()
                    .enumerate()
                    .filter_map(|(index, value)| {
                        let distance = levenshtein(query, value);
                        (distance <= radius).then_some((distance, index, value.clone()))
                    })
                    .collect::<Vec<_>>();
                expected.sort_by_key(|(distance, index, _)| (*distance, *index));

                let actual = report
                    .matches
                    .iter()
                    .map(|matched| (matched.distance, matched.value.clone()))
                    .collect::<Vec<_>>();
                let expected = expected
                    .into_iter()
                    .map(|(distance, _, value)| (distance, value))
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "query={query:?}, radius={radius}");
                assert!(report.distance_evaluations <= tree.len());
            }
        }
    }

    #[test]
    fn bk_tree_preserves_duplicate_insertion_order() {
        let mut tree = BkTree::new();
        for value in [b"cat".to_vec(), b"cat".to_vec(), b"bat".to_vec()] {
            tree.insert(value, |left, right| levenshtein(left, right));
        }

        let report = tree.search(&b"cat".to_vec(), 0, |left, right| levenshtein(left, right));
        assert_eq!(report.matches.len(), 2);
        assert!(report.matches.iter().all(|matched| matched.distance == 0));
        assert!(report
            .matches
            .iter()
            .all(|matched| matched.value.as_slice() == b"cat"));
    }

    #[test]
    fn bk_tree_exposes_candidate_pruning_separately_from_metric_cost() {
        let mut tree = BkTree::new();
        for value in std::iter::once(50_i32).chain(0..=100).filter(|value| *value != 50) {
            tree.insert(value, |left, right| left.abs_diff(*right) as usize);
        }

        let report = tree.search(&50, 0, |left, right| left.abs_diff(*right) as usize);
        assert_eq!(report.matches.len(), 1);
        assert_eq!(*report.matches[0].value, 50);
        assert!(report.distance_evaluations < tree.len() / 4);
    }

    #[test]
    fn myers_matches_generic_levenshtein_for_known_and_boundary_cases() {
        for (pattern, text) in [
            (b"".as_slice(), b"abc".as_slice()),
            (b"kitten".as_slice(), b"sitting".as_slice()),
            (b"same".as_slice(), b"same".as_slice()),
            (b"abc".as_slice(), b"".as_slice()),
            (&[b'x'; 64][..], &[b'x'; 63][..]),
        ] {
            assert_eq!(
                myers_levenshtein_bytes(pattern, text),
                Ok(levenshtein(pattern, text))
            );
        }
    }

    #[test]
    fn myers_matches_generic_levenshtein_exhaustively_for_small_byte_sequences() {
        for pattern_len in 0_usize..=4 {
            for pattern_case in 0_usize..3_usize.pow(pattern_len as u32) {
                let pattern = ternary_sequence(pattern_case, pattern_len);
                for text_len in 0_usize..=4 {
                    for text_case in 0_usize..3_usize.pow(text_len as u32) {
                        let text = ternary_sequence(text_case, text_len);
                        assert_eq!(
                            myers_levenshtein_bytes(&pattern, &text),
                            Ok(levenshtein(&pattern, &text)),
                            "pattern={pattern:?}, text={text:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn myers_rejects_patterns_beyond_the_single_word_boundary() {
        let pattern = vec![b'x'; 65];
        assert_eq!(
            myers_levenshtein_bytes(&pattern, b"text"),
            Err(MyersError::PatternTooLong {
                length: 65,
                maximum: 64
            })
        );
    }

    fn all_fixed_width_sequences(width: usize) -> Vec<Vec<u8>> {
        (0..3_usize.pow(width as u32))
            .map(|case| ternary_sequence(case, width))
            .collect()
    }

    fn ternary_sequence(mut encoded: usize, len: usize) -> Vec<u8> {
        let mut output = Vec::with_capacity(len);
        for _ in 0..len {
            output.push(b'a' + (encoded % 3) as u8);
            encoded /= 3;
        }
        output
    }
}
