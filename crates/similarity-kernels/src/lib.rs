//! Deterministic exact similarity and sorted-set kernels.
//!
//! This crate owns reusable mechanics only. Tokenization, normalization, query
//! interpretation, ranking policy, and approximate-index policy remain with
//! callers.

mod fingerprint;
mod fuzzy;
mod shingling;

use std::cmp::Ordering;

pub use fingerprint::{
    MinHashError, MinHashSignature, hamming_distance64, minhash_jaccard_estimate,
    minhash_signature, simhash64, simhash64_weighted,
};
pub use fuzzy::{BkMatch, BkSearchReport, BkTree, MyersError, myers_levenshtein_bytes};
pub use shingling::{RollingHashes, Shingles, rolling_hashes, shingles};

/// Computes Levenshtein edit distance between two generic sequences.
///
/// Insertions, deletions, and substitutions each cost one. The implementation
/// keeps only two rows whose width is the shorter input, so auxiliary memory is
/// `O(min(left.len(), right.len()))`.
#[must_use]
pub fn levenshtein<T: Eq>(left: &[T], right: &[T]) -> usize {
    if left.is_empty() {
        return right.len();
    }
    if right.is_empty() {
        return left.len();
    }

    let (shorter, longer) = if left.len() <= right.len() {
        (left, right)
    } else {
        (right, left)
    };

    let mut previous = (0..=shorter.len()).collect::<Vec<_>>();
    let mut current = vec![0_usize; shorter.len() + 1];

    for (long_index, long_value) in longer.iter().enumerate() {
        current[0] = long_index + 1;
        for (short_index, short_value) in shorter.iter().enumerate() {
            let substitution = previous[short_index] + usize::from(long_value != short_value);
            let deletion = previous[short_index + 1] + 1;
            let insertion = current[short_index] + 1;
            current[short_index + 1] = substitution.min(deletion).min(insertion);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[shorter.len()]
}

/// Returns the union of two strictly sorted, duplicate-free slices.
///
/// Inputs are assumed to be sorted-unique already. This function deliberately
/// does not sort, deduplicate, or otherwise normalize caller-owned data.
#[must_use]
pub fn sorted_unique_union<T: Clone + Ord>(left: &[T], right: &[T]) -> Vec<T> {
    let mut output = Vec::with_capacity(left.len() + right.len());
    let mut left_index = 0_usize;
    let mut right_index = 0_usize;

    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            Ordering::Less => {
                output.push(left[left_index].clone());
                left_index += 1;
            }
            Ordering::Greater => {
                output.push(right[right_index].clone());
                right_index += 1;
            }
            Ordering::Equal => {
                output.push(left[left_index].clone());
                left_index += 1;
                right_index += 1;
            }
        }
    }

    output.extend(left[left_index..].iter().cloned());
    output.extend(right[right_index..].iter().cloned());
    output
}

/// Returns the intersection of two strictly sorted, duplicate-free slices.
#[must_use]
pub fn sorted_unique_intersection<T: Clone + Ord>(left: &[T], right: &[T]) -> Vec<T> {
    let mut output = Vec::with_capacity(left.len().min(right.len()));
    let mut left_index = 0_usize;
    let mut right_index = 0_usize;

    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            Ordering::Less => left_index += 1,
            Ordering::Greater => right_index += 1,
            Ordering::Equal => {
                output.push(left[left_index].clone());
                left_index += 1;
                right_index += 1;
            }
        }
    }

    output
}

/// Returns values present in `left` but absent from `right`.
///
/// Both inputs must be strictly sorted and duplicate-free.
#[must_use]
pub fn sorted_unique_difference<T: Clone + Ord>(left: &[T], right: &[T]) -> Vec<T> {
    let mut output = Vec::with_capacity(left.len());
    let mut left_index = 0_usize;
    let mut right_index = 0_usize;

    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            Ordering::Less => {
                output.push(left[left_index].clone());
                left_index += 1;
            }
            Ordering::Greater => right_index += 1,
            Ordering::Equal => {
                left_index += 1;
                right_index += 1;
            }
        }
    }

    output.extend(left[left_index..].iter().cloned());
    output
}

/// Returns values present in exactly one of the two sorted-unique inputs.
#[must_use]
pub fn sorted_unique_symmetric_difference<T: Clone + Ord>(left: &[T], right: &[T]) -> Vec<T> {
    let mut output = Vec::with_capacity(left.len() + right.len());
    let mut left_index = 0_usize;
    let mut right_index = 0_usize;

    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            Ordering::Less => {
                output.push(left[left_index].clone());
                left_index += 1;
            }
            Ordering::Greater => {
                output.push(right[right_index].clone());
                right_index += 1;
            }
            Ordering::Equal => {
                left_index += 1;
                right_index += 1;
            }
        }
    }

    output.extend(left[left_index..].iter().cloned());
    output.extend(right[right_index..].iter().cloned());
    output
}

/// Counts the intersection of two strictly sorted, duplicate-free slices
/// without allocating an output collection.
#[must_use]
pub fn sorted_unique_intersection_count<T: Ord>(left: &[T], right: &[T]) -> usize {
    let mut count = 0_usize;
    let mut left_index = 0_usize;
    let mut right_index = 0_usize;

    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            Ordering::Less => left_index += 1,
            Ordering::Greater => right_index += 1,
            Ordering::Equal => {
                count += 1;
                left_index += 1;
                right_index += 1;
            }
        }
    }

    count
}

/// Counts the union of two strictly sorted, duplicate-free slices without
/// allocating an output collection.
#[must_use]
pub fn sorted_unique_union_count<T: Ord>(left: &[T], right: &[T]) -> usize {
    let mut count = 0_usize;
    let mut left_index = 0_usize;
    let mut right_index = 0_usize;

    while left_index < left.len() && right_index < right.len() {
        count += 1;
        match left[left_index].cmp(&right[right_index]) {
            Ordering::Less => left_index += 1,
            Ordering::Greater => right_index += 1,
            Ordering::Equal => {
                left_index += 1;
                right_index += 1;
            }
        }
    }

    count += left.len() - left_index;
    count += right.len() - right_index;
    count
}

/// Computes exact Jaccard similarity for two sorted-unique sets.
///
/// Two empty sets are defined as identical and therefore have similarity `1.0`.
#[must_use]
pub fn jaccard_similarity_sorted_unique<T: Ord>(left: &[T], right: &[T]) -> f64 {
    let union = sorted_unique_union_count(left, right);
    if union == 0 {
        return 1.0;
    }

    let intersection = sorted_unique_intersection_count(left, right);
    intersection as f64 / union as f64
}

/// Computes Jaccard distance (`1 - similarity`) for two sorted-unique sets.
#[must_use]
pub fn jaccard_distance_sorted_unique<T: Ord>(left: &[T], right: &[T]) -> f64 {
    1.0 - jaccard_similarity_sorted_unique(left, right)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{
        jaccard_distance_sorted_unique, jaccard_similarity_sorted_unique, levenshtein,
        sorted_unique_difference, sorted_unique_intersection, sorted_unique_intersection_count,
        sorted_unique_symmetric_difference, sorted_unique_union, sorted_unique_union_count,
    };

    #[test]
    fn levenshtein_matches_known_examples_and_generic_sequences() {
        assert_eq!(levenshtein(b"kitten", b"sitting"), 3);
        assert_eq!(levenshtein::<u8>(&[], b"abc"), 3);
        assert_eq!(levenshtein(&[1, 2, 3, 4], &[1, 3, 4, 5]), 2);
    }

    #[test]
    fn levenshtein_matches_full_matrix_oracle_for_small_sequences() {
        for left_len in 0_usize..=4 {
            for left_case in 0_usize..3_usize.pow(left_len as u32) {
                let left = ternary_sequence(left_case, left_len);
                for right_len in 0_usize..=4 {
                    for right_case in 0_usize..3_usize.pow(right_len as u32) {
                        let right = ternary_sequence(right_case, right_len);
                        assert_eq!(
                            levenshtein(&left, &right),
                            reference_levenshtein(&left, &right),
                            "left={left:?}, right={right:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn levenshtein_is_symmetric() {
        let fixtures = [
            (b"".as_slice(), b"abc".as_slice()),
            (b"search".as_slice(), b"research".as_slice()),
            (b"distance".as_slice(), b"instance".as_slice()),
        ];
        for (left, right) in fixtures {
            assert_eq!(levenshtein(left, right), levenshtein(right, left));
        }
    }

    #[test]
    fn sorted_set_operations_match_btree_set_oracles_exhaustively() {
        for left_mask in 0_u8..64 {
            let left = set_from_mask(left_mask);
            let left_set = left.iter().copied().collect::<BTreeSet<_>>();
            for right_mask in 0_u8..64 {
                let right = set_from_mask(right_mask);
                let right_set = right.iter().copied().collect::<BTreeSet<_>>();

                let union = left_set.union(&right_set).copied().collect::<Vec<_>>();
                let intersection = left_set
                    .intersection(&right_set)
                    .copied()
                    .collect::<Vec<_>>();
                let difference = left_set.difference(&right_set).copied().collect::<Vec<_>>();
                let symmetric = left_set
                    .symmetric_difference(&right_set)
                    .copied()
                    .collect::<Vec<_>>();

                assert_eq!(sorted_unique_union(&left, &right), union);
                assert_eq!(sorted_unique_intersection(&left, &right), intersection);
                assert_eq!(sorted_unique_difference(&left, &right), difference);
                assert_eq!(sorted_unique_symmetric_difference(&left, &right), symmetric);
                assert_eq!(
                    sorted_unique_intersection_count(&left, &right),
                    intersection.len()
                );
                assert_eq!(sorted_unique_union_count(&left, &right), union.len());
            }
        }
    }

    #[test]
    fn jaccard_uses_exact_set_counts_and_explicit_empty_semantics() {
        assert_eq!(jaccard_similarity_sorted_unique::<u8>(&[], &[]), 1.0);
        assert_eq!(jaccard_distance_sorted_unique::<u8>(&[], &[]), 0.0);
        assert_eq!(
            jaccard_similarity_sorted_unique(&[1, 2], &[2, 3]),
            1.0 / 3.0
        );
        assert_eq!(
            jaccard_distance_sorted_unique(&[1, 2], &[2, 3]),
            1.0 - 1.0 / 3.0
        );
    }

    #[test]
    fn jaccard_matches_btree_set_counts_for_all_small_sets() {
        for left_mask in 0_u8..64 {
            let left = set_from_mask(left_mask);
            let left_set = left.iter().copied().collect::<BTreeSet<_>>();
            for right_mask in 0_u8..64 {
                let right = set_from_mask(right_mask);
                let right_set = right.iter().copied().collect::<BTreeSet<_>>();
                let union = left_set.union(&right_set).count();
                let intersection = left_set.intersection(&right_set).count();
                let expected = if union == 0 {
                    1.0
                } else {
                    intersection as f64 / union as f64
                };

                assert_eq!(jaccard_similarity_sorted_unique(&left, &right), expected);
                assert_eq!(
                    jaccard_similarity_sorted_unique(&left, &right),
                    jaccard_similarity_sorted_unique(&right, &left)
                );
            }
        }
    }

    fn ternary_sequence(mut encoded: usize, len: usize) -> Vec<u8> {
        let mut output = Vec::with_capacity(len);
        for _ in 0..len {
            output.push((encoded % 3) as u8);
            encoded /= 3;
        }
        output
    }

    fn reference_levenshtein<T: Eq>(left: &[T], right: &[T]) -> usize {
        let width = right.len() + 1;
        let mut matrix = vec![0_usize; (left.len() + 1) * width];
        for left_index in 0..=left.len() {
            matrix[left_index * width] = left_index;
        }
        for (right_index, cell) in matrix.iter_mut().take(width).enumerate() {
            *cell = right_index;
        }

        for left_index in 1..=left.len() {
            for right_index in 1..=right.len() {
                let substitution = matrix[(left_index - 1) * width + right_index - 1]
                    + usize::from(left[left_index - 1] != right[right_index - 1]);
                let deletion = matrix[(left_index - 1) * width + right_index] + 1;
                let insertion = matrix[left_index * width + right_index - 1] + 1;
                matrix[left_index * width + right_index] =
                    substitution.min(deletion).min(insertion);
            }
        }

        matrix[left.len() * width + right.len()]
    }

    fn set_from_mask(mask: u8) -> Vec<u8> {
        (0_u8..6).filter(|value| mask & (1 << value) != 0).collect()
    }
}
