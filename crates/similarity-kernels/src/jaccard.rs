use core::cmp::Ordering;

/// Computes Jaccard similarity for two sorted-unique slices.
///
/// Inputs must be strictly increasing according to `Ord`. Similarity is
/// `|A ∩ B| / |A ∪ B|`; two empty sets have similarity `1.0`.
pub fn jaccard_similarity<T: Ord>(a: &[T], b: &[T]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }

    let (intersection, union) = intersection_and_union_counts(a, b);
    intersection as f64 / union as f64
}

/// Computes Jaccard distance (`1 - similarity`) for sorted-unique slices.
pub fn jaccard_distance<T: Ord>(a: &[T], b: &[T]) -> f64 {
    1.0 - jaccard_similarity(a, b)
}

fn intersection_and_union_counts<T: Ord>(a: &[T], b: &[T]) -> (usize, usize) {
    let mut i = 0;
    let mut j = 0;
    let mut intersection = 0;
    let mut union = 0;

    while i < a.len() && j < b.len() {
        union += 1;
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                intersection += 1;
                i += 1;
                j += 1;
            }
        }
    }

    union += (a.len() - i) + (b.len() - j);
    (intersection, union)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{jaccard_distance, jaccard_similarity};

    fn reference(a: &[u8], b: &[u8]) -> f64 {
        let a: BTreeSet<_> = a.iter().copied().collect();
        let b: BTreeSet<_> = b.iter().copied().collect();
        let union = a.union(&b).count();
        if union == 0 {
            1.0
        } else {
            a.intersection(&b).count() as f64 / union as f64
        }
    }

    #[test]
    fn defines_empty_set_semantics() {
        assert_eq!(jaccard_similarity::<u8>(&[], &[]), 1.0);
        assert_eq!(jaccard_distance::<u8>(&[], &[]), 0.0);
        assert_eq!(jaccard_similarity(&[], &[1]), 0.0);
        assert_eq!(jaccard_distance(&[], &[1]), 1.0);
    }

    #[test]
    fn handles_identical_disjoint_and_partial_sets() {
        assert_eq!(jaccard_similarity(&[1, 2, 3], &[1, 2, 3]), 1.0);
        assert_eq!(jaccard_similarity(&[1, 2], &[3, 4]), 0.0);
        assert_eq!(jaccard_similarity(&[1, 2, 4], &[2, 3, 4]), 0.5);
        assert_eq!(jaccard_distance(&[1, 2, 4], &[2, 3, 4]), 0.5);
    }

    #[test]
    fn is_symmetric() {
        let a = [1, 3, 4, 8];
        let b = [1, 2, 4, 9];
        assert_eq!(jaccard_similarity(&a, &b), jaccard_similarity(&b, &a));
    }

    #[test]
    fn matches_btree_set_reference() {
        for mask_a in 0u16..128 {
            for mask_b in 0u16..128 {
                let a: Vec<u8> = (0..7)
                    .filter(|bit| mask_a & (1 << bit) != 0)
                    .collect();
                let b: Vec<u8> = (0..7)
                    .filter(|bit| mask_b & (1 << bit) != 0)
                    .collect();
                assert_eq!(jaccard_similarity(&a, &b), reference(&a, &b));
            }
        }
    }
}
